"""Joint training of RestorationNet and EnhancementNet."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from load_data import UltraISPDataset_pro
from mssim import MSSSIM
from model_config import resolve_model_config
from dual_stage_isp import DualStageISP
from utils import BaseOptions, get_device, load_model_weights, normalize_batch
from vgg import vgg_19


def rgb_to_luma(x):
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def sobel_edges(x):
    channels = x.shape[1]
    kx = x.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    ky = x.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return (F.conv2d(x, kx, padding=1, groups=channels),
            F.conv2d(x, ky, padding=1, groups=channels))


def edge_loss_sobel(pred, target):
    gx_p, gy_p = sobel_edges(rgb_to_luma(pred))
    gx_t, gy_t = sobel_edges(rgb_to_luma(target))
    mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-6)
    mag_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-6)
    return torch.abs(mag_p - mag_t).mean()


def train_model(opt):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = get_device(opt.device, local_rank)
    if distributed:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    rank = dist.get_rank() if distributed else 0
    main_process = rank == 0

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    checkpoint = torch.load(opt.ckpt, map_location=device, weights_only=True) if opt.ckpt else None
    model_config = resolve_model_config(opt.config, checkpoint)
    generator = DualStageISP(model_config).to(device)
    if distributed:
        generator = DDP(generator, device_ids=[local_rank] if device.type == 'cuda' else None)
    model = generator.module if distributed else generator

    optimizer = Adam(generator.parameters(), lr=opt.lr)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1., total_iters=opt.warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=opt.epochs - opt.warmup_epochs, eta_min=1e-6),
        ],
        milestones=[opt.warmup_epochs],
    )
    start_epoch, best_psnr = 0, 0.
    if opt.ckpt:
        load_model_weights(model, checkpoint)
        for key in ('optimizer', 'scheduler', 'epoch'):
            if key not in checkpoint:
                raise ValueError(f'Resuming training requires {key} in the checkpoint.')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint.get('best_psnr', 0.)

    train_dataset = UltraISPDataset_pro(opt.dataroot, opt.bl, opt.wl, test=False)
    val_dataset = UltraISPDataset_pro(opt.dataroot, opt.bl, opt.wl, test=True)
    train_sampler = DistributedSampler(train_dataset, seed=opt.seed) if distributed else None
    loader_options = dict(num_workers=opt.num_workers, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                              shuffle=train_sampler is None, drop_last=True, **loader_options)
    if not len(train_loader):
        raise ValueError('The training set is too small for this batch size and number of processes.')
    # Do not pad validation indices: each sample contributes exactly once.
    if distributed:
        val_dataset = Subset(val_dataset, range(rank, len(val_dataset), world_size))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_options)

    perceptual_model = vgg_19(device, opt.vgg_weights).eval()
    ssim = MSSSIM()
    l1 = nn.L1Loss()
    lpips_fn = None
    if opt.lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    writer = None
    if main_process:
        Path(opt.save_model_path).mkdir(parents=True, exist_ok=True)
        (Path(opt.save_model_path) / 'model_config.json').write_text(json.dumps(model.model_config, indent=2) + '\n')
        writer = SummaryWriter(opt.tb_log_dir)
        print(f'Model configuration: {model.model_config["name"]}')
        print(f'Training LC-ISP on {device}; processes={world_size}; batch size per process={opt.batch_size}')
        print(f'Train samples={len(train_dataset)}; resume epoch={start_epoch}')

    for epoch in range(start_epoch, opt.epochs):
        generator.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{opt.epochs}', disable=not main_process)
        for step, (inputs, proraw, targets) in enumerate(progress):
            inputs, proraw, targets = (x.to(device, non_blocking=True) for x in (inputs, proraw, targets))
            outputs, restored = generator(inputs)
            l1_loss = l1(outputs, targets)
            vgg_loss = l1(perceptual_model(normalize_batch(outputs)), perceptual_model(normalize_batch(targets)))
            ssim_loss = 1 - ssim(outputs, targets)
            inter_loss = l1(restored, proraw)
            edge_loss = edge_loss_sobel(restored, proraw)
            vgg_inter = l1(perceptual_model(normalize_batch(restored)), perceptual_model(normalize_batch(proraw)))
            total_loss = (l1_loss + vgg_loss + 0.15 * ssim_loss
                          + 0.1 * inter_loss + 0.1 * edge_loss + 0.01 * vgg_inter)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            if main_process:
                progress.set_postfix(loss=f'{total_loss.item():.4f}')
                global_step = epoch * len(train_loader) + step
                for name, value in {'Loss': total_loss, 'L1': l1_loss, 'VGG': vgg_loss,
                                    'SSIM': ssim_loss, 'Inter': inter_loss, 'EDGE': edge_loss}.items():
                    writer.add_scalar(f'Train/{name}', value.item(), global_step)

        # Evaluate the unwrapped model so ranks can process different sample counts.
        model.eval()
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        with torch.no_grad():
            for inputs, _, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                outputs = (outputs * 255).clamp(0, 255).byte()
                targets = (targets * 255).clamp(0, 255).byte()
                mse = F.mse_loss(outputs.float() / 255, targets.float() / 255)
                totals[0] += 10 * torch.log10(1. / (mse + 1e-10))
                totals[2] += 1
                if lpips_fn is not None:
                    totals[1] += lpips_fn(outputs.float() / 127.5 - 1,
                                          targets.float() / 127.5 - 1).sum()
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[2].item() == 0:
            raise ValueError('The validation split is empty.')
        avg_psnr = (totals[0] / totals[2]).item()
        avg_lpips = (totals[1] / totals[2]).item() if lpips_fn is not None else None

        # Save scheduler state for the next epoch, including on resumed runs.
        scheduler.step()
        if main_process:
            improved = avg_psnr > best_psnr
            best_psnr = max(best_psnr, avg_psnr)
            checkpoint = {'epoch': epoch + 1, 'model': model.state_dict(),
                          'model_config': model.model_config,
                          'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                          'best_psnr': best_psnr}
            writer.add_scalar('Val/PSNR', avg_psnr, epoch)
            print(f'Epoch {epoch + 1}: PSNR={avg_psnr:.4f} dB')
            if avg_lpips is not None:
                writer.add_scalar('Val/LPIPS', avg_lpips, epoch)
                print(f'LPIPS={avg_lpips:.4f}')
            if improved:
                torch.save(checkpoint, Path(opt.save_model_path) / 'best_model.pth')
            if (epoch + 1) % opt.save_interval == 0:
                torch.save(checkpoint, Path(opt.save_model_path) / f'epoch_{epoch + 1}.pth')
            torch.save(checkpoint, Path(opt.save_model_path) / 'last_model.pth')

    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    train_model(BaseOptions(training=True).parse())
