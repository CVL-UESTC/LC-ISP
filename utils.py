"""Command-line options and checkpoint helpers for LC-ISP."""

import argparse
from collections.abc import Mapping

import torch


class BaseOptions:
    def __init__(self, training=True):
        self.training = training

    def parse(self):
        parser = argparse.ArgumentParser(description="Train or evaluate LC-ISP.")
        parser.add_argument('--config', help='Model JSON file; defaults to the checkpoint configuration, or Base for legacy/no checkpoint.')
        parser.add_argument('--dataroot', required=True, help='Dataset directory containing data/, split IDs, and metadata JSON files.')
        parser.add_argument('--ckpt', required=not self.training, help='Training checkpoint; resumes training or supplies evaluation weights.')
        parser.add_argument('--batch_size', type=int, default=6 if self.training else 1)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--bl', type=int, default=528, help='RAW black level.')
        parser.add_argument('--wl', type=int, default=4095, help='RAW white level.')
        parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
        parser.add_argument('--lpips', action='store_true', help='Also compute LPIPS (requires lpips and AlexNet weights).')
        if self.training:
            parser.add_argument('--epochs', type=int, default=150)
            parser.add_argument('--warmup_epochs', type=int, default=5)
            parser.add_argument('--lr', type=float, default=0.0001)
            parser.add_argument('--seed', type=int, default=0)
            parser.add_argument('--save_interval', type=int, default=10)
            parser.add_argument('--save_model_path', default='./checkpoints')
            parser.add_argument('--tb_log_dir', default='./logs')
            parser.add_argument('--vgg_weights', help='Optional local vgg19-dcbb9e9d.pth; otherwise use the torchvision cache/download.')
        else:
            parser.add_argument('--save_img_path', default='./results')
            parser.add_argument('--ssim', action='store_true', help='Also compute SSIM and MS-SSIM (requires pytorch-msssim).')
        opt = parser.parse_args()
        if opt.batch_size < 1 or opt.num_workers < 0:
            parser.error('batch_size must be positive and num_workers must be nonnegative.')
        if opt.wl <= opt.bl:
            parser.error('wl must be greater than bl.')
        if self.training:
            if not 0 < opt.warmup_epochs < opt.epochs:
                parser.error('epochs must be greater than warmup_epochs, which must be positive.')
            if opt.lr <= 0 or opt.save_interval < 1:
                parser.error('lr and save_interval must be positive.')
        return opt


def get_device(requested='auto', local_rank=0):
    use_cuda = requested == 'cuda' or (requested == 'auto' and torch.cuda.is_available())
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable.')
        torch.cuda.set_device(local_rank)
        return torch.device('cuda', local_rank)
    return torch.device('cpu')


def load_model_weights(model, checkpoint):
    """Accept the original DDP/DataParallel checkpoints and plain state dicts."""
    state_dict = checkpoint.get('model', checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError('Expected a model state dict or a checkpoint containing model.')
    state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def normalize_batch(batch):
    # ImageNet normalization used by the original perceptual loss.
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std
