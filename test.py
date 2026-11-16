"""Inference and evaluation on the LC-ISP test triplets."""

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_data import UltraISPDataset_pro
from model_config import resolve_model_config
from dual_stage_isp import DualStageISP
from utils import BaseOptions, get_device, load_model_weights


def test_model(opt):
    device = get_device(opt.device)
    checkpoint = torch.load(opt.ckpt, map_location=device, weights_only=True)
    model_config = resolve_model_config(opt.config, checkpoint)
    model = DualStageISP(model_config).to(device)
    load_model_weights(model, checkpoint)
    print(f'Model configuration: {model.model_config["name"]}')
    model.eval()
    torch.backends.cudnn.deterministic = True
    dataset = UltraISPDataset_pro(opt.dataroot, opt.bl, opt.wl, test=True, ret_path=True)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.num_workers, pin_memory=device.type == 'cuda')
    output_dir = Path(opt.save_img_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    calc_ssim, calc_ms_ssim, lpips_fn = None, None, None
    if opt.ssim:
        from pytorch_msssim import ssim as calc_ssim, ms_ssim as calc_ms_ssim
    if opt.lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    totals = {'psnr': 0.}
    if opt.ssim:
        totals.update(ssim=0., ms_ssim=0.)
    if opt.lpips:
        totals['lpips'] = 0.
    count, elapsed = 0, 0.
    with torch.no_grad():
        for raw, proraw, target, raw_paths in tqdm(loader):
            raw, target = raw.to(device), target.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            enhanced, restored = model(raw)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            for index, raw_path in enumerate(raw_paths):
                prediction = enhanced[index].clamp(0, 1)
                reference = target[index].clamp(0, 1)
                # Keep the original round-to-8-bit evaluation protocol.
                mse = F.mse_loss(torch.round(prediction * 255) / 255,
                                 torch.round(reference * 255) / 255).item()
                totals['psnr'] += float('inf') if mse == 0 else -10 * math.log10(mse)
                if calc_ssim is not None:
                    totals['ssim'] += calc_ssim(prediction[None], reference[None], data_range=1.).item()
                    totals['ms_ssim'] += calc_ms_ssim(prediction[None], reference[None], data_range=1.).item()
                if lpips_fn is not None:
                    totals['lpips'] += lpips_fn(prediction[None] * 2 - 1, reference[None] * 2 - 1).item()
                # Save predictions and intermediate visualizations without JPEG loss.
                stem = Path(raw_path).stem
                images = {'': prediction, '_RESTORED': restored[index],
                          '_PRORAW': proraw[index], '_GT': reference}
                for suffix, image in images.items():
                    image = image.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    image = np.round(image * 255).astype(np.uint8)[..., ::-1]
                    path = output_dir / f'{stem}{suffix}.png'
                    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                        raise OSError(f'Could not save {path}')
                count += 1
    metrics = {name: value / count for name, value in totals.items()}
    metrics.update(images=count, seconds_per_image=elapsed / count)
    # Perfect matches are printed as infinity, while JSON stays standards-compliant.
    json_metrics = {key: value if not isinstance(value, float) or math.isfinite(value) else str(value)
                    for key, value in metrics.items()}
    (output_dir / 'metrics.json').write_text(json.dumps(json_metrics, indent=2) + '\n')
    print(json.dumps(json_metrics, indent=2))


if __name__ == '__main__':
    test_model(BaseOptions(training=False).parse())
