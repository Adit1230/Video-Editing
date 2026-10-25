"""Edit-quality metrics for one or more run.py output folders.

All runs must use the same source frames. Background PSNR uses one shared mask
(union of every run's foreground mask) so runs are compared on the same pixels.

Metrics
  clip_text_target       100 * cos(CLIP(edited frame), CLIP(target prompt))  higher = matches prompt
  clip_directional       cos(edit - source image, target - source text)       higher = edit goes the right way
  clip_frame_consistency 100 * cos(CLIP(frame t), CLIP(frame t+1))            higher = steadier
  warp_error             mean |warp(edit_t) - edit_t+1| on [0,255], using source optical flow; lower = steadier
  background_psnr_db     PSNR(source, edit) outside the dilated shared mask   higher = background kept
  foreground_change      mean |edit - source| inside the undilated shared mask higher = stronger edit
  foreground_hue_deg     circular mean hue of saturated pixels inside the undilated mask (blue ~ 220-240, purple ~ 270)
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def frames(folder):
    return [np.asarray(Image.open(p).convert('RGB')) for p in sorted(Path(folder).glob('*.png'))]


def masks(run, count, size):
    files = sorted((run / 'masks').glob('*.png'))
    if len(files) != count:
        return np.zeros((count, size, size), bool)
    return np.stack([np.asarray(Image.open(p).convert('L')) >= 128 for p in files])


def warp_error(source, edited):
    errors = []
    for t in range(len(source) - 1):
        a = cv2.cvtColor(source[t], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(source[t + 1], cv2.COLOR_RGB2GRAY)
        # Flow from frame t+1 back to frame t, so edit_t can be pulled onto t+1.
        flow = cv2.calcOpticalFlowFarneback(b, a, None, .5, 3, 21, 3, 5, 1.1, 0)
        h, w = a.shape
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x, map_y = grid_x + flow[..., 0], grid_y + flow[..., 1]
        valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
        warped = cv2.remap(edited[t], map_x, map_y, cv2.INTER_LINEAR)
        diff = np.abs(warped.astype(float) - edited[t + 1].astype(float)).mean(-1)
        errors.append(diff[valid].mean())
    return float(np.mean(errors))


def hue_degrees(images, region):
    hues = []
    for image, keep in zip(images, region):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        selected = keep & (hsv[..., 1] > 60) & (hsv[..., 2] > 40)  # ignore grey and dark pixels
        hues.append(hsv[..., 0][selected].astype(float) * 2)
    hues = np.concatenate(hues) if hues else np.array([])
    if not hues.size:
        return None
    angle = np.deg2rad(hues)
    return float(np.rad2deg(math.atan2(np.sin(angle).mean(), np.cos(angle).mean())) % 360)


def main(runs, output, model_id, development_cpu=False):
    from transformers import CLIPModel, CLIPProcessor
    if not development_cpu and not torch.cuda.is_available():raise RuntimeError('Metrics require CUDA; CPU is only available with --development-cpu for tests')
    device = 'cpu' if development_cpu else 'cuda'
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def embed_images(images):
        inputs = processor(images=[Image.fromarray(x) for x in images], return_tensors='pt').to(device)
        return torch.nn.functional.normalize(model.get_image_features(**inputs).float(), dim=-1)

    @torch.no_grad()
    def embed_text(text):
        inputs = processor(text=[text], return_tensors='pt', padding=True, truncation=True).to(device)
        return torch.nn.functional.normalize(model.get_text_features(**inputs).float(), dim=-1)[0]

    runs = [Path(r) for r in runs]
    source = frames(runs[0] / 'source_frames')
    size = source[0].shape[0]
    for run in runs[1:]:
        other = frames(run / 'source_frames')
        if len(other) != len(source) or any(not np.array_equal(a, b) for a, b in zip(source, other)):
            raise ValueError(f'{run} uses different source frames from {runs[0]}')
    foreground = np.zeros((len(source), size, size), bool)
    for run in runs:
        foreground |= masks(run, len(source), size)
    core = foreground.copy()  # edit region itself, for foreground change and hue
    # Grow the shared mask so soft edit borders do not count as background.
    kernel = np.ones((15, 15), np.uint8)
    foreground = np.stack([cv2.dilate(m.astype(np.uint8), kernel) > 0 for m in foreground])
    background = ~foreground
    source_image = embed_images(source)
    results = {'shared_foreground_fraction': float(foreground.mean()), 'runs': {}}
    source_consistency = (source_image[:-1] * source_image[1:]).sum(-1).mean().item() * 100
    results['source_clip_frame_consistency'] = source_consistency
    results['source_warp_error'] = warp_error(source, source)
    for run in runs:
        config = json.loads((run / 'run_report.json').read_text())['config']
        edited = frames(run / 'edited_frames')
        edit_image = embed_images(edited)
        target_text = embed_text(config['target_prompt'])
        source_text = embed_text(config['source_prompt'])
        direction = torch.nn.functional.normalize(edit_image - source_image, dim=-1)
        text_direction = torch.nn.functional.normalize(target_text - source_text, dim=-1)
        diff = np.abs(np.stack(edited).astype(float) - np.stack(source).astype(float))
        mse = (diff ** 2).mean(-1)[background].mean() if background.any() else float('nan')
        results['runs'][run.name] = {
            'clip_text_target': (edit_image @ target_text).mean().item() * 100,
            'clip_directional': (direction @ text_direction).mean().item(),
            'clip_frame_consistency': (edit_image[:-1] * edit_image[1:]).sum(-1).mean().item() * 100,
            'warp_error': warp_error(source, edited),
            'background_psnr_db': 10 * math.log10(255 ** 2 / max(mse, 1e-10)),
            'foreground_change': float(diff.mean(-1)[core].mean()) if core.any() else None,
            'foreground_hue_deg': hue_degrees(edited, core),
        }
        print(run.name, json.dumps(results['runs'][run.name]), flush=True)
    results['source_foreground_hue_deg'] = hue_degrees(source, core)
    Path(output).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='run.py output folders that share source frames')
    parser.add_argument('--output', required=True)
    parser.add_argument('--clip-model', default='openai/clip-vit-large-patch14')
    parser.add_argument('--development-cpu',action='store_true')
    args = parser.parse_args()
    main(args.runs, args.output, args.clip_model,args.development_cpu)
