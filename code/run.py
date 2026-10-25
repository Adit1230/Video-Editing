"""Single-GPU runner for our independent Video Editing-method implementation."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from cache_manager import CacheManager, atomic_save, cleanup_incomplete

ROOT = Path(__file__).resolve().parent



def fingerprint(config, frame_paths):
    keys = ('source_prompt', 'model_id', 'model_revision', 'model_path', 'steps', 'size', 'frames', 'seed',
            'inversion_method', 'inversion_iterations', 'inversion_relaxation')
    h = hashlib.sha256(json.dumps({k: config.get(k) for k in keys}, sort_keys=True).encode())
    for path in [ROOT / 'engine.py', ROOT / 'inversion.py', Path(__file__)]:
        h.update(path.read_bytes())
    for path in frame_paths:
        h.update(path.read_bytes())
    return h.hexdigest()[:20]


def resolve_video(c):
    url = c.get('video_url', '').strip()
    if not url:
        return Path(c['video_path'])
    if urlsplit(url).scheme not in ('http', 'https'):
        raise ValueError('video_url must be an HTTP(S) direct video-file URL.')
    folder = Path(c.get('download_dir', '/tmp/video_editing_downloads'))
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (hashlib.sha256(url.encode()).hexdigest()[:20] + '.mp4')
    if target.is_file() and target.stat().st_size:
        return target
    partial = target.with_suffix('.part')
    try:
        print('Downloading input video...', flush=True)
        with urlopen(Request(url, headers={'User-Agent': 'FateZero-Kaggle/1.0'}), timeout=60) as response:
            if 'text/html' in response.headers.get('Content-Type', '').lower():
                raise ValueError('Link returned a webpage. Supply a direct MP4/video download link, not a watch/share page.')
            total = 0
            with partial.open('wb') as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > c.get('max_download_mb', 1024) * 1024**2:
                        raise ValueError('Video exceeds max_download_mb. Use a shorter source or raise the limit.')
                    output.write(chunk)
        import cv2
        capture = cv2.VideoCapture(str(partial))
        try:
            ok, _ = capture.read()
            if not ok:
                raise ValueError('Downloaded file is not a decodable video. Use a direct video-file URL.')
        finally:
            capture.release()
        partial.replace(target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def clip_selection(c, fps, count):
    n, start, stride = c['frames'], c['start_frame'], c['frame_stride']
    seconds = c.get('clip_seconds')
    if seconds is None:
        return [start + i * stride for i in range(n)], fps / stride
    remaining = max(0, int(count) - start)
    automatic = seconds == 'auto'
    if automatic:
        # Use the preset's frame count with distinct source frames. Extend the
        # opening interval when needed rather than repeating/interpolating frames.
        preferred = 2. if c.get('backend') == 'cogvideox_2b' else 1.
        available = min(max(int(math.floor(preferred * fps)), n), remaining)
    else:
        available = min(int(math.floor(seconds * fps)), remaining)
    if available < n:
        minimum = math.ceil(n / fps * 1000) / 1000
        detail = (f'At {fps:g} FPS, frames={n} needs at least {minimum:.3f} seconds '
                  f'and {n} distinct source frames. Only {available} are available in '
                  f'the selected clip ({remaining} remain after start_frame={start}). ')
        if remaining < n:
            raise ValueError(detail + 'Use an earlier start, another video, or fewer frames'
                + (' (CogVideoX requires 4*k+1, e.g. 5, 9, 13, 17).' if c.get('backend')=='cogvideox_2b' else '.'))
        raise ValueError(detail + f'Set clip_seconds="auto" or increase clip_seconds to at least {minimum:.3f}.')
    if automatic:
        print(f'Automatic opening clip: {available / fps:.3f} s at {fps:g} source FPS; '
              f'sampling {n} distinct frames.', flush=True)
    # Sample the opening interval without loading all its frames into memory.
    return [start + int(i * available / n) for i in range(n)], n * fps / available


def prepare_frames(c, out):
    import cv2
    from PIL import Image, ImageOps
    path = resolve_video(c)
    if not path.exists():
        raise FileNotFoundError(f'Input not found: {path}. Set video_path to your Kaggle dataset file.')
    out.mkdir(parents=True, exist_ok=True)
    n, start, stride = c['frames'], c['start_frame'], c['frame_stride']
    fps = None
    images = []
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
        if not c.get('input_fps'):
            raise ValueError('Set input_fps for a frame folder to preserve playback timing.')
        fps = float(c['input_fps'])
        source_indices, output_fps = clip_selection(c, fps, len(files))
        if not files or source_indices[-1] >= len(files):
            raise ValueError('Frame folder is too short for frames/start_frame/frame_stride.')
        for i in source_indices:
            with Image.open(files[i]) as im:
                images.append(im.convert('RGB'))
    else:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError('OpenCV could not open the video.')
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                if not c.get('input_fps'):
                    raise ValueError('Video FPS is unavailable; set input_fps.')
                fps = float(c['input_fps'])
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if c.get('clip_seconds') is not None and (not math.isfinite(count) or count <= 0):
                raise ValueError('Video frame count is unavailable; set clip_seconds=None to use explicit frame_stride.')
            source_indices, output_fps = clip_selection(c, fps, count)
            wanted = set(source_indices)
            for i in range(source_indices[-1] + 1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Video ended at frame {i}; reduce start_frame/frames/frame_stride.')
                if i in wanted:
                    images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
    paths = []
    for old in out.glob('[0-9][0-9][0-9][0-9][0-9].png'):old.unlink()
    for i, image in enumerate(images):
        image = ImageOps.fit(image, (c['size'], c['size']), method=Image.Resampling.LANCZOS)
        target = out / f'{i:05d}.png'
        image.save(target)
        paths.append(target)
    return paths, output_fps, source_indices


def resolve_model(c):
    if c.get('model_path'):
        path = Path(c['model_path'])
    else:
        from huggingface_hub import snapshot_download
        # No training checkpoints, ONNX, safety checker or duplicate weight formats.
        path = Path(snapshot_download(
            c['model_id'], revision=c['model_revision'],
            allow_patterns=['model_index.json', 'unet/config.json', 'unet/diffusion_pytorch_model.bin',
                            'vae/config.json', 'vae/diffusion_pytorch_model.bin',
                            'text_encoder/config.json', 'text_encoder/pytorch_model.bin',
                            'tokenizer/*', 'scheduler/scheduler_config.json'],
            token=os.environ.get('HF_TOKEN') or None))
    for relative in ('unet/config.json', 'unet/diffusion_pytorch_model.bin',
                     'vae/diffusion_pytorch_model.bin', 'text_encoder/pytorch_model.bin',
                     'tokenizer/tokenizer_config.json', 'scheduler/scheduler_config.json'):
        if not (path / relative).is_file():
            raise FileNotFoundError(f'Missing {path / relative}. Use a complete SD 1.x Diffusers .bin directory; '
                                    'this legacy loader does not load standalone safetensors or SDXL.')
    return path


def export_video(images, path, fps):
    import imageio.v2 as imageio
    import numpy as np
    with imageio.get_writer(str(path), fps=fps, codec='libx264', quality=9,
                            pixelformat='yuv420p', macro_block_size=None) as writer:
        for frame in images:
            writer.append_data(np.asarray(frame.convert('RGB')))


def validate(c):
    from inversion import validate_fixed_point
    if c.get('inversion_method') != 'fixed_point':
        raise ValueError('Set inversion_method=fixed_point. Update the source dataset and config together.')
    validate_fixed_point(c.get('inversion_iterations'), c.get('inversion_relaxation', 1.))
    if c.get('clip_seconds') not in (None,'auto') and (not isinstance(c['clip_seconds'],(int,float)) or not math.isfinite(c['clip_seconds']) or c['clip_seconds'] <= 0):
        raise ValueError('clip_seconds must be positive, "auto", or null.')
    if c['size'] not in (256, 320, 384, 512):
        raise ValueError('Use size=256, 320, 384 or 512.')
    if not 2 <= c['frames'] <= 16 or not 10 <= c['steps'] <= 100:
        raise ValueError('Use 2–16 frames and 10–100 steps.')
    if c['start_frame'] < 0 or c['frame_stride'] < 1:
        raise ValueError('start_frame must be >=0, frame_stride must be >=1.')
    for key in ('self_replace', 'cross_replace'):
        if not 0 <= c[key] <= 1:
            raise ValueError(f'{key} must be between 0 and 1.')
    if not c['guidance'] > 1:
        raise ValueError('Editing requires guidance >1 because the official controller splits CFG batches.')
    if not c['source_prompt'].strip() or not c['target_prompt'].strip():
        raise ValueError('Provide source_prompt and target_prompt.')


def main(c):
    validate(c)
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from engine import VideoUNet, AttentionCache, Editor

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select a Kaggle GPU accelerator before running.')
    device = torch.device('cuda:0')
    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print('GPU:', torch.cuda.get_device_name(0), 'VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30, flush=True)
    if torch.cuda.get_device_properties(0).total_memory < 14 * 2**30:
        print('Less than 14 GiB VRAM: start with 384px and 4 frames.', flush=True)
    if c.get('cache_auto_cleanup', True):
        cleanup_incomplete(c['cache_path'])
    out = Path(c['output_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths, fps, indices = prepare_frames(c, out / 'source_frames')
    source = [Image.open(p).convert('RGB') for p in paths]
    export_video(source, out / 'source.mp4', fps)
    model_path = resolve_model(c)
    # Inversion cache must also change if a local checkpoint is replaced.
    local_signature = []
    for name in ('unet/diffusion_pytorch_model.bin', 'vae/diffusion_pytorch_model.bin', 'text_encoder/pytorch_model.bin'):
        p = model_path / name
        local_signature.append((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns))
    c = dict(c, resolved_model_signature=local_signature)
    cache_key = hashlib.sha256((fingerprint(c, paths) + json.dumps(local_signature)).encode()).hexdigest()[:20]
    with CacheManager(c['cache_path'], cache_key, c) as cache_disk:
        cache_root = cache_disk.path
        start = time.time()
        tokenizer = CLIPTokenizer.from_pretrained(str(model_path), subfolder='tokenizer')
        for name in ('source_prompt', 'target_prompt'):
            if len(tokenizer.encode(c[name])) > tokenizer.model_max_length:
                raise ValueError(f'{name} exceeds CLIP token limit. Shorten it to avoid silent truncation.')
        text_encoder = CLIPTextModel.from_pretrained(str(model_path), subfolder='text_encoder', torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(str(model_path), subfolder='vae', torch_dtype=torch.float16)
        backbone = UNet2DConditionModel.from_pretrained(str(model_path), subfolder='unet', torch_dtype=torch.float16)
        unet = VideoUNet(backbone, rank=160, least_channel=640).to(device=device, dtype=torch.float16).eval()
        scheduler = DDIMScheduler.from_pretrained(str(model_path), subfolder='scheduler')
        if scheduler.config.prediction_type != 'epsilon':
            raise ValueError('An epsilon-prediction SD1.x checkpoint is required.')
        for component in (unet, vae, text_encoder):
            component.to(device).eval().requires_grad_(False)
            assert next(component.parameters()).is_cuda
        scheduler.set_timesteps(c['steps'], device=device)
        cache = AttentionCache(cache_root, c['steps'], c['self_replace'], c['cross_replace'])
        cache.disk_manager = cache_disk
        cache.mask_threshold = c.get('mask_threshold', .3)
        # Token positions for the requested foreground, including BPE subwords.
        source_ids = tokenizer(c['source_prompt']).input_ids
        for word in c.get('edit_words', []):
            ids = tokenizer(word, add_special_tokens=False).input_ids
            for i in range(len(source_ids)-len(ids)+1):
                if source_ids[i:i+len(ids)] == ids:
                    cache.mask_tokens.extend(range(i, i+len(ids)))
        if c.get('edit_words') and not cache.mask_tokens:
            raise ValueError('edit_words must occur in source_prompt (e.g. flowers).')
        editor = Editor(unet, vae, text_encoder, tokenizer, scheduler, cache)
        tensor = torch.from_numpy(np.stack([np.asarray(x) for x in source]).copy()).permute(0, 3, 1, 2)
        tensor = tensor.to(device=device, dtype=torch.float16) / 127.5 - 1
        cache_state = cache_root / 'inversion.pt'
        with torch.inference_mode():
            state = None
            if c.get('reuse_cache', True) and cache_disk.reusable():
                try:
                    state = torch.load(cache_state, map_location='cpu')
                    attention_paths = [cache_root / name for name in state['attention_files']]
                    if len(attention_paths) != c['steps'] or not all(p.is_file() and p.stat().st_size for p in attention_paths):
                        raise ValueError('Missing attention snapshots')
                    if not torch.is_tensor(state['noisy']):
                        raise ValueError('Invalid latent state')
                except Exception as exc:
                    print(f'Rebuilding unreadable/incomplete inversion cache ({type(exc).__name__}).', flush=True)
                    cache_disk.reset()
                    state = None
            if state is not None:
                noisy = state['noisy'].to(device)
                print('Reusing matching inversion cache:', cache_key, flush=True)
            else:
                encoded = torch.cat([vae.encode(frame).latent_dist.mode() for frame in tensor.split(1)]) * .18215
                clean = encoded.permute(1, 0, 2, 3).unsqueeze(0)
                noisy = editor.invert(clean, c['source_prompt'], c['inversion_iterations'], c.get('inversion_relaxation', 1.))
                state = {'noisy': noisy.cpu(), 'attention_files': [f'{i:03d}.pt' for i in range(c['steps'])]}
                atomic_save(state, cache_state, torch.save)
                cache_disk.mark_complete()
                del encoded, clean
            reconstruction = None
            if c.get('reconstruct', False):
                reconstruction = editor.decode(editor.denoise(noisy, c['source_prompt'], guidance=1.))
                export_video(reconstruction, out / 'reconstruction.mp4', fps)
            edited_latents = editor.denoise(noisy, c['target_prompt'], c['source_prompt'],
                                            c['guidance'], c.get('negative_prompt', ''))
            result = editor.decode(edited_latents)
        if len(result) != c['frames']:
            raise RuntimeError('Output frame count does not match the input clip.')
        frames_dir = out / 'edited_frames'
        frames_dir.mkdir(exist_ok=True)
        for i, image in enumerate(result):
            image.save(frames_dir / f'{i:05d}.png')
        export_video(result, out / 'edited.mp4', fps)
        comparisons = []
        for i in range(len(source)):
            panels = [source[i]] + ([reconstruction[i]] if reconstruction else []) + [result[i]]
            canvas = Image.new('RGB', (c['size'] * len(panels), c['size']))
            for j, panel in enumerate(panels):
                canvas.paste(panel, (j * c['size'], 0))
            comparisons.append(canvas)
        export_video(comparisons, out / 'comparison.mp4', fps)
        comparisons[0].save(out / 'comparison_first_frame.png')
        report = {'config': c, 'implementation': 'independent-fatezero-fixed-point-v2', 'architecture': {'temporal_conv_rank':160, 'sparse_reference':'middle', 'least_sc_channel':640}, 'unet_calls': unet.calls, 'unet_device':unet.last_device, 'attention_stats':cache.stats,
                  'fps': fps, 'source_frame_indices': indices, 'runtime_seconds': time.time()-start,
                  'peak_gpu_GiB': torch.cuda.max_memory_allocated()/2**30, 'cache_key': cache_key,
                  'comparison_order': ['source', 'reconstruction', 'edit'] if reconstruction else ['source', 'edit']}
        report['edit_mean_absolute_pixel_change'] = float(np.mean([np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float)).mean() for a,b in zip(source,result)]))
        if report['edit_mean_absolute_pixel_change'] < 2:
            print('Edit is very close to source. Try self_replace=.3, cross_replace=.15, guidance=9; inspect frames.', flush=True)
        if reconstruction:
            mse = np.mean([(np.asarray(a).astype(float)-np.asarray(b).astype(float))**2 for a,b in zip(source,reconstruction)])
            report['reconstruction_psnr_db'] = 10 * math.log10(255**2 / max(mse, 1e-10))
            print('Reconstruction PSNR:', report['reconstruction_psnr_db'], '(diagnostic, not an edit-quality score)', flush=True)
        report['cache_evicted_runs'] = cache_disk.removed
        (out / 'run_report.json').write_text(json.dumps(report, indent=2))
        if not c.get('keep_cache', False):
            cache_disk.release()
        print('Saved edited.mp4, comparison.mp4, PNG frames and run_report.json in', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        main(json.loads(Path(args.config).read_text()))
    except Exception as exc:
        print(f'FateZero failed: {exc}', file=sys.stderr, flush=True)
        raise
