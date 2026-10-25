"""Gradio interface: upload a video, describe it, write the edited description, get the edited clip and metrics.

Runs in the notebook kernel (Gradio, OpenCV, Pillow). Each edit calls launch.py and metrics.py in the
isolated modern environment (torch 2.5, diffusers 0.35), so the interface never imports torch itself.
Backbones (SD1.4, SD2.1, CogVideoX-2B) and mask sources (attention, Grounded-SAM 2, hybrid) come from
options.py / run_extended.py; every edit starts from the package's "improved" profile.

The edited description is typed by the user as-is: nothing in this file rewrites or generates prompts.
"""
import json
import math
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

APPEARANCE, TEXTURE, OBJECT, STYLE = ('Colour (local)', 'Texture / material (local)', 'Object replacement (local)',
                                      'Style transfer (whole frame)')
ALL_LAYERS, DEEP_ONLY = 'All resolutions (steadiest object)', 'Deep layers only (less background warping)'
SHARING = {ALL_LAYERS: 320, DEEP_ONLY: 640}
# Defaults per edit type. Colour/object come from the flower ablation and app tests, style from the style
# sweeps, texture and the sharing defaults from the Video Editing-clip demo sweep (kaggle_demo_sweep/).
EDIT_TYPES = {
    APPEARANCE: dict(self_replace=.7, cross_replace=.5, latent_blend=True, blend_dilate=1, guidance=7.5,
                     negative_inversion=True, sharing=ALL_LAYERS),
    TEXTURE: dict(self_replace=.6, cross_replace=.6, latent_blend=True, blend_dilate=1, guidance=10.,
                  negative_inversion=True, sharing=ALL_LAYERS),
    OBJECT: dict(self_replace=.4, cross_replace=.3, latent_blend=True, blend_dilate=2, guidance=7.5,
                 negative_inversion=True, sharing=ALL_LAYERS),
    STYLE: dict(self_replace=.4, cross_replace=.2, latent_blend=False, blend_dilate=1, guidance=12.,
                negative_inversion=True, sharing=ALL_LAYERS),
}
BACKBONES = {'Stable Diffusion 1.4': 'sd14', 'Stable Diffusion 2.1': 'sd21',
             'CogVideoX-2B (native video, experimental)': 'cogvideox_2b'}
MASK_SOURCES = {'Attention (from region words)': 'attention', 'Grounded-SAM 2 (detect region words)': 'grounded_sam2',
                'Hybrid (SAM 2 inside attention region)': 'hybrid'}
# Grounded-SAM 2 masks gave cleaner regions and steadier edits than attention masks on the kitten, bear,
# squirrel and swan clips of the demo sweep; attention remains the fallback when detection finds nothing.
DEFAULT_MASK = 'Grounded-SAM 2 (detect region words)'
FAST, STANDARD = 'Fast (384 px, 20 steps)', 'Standard (512 px, 30 steps, ~4-6 min on a T4)'
PRESETS = {  # image backbones / native CogVideoX (whose frame count must be 4k+1)
    FAST: dict(image=dict(size=384, steps=20, inversion_iterations=2),
               native=dict(size=320, steps=15, inversion_iterations=2, frames=9)),
    STANDARD: dict(image=dict(size=512, steps=30, inversion_iterations=3),
                   native=dict(size=320, steps=25, inversion_iterations=3, frames=17)),
}

METRIC_ROWS = [
    ('clip_text_target', 'CLIP text alignment', 'higher', 'Edited frames vs. target description (×100)'),
    ('clip_directional', 'CLIP directional similarity', 'higher', 'Image change points the same way as the text change'),
    ('clip_frame_consistency', 'CLIP frame consistency', 'higher', 'Similarity of consecutive edited frames (×100)'),
    ('warp_error', 'Optical-flow warp error', 'lower', 'Flicker after following the source motion (0–255 scale)'),
    ('background_psnr_db', 'Background PSNR (dB)', 'higher', 'Preservation outside the edit region (whole frame for style edits)'),
    ('foreground_change', 'Edit-region change', 'higher', 'Mean pixel change inside the edit region'),
    ('reconstruction_psnr_db', 'Reconstruction PSNR (dB)', 'higher', 'Inversion fidelity: source rebuilt from its inversion'),
]


def find_phrase(source, phrase):
    return re.search(r'(?<![\w-])' + re.escape(phrase.strip()) + r'(?![\w-])', source, re.I)


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Could not open the video. Upload an MP4/MOV/WEBM file.')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError('The video has no readable frame rate or frame count.')
    return fps, count, width, height


def overlay(frame_path, mask_path, target):
    """Save the source frame with the edit region tinted red as a PNG and return its path."""
    with Image.open(frame_path) as image:
        frame = np.asarray(image.convert('RGB'), dtype=np.float32)
    with Image.open(mask_path) as image:
        size = (frame.shape[1], frame.shape[0])
        mask = np.asarray(image.convert('L').resize(size), dtype=np.float32)[..., None] / 255
    tint = np.array([255, 60, 60], np.float32)
    blended = np.clip(frame * (1 - .45 * mask) + tint * .45 * mask, 0, 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(blended), 'RGB').save(target)
    return str(target)


def metrics_table(job, metrics, report):
    values = dict(metrics['runs'][job.name]) if metrics else {}
    values['reconstruction_psnr_db'] = report.get('reconstruction_psnr_db')
    reference = {'clip_frame_consistency': metrics and metrics.get('source_clip_frame_consistency'),
                 'warp_error': metrics and metrics.get('source_warp_error')}
    rows = []
    for key, name, better, meaning in METRIC_ROWS:
        value = values.get(key)
        if value is None:
            continue
        source = reference.get(key)
        rows.append([name, round(value, 3), better, '' if source is None else round(source, 3), meaning])
    return rows


class Editor:
    def __init__(self, runner_python, env, work_dir, cache_dir='/tmp/video_editing_app_cache',
                 mask_cache_dir='/tmp/video_editing_app_mask_cache'):
        self.python, self.env = str(runner_python), env
        self.work = Path(work_dir)
        self.cache, self.mask_cache = Path(cache_dir), Path(mask_cache_dir)
        self.work.mkdir(parents=True, exist_ok=True)

    def _stream(self, command, log, label):
        process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        last = 0.
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            if log and '%|' in line and '%|' in log[-1]:
                log[-1] = line  # replace the previous progress-bar line
            else:
                log.append(line)
            if time.time() - last > 1:
                last = time.time()
                yield
        if process.wait():
            tail = '\n'.join(log[-8:])
            raise RuntimeError(f'{label} failed:\n{tail}')

    def config(self, video, source, target, edit_type, region_words, backbone, mask_source, preset, frames,
               start_frame, seconds, guidance, self_replace, cross_replace, latent_blend, negative_inversion,
               sharing, seed, reconstruct, output_dir):
        backend = BACKBONES[backbone]
        config = json.loads((ROOT / 'configs' / f'improved_{backend}.json').read_text())
        family = 'native' if backend == 'cogvideox_2b' else 'image'
        config.update(PRESETS[preset][family])
        if family == 'image':
            config['frames'] = int(frames)
        mask_backend = MASK_SOURCES[mask_source] if region_words else 'attention'
        config.update(
            video_url='', video_path=str(video), input_fps=None, clip_seconds=float(seconds), start_frame=start_frame,
            frame_stride=1, seed=int(seed), source_prompt=source, target_prompt=target, edit_words=region_words,
            mask_backend=mask_backend, mask_prompt=' '.join(region_words), mask_threshold=.3,
            guidance=float(guidance), self_replace=float(self_replace), cross_replace=float(cross_replace),
            mask_latent_blend=bool(latent_blend and region_words), blend_fraction=1.,
            blend_dilate=EDIT_TYPES[edit_type]['blend_dilate'], negative_prompt='',
            negative_prompt_inversion=bool(negative_inversion), reconstruct=bool(reconstruct),
            keep_cache=True, reuse_cache=True, cache_auto_cleanup=True, cache_keep_runs=3,
            cache_path=str(self.cache), mask_cache_path=str(self.mask_cache), output_dir=str(output_dir),
            edit_profile='improved', compute_metrics=False,
            stream_profile=False)  # the shipped configs profile with GPU syncs; ordinary runs should not
        if family == 'image':
            config.update(sparse_least_channel=SHARING[sharing], zero_temporal_bias=True)
        return config

    def edit(self, video, source, target, edit_type, region, backbone, mask_source, preset, frames, start, seconds,
             guidance, self_replace, cross_replace, latent_blend, negative_inversion, sharing, seed,
             reconstruct, compute_metrics):
        """Generator yielding (log, comparison, edited, metrics rows, frames, mask overlays, zip, report)."""
        log = []
        empty = (None,) * 7

        def update(*outputs):
            return ('\n'.join(log[-40:]),) + (outputs or empty)

        if not video:
            raise ValueError('Upload a video first.')
        source, target, region = (source or '').strip(), (target or '').strip(), (region or '').strip()
        if not source or not target:
            raise ValueError('Fill in both descriptions: the original video and the edited video.')
        if source == target:
            raise ValueError('The edited description is identical to the original one. Change the words you want edited.')
        region_words = [w.strip() for w in region.split(',') if w.strip()]
        for word in region_words:
            if not find_phrase(source, word):
                raise ValueError(f'Edit region word "{word}" is not in the original description.')
        if edit_type != STYLE and not region_words:
            raise ValueError('Local edits need edit region words from the original description (e.g. "cat").')
        if edit_type == STYLE:
            region_words, latent_blend = [], False
        backend = BACKBONES[backbone]
        needed = PRESETS[preset]['native']['frames'] if backend == 'cogvideox_2b' else int(frames)
        fps, count, width, height = video_info(video)
        start_frame = int(round(start * fps))
        available = min(int(math.floor(seconds * fps)), count - start_frame)
        if available < needed:
            raise ValueError(f'The video is {count / fps:.1f} s at {fps:.0f} fps; from {start:.1f} s a {seconds:.1f} s clip '
                             f'has {max(available, 0)} frames, fewer than {needed}. Use an earlier start or a longer clip.')
        job = self.work / time.strftime('edit_%Y%m%d_%H%M%S')
        job.mkdir(parents=True)
        local_video = job / ('input' + (Path(video).suffix or '.mp4'))
        shutil.copy(video, local_video)
        config = self.config(local_video, source, target, edit_type, region_words, backbone, mask_source, preset,
                             frames, start_frame, seconds, guidance, self_replace, cross_replace, latent_blend,
                             negative_inversion, sharing, seed, reconstruct, job)
        (job / 'request.json').write_text(json.dumps(config, indent=2))
        log.append(f'Input: {width}x{height}, {count / fps:.1f} s at {fps:.1f} fps. Editing {config["frames"]} frames '
                   f'from {start:.1f}–{start + seconds:.1f} s with {backbone} ({preset}).')
        log.append(f'Original: {source}')
        log.append(f'Edited:   {target}')
        if region_words:
            log.append(f'Edit region: {", ".join(region_words)} via {config["mask_backend"]} mask')
        yield update()
        started = time.time()
        for _ in self._stream([self.python, str(ROOT / 'launch.py'), '--config', str(job / 'request.json')], log, 'Editing'):
            yield update()
        report = json.loads((job / 'run_report.json').read_text())
        metrics = None
        if compute_metrics:
            log.append('Computing metrics (CLIP ViT-L/14, optical flow)...')
            yield update(str(job / 'comparison.mp4'), str(job / 'edited.mp4'), None, None, None, None, None)
            for _ in self._stream([self.python, str(ROOT / 'metrics.py'), str(job), '--output', str(job / 'metrics.json')],
                                  log, 'Metrics'):
                yield update(str(job / 'comparison.mp4'), str(job / 'edited.mp4'), None, None, None, None, None)
            metrics = json.loads((job / 'metrics.json').read_text())
        frames_out = sorted((job / 'edited_frames').glob('*.png'))
        sources = sorted((job / 'source_frames').glob('*.png'))
        masks = sorted((job / 'masks').glob('*.png'))
        (job / 'mask_overlays').mkdir(exist_ok=True)
        overlays = [(overlay(s, m, job / 'mask_overlays' / s.name), f'frame {i + 1}')
                    for i, (s, m) in enumerate(zip(sources, masks))]
        archive = job / f'{job.name}.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(job.rglob('*')):
                if path.is_file() and path != archive and path != local_video:
                    bundle.write(path, path.relative_to(job))
        summary = {'seconds': round(time.time() - started, 1), 'output_fps': round(report['fps'], 2),
                   'peak_gpu_GiB': round(report['peak_gpu_GiB'], 2), 'backend': config['backend'],
                   'mask_backend': config['mask_backend'], 'cache_key': report['cache_key'],
                   'reused_inversion': bool(report.get('cache_reused')), 'edit_type': edit_type,
                   'edit_region_words': region_words, 'output_dir': str(job)}
        stats = report.get('attention_stats', {})
        if 'stream_peak_gpu_bytes' in stats:  # v3 streaming cache placement
            summary['attention_cache_peak_gpu_GiB'] = round(stats['stream_peak_gpu_bytes'] / 2**30, 2)
            summary['attention_cache_oom_fallbacks'] = stats.get('gpu_cache_oom_fallbacks', 0)
        log.append(f'Done in {summary["seconds"]} s. Same clip + original description reuses the inversion next time.')
        yield update(str(job / 'comparison.mp4'), str(job / 'edited.mp4'), metrics_table(job, metrics, report) or None,
                     [(str(p), f'frame {i + 1}') for i, p in enumerate(frames_out)], overlays or None, str(archive), summary)


GUIDE = """
### Diffusion-inversion video editing
1. **Upload a short video** (or pick an example below). Only a short clip is edited: choose its start and length.
2. **Describe the original video** in one sentence.
3. **Write the edited description**: the same sentence with only the words you want changed.
   Colour `a black cat …` → `a ginger cat …` · texture `a black swan …` → `a Swarovski crystal swan …` ·
   object `a brown bear …` → `a tiger …`. For a **style**, start with the medium and keep the subject:
   `a watercolor painting of a pink water lily, soft washes of colour on textured paper`.
4. Pick the **edit type** (it sets sensible defaults for the sliders) and, for local edits, type the **edit region
   words**: the word(s) in the *original* description that name the thing being changed, e.g. `cat`.
5. Click **Edit video**. The clip is inverted into the chosen diffusion model (fixed-point DDIM inversion), then
   regenerated from the edited description while reusing the original's attention (FateZero-style fusion).
   For local edits, everything outside the edit region is copied from the inverted original at every step;
   Grounded-SAM 2 can detect that region instead of using attention maps.

Another edit of the same clip with the same original description reuses the inversion and is faster.
If Grounded-SAM 2 finds nothing for the region words, switch *Edit region from* to *Attention*.
If moving backgrounds warp, set *Cross-frame sharing* to *Deep layers only*.
"""


def build_ui(editor, examples=(), showcase=()):
    """examples: [video, original description, edited description, edit type, region words, start s, length s,
    mask source label] rows. showcase: (title, comparison video path, caption) tuples for a pre-computed tab."""
    import gradio as gr

    def apply_type(edit_type):
        settings = EDIT_TYPES[edit_type]
        return (settings['self_replace'], settings['cross_replace'], settings['latent_blend'],
                settings['guidance'], settings['negative_inversion'], settings['sharing'])

    def run_edit(video, source, target, edit_type, region, backbone, mask_source, preset, frames, start, seconds,
                 guidance, self_replace, cross_replace, latent_blend, negative_inversion, sharing, seed, reconstruct,
                 compute_metrics):
        try:
            yield from editor.edit(video, source, target, edit_type, region, backbone, mask_source, preset, frames,
                                   start, seconds, guidance, self_replace, cross_replace, latent_blend,
                                   negative_inversion, sharing, seed, reconstruct, compute_metrics)
        except (ValueError, RuntimeError) as error:
            raise gr.Error(str(error)[-1500:])

    with gr.Blocks(title='Video editing with diffusion inversion') as demo:
        with gr.Tabs():
            with gr.Tab('Edit a video'):
                gr.Markdown(GUIDE)
                with gr.Row():
                    with gr.Column(scale=5):
                        video = gr.Video(label='Input video', sources=['upload'])
                        source = gr.Textbox(label='Original description',
                                            placeholder='a black cat walking on the floor next to a wall')
                        target = gr.Textbox(label='Edited description (what the output should show)',
                                            placeholder='a ginger cat walking on the floor next to a wall')
                        with gr.Row():
                            edit_type = gr.Radio(list(EDIT_TYPES), value=APPEARANCE, label='Edit type')
                            region = gr.Textbox(label='Edit region words (from the original description, comma separated; '
                                                      'not used for style edits)', placeholder='cat')
                        with gr.Row():
                            backbone = gr.Radio(list(BACKBONES), value='Stable Diffusion 1.4', label='Diffusion model')
                            mask_source = gr.Radio(list(MASK_SOURCES), value=DEFAULT_MASK,
                                                   label='Edit region from (local edits)')
                        with gr.Row():
                            preset = gr.Radio(list(PRESETS), value=STANDARD, label='Quality')
                            frames = gr.Slider(4, 16, value=8, step=2, label='Frames (SD models; CogVideoX uses 9/17)')
                        with gr.Row():
                            start = gr.Number(value=0, label='Clip start (s)', minimum=0)
                            seconds = gr.Number(value=1, label='Clip length (s)', minimum=.2)
                        with gr.Accordion('Advanced', open=False):
                            with gr.Row():
                                guidance = gr.Slider(3, 15, value=7.5, step=.5, label='Guidance scale')
                                seed = gr.Number(value=0, precision=0, label='Seed')
                            with gr.Row():
                                self_replace = gr.Slider(0, 1, value=.7, step=.05, label='Self-attention replace (structure)')
                                cross_replace = gr.Slider(0, 1, value=.5, step=.05, label='Cross-attention replace (layout)')
                            with gr.Row():
                                latent_blend = gr.Checkbox(value=True, label='Keep background (latent blending)')
                                negative_inversion = gr.Checkbox(value=True, label='Negative-prompt inversion')
                            sharing = gr.Radio(list(SHARING), value=ALL_LAYERS, label='Cross-frame sharing (SD models)')
                            with gr.Row():
                                reconstruct = gr.Checkbox(value=False, label='Also reconstruct (inversion PSNR)')
                                compute_metrics = gr.Checkbox(value=True, label='Compute metrics')
                        run = gr.Button('Edit video', variant='primary')
                    with gr.Column(scale=6):
                        status = gr.Textbox(label='Progress', lines=10, max_lines=10, autoscroll=True)
                        comparison = gr.Video(label='Original | (reconstruction) | edited', autoplay=True, loop=True)
                        edited = gr.Video(label='Edited clip', autoplay=True, loop=True)
                        metrics = gr.Dataframe(headers=['Metric', 'Edited', 'Better', 'Original video', 'Meaning'],
                                               label='Quality metrics', wrap=True)
                        with gr.Tabs():
                            with gr.Tab('Edited frames'):
                                gallery = gr.Gallery(columns=4, height=300, format='png')
                            with gr.Tab('Edit region'):
                                masks = gr.Gallery(columns=4, height=300, format='png')
                        download = gr.File(label='Download frames, videos, masks, config and report')
                        summary = gr.JSON(label='Run summary')
                if examples:
                    gr.Examples(examples, [video, source, target, edit_type, region, start, seconds, mask_source],
                                label='Examples (fill in the form; then click Edit video)', examples_per_page=15)
            if showcase:
                with gr.Tab('Showcase (pre-computed)'):
                    gr.Markdown('Results produced with this interface\'s default settings. Left: original, right: edited.')
                    for title, path, caption in showcase:
                        gr.Markdown(f'**{title}** — {caption}')
                        gr.Video(value=str(path), autoplay=True, loop=True, interactive=False, show_label=False)
        edit_type.change(apply_type, edit_type,
                         [self_replace, cross_replace, latent_blend, guidance, negative_inversion, sharing],
                         api_name='edit_defaults')
        run.click(run_edit, [video, source, target, edit_type, region, backbone, mask_source, preset, frames, start,
                             seconds, guidance, self_replace, cross_replace, latent_blend, negative_inversion, sharing,
                             seed, reconstruct, compute_metrics],
                  [status, comparison, edited, metrics, gallery, masks, download, summary], api_name='edit')
    return demo
