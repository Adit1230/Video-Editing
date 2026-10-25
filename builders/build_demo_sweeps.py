"""Builds two demo-sweep notebooks (local edits, style edits) that share one job list.

Each job is one edit of one clip. Jobs on the same clip and source prompt reuse the inversion.
Outputs per job: edited.mp4, comparison.mp4, frames, masks, run_report.json, a 4-frame grid PNG,
plus metrics.json per clip. Clips are Video Editing's released demo data (attribute.zip / style.zip)
and the MDN flower video.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# name: (source, folder inside zip or URL, frame stride, first frame)
CLIPS = {
    'flower': ('url', 'https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4', None, 0),
    'bus': ('attribute', 'bus_gpu', 1, 0),
    'cat': ('attribute', 'cat_tiger_leopard_grass', 1, 0),
    'swan': ('attribute', 'swan_swarov', 1, 0),
    'duck': ('attribute', 'duck_rubber', 1, 0),
    'dog': ('attribute', 'gray_dog', 1, 0),
    'bear': ('attribute', 'bear_tiger_lion_leopard', 1, 0),
    'fox': ('attribute', 'fox_wolf_snow', 1, 0),
    'squirrel': ('attribute', 'squirrel_carrot', 1, 0),
    'rabbit_strawberry': ('attribute', 'rabbit_strawberry', 1, 0),
    'sunflower': ('style', 'sunflower', 1, 0),
    'lily': ('style', 'red_water_lily_opening', 20, 0),
    'blackswan': ('style', 'blackswan', 2, 0),
    'surf': ('style', 'surf', 1, 0),
    'rabbit_watermelon': ('style', 'rabit', 3, 0),
    'train': ('style', 'train', 4, 0),
}

SOURCE = {
    'flower': 'a red flower bud on a green stem with purple-edged leaves, blurred garden background',
    'bus': 'a white and blue bus on the road',
    'cat': 'a black cat walking on the floor next to a wall',
    'swan': 'a black swan with a red beak swimming in a river near a wall and bushes',
    'duck': 'a sleepy white duck',
    'dog': 'a gray dog sitting on the mat',
    'bear': 'a brown bear walking on the rock against a wall',
    'fox': 'a white fox sitting in the grass',
    'squirrel': 'a squirrel is eating a carrot',
    'rabbit_strawberry': 'a rabbit is eating strawberries',
    'sunflower': 'a yellow sunflower',
    'lily': 'a pink water lily',
    'blackswan': 'a black swan with a red beak swimming in a river near a wall and bushes',
    'surf': 'a man with round helmet surfing on a white wave in blue ocean with a rope',
    'rabbit_watermelon': 'a black and white rabbit is eating a watermelon',
    'train': 'a train traveling down tracks next to a forest filled with trees and flowers and a man on the side of the track',
}

COLOUR = dict(self_replace=.7, cross_replace=.5, guidance=7.5, negative_prompt_inversion=True,
              mask_latent_blend=True, blend_dilate=1)
TEXTURE = dict(self_replace=.6, cross_replace=.6, guidance=10., negative_prompt_inversion=True,
               mask_latent_blend=True, blend_dilate=1)
OBJECT = dict(self_replace=.4, cross_replace=.3, guidance=7.5, negative_prompt_inversion=True,
              mask_latent_blend=True, blend_dilate=2)
STYLE_A = dict(self_replace=.4, cross_replace=.2, guidance=12., negative_prompt_inversion=True,
               mask_latent_blend=False)
STYLE_B = dict(self_replace=.6, cross_replace=.4, guidance=10., negative_prompt_inversion=True,
               mask_latent_blend=False)


def job(name, clip, kind, target, words=(), settings=None, mask='attention', **extra):
    return dict(name=name, clip=clip, kind=kind, source=SOURCE[clip], target=target, edit_words=list(words),
                mask_backend=mask, settings=dict(settings or {}, **extra))


LOCAL = [
    job('colour_flower_blue', 'flower', 'colour', SOURCE['flower'].replace('a red', 'a blue'), ['flower', 'bud'], COLOUR),
    job('colour_bus_red_yellow', 'bus', 'colour', 'a red and yellow bus on the road', ['bus'], COLOUR),
    job('colour_bus_red_yellow__sam', 'bus', 'colour', 'a red and yellow bus on the road', ['bus'], COLOUR, mask='grounded_sam2'),
    job('colour_cat_ginger', 'cat', 'colour', 'a ginger cat walking on the floor next to a wall', ['cat'], COLOUR),
    job('colour_cat_ginger__sam', 'cat', 'colour', 'a ginger cat walking on the floor next to a wall', ['cat'], COLOUR, mask='grounded_sam2'),
    job('texture_swan_crystal', 'swan', 'texture',
        'a Swarovski crystal swan with a red beak swimming in a river near a wall and bushes', ['swan'], TEXTURE),
    job('texture_swan_crystal__sam', 'swan', 'texture',
        'a Swarovski crystal swan with a red beak swimming in a river near a wall and bushes', ['swan'], TEXTURE, mask='grounded_sam2'),
    job('texture_swan_crystal__deep_only', 'swan', 'texture',
        'a Swarovski crystal swan with a red beak swimming in a river near a wall and bushes', ['swan'], TEXTURE,
        sparse_least_channel=640),
    job('texture_swan_crystal__sd21', 'swan', 'texture',
        'a Swarovski crystal swan with a red beak swimming in a river near a wall and bushes', ['swan'], TEXTURE,
        backend='sd21'),
    job('texture_duck_rubber', 'duck', 'texture', 'a sleepy yellow rubber duck', ['duck'], TEXTURE),
    job('texture_duck_rubber__sam', 'duck', 'texture', 'a sleepy yellow rubber duck', ['duck'], TEXTURE, mask='grounded_sam2'),
    job('texture_dog_robotic', 'dog', 'texture', 'a robotic dog sitting on the mat', ['dog'], TEXTURE),
    job('object_bear_tiger', 'bear', 'object', 'a tiger walking on the rock against a wall', ['bear'], OBJECT),
    job('object_bear_tiger__sam', 'bear', 'object', 'a tiger walking on the rock against a wall', ['bear'], OBJECT, mask='grounded_sam2'),
    job('object_fox_wolf', 'fox', 'object', 'a grey wolf sitting in the grass', ['fox'], OBJECT),
    job('background_fox_snow', 'fox', 'background', 'a white fox sitting in the snow', ['grass'], OBJECT),
    job('object_squirrel_eggplant', 'squirrel', 'object', 'a squirrel is eating an eggplant', ['carrot'], OBJECT),
    job('object_squirrel_eggplant__sam', 'squirrel', 'object', 'a squirrel is eating an eggplant', ['carrot'], OBJECT, mask='grounded_sam2'),
    job('object_rabbit_oranges', 'rabbit_strawberry', 'object', 'a rabbit is eating oranges', ['strawberries'], OBJECT),
]

STYLE = [
    job('style_flower_watercolor', 'flower', 'style',
        'a watercolor painting of red flower bud on a green stem with purple-edged leaves, soft washes of colour on textured paper',
        settings=STYLE_A),
    job('style_sunflower_vangogh', 'sunflower', 'style',
        'a Van Gogh oil painting of yellow sunflower, thick swirling brush strokes', settings=STYLE_A),
    job('style_sunflower_vangogh__B', 'sunflower', 'style',
        'a Van Gogh oil painting of yellow sunflower, thick swirling brush strokes', settings=STYLE_B),
    job('style_sunflower_vangogh__sd21', 'sunflower', 'style',
        'a Van Gogh oil painting of yellow sunflower, thick swirling brush strokes', settings=STYLE_A, backend='sd21'),
    job('style_lily_monet', 'lily', 'style',
        'a Claude Monet impressionist painting of pink water lily, loose dabs of colour and soft light', settings=STYLE_A),
    job('style_lily_monet__B', 'lily', 'style',
        'a Claude Monet impressionist painting of pink water lily, loose dabs of colour and soft light', settings=STYLE_B),
    job('style_blackswan_cartoon', 'blackswan', 'style',
        'an anime illustration of black swan with a red beak swimming in a river near a wall and bushes, clean line art and flat cel shading',
        settings=STYLE_A),
    job('style_blackswan_cartoon__B', 'blackswan', 'style',
        'an anime illustration of black swan with a red beak swimming in a river near a wall and bushes, clean line art and flat cel shading',
        settings=STYLE_B),
    job('style_blackswan_cartoon__deep_only', 'blackswan', 'style',
        'an anime illustration of black swan with a red beak swimming in a river near a wall and bushes, clean line art and flat cel shading',
        settings=STYLE_A, sparse_least_channel=640),
    job('style_surf_ukiyoe', 'surf', 'style',
        'a ukiyo-e woodblock print of man with round helmet surfing on a white wave in blue ocean with a rope, bold outlines and flat colours',
        settings=STYLE_A),
    job('style_surf_ukiyoe__deep_only', 'surf', 'style',
        'a ukiyo-e woodblock print of man with round helmet surfing on a white wave in blue ocean with a rope, bold outlines and flat colours',
        settings=STYLE_A, sparse_least_channel=640),
    job('style_rabbit_pokemon', 'rabbit_watermelon', 'style',
        'a pokemon cartoon of black and white rabbit eating a watermelon, clean line art and flat cel shading', settings=STYLE_A),
    job('style_train_shinkai', 'train', 'style',
        'a Makoto Shinkai anime painting of train traveling down tracks next to a forest filled with trees and flowers, '
        'vivid colours and detailed light', settings=STYLE_A),
]

SETUP = r'''
import os, sys, json, hashlib, shutil, subprocess, time, zipfile, urllib.request
from pathlib import Path
candidates = [p.parent for p in Path('/kaggle/input').rglob('launch.py') if (p.parent / 'run_extended.py').is_file()]
assert len(candidates) == 1, f'Attach fatezero-video-editing-code-v2 (found {len(candidates)})'
PROJECT = Path('/tmp/fz_code'); shutil.copytree(candidates[0], PROJECT, dirs_exist_ok=True)
WORK = Path('/kaggle/working'); RUNS = WORK / 'runs'; RUNS.mkdir(exist_ok=True, parents=True)
BASE = Path('/tmp/video_editing_runtime'); BASE.mkdir(exist_ok=True)
ENV = os.environ.copy()
ENV.update(PYTHONNOUSERSITE='1', TOKENIZERS_PARALLELISM='false', HF_HOME=str(BASE / 'huggingface'),
           HF_HUB_DISABLE_TELEMETRY='1', UV_PYTHON_INSTALL_DIR=str(BASE / 'managed_python'),
           UV_CACHE_DIR=str(BASE / 'uv_cache'), CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', SAM2_BUILD_CUDA='0')
ENV.pop('PYTHONPATH', None)
uv = BASE / 'bootstrap' / 'bin' / 'uv'
if not uv.exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', str(BASE / 'bootstrap'), 'uv==0.6.17'], check=True)
venv = BASE / 'modern_venv'; python = venv / 'bin' / 'python'
if not python.exists():
    subprocess.run([str(uv), 'venv', '--python', '3.10.16', str(venv)], check=True, env=ENV)
    subprocess.run([str(uv), 'pip', 'install', '--python', str(python), '-r', str(PROJECT / 'requirements-modern.txt'),
                    '--index-url', 'https://pypi.org/simple', '--extra-index-url', 'https://download.pytorch.org/whl/cu121',
                    '--index-strategy', 'unsafe-best-match'], check=True, env=ENV)
    subprocess.run([str(uv), 'pip', 'install', '--python', str(python), '--no-deps', '-r',
                    str(PROJECT / 'requirements-segmentation.txt')], check=True, env=ENV)
subprocess.run([str(python), '-c', 'import torch,diffusers;print(torch.__version__,diffusers.__version__,torch.cuda.get_device_name(0))'],
               check=True, env=ENV)
'''

DATA = r'''
DATA = Path('/tmp/fz_data'); DATA.mkdir(exist_ok=True)
for z in ('attribute', 'style'):
    if not (DATA / z).exists():
        urllib.request.urlretrieve(f'https://github.com/ChenyangQiQi/FateZero/releases/download/v0.0.1/{z}.zip', DATA / f'{z}.zip')
        zipfile.ZipFile(DATA / f'{z}.zip').extractall(DATA / z)
VIDEOS = WORK / 'clips'; VIDEOS.mkdir(exist_ok=True)
def clip_path(name):
    kind, where, stride, start = CLIPS[name]
    target = VIDEOS / f'{name}.mp4'
    if target.exists():
        return target
    if kind == 'url':
        urllib.request.urlretrieve(where, target)
        return target
    folder = next(p for p in (DATA / kind).rglob(where) if p.is_dir())
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))[start::stride][:8]
    subprocess.run([str(python), '-c', f"""
import imageio.v2 as imageio, numpy as np
from PIL import Image
files = {[str(f) for f in files]!r}
with imageio.get_writer({str(target)!r}, fps=8, codec='libx264', quality=10, pixelformat='yuv420p', macro_block_size=None) as w:
    for f in files:
        im = Image.open(f).convert('RGB'); s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2)).resize((512, 512), Image.LANCZOS)
        w.append_data(np.asarray(im))
"""], check=True, env=ENV)
    return target
'''

RUN = r'''
base_cfg = {b: json.loads((PROJECT / 'configs' / f'improved_{b}.json').read_text()) for b in ('sd14', 'sd21')}
def run_job(j):
    backend = j['settings'].get('backend', 'sd14')
    c = dict(base_cfg[backend])
    settings = {k: v for k, v in j['settings'].items() if k != 'backend'}
    out = RUNS / j['name']
    c.update(video_url='', video_path=str(clip_path(j['clip'])), clip_seconds=1.0, start_frame=0, frame_stride=1,
             frames=8, size=512, steps=30, inversion_iterations=3, seed=0, source_prompt=j['source'], target_prompt=j['target'],
             edit_words=j['edit_words'], mask_backend=j['mask_backend'], mask_prompt=' '.join(j['edit_words']),
             cache_path='/tmp/fz_cache', mask_cache_path='/tmp/fz_mask_cache', keep_cache=True, reuse_cache=True,
             cache_keep_runs=3, output_dir=str(out), reconstruct=False, edit_profile='improved')
    if not j['edit_words']:
        c['mask_latent_blend'] = False
    c.update(settings)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'request.json').write_text(json.dumps(c, indent=2))
    start = time.time()
    with (out / 'log.txt').open('w') as log:
        code = subprocess.run([str(python), str(PROJECT / 'launch.py'), '--config', str(out / 'request.json')],
                              cwd=PROJECT, env=ENV, stdout=log, stderr=subprocess.STDOUT).returncode
    tail = (out / 'log.txt').read_text()[-1500:]
    print(f"{j['name']}: exit {code} in {time.time() - start:.0f}s", flush=True)
    if code:
        print(tail, flush=True)
    return code == 0

done = {}
for j in sorted(JOBS, key=lambda j: (j['clip'], j['settings'].get('backend', 'sd14'), j['settings'].get('sparse_least_channel', 320))):
    try:
        done[j['name']] = run_job(j)
    except Exception as error:
        print(j['name'], 'FAILED', repr(error), flush=True); done[j['name']] = False
print(json.dumps(done))
'''

REPORT = r'''
from PIL import Image, ImageDraw
summary = {}
(WORK / 'grids').mkdir(exist_ok=True)
for clip in sorted({j['clip'] for j in JOBS}):
    runs = [RUNS / j['name'] for j in JOBS if j['clip'] == clip and done.get(j['name'])]
    if not runs:
        continue
    for run in runs:  # one metrics call per run: masks differ, so background regions must not be unioned
        result = subprocess.run([str(python), str(PROJECT / 'metrics.py'), str(run), '--output', str(run / 'metrics.json')],
                                cwd=PROJECT, env=ENV, capture_output=True, text=True)
        if result.returncode:
            print(run.name, 'metrics failed', result.stdout[-500:], result.stderr[-800:], flush=True); continue
        m = json.loads((run / 'metrics.json').read_text())
        r = json.loads((run / 'run_report.json').read_text())
        summary[run.name] = dict(m['runs'][run.name], source_warp=m['source_warp_error'],
                                 source_consistency=m['source_clip_frame_consistency'],
                                 seconds=round(r['runtime_seconds']), vram=round(r['peak_gpu_GiB'], 2))
        print('METRIC', run.name, json.dumps(summary[run.name]), flush=True)
        src = sorted((run / 'source_frames').glob('*.png')); edt = sorted((run / 'edited_frames').glob('*.png'))
        msk = sorted((run / 'masks').glob('*.png'))
        rows = [src, edt] + ([msk] if msk else [])
        T = 192
        grid = Image.new('RGB', (T * 4, T * len(rows) + 18), 'white')
        ImageDraw.Draw(grid).text((4, 2), run.name, fill='black')
        for r_i, row in enumerate(rows):
            for c_i, path in enumerate(row[::2][:4]):
                grid.paste(Image.open(path).convert('RGB').resize((T, T)), (c_i * T, 18 + r_i * T))
        grid.save(WORK / 'grids' / f'{run.name}.png')
(WORK / 'summary.json').write_text(json.dumps(summary, indent=2))
print('SWEEP_COMPLETE', flush=True)
'''


def notebook(jobs, title):
    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': f'# {title}\nDemo-selection sweep; see build_notebooks.py.'},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': SETUP.strip()},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': f'CLIPS = {CLIPS!r}\nJOBS = {jobs!r}\n' + DATA.strip()},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': RUN.strip()},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': REPORT.strip()},
    ]
    for i, c in enumerate(cells):
        c['id'] = f'cell-{i:02d}'
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                         'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


for slug, title, jobs in [('video-editing-demo-sweep-local', 'FateZero Demo Sweep Local', LOCAL),
                          ('video-editing-demo-sweep-style', 'FateZero Demo Sweep Style', STYLE)]:
    folder = HERE / slug
    folder.mkdir(exist_ok=True)
    (folder / f'{slug}.ipynb').write_text(json.dumps(notebook(jobs, title), indent=1))
    (folder / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'shreyaansharma123/{slug}', 'title': title, 'code_file': f'{slug}.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
        'dataset_sources': ['shreyaansharma123/fatezero-video-editing-code-v2'], 'kernel_sources': [],
        'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'}, indent=2))
    print(slug, len(jobs), 'jobs')
