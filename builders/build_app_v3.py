"""Builds video-editing-app-v3.ipynb next to this script.

Examples and showcase entries come from examples_app_v3.json (chosen from the kaggle_demo_sweep results).
Example clips are Video Editing's released demo data (attribute.zip / style.zip, 8 frames each) and the MDN flower video.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SPEC = json.loads((HERE / 'examples_app_v3.json').read_text())
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip()})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.strip()})


md("""
# Video editing with diffusion inversion: interactive interface (v3)

**To use it:** in the right-hand panel choose **GPU T4** and turn **Internet on**, then click **Run All**.
After 6–8 minutes (installs and model downloads) the last cell prints a `https://….gradio.live` link and a
username/password. Open the link. The cell keeps running while the interface is up; stop it to shut the interface down.

**In the interface:**
1. Upload a video (or click an example, which fills the whole form) and choose which short clip to edit.
2. Type the **original description** of the video, e.g. `a black cat walking on the floor next to a wall`.
3. Type the **edited description**: the same sentence with only the words you want changed.
   - colour: `a ginger cat walking on the floor next to a wall`
   - texture / material: `a Swarovski crystal swan with a red beak swimming in a river`
   - object: `a tiger walking on the rock against a wall`
   - style (start with the medium): `a watercolor painting of a pink water lily, soft washes of colour on textured paper`
4. Pick the **edit type** (it sets the slider defaults) and, for local edits, the **edit region words**: the word(s) in the
   original description that name what changes, e.g. `cat`.
5. Click **Edit video**. You get original | edited side by side, the edited clip and frames, the edit region, and
   metrics: CLIP text alignment, CLIP directional similarity, temporal consistency (CLIP and optical-flow warp error),
   background preservation (PSNR) and optionally inversion fidelity.
6. The **Showcase** tab shows pre-computed results for every edit type.

**Method:** fixed-point DDIM inversion into Stable Diffusion 1.4 (or SD2.1 / CogVideoX-2B), then denoising toward the
edited description while fusing the source's attention (Video Editing-style), with cross-frame attention, negative-prompt
inversion and latent blending inside an attention or Grounded-SAM 2 mask. No training or fine-tuning.
Code: dataset `shreyaansharma123/fatezero-video-editing-code-v3`. The showcase videos were made with the v2 code.
""")

code(r"""
import os
INTERACTIVE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') != 'Batch'  # 'Batch' = Save Version / API run
LAUNCH = INTERACTIVE
SMOKE_TEST_LINK = not INTERACTIVE  # batch runs: open the public link, send the test edits through it, close it
print('Interactive session' if INTERACTIVE else 'Batch run: link test with example edits')
""")

code(r"""
from pathlib import Path
import hashlib, json, secrets, shutil, subprocess, sys, urllib.request, zipfile

candidates = [p.parent for p in Path('/kaggle/input').rglob('app.py') if (p.parent / 'launch.py').is_file()]
if len(candidates) != 1:
    raise RuntimeError(f'Attach the fatezero-video-editing-code-v3 dataset (found {len(candidates)} code folders)')
PROJECT = Path('/tmp/video_editing_code')
shutil.copytree(candidates[0], PROJECT, dirs_exist_ok=True)
WORK = Path('/kaggle/working')
SHOWCASE_DIR = next((p.parent for p in Path('/kaggle/input').rglob('showcase.json')), None)
print('Code:', candidates[0], '| showcase:', SHOWCASE_DIR)
""")

md("## Isolated editing environment (Python 3.10, torch 2.5.1 cu121, diffusers 0.35.1, SAM 2)")

code(r"""
BASE = Path('/tmp/video_editing_runtime'); BASE.mkdir(exist_ok=True)
ENV = os.environ.copy()
ENV.update(PYTHONNOUSERSITE='1', TOKENIZERS_PARALLELISM='false', HF_HOME=str(BASE / 'huggingface'),
           HF_HUB_DISABLE_TELEMETRY='1', UV_PYTHON_INSTALL_DIR=str(BASE / 'managed_python'),
           UV_CACHE_DIR=str(BASE / 'uv_cache'), CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', SAM2_BUILD_CUDA='0')
ENV.pop('PYTHONPATH', None)
try:
    from kaggle_secrets import UserSecretsClient
    _token = UserSecretsClient().get_secret('HF_TOKEN')
    if _token:
        ENV['HF_TOKEN'] = _token
    del _token
except Exception:
    pass  # all default models are public; a token is optional.
uv = BASE / 'bootstrap' / 'bin' / 'uv'
if not uv.exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', str(BASE / 'bootstrap'), 'uv==0.6.17'], check=True)
venv = BASE / 'modern_venv'; python = venv / 'bin' / 'python'
stamp = BASE / 'modern.sha256'
signature = hashlib.sha256((PROJECT / 'requirements-modern.txt').read_bytes() +
                           (PROJECT / 'requirements-segmentation.txt').read_bytes() + b'python3.10.16-uv0.6.17').hexdigest()
if not python.exists() or not stamp.exists() or stamp.read_text() != signature:
    subprocess.run([str(uv), 'venv', '--python', '3.10.16', str(venv)], check=True, env=ENV)
    subprocess.run([str(uv), 'pip', 'install', '--python', str(python), '-r', str(PROJECT / 'requirements-modern.txt'),
                    '--index-url', 'https://pypi.org/simple', '--extra-index-url', 'https://download.pytorch.org/whl/cu121',
                    '--index-strategy', 'unsafe-best-match'], check=True, env=ENV)
    subprocess.run([str(uv), 'pip', 'install', '--python', str(python), '--no-deps', '-r',
                    str(PROJECT / 'requirements-segmentation.txt')], check=True, env=ENV)
    stamp.write_text(signature)
for test in ('test_options.py', 'test_frame_selection.py', 'test_improvements.py'):
    subprocess.run([str(python), str(PROJECT / test)], cwd=PROJECT, env=ENV, check=True)
# GPU cache placement / OOM-fallback contracts of the streaming path: report, but do not block the interface.
if subprocess.run([str(python), str(PROJECT / 'test_stream_performance.py')], cwd=PROJECT, env=ENV).returncode:
    print('WARNING: test_stream_performance.py failed; see its output above.', flush=True)
""")

md("## Download the default models once (SD1.4, CLIP ViT-L/14, Grounding DINO, SAM 2.1 tiny)")

code(r"""
subprocess.run([str(python), '-c', '''
import json, sys
sys.path.insert(0, "/tmp/video_editing_code")
from options import normalized_config
from backends import resolve_checkpoint
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor
c = normalized_config(json.load(open("/tmp/video_editing_code/configs/improved_sd14.json")))
print("SD1.4:", resolve_checkpoint(c)[0])
CLIPModel.from_pretrained("openai/clip-vit-large-patch14"); CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
snapshot_download(c["grounding_model_id"], revision=c["grounding_revision"])
snapshot_download(c["sam_model_id"], revision=c["sam_revision"])
print("models ready")
'''], cwd=PROJECT, env=ENV, check=True)
""")

md("## Example clips")

code(r"""
sys.path.insert(0, str(PROJECT))
import app
SPEC = """ + repr(SPEC) + r"""
CLIPS_DIR = WORK / 'samples'; CLIPS_DIR.mkdir(exist_ok=True)
DATA = Path('/tmp/fz_data'); DATA.mkdir(exist_ok=True)
for z in ('attribute', 'style'):
    if not (DATA / z).exists():
        urllib.request.urlretrieve(f'https://github.com/ChenyangQiQi/FateZero/releases/download/v0.0.1/{z}.zip', DATA / f'{z}.zip')
        zipfile.ZipFile(DATA / f'{z}.zip').extractall(DATA / z)

def clip_path(name):
    kind, where, stride, start = SPEC['clips'][name]
    target = CLIPS_DIR / f'{name}.mp4'
    if target.exists():
        return target
    if kind == 'url':
        urllib.request.urlretrieve(where, target)
        return target
    folder = next(p for p in (DATA / kind).rglob(where) if p.is_dir())
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))[start::stride][:8]
    subprocess.run([str(python), '-c', f'''
import imageio.v2 as imageio, numpy as np
from PIL import Image
files = {[str(f) for f in files]!r}
with imageio.get_writer({str(target)!r}, fps=8, codec="libx264", quality=10, pixelformat="yuv420p", macro_block_size=None) as w:
    for f in files:
        im = Image.open(f).convert("RGB"); s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2)).resize((512, 512), Image.LANCZOS)
        w.append_data(np.asarray(im))
'''], check=True, env=ENV)
    return target

TYPES = {'APPEARANCE': app.APPEARANCE, 'TEXTURE': app.TEXTURE, 'OBJECT': app.OBJECT, 'STYLE': app.STYLE}
EXAMPLES = [[str(clip_path(e['clip'])), e['source'], e['target'], TYPES[e['kind']], e['region'], 0, e['seconds'], e['mask']]
            for e in SPEC['examples']]
SHOWCASE = []
if SHOWCASE_DIR:
    for item in json.loads((SHOWCASE_DIR / 'showcase.json').read_text()):
        SHOWCASE.append((item['title'], str(SHOWCASE_DIR / item['video']), item['caption']))
print(len(EXAMPLES), 'examples,', len(SHOWCASE), 'showcase videos')
""")

md("## Interface")

code(r"""
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio>=4.44,<6'], check=True)
editor = app.Editor(python, ENV, WORK / 'app_runs')
demo = app.build_ui(editor, EXAMPLES, SHOWCASE)
USER, PASSWORD = 'editor', secrets.token_urlsafe(9)
""")

code(r"""
if SMOKE_TEST_LINK:
    from gradio_client import Client, handle_file
    demo.queue(max_size=4).launch(share=True, prevent_thread_lock=True, auth=(USER, PASSWORD),
                                  allowed_paths=[str(WORK), '/kaggle/input'], max_file_size='200mb', show_error=True)
    print('SHARE_URL', demo.share_url, flush=True)
    # The public link is checked with the quick edit_defaults call; edits go through the local server so a
    # dropped gradio.live tunnel (a share-service outage) does not abort the test of the app itself.
    client = Client(demo.local_url, auth=(USER, PASSWORD), verbose=False)
    try:
        share = Client(demo.share_url, auth=(USER, PASSWORD), verbose=False)
    except Exception as error:
        share = None
        print('SHARE_LINK_UNAVAILABLE', repr(error)[:300], flush=True)
    results = WORK / 'link_test'; results.mkdir(exist_ok=True)
    for test in SPEC['link_tests']:
        e = SPEC['examples'][test['example']]
        edit_type = TYPES[e['kind']]
        try:  # the type-defaults event, over the public link
            defaults = share.predict(edit_type, api_name='/edit_defaults')
            print('SHARE_LINK_OK', flush=True)
        except Exception as error:
            print('SHARE_LINK_FAILED', repr(error)[:300], flush=True)
            defaults = client.predict(edit_type, api_name='/edit_defaults')
        self_r, cross_r, blend, guidance, npi, sharing = defaults
        print('REQUEST', json.dumps(dict(source=e['source'], target=e['target'], edit_type=edit_type, region=e['region'],
                                         mask=e['mask'], defaults=defaults)), flush=True)
        out = client.predict({'video': handle_file(str(clip_path(e['clip'])))}, e['source'], e['target'], edit_type,
                             e['region'], test.get('backbone', 'Stable Diffusion 1.4'), e['mask'],
                             app.STANDARD, 8, 0, e['seconds'], guidance, self_r, cross_r, blend, npi, sharing, 0, False, True,
                             api_name='/edit')
        summary, metrics = out[7], out[3]
        print('LINK_TEST_SUMMARY', json.dumps(summary), flush=True)
        print('LINK_TEST_METRICS', json.dumps(metrics), flush=True)
        job = Path(summary['output_dir'])
        report = json.loads((job / 'run_report.json').read_text())
        stats = report['attention_stats']
        print('LINK_TEST_STREAM', json.dumps(dict(name=e['name'], stream_performance=report.get('stream_performance'),
              gpu_peak_cache=stats.get('stream_peak_gpu_bytes'), d2h=stats.get('d2h_bytes'), h2d=stats.get('h2d_bytes'),
              oom_fallbacks=stats.get('gpu_cache_oom_fallbacks'))), flush=True)
        name = f"{test['example']:02d}_{e['name']}"
        shutil.copy(job / 'comparison.mp4', results / f'{name}.mp4')
        shutil.copy(job / 'comparison_first_frame.png', results / f'{name}.png')
    demo.close()
    print('LINK_TEST_COMPLETE', flush=True)
elif LAUNCH:
    print(f'Username: {USER}\nPassword: {PASSWORD}\nOpen the public URL printed below. Stop this cell to shut the interface down.',
          flush=True)
    demo.queue(max_size=4).launch(share=True, auth=(USER, PASSWORD), allowed_paths=[str(WORK), '/kaggle/input'],
                                  max_file_size='200mb', show_error=True, debug=True)
""")

notebook = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:02d}'
(HERE / 'video-editing-app-v3.ipynb').write_text(json.dumps(notebook, indent=1))
sources = ['shreyaansharma123/fatezero-video-editing-code-v3'] + (['shreyaansharma123/fatezero-demo-showcase'] if SPEC.get('showcase') else [])
(HERE / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'shreyaansharma123/fatezero-video-editor-app-v3', 'title': 'FateZero Video Editor App v3',
    'code_file': 'video-editing-app-v3.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
    'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True, 'dataset_sources': sources, 'kernel_sources': [],
    'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'}, indent=2))
print('wrote', len(cells), 'cells')
