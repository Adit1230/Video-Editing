"""Single-edit notebook: red bud -> blue bud, saving source, reconstruction and edited frames for a comparison grid.

Settings match the earlier colour_flower_blue sweep run (attention mask, colour defaults, 8 frames, 512 px, 30 steps),
plus reconstruct=True so the reconstruction frames can be shown.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SWEEP = HERE / 'build_demo_sweeps.py'
ns = {'__file__': str(SWEEP)}
_sweep = SWEEP.read_text(encoding='utf-8')
exec(_sweep.split("LOCAL = [")[0], ns)  # SOURCE and COLOUR
SETUP = _sweep.split("SETUP = r'''")[1].split("'''")[0]  # environment cell body
DATA = _sweep.split("DATA = r'''")[1].split("'''")[0]  # dataset-download and clip-writing cell body

CLIPS = {'flower': ('url', 'https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4', None, 0)}
JOB = dict(name='flower_blue_reconstruction', clip='flower', source=ns['SOURCE']['flower'],
           target=ns['SOURCE']['flower'].replace('a red', 'a blue'), edit_words=['flower', 'bud'],
           mask_backend='attention', settings=dict(ns['COLOUR'], reconstruct=True, stream_profile=False))

RUN = r'''
import json, time, subprocess
from pathlib import Path
base = json.loads((PROJECT / 'configs' / 'improved_sd14.json').read_text())
j = JOB
out = WORK / 'flower' / j['name']
c = dict(base, video_url='', video_path=str(clip_path('flower')), clip_seconds=1.0, start_frame=0, frame_stride=1,
         frames=8, size=512, steps=30, inversion_iterations=3, seed=0, source_prompt=j['source'], target_prompt=j['target'],
         edit_words=j['edit_words'], mask_backend=j['mask_backend'], mask_prompt=' '.join(j['edit_words']),
         cache_path='/tmp/fz_cache', mask_cache_path='/tmp/fz_mask_cache', keep_cache=False, reuse_cache=False,
         output_dir=str(out), edit_profile='improved', negative_prompt='')
c.update(j['settings'])
out.mkdir(parents=True, exist_ok=True)
(out / 'request.json').write_text(json.dumps(c, indent=2))
start = time.time()
with (out / 'log.txt').open('w') as log:
    code = subprocess.run([str(python), str(PROJECT / 'launch.py'), '--config', str(out / 'request.json')],
                          cwd=PROJECT, env=ENV, stdout=log, stderr=subprocess.STDOUT).returncode
print('exit', code, f'{time.time() - start:.0f}s', flush=True)
if code:
    print((out / 'log.txt').read_text()[-3000:], flush=True)
    raise SystemExit(code)
'''

GRID = r'''
import cv2, numpy as np
from PIL import Image, ImageDraw
out = WORK / 'flower' / JOB['name']
subprocess.run([str(python), str(PROJECT / 'metrics.py'), str(out), '--output', str(out / 'metrics.json')],
               cwd=PROJECT, env=ENV, check=True)
def frames_from_video(path):
    capture = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames
rows = [('source_frames', sorted((out / 'source_frames').glob('*.png'))),
        ('reconstruction_frames', frames_from_video(out / 'reconstruction.mp4')),
        ('edited_frames', sorted((out / 'edited_frames').glob('*.png')))]
rows = [(name, [Image.open(p).convert('RGB') if isinstance(p, Path) else p for p in items]) for name, items in rows]
T, LABEL = 320, 24
pick = [0, 2, 4, 6]
grid = Image.new('RGB', (T * len(pick), (T + LABEL) * len(rows)), 'white')
draw = ImageDraw.Draw(grid)
for r, (name, items) in enumerate(rows):
    draw.text((4, r * (T + LABEL) + 6), name, fill='black')
    for c, i in enumerate(pick):
        grid.paste(items[i].resize((T, T)), (c * T, r * (T + LABEL) + LABEL))
grid.save(WORK / 'flower_blue_comparison.png')
print((out / 'metrics.json').read_text()[:1500], flush=True)
print('GRID_DONE', flush=True)
'''


def cell(source):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.strip()}


cells = [
    {'cell_type': 'markdown', 'metadata': {}, 'source': '# Video Editing flower run: red bud → blue bud with reconstruction\n'
     'One edit with the colour defaults from the demo sweep; saves source, reconstruction and edited frames.'},
    cell(SETUP),
    cell(f'CLIPS = {CLIPS!r}\nJOB = {JOB!r}\n' + DATA.strip()),
    cell(RUN),
    cell(GRID),
]
for i, c in enumerate(cells):
    c['id'] = f'cell-{i:02d}'
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
(HERE / 'video-editing-flower-run.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
(HERE / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'shreyaansharma123/fatezero-flower-run', 'title': 'FateZero Flower Run',
    'code_file': 'video-editing-flower-run.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
    'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
    'dataset_sources': ['shreyaansharma123/fatezero-video-editing-code-v3'], 'kernel_sources': [],
    'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'}, indent=2), encoding='utf-8')
print('wrote', len(cells), 'cells')
