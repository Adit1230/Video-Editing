# Editable FateZero-method video editor

Upload FateZero_Kaggle_Package.zip as a Kaggle dataset, attach it to the accompanying
notebook, enable GPU and Internet, then run all cells. The notebook reads the extracted files under
`/kaggle/input/datasets/aditprabhu/fatezero-zip` and copies them to a writable folder.
Update that dataset with the new source files, including `inversion.py`, before running. Kaggle may unpack the ZIP;
both directories and ZIP files are accepted. There is no inline encoded payload.

## Our implementation

`engine.py` implements the video architecture, token alignment, fixed-point DDIM inversion,
reverse attention replay, foreground masks and guided denoising. `run.py` handles
video download, opening-clip sampling, model loading and outputs. `selftest.py`
checks the custom pipeline using tiny random weights without downloading a model.
No FateZero clone, vendored modules, or runtime imports are required.
Standard pretrained SD components are loaded using Diffusers, Transformers and
PyTorch. This is editable source rather than a wrapper around FateZero's pipeline.

Reference: https://github.com/ChenyangQiQi/FateZero at commit
9b6cb7b4dea2b2154df71ad7be0f61a66ca46d88.

The model retains the SD1.4 spatial UNet, rank-160 residual temporal convolution,
spatial attention with middle-frame keys/values at channels >=640, and temporal
attention after the feed-forward blocks. Residual GroupNorm includes the frame
dimension; transformer GroupNorm remains per frame. Temporal up projections are
zero initialized. The VAE and CLIP use the original SD1.4 checkpoint weights.

A tiny architecture comparison with identical weights matches the official UNet
within 1e-6 on CPU. This does NOT establish full-size, fp16 GPU or visual parity.
New temporal parameters initialize independently unless an identical state is
loaded. No paper-result or pixel-identical reproduction is claimed.
The token aligner, resolution-independent mask aggregation and video I/O are our
implementations. Fast settings intentionally differ from published experiments.

## Visible edit and speed

The sample link is a flower video. Source prompt describes pink flowers; target
prompt requests vivid blue flowers with the same green leaves. Changed tokens use
target cross attention rather than having source colour attention forced onto them.
`edit_words=['flowers']` derives a foreground mask from source cross attention,
allowing target self attention there and preserving source attention outside.
No postprocessing recolouring is applied: the result comes from diffusion editing.
Neither the colour change nor perfect background preservation is guaranteed.

Fast default: 4 frames, 320px square, 15 steps, 2 fixed-point iterations per step, self_replace=.5, cross_replace=.3,
guidance=7.5, reconstruction=False, keep_cache=True. The old default ran 30 steps
at 512px plus a reconstruction pass. The new default uses 45 UNet calls (30 inversion + 15 edit) rather
than the old 90 on a fresh run; actual wall-time speedup is hardware dependent. On reuse,
only the 15-step edit runs. Four frames over two seconds remains only 2 FPS.

The `reference` preset restores 512px/30 steps and conservative fusion, with the
foreground mask disabled. It is a comparison configuration, not an exact paper
experiment. No smaller backbone is enabled: that changes architecture and quality.
The optional quality preset may require substantially more memory and disk.

If editing is too weak, lower self_replace to .3 and cross_replace to .15, or try
guidance=9. If structure drifts, increase fusion. For another video, update both
prompts and edit_words. Keep the nouns and scene description consistent and change
one obvious attribute at a time. Inspect comparison_first_frame.png and the entire
comparison.mp4. Pixel change in run_report.json is only a diagnostic, not proof
that the target attribute changed.

## GPU and cache

Inference and the notebook model test require CUDA; they fail without a GPU.
CPU handles download, frame/video I/O and disk caches. The explicit developer-only
`selftest.py --development-cpu` path allows tiny wiring tests locally and is never
selected by the Kaggle notebook. GPU memory and pretrained visual quality remain
unverified in this development environment.

At every DDIM noise level, fixed-point inversion solves
`x_t = A*x_previous + B*epsilon(x_t,t)` by repeatedly evaluating epsilon at the
current guess, with x_previous held constant. The guess starts at x_previous.
Solver updates use fp32; UNet calls retain fp16 CUDA inference.
This uses deterministic DDIM algebra, but inversion is now iterative, not a
single direct DDIM estimate. It is not claimed to reproduce original FateZero's
inversion results.

| Preset | Fixed-point iterations per inversion step |
|---|---:|
| smoke | 1 |
| fast | 2 |
| default | 3 |
| balanced | 3 |
| reference | 5 |
| quality | 5 |

Override `config['inversion_iterations']` to choose another positive integer.
`inversion_relaxation` defaults to 1; try .5 if the iteration is unstable.
There is no automatic early stopping: the count is exact and includes the final
attention-recording evaluation. More iterations do not guarantee convergence.
The last recorded maps are from the final iteration's UNet evaluation, before
that iteration updates the guess; there is no additional UNet pass after solving.
All earlier iterations use attention with recording disabled and produce no CPU
attention copies or files. Each noise level writes exactly one CPU .pt snapshot. Editing loads one
snapshot at a time in reverse order, transferring needed tensors to CUDA.
`inversion.pt` stores the final noisy latent and snapshot names. Cache identity
includes source frames, source prompt, code, steps, resolution, seed, checkpoint, iteration count and relaxation. Changing any inversion
setting invalidates old caches; changing target-only controls does not.
Changing target prompt or target-only editing controls reuses inversion. A failed cache is
rejected. KEEP_CACHE=True preserves it for later edits; remove old fingerprint
folders manually when no longer needed.

Use the isolated Python 3.10 / torch 2.0.1 CUDA 11.8 environment installed by the
notebook. Keep the pinned Diffusers 0.11.1 API. A single 16GB T4/P100 is the intended
target, not a measured memory guarantee. Two GPUs do not pool VRAM. No training is
performed, and audio is not retained.

Outputs: edited.mp4, source.mp4, comparison.mp4, PNG frames, run_report.json,
and optional reconstruction.mp4. The report records CUDA device, UNet calls,
attention-fusion counts, timing, memory and mean absolute pixel change.

`test_inversion.py` contains no-download analytical solver tests; `selftest.py` additionally tests the actual model, cache modes and exact snapshot count on GPU.
