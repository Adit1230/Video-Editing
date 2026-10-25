# Configurable video editing with diffusion inversion

Update your Kaggle dataset with this ZIP's extracted files and import the included
FateZero_Kaggle.ipynb. The source path defaults to
/kaggle/input/datasets/aditprabhu/fatezero-zip. Enable GPU and Internet.
All source is plain editable Python; no inline source archive or FateZero runtime
imports are used. Diffusers supplies standard pretrained model definitions.

## Model and runtime selection

| backend | Model | Runtime |
|---|---|---|
| sd14 | Original adapted SD1.4 | Original legacy runtime when masks=attention |
| sd21 | Adapted SD2.1 base, including its own CLIP text encoder | Modern |
| cogvideox_2b | Native CogVideoX-2B video transformer and video VAE | Modern |

The original SD1.4 model architecture and editing behavior are retained.
engine.py and run.py now include disk-cache lifecycle fixes; inversion.py,
selftest.py and requirements.txt retain their original code/pins. launch.py dispatches to that runner for the baseline.
SD1.4 with external masks uses the current Diffusers image adapter and modern
runtime; choose original profile + sd14 + attention + disk to return to the original implementation.
Existing config files without the new keys continue to select the baseline.
Direct inversion is explicitly selectable as inversion_method=direct; the legacy
runner receives the equivalent one-iteration/unit-relaxation configuration.

SD2.1 and CogVideoX are alternative complete backbones. Their weights cannot be
mixed by switching individual UNet/text-encoder components. A two-model keyframe
editing/propagation system is a separate pipeline and is not implemented here.
Wan/flow-matching and SDXL are not supported by these adapters.

All diffusion and segmentation model inference requires CUDA in the notebook.
Weight offloading moves inactive models through CPU memory, but does not switch
model computation to CPU. GPU VRAM numbers and visual quality require measurement
on Kaggle. CPU-only developer tests use an explicit --development-cpu flag.

With model_cpu_offload=True (the notebook default), modern adapters explicitly
offload every inactive hooked component before entering a new stage. This also
covers VAE encoding before text encoding and returning to T5 after inversion;
the default one-way hook chain alone does not cover those transitions. Source,
target and negative prompts are encoded once before inversion. Their small
embeddings are cached on CPU and copied to CUDA when needed, so repeated calls
do not reload T5. Active model computation stays on CUDA. Attention head/query
coverage and model weights are unchanged by this memory-management fix.

## Configuration examples

Change these variables in the notebook and rerun configuration onward:

    BACKEND = 'sd14'          # original framework
    MASK_BACKEND = 'attention'

    BACKEND = 'sd21'          # newer image backbone
    MASK_BACKEND = 'grounded_sam2'

    BACKEND = 'cogvideox_2b'   # native video model
    MASK_BACKEND = 'hybrid'

    INVERSION_METHOD = 'direct'       # one inverse DDIM evaluation
    INVERSION_METHOD = 'fixed_point'  # use preset iteration count

    ATTENTION_STORAGE = 'stream'      # paired source/edit denoising, current step only
    ATTENTION_HEADS = 0               # all heads; positive values explicitly limit heads
    ATTENTION_MAX_QUERIES = 0         # stream: use the largest query grids that fit

The source bundle also provides ready-made JSON files under configs/. For CLI
use, run launch.py --config configs/sd21_segmented.json in its matching runtime.
Legacy requirements use PyTorch 2.0.1 CUDA 11.8 / Diffusers 0.11.1; modern requirements
use PyTorch 2.5.1 CUDA 12.1 / Diffusers 0.35.1 / Transformers 4.49.0. The notebook
installs them in separate Python 3.10 environments without modifying its kernel.
SAM 2 is installed only when needed, from a pinned official commit, with the
optional CUDA postprocessing extension disabled. Initial setup needs Internet.
The modern smoke test uses an injected segmentation predictor and does not
require SAM 2 for SD2.1 or CogVideoX with attention/file masks.

## Independent mask options

| mask_backend | Behavior |
|---|---|
| attention | Existing source cross-attention-derived mask |
| sam2 | SAM 2.1 video segmentation initialized with points or boxes |
| grounded_sam2 | Grounding DINO text detection followed by SAM 2.1 tracking |
| hybrid | Intersection of external mask and available attention-derived region |
| file | User-provided source-frame masks |

mask_prompt selects the object for text detection, e.g. flowers. Use sam2.1-hiera-tiny
or small first. mask_instance='all' unions detections; an integer chooses the object
at that zero-based rank in descending detection confidence. Multiple initialized
objects are tracked and unioned. SAM points/boxes refer to the original, uncropped
source image in pixel coordinates. Point labels: 1 foreground, 0 background.

    config.update(mask_boxes=[[100,50,400,300]])
    config.update(mask_points=[[250,150]], mask_labels=[1])

mask_track_stride=1 tracks the dense opening source clip before selecting the
editing frames. Larger strides are faster but can hurt tracking. Corrections use
original source frame indices and must lie within the selected clip interval:

    config['mask_corrections'] = [dict(frame_index=15, object_id=1,
                                      points=[[250,150]], labels=[1])]

For file masks, supply PNGs named by ORIGINAL source frame index (00000.png,
00015.png, etc.) in mask_dir. They must use original-frame coordinates. Their
crop/resize exactly matches video preprocessing. mask_dilation and mask_feather
are applied in output pixels. Empty masks stop by default; mask_allow_empty=True
allows a known full occlusion. Segmentation can still fail; mask accuracy is not
claimed from model names alone. Inspect mask_preview.mp4 and mask_frames/.

The notebook's separate mask-only cell runs before diffusion. Masks are cached
independently of the diffusion checkpoint, target prompt and inversion settings.
Changing mask prompts, tracking inputs, source frames or segmentation weights
invalidates the relevant mask cache. Dilation and feathering reuse raw masks.
Segmentation models are deleted before the diffusion model is loaded.

mask_latent_blend optionally replaces the background with matching source latents
at EVERY denoising step. It is separate from mask-guided attention replay.
mask_composite optionally composites the final pixel edit with the source outside
the external mask, and saves uncomposited edited frames separately. Both default
to False to preserve the previous editing behavior. Tight masks can restrict
shape changes; allow additional space by dilation when appropriate.

## Inversion and attention caches

Fixed-point inversion solves x_t = A*x_previous + B*epsilon(x_t,t) while keeping
x_previous fixed within each timestep. The model adapter converts velocity to
noise using the current noisy guess when needed. Each backend uses its actual
checkpoint schedule with deterministic DDIM steps and sample clipping disabled.
Direct inversion uses exactly one evaluation at the initial cleaner-latent guess.
Fixed-point iteration count and relaxation are configurable per preset.
More iterations do not guarantee convergence. Two attention lifecycles are selectable:

- attention_storage=stream: invert with attention recording disabled on every
  iteration. Starting from the same noisy latent, advance an original-prompt
  source path (guidance 1) and a target-prompt edited path at each timestep.
  Record source attention, use it in that timestep's edited path, then release
  all maps before the next timestep. These are two ordered CUDA forwards per
  timestep. Only the current step is retained, on GPU then RAM; overflow spills to temporary
  disk files which are removed after the edit step or on errors. The cached
  inversion consists only of the noisy latent. Optional reconstruction uses the
  source path's result, and latent preservation uses its updated current latent.
- attention_storage=disk: preserve the previous full inversion-map archive.
  Only the final fixed-point evaluation records attention (before its final
  update); direct inversion records its single evaluation. The edit replays
  inversion maps in reverse order and latent preservation uses inversion anchors.

The notebook defaults to stream for modern backends and disk for the original
SD1.4 baseline. Explicit stream also works with SD1.4 in the modern runtime.
The reference maps differ between these modes: source denoising versus source
inversion. They are not numerically identical editing algorithms, and retaining
more attention does not guarantee better pretrained visual quality. Streaming
trades an extra source denoising pass for bounded attention storage; it is not a
guaranteed speedup.

Modern presets select attention_heads=0 (every head).
attention_max_queries=0 selects every query grid if the current-step estimate
fits available GPU + RAM + spill disk, with reserves; otherwise it logs the largest
grid that fits and keeps every head. There is no arbitrary 8 GiB archive cap in
stream mode. Eligible old inversion caches are reclaimed first when automatic
cleanup is enabled. Positive attention_max_queries values explicitly bound recorded
spatial queries. stream_cache_ram_gib=0 automatically uses available RAM with a
2 GiB working reserve; a positive value sets a smaller RAM cap and spills the
rest. Query chunks bound GPU intermediates without dropping rows or heads.

The native backend retains trained video attention instead of replacing it with
middle-frame image attention. It selectively fuses a limited set of video-query
rows and attention heads in selected joint-attention layers. Text-query outputs
and unselected rows/heads keep the model's own attention. Recorded video-to-text
columns support source-token alignment; changed tokens retain target attention.
Selected video-to-video columns preserve source structure outside editable regions.
This is an experimental adaptation of FateZero's principle, not a validated
reproduction of the original paper on CogVideoX.

Native stream defaults: layers [0,10,20], all heads, automatic query sizing.
A regular spatial query grid is subsampled per latent frame only if needed for
the current-step storage budget or explicitly requested. Disk cache
keys include these recording choices; streaming can reuse its noisy inversion
when only attention bounds change. attention_query_chunk bounds computation memory.
Native layer selection remains explicit because full joint attention across
every video token/layer can exceed available resources even for a single timestep.
cache_max_gib=8 applies only to the full inversion archive in disk mode;
image preflight checks it before recording and the first saved step verifies it.
Recording complete joint-attention matrices is intentionally not the default.
cache_attention=False selects inversion + prompt editing without source replay
in the modern runtime. It does not disable the trained model's own attention.

Source inversion caches include the checkpoint revision, source prompt/frames,
solver settings, code and recording configuration. External masks and target-only
controls reuse inversion. Disk mode also saves source latent trajectories;
stream mode keeps the current source trajectory in memory. KEEP_CACHE=True
retains the successful inversion; changing steps/resolution
or model creates a new cache. Interrupted and failed inversion caches are cleaned automatically; see cache retention below.

## Presets and quality checks

Image presets retain their existing frame counts, sizes and step counts. Native
presets use frame counts of 4*k+1. Fast native uses 9 frames at 320px/15 steps/2
iterations. These low-resolution native previews are performance and wiring
experiments, below the model's original training resolution. Native models are
not promised to be faster or fit 16GB with every configuration. Start small,
inspect reconstruction, and benchmark memory before increasing resolution/frames.
Set reconstruct=True for an additional source-only pass. VAE latent masks account
for the native VAE's temporal compression: first frame plus groups of four frames.
CogVideoX VAE computation defaults to float32 for numerical stability, with FP16
transformer and text-encoder inference; vae_fp32 can be configured.

The sample flower video uses pink flowers -> vivid blue flowers with green leaves.
For another video, describe its actual content, set edit_words/mask_prompt and
change one clear attribute. Weak edits: reduce fusion or increase guidance. Drift:
increase fusion or use a reliable external mask with mask_latent_blend=True.
Few frames over two seconds still mean low playback FPS. Audio is not retained.

Outputs include source.mp4, edited.mp4, comparison.mp4, PNG frames, optional
reconstruction.mp4 and mask previews. run_report.json records checkpoint revision,
prediction type, device, model calls, runtime, VRAM, cache reuse, fusion counts,
pixel change and outside-mask change when masks are supplied. PSNR and pixel
change are diagnostics, not semantic success or temporal quality guarantees.

## Source organization

- Original baseline: engine.py, inversion.py, run.py, selftest.py.
- Configuration/dispatch: options.py, launch.py.
- Model adapters: backends.py.
- Attention replay and spatial-temporal mask mapping: attention_ext.py.
- Segmentation/tracking/cache/previews: masks.py.
- Shared modern inversion/edit loop: run_extended.py.
- Regression tests: test_options.py, test_inversion.py, selftest_extended.py.

See VALIDATION.md for test evidence and limits. LEGACY_NOTES.md retains prior
baseline documentation. Reference algorithm: https://github.com/ChenyangQiQi/FateZero.

## Rerunning without resetting the environment

The pipeline now manages inversion-cache disk space in both the legacy SD1.4
runner and the optional backends. Change the video or config and rerun inference
in the same session. The environment setup only reinstalls when its requirement
signature changes; these cache fixes add no package dependencies.

    config.update(cache_auto_cleanup=True, cache_keep_runs=2, cache_reserve_gib=2.0)

cache_keep_runs includes the protected current cache. Two retains at most the
current and one recent older inversion; older inactive caches may be evicted
further when the new run needs disk space. Use 1 to keep only the current
configuration. KEEP_CACHE=True still preserves that matching inversion for new
target prompts. Set cache_auto_cleanup=False to disable eviction of other runs,
in which case low disk space requires manual cleanup.

Incomplete caches are removed before preprocessing/model loading and on inversion
failure. A completed inversion survives later denoising/export failures when
KEEP_CACHE=True. Reuse validates required files, and incomplete/unreadable current
caches rebuild automatically. Snapshot and inversion-state writes are atomic, so
a disk-full error does not leave a partially committed .pt file. Active cache
directories are protected by OS file locks.

Cleanup only targets recognized pipeline tensor/metadata files in fingerprint
directories under cache_path. It never targets model weights, environments,
segmentation masks, result videos or unrelated directories. The space estimate
requests only future snapshot bytes plus the reserve: the first serialized
snapshot is already on disk and is not counted twice. Each later snapshot also
checks available space.

If a single inversion still cannot fit after older caches are removed, the error
reports the remaining required and available space. Use a larger cache_path
volume, free other large files, or reduce frames/resolution/steps. Restarting the
Python kernel is not a disk-space fix. Attention precision and model settings
are not silently reduced by cleanup.

## Improved editing profile

The notebook now defaults to `EDIT_PROFILE='improved'` with paired streaming.
Choose `EDIT_PROFILE='original'` to restore the original settings; automatic
storage then uses the legacy disk path for SD1.4 with attention masks. Every
improvement is independently overridable. Raw JSON configs retain their existing
explicit values; profile defaults fill only omitted keys. The notebook explicitly
applies the chosen profile before your overrides. See configs/improved_*.json.

| Option | Original | Improved |
|---|---:|---:|
| self_replace / cross_replace | .5 / .3 | .7 / .5 |
| sparse_least_channel | 640 | 320 |
| zero_temporal_bias | false | true |
| negative_prompt_inversion | false | true |
| mask_latent_blend | false | true |
| blend_fraction | 1 | 1 |
| blend_dilate / blend_feather | 0 / 0 | 1 / 1 |

The sample now describes a red flower **bud** changing to a blue bud and uses
`edit_words=['flower','bud']`. Describe your actual clip when replacing it.
The clip duration is automatic by default (roughly one second for image models
and two for native video, extending if needed for distinct frames); resolution, steps,
and inversion iterations remain controlled by the selected speed preset.

SD1.4 and SD2.1 implement the middle-frame K/V threshold and zero output bias on
new, untrained temporal attention blocks. Zero weight plus zero bias makes these
added residual branches identity initially. CogVideoX keeps its pretrained joint
spatiotemporal attention intact: these two image-adapter options are inapplicable
and do not zero or replace its learned temporal attention. Prompt guidance,
attention fusion, masked latent blending and streaming work with all three models.
The report records whether image adaptation was applied.

`negative_prompt_inversion=True` changes ONLY the target CFG reference to the
source prompt: epsilon_source + guidance * (epsilon_target - epsilon_source).
It does not modify the inversion solver or source replay. Disable it to use your
ordinary target `negative_prompt`. CFG=1 source evaluations do not evaluate an
unconditional branch; their effective negative prompt is always empty.

Source inversion, source reconstruction and streamed source replay share a source
prediction helper: source prompt only, guidance=1, empty negative prompt, same
model/precision/input scaling and deterministic DDIM eta=0. Paired denoising has
independent source and target scheduler instances. Target guidance, negative
prompt, fusion and masks never modify the source path. Each timestep executes
source forward -> source step -> target forward/step using that source attention
-> optional target latent blending against the **new source latent** -> delete
all current-step maps/spills. The two trajectories advance in lockstep; they do
not execute concurrently on the GPU or keep two model copies. No attention is
cached during streamed inversion, including its final fixed-point iteration.
Disk mode still records only the final inversion iteration and blends against
its stored inversion trajectory, matching the friend's implementation.

`mask_latent_blend` also accepts the alias `latent_blend`. `blend_fraction` selects
the first fraction of denoising steps to blend. Dilation and feather radii affect
only the target blend mask; source attention and the source path remain intact.
For attention masks, dilation operates on the 32x32 heat grid before resize and
feathering operates at latent spatial resolution, matching the SD1.4 sequence.
For segmentation/hybrid masks, dilation/feathering operate at latent spatial
resolution after the existing crop, temporal pooling and mask processing. SAM,
Grounded SAM, file masks or hybrid masks remain selectable; no SAM install is
needed for attention-only tests. Diffusion masking is separate from optional
final pixel compositing (`mask_composite`).

Modern runs export run_config.json and consensus/segmentation masks in masks/.
Set `COMPUTE_METRICS=True` for the friend's CLIP/flow/background/hue diagnostics;
metrics run in a separate CUDA process after editing. Use metrics.py with multiple
run folders to compare identical source frames under one shared foreground mask.
Changed prompts, differing masks and CLIP scores can confound quality comparisons.
No Gradio interface is included. No new model weights or quality guarantees are
introduced. Full pretrained CUDA quality/VRAM must still be tested on Kaggle.


## Automatic clip duration

The notebook defaults to `CLIP_SECONDS='auto'`. Image presets prefer one second
and CogVideoX prefers two, extending to include the requested number of distinct
source frames when the input FPS is low. A shorter video is accepted if enough
frames remain after START_FRAME. Sampling never duplicates frames or silently
reduces the model's frame count. If the whole remaining video is too short, choose
another video, an earlier start or a smaller valid frame count (CogVideoX uses
4*k+1). Explicit numeric durations remain strict and errors report the source
FPS and minimum duration. `None` retains the previous frame-stride behavior.


## Streaming throughput and GPU cache

The old path wrote each selected source attention chunk to pageable CPU RAM and
copied it back during the target forward. At the reported 5.46 GiB/step this
routes roughly 10.9 GiB per step over PCIe, even after fusion windows expire.
Explicit attention also materializes large score/probability matrices, unlike
fused inversion attention. Numerous 128-row calls and repeated mask interpolation
add overhead. This diagnosis comes from the code and reported payload, not a
local T4 profiler trace.

The notebook now uses:

    STREAM_CACHE_DEVICE = 'auto'
    STREAM_CACHE_GPU_GIB = 0.       # automatic safe VRAM use; positive = cache cap
    STREAM_GPU_RESERVE_GIB = 4.     # activation/workspace headroom
    ATTENTION_QUERY_CHUNK = 512
    PROFILE_STREAMING = True

GPU storage is tried first for each current-step map. Live free VRAM plus reusable
allocator blocks, the configured reserve and optional cap determine placement.
Overflow goes to RAM, then disk, retaining all heads and configured query rows.
`STREAM_CACHE_DEVICE='cpu'` forces CPU placement for comparisons. The reserve is
not an allocation; activations can use that memory. Keep headroom for target CFG,
which uses more activation memory than the source forward.

If a forward runs out of memory while cache tensors reside on GPU, the cache is
moved to RAM/disk and the same forward is retried once. All maps, including ones
already read by a partially completed target forward, survive for the retry.
The source and target scheduler steps advance only after successful predictions.
A smaller learned GPU cap is used on later timesteps instead of permanently
abandoning VRAM. A genuine model OOM with no GPU cache still raises normally.
No model inference moves to CPU.

Self/cross probability recording and target explicit attention stop after their
respective fusion windows. Required attention-mask heat maps continue so masking,
latent blending and exported consensus masks keep working. External segmentation
needs no source heat maps. Mask resampling and token alignment are reused instead
of repeated for each chunk. Full fused SDPA handles unaffected attention where a
CUDA fused kernel is supported; a chunked fallback remains for other devices.
The larger explicit query chunk changes computation batching, not query coverage.
CogVideoX's joint attention uses the same cache placement and lifecycle rules.

`run_report.json` includes `stream_performance`: cumulative source-forward/cache,
target-forward/replay and total paired times, plus a synchronization flag. With
profiling enabled, GPU synchronization bounds each measured forward. Disable
profiling for ordinary runs after comparison. Attention statistics include peak
GPU/RAM cache bytes, attention-cache D2H/H2D transfer bytes, spill volume and OOM
fallbacks. They do not count model-loading, embedding or video-I/O transfers.
Logs also report cache placement. Model parameters, source guidance=1, empty
source negative prompt, deterministic DDIM and target guidance remain unchanged.
Fused/chunked kernels can differ slightly in floating-point rounding; no bitwise
identity or measured Kaggle speedup is claimed without a GPU run.

Existing raw configs retain their query chunk; the notebook and improved JSON
examples select 512. Disk-cache fingerprints include chunk size because snapshot
keys encode chunk boundaries. Streamed noisy-latent reuse is independent of cache
placement, profiling and chunk size. Code changes still invalidate old inversions.


## Interface (`app.py`)

`app.py` is a Gradio interface (notebook `video-editing-app-v3`). The user uploads a clip, types the original
and edited descriptions, picks an edit type and, for local edits, the edit-region words. Each edit runs `launch.py`
(and `metrics.py`) in the modern environment, starting from `configs/improved_<model>.json`. The edit type only sets the
slider defaults (`EDIT_TYPES` in `app.py`); every value stays editable under Advanced. `stream_profile` is switched off for
interface runs. See `REFERENCES.md` for sources.
