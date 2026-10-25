# Validation of independent implementation

Passed locally using torch 2.0.1 CPU and Diffusers 0.11.1:
- Tiny custom video UNet: DDIM inversion, reconstruction, attention-fused edit,
  VAE frame decoding and source-mask generation complete with finite output.
- Changed target token retains its own attention; unchanged aligned tokens use
  source attention. Unconditional CFG attention remains unchanged.
- Attention snapshots load in reverse order, independent of live tensors.
- Tiny video UNet output parity with official FateZero using a fully mapped,
  identical state dictionary (no unmatched keys): max absolute error <1e-6.
  This comparison used the reference source in the development workspace only;
  the deliverable has no FateZero runtime code or dependency.
- CPU-only notebook smoke-test execution is rejected, as intended.
- Notebook cells and Python modules compile; package contains plain source.

Not validated: pretrained full-model GPU execution, Kaggle peak VRAM, wall-time
speedup, prompt fidelity or visual quality. Tiny numerical parity does not imply
full-pipeline or paper-result parity. Source-mask aggregation, token alignment,
fast resolution/steps and colour editing settings differ from published configs.

## Fixed-point update
Analytical tests check the implicit inverse DDIM equation against a known linear
noise predictor, reverse reconstruction residual, fixed anchor, relaxation,
argument validation and final-evaluation recording flags. The GPU smoke test now
checks three UNet evaluations per inversion step with cache modes off/off/invert,
and exactly one snapshot write per step. Historical tiny architecture parity
above applies to the unchanged UNet, not to the new inversion algorithm.

The updated complete tiny-model test passed locally on CPU with torch 2.2.2,
Diffusers 0.11.1 and Transformers 4.38.2 (developer environment only). It verified
six inversion calls for two steps at three iterations, cache modes off/off/invert
for each step, exactly two torch.save calls, finite latents and successful
reconstruction/edit/decoding. The Kaggle environment pins remain unchanged:
Python 3.10, torch 2.0.1 CUDA 11.8 and Transformers 4.25.1. No GPU is available
here to validate those CUDA paths. Notebook setup was also executed against a
nested extracted dataset fixture, and all six preset cells were executed.

## Optional backends update (2026-10-23)

The original engine.py, run.py, inversion.py, selftest.py and requirements.txt
were compared byte-for-byte with their pre-change copies and are unchanged.

New checks passed locally on CPU using torch 2.5.1+cpu, torchvision 0.20.1+cpu,
Diffusers 0.35.1, Transformers 4.49.0 and the pinned SAM 2 source:
- Actual DDIMScheduler and CogVideoXDDIMScheduler reverse/inverse round trips
  for both epsilon and velocity parameterizations.
- Custom CogVideoX joint-attention processor matches its library reference
  when fusion is disabled, including the rotary-embedding path.
- Tiny image model with 48-wide text conditioning and velocity prediction:
  VAE encode, three-iteration inversion, source-attention replay, externally
  masked latent preservation, denoising and VAE decode all complete.
- Tiny native CogVideoX transformer + video VAE + T5 text encoder: complete
  video encoding/inversion/editing/decoding succeeds, including temporal
  compression and background latent preservation.
- Exactly one attention snapshot and one source-latent anchor per timestep;
  model modes are off/off/invert for three fixed-point evaluations.
- Source mask crop/resize, PNG loading by original frame index, native temporal
  max pooling, target-independent mask reuse and video previews pass.
- All model/mask choices pass config checks with appropriate prompts; invalid
  native frame counts, missing manual prompts, and incompatible runtime choices
  fail. Direct inversion dispatches to one unit-relaxation legacy iteration.
- 54 model/preset/automatic-mask notebook configurations execute and compile.

Live Hub metadata/config checks confirmed the four configured model repositories
and their pinned revisions are accessible. The selected SD2.1-base scheduler uses
epsilon prediction; CogVideoX-2B uses v_prediction. The installed SAM 2 checkpoint
mappings and Grounding DINO postprocessing signature were checked against the
pinned libraries. No pretrained diffusion or segmentation weights were used in
the integration tests.

Not established: full pretrained CUDA execution; Kaggle peak memory or speed;
SAM/Grounding DINO mask quality or runtime; native-video reconstruction fidelity;
prompt adherence; absence of flicker. Tiny CPU tests do not establish these. The
native processor deliberately caches a bounded subset of attention; it is an
experimental editing adapter, not a claim of original FateZero result parity.
The notebook defaults to requiring CUDA; developer CPU testing is explicit.

Additional mask-stage contract test passed with an injected SAM predictor: dense
source frames are passed in order, detector boxes initialize the object, a later
point correction reaches the predictor, and only selected source indices are
exported. This uses a test predictor, not pretrained SAM predictions. Native VAE
precision handling and offload-compatible model properties are covered by the
current adapter code; mixed-precision CUDA/offload execution still needs Kaggle.

## Repeated-run cache disk fix

Nine deterministic filesystem regression tests pass without Torch or a GPU.
Both actual attention-recorder classes are executed with a fixed-size test
serializer and a simulated disk to reproduce the reported first-snapshot
failure. Older completed data is evicted, all remaining snapshots fit, and the
next matching run reuses the committed inversion. Separate checks cover
first-snapshot accounting, interrupted-run cleanup, preservation after an edit
failure, stale caches, unknown user files/symlinks, active OS locks, configurable
retention, disabled eviction, atomic-write ENOSPC failure and KEEP_CACHE=False.
The changes affect disk lifecycle and serialization, not diffusion equations,
model weights or attention fusion. Full pretrained GPU execution remains
unverified here; this reported disk failure is reproduced and tested locally.

## SD2.1 smoke test without optional SAM 2

Reproduced the reported ModuleNotFoundError in a clean modern test environment
with no sam2 package: patching a sam2 module path imports that optional module
before the mock can take effect. The tracking contract test now injects its
predictor factory; production tracking still imports the real predictor lazily.
The regression explicitly blocks sam2 imports, verifies attention masks and
mocked tracking need none, and checks that actual SAM tracking reports the
missing segmentation requirements with its original exception preserved.

The complete selftest_extended.py --development-cpu passes in that environment
(Python 3.12, torch 2.5.1+cpu; all other modern requirements at their pinned
versions), including tiny SD2.1 and CogVideoX encode/invert/edit/decode. The
nine cache tests, three options tests and three inversion tests also pass.
Notebook and source compilation pass. No pretrained GPU inference was run;
the notebook retains its CUDA requirement and Python 3.10 runtime pin.

## Current-step source/edit streaming

Added selectable attention_storage=stream alongside the previous disk inversion
archive. All inversion evaluations run with attention recording off; no timestep
maps or latent-anchor files are written during streaming inversion. Paired source
and target denoising start from the same inverted latent. Each source step records
attention for its matching edit step, then all maps and spill files are released.
The source path uses guidance 1; latent preservation uses its updated latent.

Passed with the same no-SAM modern CPU development environment:
- Tiny SD2.1 and CogVideoX stream outputs match an independent offline oracle
  that saves all source-denoising maps first and replays them later. Both RAM
  storage and forced disk spill pass with all heads and external latent masks.
- Fixed-point stream inversion performs all iterations without recording;
  direct inversion performs exactly one evaluation per step without recording.
  Existing disk-mode final-iteration-only attention and anchor checks still pass.
- Image processors match an explicit full attention reference with query chunks,
  middle-frame self attention, cross attention, selected/all heads, and unchanged
  unconditional CFG. Native reference/RoPE checks and unconditional CFG pass.
- Chunking preserves every configured query row/head; forced spill preserves
  tensors and deletes temporary files after each step and injected edit failure.
- The image and native planners retain full query grids when capacity permits; simulated
  capacity shortage selects the largest fitting resolution with all heads kept.
  Disk preflight rejects an undersized archive budget before writing snapshots.
- Stream inversion ignores the archive cap (tested with a near-zero archive cap),
  and cached noisy latents reuse across target/mask/attention-bound changes.
  Solver/checkpoint changes invalidate inversion. A failed streaming edit keeps
  the completed noisy latent reusable.
- Ten disk lifecycle, four options/dispatch and three analytical inversion
  regression tests pass. Notebook compilation and package checks pass.

This establishes wiring and numerical consistency on tiny random models only.
The streaming reference is source-denoising attention, which differs from the
previous inversion-attention algorithm. Full pretrained CUDA/T4 execution,
quality changes, peak memory, spill volume and speed remain unverified here.

## CogVideoX T5 reload OOM after completed inversion

Inspected the pinned Accelerate 1.2.1 CpuOffload/UserCpuOffloadHook and Diffusers
0.35.1 pipeline hooks. The default chain offloads only a component's predecessor;
revisiting the first component (T5) after the transformer does not evict it.
Modern adapters now explicitly offload all inactive hooked components at stage
transitions, including VAE encode -> text, denoiser -> text, and denoiser -> VAE.
Repeated denoiser calls retain the denoiser. Prompt embeddings are cached as
detached CPU tensors and returned as separate copies. The main runner prepares
source, target and negative embeddings before inversion, avoiding T5 reloads.

Passed with pinned modern packages and torch 2.5.1+cpu in explicit development
mode: a regression uses real Accelerate hooks with tiny CPU modules and a
simulated VRAM ledger. It reproduces the original T5 + denoiser overlap OOM,
then verifies exclusive component residency across forward/reverse transitions,
unchanged numerical outputs, and no denoiser reloads during repeated forwards.
Both image/native adapters' complete tiny pipeline tests pass in disk and
stream modes (RAM and forced spill), with each prompt encoded exactly once,
CPU-only cached embeddings and mutation-safe repeated retrieval. Existing
attention, scheduler, mask, cache lifecycle and config tests still pass.

No pretrained CUDA/T4 execution was possible here. Simulated residency proves
the reported overlap is addressed in the pinned hook contract; it does not
establish end-to-end T4 peak memory or rule out unrelated later memory limits.

## Improved profile and conditioning isolation

Passed with Python 3.10, torch 2.5.1+cpu and the pinned modern dependencies:
- Three new regression groups: source/target conditioning isolation across DDIM
  and CogVideoXDDIM schedulers, epsilon/velocity prediction, both target-reference
  choices and forced disk spills; mask dilation/feather/schedule and native temporal
  pooling; profile/alias validation. Target changes do not change source replay.
- Eighteen actual tiny-model adapter cases across SD1.4 (epsilon), SD2.1 (velocity
  in the test fixture), and native CogVideoX: disk, streamed RAM, forced spill,
  and improved streaming with file, attention and hybrid masks. The paired edit
  agrees with an independent offline source-attention replay oracle; each-step
  maps/spills are removed, including on injected target failures.
- Improved image adapters have a zero temporal output weight AND bias and select
  the configured K/V channel threshold. Native pretrained-style attention blocks
  are not modified by these image-only settings.
- Existing cache-manager (10), option/dispatch (4), inversion (3) regressions pass.

These are explicitly developer CPU checks on tiny/random models and analytical
predictors, not pretrained video-quality tests. Notebook adapter tests and all
real diffusion, segmentation and CLIP inference still require CUDA. No local GPU
is available; Kaggle VRAM, reconstruction quality and edit quality are unmeasured.

All 540 notebook combinations of profiles, backends, presets, mask providers and
storage modes execute configuration validation with appropriate mask inputs.
The final notebook cells compile and ZIP source bytes match the working modules.


## Mixed-precision latent-blend fix

Reproduced the reported promotion: an fp32 blend mask multiplied by an fp16
latent tensor makes the next edited latent fp32. The first paired step succeeds,
then the next UNet time-embedding Linear receives fp32 with fp16 weights.
Blending now performs arithmetic in fp32 and explicitly returns the edited
latent's original dtype/device. Both image and native adapter boundaries also
cast latent and text embeddings to the loaded denoiser dtype/device.

Five improvement regression groups now pass, including fp16/bf16 blend dtype
checks for attention, file and hybrid masks; four consecutive strict-fp16 paired
steps and disk replay; and both backend input boundaries with a real half-weight
Linear. Existing tiny-model adapter tests continue to pass. These checks run on
CPU explicitly for development; no pretrained GPU run was available locally.


## Native input clip length fix

Six no-model frame-selection regressions pass: explicit one-second failure at
15 FPS / 17 requested frames; native two-second automatic selection; automatic
extension at low FPS with unique frames; a shorter video with sufficient frames;
insufficient total frames and actionable alternatives; original explicit-duration
and stride behavior; and automatic/invalid configuration validation. Model frame
counts and temporal compression remain unchanged. Tests use no GPU or model
weights because the failure happens in input sampling before diffusion loads.


## Streaming throughput update

Passed on the explicit developer CPU path with pinned modern dependencies:
- GPU/RAM/disk placement policy, optional GPU cap and reserve arithmetic using
  simulated CUDA counters; CPU snapshot independence, real file spill/replacement
  and cleanup. CUDA storage, transfer counters and offload assertions are included
  in the default GPU smoke test but were not executed locally.
- Active-window-only self/cross maps for image and native processors, continuing
  heat-mask generation, bypassed target replay after fusion, and one mask resize
  per grid/device instead of per attention chunk.
- Fused SDPA equivalence and resource-preflight capacity; the existing real tiny
  model suite passes for SD1.4, SD2.1 and CogVideoX across disk, stream, spill,
  improved/file, improved/attention and improved/hybrid cases. Improved cases use
  512-row explicit chunks; small-chunk tests remain. Streamed edits still agree
  with an independent offline source-denoising replay oracle at original tolerances.
- Six conditioning/dtype/retry regression groups and cache/options/frame tests
  pass. Injected source/target CUDA-OOM exceptions on the CPU test path confirm
  that retrying neither advances a scheduler twice nor changes either trajectory. Inversion/replay source conditioning and target guidance are unchanged.

The user's 223-second T4 log and 5.46 GiB/step CPU cache identify costly transfer,
explicit-matrix and small-call paths. No T4 is attached locally. GPU occupancy,
PCIe throughput, CUDA OOM recovery and end-to-end speedup require a Kaggle run;
new synchronized timings and attention transfer/placement counters expose them.

All 540 notebook configuration combinations validate, including legacy dispatch.
The default Kaggle smoke test additionally exercises actual CUDA cache retention,
CPU overflow, transfer counters and GPU-cache offload before full inference.
