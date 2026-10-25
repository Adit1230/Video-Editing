# Video Editing: text-guided video editing with diffusion inversion

Edit a short video from a text description: invert the clip into Stable Diffusion, then regenerate it from the
edited description while reusing the source's attention, so the structure, motion and background stay put.
Training-free. A Gradio interface lets you upload a video, type the original and edited descriptions, and get the
edited clip with quality metrics.

## What is in this folder

| Folder | Contents |
|---|---|
| `code/` | The final implementation: editing engine, streaming attention, backends (SD1.4, SD2.1, CogVideoX-2B), masks (attention, Grounded-SAM 2, hybrid), metrics, the Gradio app (`app.py`), configs and tests. Start with `code/README.md`. |
| `notebooks/` | The Kaggle notebooks that were run: the interface (`video-editing-app-v3`), the two demo sweeps and the flower run. |
| `builders/` | The scripts that generate those notebooks, plus the example specification (`examples_app_v3.json`). |
| `results/` | Every output and metric from the runs: videos, frames, per-run metrics and reports, sweep grids, and `all_metrics.csv`. |

## Reproduce

1. Upload `code/` to Kaggle as a private dataset (the notebooks expect the name `fatezero-video-editing-code-v3`).
2. Open a notebook from `notebooks/`, set **GPU T4** and **Internet on**, and click **Run All**.
   - The interface notebook prints a public link and a username/password.
   - The demo sweep and flower notebooks save their outputs under `/kaggle/working`.
3. To regenerate a notebook, run its builder in `builders/`. The builders use the bundled
   `builders/examples_app_v3.json` and `builders/build_demo_sweeps.py` paths.

Local tests that need no GPU: `python code/test_options.py`, `python code/test_frame_selection.py` and
`python code/test_inversion.py`. The GPU tests are in `code/selftest_extended.py` and `code/test_stream_performance.py`.

## Results

Metrics are in `results/all_metrics.csv` (one row per run). The main ones:

- **CLIP text alignment**: how well the edited frames match the edited description (higher is better).
- **CLIP directional similarity**: whether the change points in the direction the text asks for (higher is better).
- **Warp error**: flicker between frames after following the source motion (lower is better). The source video's own
  warp error is the floor: about 1.7 for the flower clip, 5 to 27 for the others.
- **Background PSNR**: how well everything outside the edit region is kept (higher is better).
- **Reconstruction PSNR**: how faithfully the inversion rebuilds the source (higher is better).

Highlights:

| Edit | Folder | Notes |
|---|---|---|
| Red bud → blue bud | `results/flower_blue_reconstruction/` | Reconstruction 35.6 dB, background 36.3 dB, warp error 2.4 (source 1.7) |
| Black kitten → ginger | `results/showcase/`, `results/demo_sweep/local/` | Grounded-SAM 2 mask, background 37.1 dB |
| Black swan → crystal swan | `results/showcase/` | Grounded-SAM 2 mask |
| Bear → tiger | `results/showcase/`, `results/demo_sweep/local/` | Grounded-SAM 2 mask, strongest CLIP directional score (0.43) |
| Sunflower → Van Gogh | `results/showcase/`, `results/demo_sweep/style/` | Style edit, layout kept |
| Rabbit → pokemon, surfer → ukiyo-e, lily → Monet, flower → watercolor | `results/showcase/`, `results/demo_sweep/style/` | Style edits |

Edits that failed and are not in the showcase: robotic dog, bus recolour, black swan → cartoon, train → Shinkai.

Demo sweep: 32 edits (19 local, 13 style) on FateZero's own demo clips, run with the v2 code. The interface test
edits were run with v3, and their metrics match the sweep to about 0.02.

## Notes and limits

- **Sources:** `code/REFERENCES.md` lists the papers and code each part comes from, and which parts are our own.
- **Speed:** a Standard-quality edit takes about 4–6 minutes on a T4 with the v3 streaming attention cache. v2 took 9–11 minutes.
- **Memory:** the v3 attention cache peaks near 12.5 GiB of the T4's 14.5 GiB. It falls back to CPU memory if the GPU runs out.
- **Pretrained models are not included.** The code downloads Stable Diffusion 1.4 (and optionally 2.1, CogVideoX-2B, SAM 2 and Grounding DINO) from Hugging Face.
- **Tuning:** the defaults for each edit type are in `code/app.py` (`EDIT_TYPES`). They come from the sweeps above, not from a formal benchmark.
- **Videos:** the outputs are short clips (8 frames, about 1 second). Warping still appears on moving camera shots; deep-layer sharing reduces it only slightly.
- **Not included:** the teammate's original `FateZero_Kaggle.ipynb` package notebook, the superseded v1 code (`fatezero_code/`), and the Kaggle credentials.
