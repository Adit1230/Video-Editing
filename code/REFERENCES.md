# Sources for the changes

Every arXiv ID below was checked against the arXiv API. Statements about reference code were checked
against the repositories at the commits or branches linked here.

## What was already in the provided package (not our change)

| Component | Source |
|---|---|
| Overall method: DDIM inversion that stores attention, then fuses it during editing; middle-frame sparse-causal attention at ≥640 channels; zero-initialised temporal layers; rank-160 temporal convs | **FateZero** — Qi et al., *FateZero: Fusing Attentions for Zero-shot Text-based Video Editing*, ICCV 2023, [arXiv:2303.09535](https://arxiv.org/abs/2303.09535). Code: [ChenyangQiQi/FateZero @ 9b6cb7b](https://github.com/ChenyangQiQi/FateZero/tree/9b6cb7b4dea2b2154df71ad7be0f61a66ca46d88). Config keys `SparseCausalAttention_index: ['mid']`, `least_sc_channel: 640` and `lora: 160` are in `config/style/jeep_watercolor.yaml` |
| Cross-attention replacement, token alignment, `self_replace` / `cross_replace` step fractions | **Prompt-to-Prompt** — Hertz et al., *Prompt-to-Prompt Image Editing with Cross Attention Control*, ICLR 2023, [arXiv:2208.01626](https://arxiv.org/abs/2208.01626) |
| Fixed-point (iterative) DDIM inversion | Iterative inversion as in Pan et al., *Effective Real Image Editing with Accelerated Iterative Diffusion Inversion*, ICCV 2023, [arXiv:2309.04907](https://arxiv.org/abs/2309.04907) |
| Backbone | Stable Diffusion 1.4 — Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022, [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) |

## Our changes and where each came from

| Change (config key) | Source | How we used it |
|---|---|---|
| **Negative-prompt inversion** (`negative_prompt_inversion`) | Miyake et al., *Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models*, [arXiv:2305.16807](https://arxiv.org/abs/2305.16807) | Uses the source prompt as the classifier-free-guidance negative prompt, so unedited content follows the inversion. This is an optimisation-free alternative to **Null-text Inversion** — Mokady et al., CVPR 2023, [arXiv:2211.09794](https://arxiv.org/abs/2211.09794) — which identifies the guidance/inversion mismatch it fixes. |
| **Latent blending outside the mask** (`latent_blend`) | Prompt-to-Prompt's *local blending* ([arXiv:2208.01626](https://arxiv.org/abs/2208.01626)). FateZero also exposes it as `blend_latents: True` in `config/teaser/jeep_posche_local_latent_blend.yaml`; its `jeep_watercolor.yaml` comments that directly copying latents performed "not so good in our case" | We blend with the inverted source trajectory (stored during inversion), using a mask from source cross-attention. In our ablation it raised background PSNR from 28.4 to 37.5 dB. |
| **Cross-frame attention at all resolutions** (`sparse_least_channel=320`) | **Text2Video-Zero** — Khachatryan et al., ICCV 2023, [arXiv:2303.13439](https://arxiv.org/abs/2303.13439). Its `CrossFrameAttnProcessor` ([utils.py](https://github.com/Picsart-AI-Research/Text2Video-Zero/blob/main/utils.py)) replaces every frame's keys/values with the first frame's in all self-attention layers. Related: sparse-causal attention in **Tune-A-Video** — Wu et al., ICCV 2023, [arXiv:2212.11565](https://arxiv.org/abs/2212.11565) — and **Pix2Video** — Ceylan et al., ICCV 2023, [arXiv:2303.12688](https://arxiv.org/abs/2303.12688) | We kept FateZero's middle-frame anchor and lowered the channel threshold so the 64×64 layers use it too. The value 320 is our choice; FateZero's default is 640. |
| **Zero bias on untrained temporal attention** (`zero_temporal_bias`) | **Our own finding**, from the Diffusers 0.11.1 source ([`attention.py`](https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/attention.py)): `CrossAttention.to_out[0]` is `nn.Linear(inner_dim, query_dim)` with a bias | Tune-A-Video (`tuneavideo/models/attention.py`, line 201) and FateZero (`video_diffusion/models/attention.py`, line 224) zero only `to_out[0].weight`. In Tune-A-Video the layer is then trained. With untrained weights, the random bias adds an offset to every block. **This deviates from the FateZero reference.** |
| **Style prompts written style-first** (`app.py` → `style_prompt`) | FateZero's own style configs write the style first, e.g. `"watercolor painting of a silver jeep …"` in `config/style/jeep_watercolor.yaml` | The added visual cues ("soft washes of colour on textured paper") and the style settings (self 0.4 / cross 0.2 / guidance 12 / negative-prompt inversion on) come from **our own sweep** (`kaggle_app/results/style_sweep/`). FateZero instead keeps fusion high (0.8 / 0.8) and amplifies the style word's attention ×10 (`eq_params`, from Prompt-to-Prompt's attention re-weighting), which we did not implement. |
| Stronger attention fusion for colour edits (`self_replace=0.7`, `cross_replace=0.5`) | Prompt-to-Prompt / FateZero step-fraction controls | The values are our own; they were checked by the ablation in `kaggle_improved/results/`. |
| 8 frames over 1 s; attention cache in `/tmp` | Engineering choices (Kaggle T4 memory and disk) | — |

## Evaluation metrics (`metrics.py`)

| Metric | Source | Difference from source |
|---|---|---|
| CLIP text alignment | CLIP — Radford et al., ICML 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020); CLIPScore — Hessel et al., EMNLP 2021, [arXiv:2104.08718](https://arxiv.org/abs/2104.08718) | Reported as 100 × cosine, without CLIPScore's 2.5 weight and clipping |
| CLIP directional similarity | Directional CLIP loss from StyleGAN-NADA — Gal et al., SIGGRAPH 2022, [arXiv:2108.00946](https://arxiv.org/abs/2108.00946); used as an edit metric in InstructPix2Pix — Brooks et al., CVPR 2023, [arXiv:2211.09800](https://arxiv.org/abs/2211.09800) | — |
| CLIP frame consistency | FateZero's evaluation script `CLIP/frame_acc_tem_con.py` (cosine between consecutive frames' CLIP embeddings) | FateZero uses ViT-B/32; we use ViT-L/14 |
| Optical-flow warp error | Lai et al., *Learning Blind Video Temporal Consistency*, ECCV 2018, [arXiv:1808.00449](https://arxiv.org/abs/1808.00449); used for video editing in TokenFlow — Geyer et al., ICLR 2024, [arXiv:2307.10373](https://arxiv.org/abs/2307.10373), Table 1 | Both compute flow on the original video with RAFT ([arXiv:2003.12039](https://arxiv.org/abs/2003.12039)); we use OpenCV's Farnebäck flow (Farnebäck, *Two-Frame Motion Estimation Based on Polynomial Expansion*, SCIA 2003), so absolute numbers are not comparable to those papers |
| Background PSNR, edit-region change, foreground hue | Standard PSNR; the masking and the hue statistic are our own | — |

## Interface (`app.py`)

| Component | Source |
|---|---|
| Web UI and public share link | Gradio — Abid et al., *Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild*, 2019, [arXiv:1906.02569](https://arxiv.org/abs/1906.02569) |

## Software

PyTorch 2.0.1, Hugging Face Diffusers 0.11.1 and Transformers 4.25.1, OpenCV 4.10, Gradio 4.44–5.x. The model checkpoints are
`CompVis/stable-diffusion-v1-4` (revision `2880f2c`) and `openai/clip-vit-large-patch14`.

## Added in the v2 package

| Component | Source |
|---|---|
| Grounded-SAM 2 masks (`mask_backend='grounded_sam2'`) | Grounding DINO — Liu et al., ECCV 2024, [arXiv:2303.05499](https://arxiv.org/abs/2303.05499) (text → box, `IDEA-Research/grounding-dino-tiny`) and SAM 2 — Ravi et al., [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) (box → tracked video masks, `facebook/sam2.1-hiera-tiny`), combined as in Grounded SAM — Ren et al., [arXiv:2401.14159](https://arxiv.org/abs/2401.14159) |
| SD2.1 backbone (`backend='sd21'`) | Stable Diffusion 2.1 base (`sd2-community/stable-diffusion-2-1-base`), a latent diffusion model — Rombach et al., [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) |
| Native video backbone (`backend='cogvideox_2b'`, experimental) | CogVideoX — Yang et al., [arXiv:2408.06072](https://arxiv.org/abs/2408.06072) (`THUDM/CogVideoX-2b`) |
| Streaming attention (`attention_storage='stream'`) | The package's own design: source and edit are denoised together and fused with the source's current-step attention, as Prompt-to-Prompt does for generated images ([arXiv:2208.01626](https://arxiv.org/abs/2208.01626)), instead of replaying maps stored during inversion as FateZero does |
| Demo clips | FateZero's released demo data (`attribute.zip`, `style.zip`, [release v0.0.1](https://github.com/ChenyangQiQi/FateZero/releases/tag/v0.0.1)), with the source prompts from FateZero's `config/attribute` and `config/style` files; flower clip from MDN (CC0) |

| Streaming throughput work (GPU-resident attention cache, replay-window-only recording, OOM fallback) | The package author's own engineering (`test_stream_performance.py`); no external source. The v3 app only turns profiling off and reports cache placement. |

Note: the v3 interface no longer generates edited descriptions from instructions; the user types them. The example descriptions are FateZero's own source prompts, edited as in its `config/attribute` and `config/style` files, plus our style-first wording.
