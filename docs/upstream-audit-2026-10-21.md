# Component update audit — 2026-10-21

Checked 18 pinned repositories, 30 packages and the main Hugging Face model repositories against upstream. Two upgrades were applied; two upstream changes must not be adopted by a pin bump alone. Raw comparisons are in ignored `.cache/component-audit-2026-10-21/`.

## Applied

**ComfyUI `e3699a74` → `7a283320` (official master).** [Comfy-Org/ComfyUI#17240](https://github.com/Comfy-Org/ComfyUI/pull/17240) merged as `87ed5a06`. The Fibo model, text encoder, node, detection, LoRA and model-base files are identical to the PR head we pinned, except for master's top-p fix ([#17221](https://github.com/Comfy-Org/ComfyUI/pull/17221)), which keeps the token that crosses the threshold in native text generation (Qwen PE, FLUX 3 upsampler). Master also adds PDD-bank LoRA keys, a Pad Image node and an agent host that loads only with `--enable-agent` and imports nothing beyond the standard library otherwise. Requirements move to frontend **1.57.1** and workflow templates 0.11.93; Kitchen 0.2.39, aimdo 0.5.6 and embedded docs are unchanged. No release contains the merge yet (latest v0.42.1); return to one when it ships.

**SGLang-Omni `caae7eb1` → `f31cf9e9`.** Drama 3 changes:

- [#2890](https://github.com/sgl-project/sglang-omni/pull/2890): an uncapped `top_k` now applies only to the main token; codebook heads stay at top-k 30. Our Balance default (top-k 0, sent as `1 << 30`) previously uncapped the heads too, which upstream measured as lower speaker similarity. The sentinel stays within `MAXIMUM_TOP_K` (int32 max), so no payload change was needed.
- [#2810](https://github.com/sgl-project/sglang-omni/pull/2810): semantic sampling keeps the top-p crossing token, so EOS is no longer excluded in near-silent tails.

Our launch flags (`first_frame_cuda_graph_max_requests`, `window_cuda_graphs`, `cuda_graph_backend_prefill`, `codec_path`, memory fractions) still exist at the new revision. The `[tool.uv]` protobuf override is unchanged. Upstream adds `torch_memory_saver==0.0.10` on Linux x86-64; its interaction with the sleep/checkpoint path is unverified. Reference-cache fixes for uploaded voices ([#2836](https://github.com/sgl-project/sglang-omni/pull/2836)) do not apply because we send file paths.

## Do not bump without adaptation

- **LTXVideo** (`3bf3ca62`, 6 behind): removes `LTXVSDRToHDRWorkingSpace` and `LTXVSaveHLG`, both used by our SDR→HDR graph (`h3_app/workflows/upscale.py`). Upstream replaced them with core colour-space conversion and Create/Save Video. Bundled workflow JSONs also changed; filenames are unchanged.
- **Spectrum-MOSS v0.8.0** (`3fde0674`, 9 behind): adds a required `warn_unverified_sampler` input that our graph does not pass. Its ER-SDE bridge does not affect our euler sampler.

## No action

KJNodes (we use only `PathchSageAttentionKJ`), T8 audio (`MiniMaxH3AudioConditioningT8` source unchanged across 27 commits), Spectrum H3, ControlNet Aux and the latent upscaler are as assessed on 2026-10-17. Larry Turbo, Veda, Sol, SLA, Spectrum Qwen, Video Depth, ComfyUI-Drama3, Stable Audio Tools and TRT VAE match HEAD. Model updates for drama-3, Fibo-2.0 and the MOSS DMD LoRA are README-only; the LightX2V Turbo release is already adopted. `Comfy-Org/MOSS-Image` added an optional `moss_image_color_depth_control` LoRA — a feature candidate. No package change is required.

## Validation

`h3_requirements.py` selftest passed. 149 focused tests passed (upstream-upgrade incl. refreshed Comfy requirements fixture, Drama 3 serving, Drama 3, Fibo 2.0, Modal lifecycle, setup worktree); the 3 Gradio frontend tests pass in the UI venv that has Gradio installed. No GPU inference, environment rebuild or deployment was run. Rerun setup and restart ComfyUI/SGLang, or rebuild Modal, to activate the new pins.
