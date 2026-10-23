# Upstream status

Open upstream items as of the 2026-10-22 component audit. Pins live in
`h3_sources.py` and `h3_requirements.py`. Earlier dated audits are in git
history (`git log --diff-filter=D -- 'docs/upstream-audit-*'`). Update this
page when an item is resolved or a new audit finds one.

## Waiting for an upstream release

- **ComfyUI** is pinned to official master `bc5f880d` (2026-10-23) because
  Fibo 2.0 support
  ([Comfy-Org/ComfyUI#17240](https://github.com/Comfy-Org/ComfyUI/pull/17240))
  is not in a release yet. The v0.42.x tags (through v0.42.5) are cut from a
  release branch without it. Pin the first release that includes the merge.

## Do not bump without graph changes

- **LTXVideo** (pinned `3bf3ca62`; upstream was 6 commits ahead on
  2026-10-21). Upstream removes `LTXVSDRToHDRWorkingSpace` and `LTXVSaveHLG`.
  Our SDR→HDR graph in `h3_app/workflows/upscale.py` uses both. Upstream
  replaced them with core colour-space conversion and Create/Save Video.
- **Spectrum-MOSS v0.8.0** (pinned `3fde0674`). Adds a required
  `warn_unverified_sampler` input that our graph does not pass.

## Model candidates

- **Singularity V2 as default.** WarmBloodAban's official
  `Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors` (2026-10-23) is
  the opt-in "Singularity V2" base model. v1.3 stays the default and
  preloaded profile until V2 has been compared on our prompts. There is no
  official V2 W4A8 prune yet.
- **Fibo 2.0 turbo LoRA**
  (`loras/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors`) is in
  `Comfy-Org/Fibo-2.0` without a model card entry or workflow. Wait for
  documented settings.
- **MOSS-Image `moss_image_color_depth_control` LoRA** in
  `Comfy-Org/MOSS-Image` is a feature candidate.

## Feature candidates that need integration work

- **T8 Audio 1.100+**: AudioFirst diagnostics, Wuli Hybrid workflows, and
  LoRA scoping by generated modality. These need graph and UI work, plus a
  model review for Hybrid. Upstream showed no benefit for our
  voice-conditioning graph.
- **Drama 3**: the upstream editor, code persistence and optional forced
  aligner are separate from the decoding fixes we already adopted.

## Unverified

- The 2026-10-23 bump (ComfyUI `bc5f880d`, Kitchen 0.2.40, aimdo 0.6.0,
  Spectrum v0.2.29) has only passed CPU tests. Smoke-test Fibo 2.0
  text-to-image, Qwen Image 2.1, Drama 3 and Spectrum H3 forecasting on a GPU.
  Drama 3 uses Core's `FixedKVBias`/`FixedKVCache` and Qwen3.5 layers, whose
  signatures were unchanged at the new pin.
- The Drama 3 SGLang venv now includes `torch_memory_saver==0.0.10`. How it
  interacts with the sleep/checkpoint path has not been tested on a GPU.
- Drama 3 sleep on drivers 580-609 (Colab) depends on Omni's host-SHM stage
  transport and on checkpointing without `--launch-job`; neither has run on a
  GPU yet. `tests/colab_drama3_sleep.py` is the check. SGLang-Omni has no
  transport setting, so H3 overrides its CUDA platform; drop that if upstream
  adds one.
- No Linux/Python 3.12 dependency lock exists. Git sources are pinned by SHA,
  but most packages use ranges. Record a working resolution once a GPU
  environment is validated.

## Known upstream issue

- [Qwen Image 2.1 Lightning edit drift](qwen-lightning-edit-drift.md): a
  leftward shift in Max resolution edits with the LightX2V Lightning adapter.
  It is reported upstream with no fix. The official eight-step Turbo is the
  available fallback.
