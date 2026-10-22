# Upstream status

Open upstream items as of the 2026-10-22 component audit. Pins live in
`h3_sources.py` and `h3_requirements.py`. Earlier dated audits are in git
history (`git log --diff-filter=D -- 'docs/upstream-audit-*'`). Update this
page when an item is resolved or a new audit finds one.

## Waiting for an upstream release

- **ComfyUI** is pinned to official master `7a283320` because Fibo 2.0 support
  ([Comfy-Org/ComfyUI#17240](https://github.com/Comfy-Org/ComfyUI/pull/17240))
  is not in a release yet. The latest release is v0.42.1. Pin the first
  release that includes the merge.

## Do not bump without graph changes

- **LTXVideo** (pinned `3bf3ca62`; upstream was 6 commits ahead on
  2026-10-21). Upstream removes `LTXVSDRToHDRWorkingSpace` and `LTXVSaveHLG`.
  Our SDR→HDR graph in `h3_app/workflows/upscale.py` uses both. Upstream
  replaced them with core colour-space conversion and Create/Save Video.
- **Spectrum-MOSS v0.8.0** (pinned `3fde0674`). Adds a required
  `warn_unverified_sampler` input that our graph does not pass.

## Model candidates

- **Singularity V2 INT8**
  (`Minimax-h3_Singularity_ref2va_v2_int8.safetensors`, 34.0 GB). v1.3 stays
  the default. The community prune is available as the opt-in "Singularity V2"
  base model. Switch the default when an official pruned V2 ships.
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

- The Drama 3 SGLang venv now includes `torch_memory_saver==0.0.10`. How it
  interacts with the sleep/checkpoint path has not been tested on a GPU.
- No Linux/Python 3.12 dependency lock exists. Git sources are pinned by SHA,
  but most packages use ranges. Record a working resolution once a GPU
  environment is validated.

## Known upstream issue

- [Qwen Image 2.1 Lightning edit drift](qwen-lightning-edit-drift.md): a
  leftward shift in Max resolution edits with the LightX2V Lightning adapter.
  It is reported upstream with no fix. The official eight-step Turbo is the
  available fallback.
