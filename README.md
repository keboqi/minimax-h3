# MiniMax H3 launcher

A standalone Gradio interface and deployment toolkit for MiniMax H3 video
generation on NVIDIA Blackwell GPUs. It provisions ComfyUI, the required H3
models, Sol-Attn, SageAttention, and a bundled FirstBlockCache node.

## What is included

- Text/image-to-video and reference-media-to-video workflows
- Speed and quality NVFP4 model profiles
- Optional LightX2V Turbo LoRA
- H3-native zero-copy Sol sparse attention with SageAttention fallback
- Two-way feed-forward chunking for ConvRot quality checkpoints
- Hardware-aware ComfyUI memory mode selection
- FirstBlockCache and native ComfyUI EasyCache support
- Matching local and Modal deployment paths
- Version-aware, resumable Hugging Face model provisioning

## Requirements

- Linux or WSL2 with Bash
- Python 3.12
- An NVIDIA Blackwell GPU and a compatible CUDA 13 driver
- Git, FFmpeg, and enough disk space for ComfyUI and the model set
- Hugging Face access to every configured model repository

The installer pins the ABI-sensitive stack to Torch 2.11.0 + CUDA 13.0,
NumPy 1.26.4, and SciPy 1.15.3. SageAttention is installed from a prebuilt
CPython 3.12 Linux wheel; source compilation is intentionally disabled.

The Sol-Attn integration is pinned to a reviewed Apache-2.0 upstream commit so
its ComfyUI node contract remains reproducible. Sol uses the zero-copy H3 path,
keeps conditioning KV exact by default, and leaves its optional INT8 attention
approximations disabled. Quality ConvRot models additionally use bit-preserving
two-way feed-forward chunking above 8K packed tokens.

## Run locally

```bash
git clone <repository-url>
cd minimax-h3
bash run_h3.sh
```

The first run creates `h3/`, installs ComfyUI and dependencies, and downloads
the models. Later runs check remote model metadata and refresh only stale files.
The UI listens on `http://127.0.0.1:7860` by default. Local launch uses ComfyUI
Dynamic VRAM on GPUs below 64 GiB, allowing current releases to manage model
residency, prefetching, and host-memory pressure. It selects `--gpu-only` on
64 GiB+ GPUs, where keeping the full stack resident avoids unnecessary loading.
The 96 GiB Modal target likewise remains GPU-only.

`run_h3.sh` binds Gradio to `0.0.0.0`, so use host firewall rules or a trusted
network when the machine is reachable by other devices.

To provision without launching:

```bash
python3 setup_h3.py --install-dir ./h3
```

To refresh an existing installation without reinstalling its environment:

```bash
python3 setup_h3.py --install-dir ./h3 --skip-env
```

## Deploy with Modal

Install and authenticate the Modal CLI, then run:

```bash
modal deploy modal_h3.py
```

Useful environment variables include `H3_MODAL_APP_NAME`,
`H3_MODAL_VOLUME`, `H3_MODAL_MIN_CONTAINERS`,
`H3_MODAL_SCALEDOWN_WINDOW`, and `H3_MODAL_PROXY_AUTH`. Set
`H3_MODAL_PROXY_AUTH=1` for Modal proxy authentication.

## Validation

The fast checks do not download models or require a GPU:

```bash
python3 -m py_compile \
  gradio_app.py h3_attention.py h3_models.py h3_requirements.py \
  modal_h3.py setup_h3.py custom_nodes/H3Acceleration/__init__.py
python3 h3_requirements.py
python3 h3_models.py
python3 h3_attention.py --selftest
python3 gradio_app.py --selftest
bash -n run_h3.sh
```

## Repository layout

- `gradio_app.py` — UI, workflow construction, and ComfyUI API client
- `setup_h3.py` — local environment and model provisioning
- `modal_h3.py` — Modal image, volume, and service lifecycle
- `h3_models.py` — shared model inventory and download logic
- `h3_requirements.py` — shared dependency compatibility policy
- `h3_attention.py` — runtime SageAttention capability probe
- `custom_nodes/H3Acceleration` — bundled FirstBlockCache node

See [REVIEW.md](REVIEW.md) for review findings and refactor history.
