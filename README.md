# MiniMax H3 launcher

A standalone Gradio interface and deployment toolkit for MiniMax H3 video
generation on NVIDIA Blackwell GPUs. It provisions ComfyUI, the required H3
models, Sol-Attn, SageAttention, Spectrum, and a bundled FirstBlockCache node.

## What is included

- Text/image-to-video and reference-media-to-video workflows
- Live queue position, workflow stage, node count, and sampler-step progress
- Thumbnail gallery that loads a video player only after a generated video is selected
- Speed and quality NVFP4 model profiles
- Selectable Larry v4-600 EMA and LightX2V v0.1 Turbo LoRAs
- H3-native zero-copy Sol v0.6.0 sparse attention with SageAttention fallback
- Bit-exact fused H3 modulation projections for LightX2V Turbo
- Two-way feed-forward chunking for ConvRot quality checkpoints
- Hardware-aware ComfyUI memory mode selection
- Spectrum v0.2.5 as the normal-generation default and experimental Turbo option
- FirstBlockCache and native ComfyUI EasyCache alternatives
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

The Sol-Attn integration is pinned to the reviewed v0.6.0 commit
`bd28909e6a152ba5db4e3590d2d9df1c249943f2` so its ComfyUI node contract
remains reproducible. Sol uses the zero-copy H3 path, keeps conditioning KV
exact by default, and leaves its optional INT8 attention approximations
disabled. LightX2V Turbo additionally uses v0.6.0's bit-exact fused modulation
node after its LoRA. Larry Turbo keeps its AdaLN path unfused pending composition
validation. Provisioning applies a fail-closed patch to Larry's pinned node so
its E-grid adapter derives the same dynamic timestep set as ComfyUI, including
visual and audio reference-conditioning rows. Quality ConvRot models use
bit-preserving two-way feed-forward chunking above 8K packed tokens.

Spectrum is pinned to v0.2.5 and is applied after LoRA, Sol-Attn, and ConvRot
feed-forward patches. Its default uses system-RAM history and replay archives,
degree-1 forecasting, offline smoothing replay, and zero spectral audio blending.
Spectrum, FirstBlockCache, and EasyCache are mutually exclusive acceleration
choices. Turbo defaults to Spectrum through v0.2.5's reviewed Larry Turbo and
RES multistep sampler paths. Turbo continues to reject FirstBlockCache and
EasyCache. Its attention default is Auto: jobs at or above 8K estimated packed
tokens use Sol, while smaller jobs stay dense. Reference-media jobs always use
Sol because their conditioning rows cannot be estimated before ComfyUI encodes
the uploads.
Spectrum exposes one continuous capture-and-replay progress range to ComfyUI,
so the Gradio live progress stream remains active during both passes.

Turbo Spectrum remains approximate. Its conservative policy permits at most one
forecast before a completed native refresh, which limits both acceleration and
trajectory error at four to eight steps. Compare the same prompt and seed with
Acceleration Off before relying on it for quality-critical output.

Turbo defaults to Larry v4-600 EMA at six steps and strength 1.0. Its pinned
custom node uses a quantization-aware bypass loader plus the adaptive H3 Turbo
sampler. LightX2V v0.1 remains selectable as the simpler four-step option using
the core LoRA loader at strength 0.75. Turbo step defaults are applied by an
immediate mode/variant UI update before generation is queued, preventing
Larry's six-step default from surviving a switch to LightX2V. The resulting
step control remains editable so users can increase either Turbo variant's
count for clips that benefit from additional refinement.

Reference mode currently reuses each option's FL2VA-trained Turbo LoRA and is
experimental. Community runs show that this can work, but also report occasional
audio-sync, prompt-adherence, and visual issues. The generated model configuration
keeps separate Ref2VA keys so dedicated releases can replace either shared file
without changing workflow construction.

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

The **API** tab includes a copy-ready Python example. Its `/generate_video`
endpoint only requires a prompt and uses the same defaults shown in the Generate
tab. It returns a public HTTP download URL instead of a client-local temporary
file path. The `/generate_video_advanced` endpoint exposes every generation
control; its current request schema is linked from the API tab.

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

The Gradio service also exposes the full ComfyUI interface at `/comfyui/`
on the same public URL. HTTP, uploads, and live WebSocket progress are proxied
to the private ComfyUI backend on port 8188.

Runtime-only Python files (`gradio_app.py`, `h3_models.py`, `h3_attention.py`,
and the bundled H3Acceleration node) are mounted into Modal containers at
startup after the expensive ComfyUI image layer is built. Changes to those files
therefore reuse the cached ComfyUI, CUDA, Torch, and dependency layers. Only
`h3_requirements.py`, which controls build-time package installation and ABI
pins, is copied into an earlier image layer.

## Validation

The fast checks do not download models or require a GPU:

```bash
python3 -m py_compile \
  gradio_app.py h3_attention.py h3_models.py h3_node_patches.py h3_requirements.py \
  modal_h3.py setup_h3.py custom_nodes/H3Acceleration/__init__.py
python3 h3_requirements.py
python3 h3_models.py
python3 h3_node_patches.py --selftest
python3 h3_attention.py --selftest
python3 gradio_app.py --selftest
bash -n run_h3.sh
```

## Repository layout

- `gradio_app.py` — UI, workflow construction, and ComfyUI API client
- `setup_h3.py` — local environment and model provisioning
- `modal_h3.py` — Modal image, volume, and service lifecycle
- `h3_models.py` — shared model inventory and download logic
- `h3_node_patches.py` — verified compatibility patches for pinned external nodes
- `h3_requirements.py` — shared dependency compatibility policy
- `h3_attention.py` — runtime SageAttention capability probe
- `custom_nodes/H3Acceleration` — bundled FirstBlockCache node
- Larry's pinned `ComfyUI-MiniMax-H3-Turbo` — quantization-aware Turbo loader/sampler

See [REVIEW.md](REVIEW.md) for review findings and refactor history.
