# MiniMax H3 launcher

A standalone Gradio interface and deployment toolkit for MiniMax H3 video
generation on NVIDIA Blackwell GPUs. It provisions ComfyUI, the required H3
models, Sol-Attn, SageAttention, Spectrum, and a bundled FirstBlockCache node.

## What is included

- Text/image-to-video and reference-media-to-video workflows
- A dedicated LTX-2.5 text/image-to-video tab with synchronized audio
- Live queue position, workflow stage, node count, overall work, and sampling schedule
- Resolution-aware thumbnail gallery that loads a video only after it is selected
- Speed and quality NVFP4 profiles plus the official Original BF16 profile
- Per-video SeedVR2 upscale and frame interpolation in the gallery
- Selectable Larry v4-600 EMA and official LightX2V v1.0 4-step/8-step Turbo LoRAs
- H3-native zero-copy Sol v0.6.1 sparse attention with SageAttention fallback
- Bit-exact fused H3 modulation projections for LightX2V Turbo
- Two-way feed-forward chunking for ConvRot quality checkpoints
- Optional experimental INT8 ConvRot video VAE, lazy-downloaded on first use
- Hardware-aware ComfyUI memory mode selection
- One-click model unloading and VRAM cache release from the UI
- Spectrum v0.2.7 in legacy mode as the normal-generation default and experimental Turbo option
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

The Sol-Attn integration is pinned to the reviewed v0.6.1 commit
`e1d211026583064d33dc4326207c6502e2442208` so its ComfyUI node contract
remains reproducible. Sol uses the zero-copy H3 path, keeps conditioning KV
exact by default, and leaves its optional INT8 attention approximations
disabled. LightX2V Turbo additionally uses v0.6.0's bit-exact fused modulation
node after its LoRA. Larry Turbo keeps its AdaLN path unfused pending composition
validation. Provisioning applies a fail-closed patch to Larry's pinned node so
its E-grid adapter derives the same dynamic timestep set as ComfyUI, including
visual and audio reference-conditioning rows. Quality ConvRot models use
bit-preserving two-way feed-forward chunking above 8K packed tokens.

Spectrum is pinned to v0.2.7 and is applied after LoRA, Sol-Attn, and ConvRot
feed-forward patches. Its default uses system-RAM history and replay archives,
degree-1 forecasting, offline smoothing replay, zero spectral audio blending,
and explicit legacy (`model_aware_mode=off`) scheduling. v0.2.7 also contains
native ER-SDE support and its protected two-actual-step replay tail, while the
current H3 graphs continue to use the reviewed Larry and RES sampler paths.
Spectrum, FirstBlockCache, and EasyCache are mutually exclusive acceleration
choices. Turbo defaults to Spectrum through the reviewed Larry Turbo and
RES multistep sampler paths. EasyCache is also available as an experimental,
default-off Turbo option after ComfyUI's H3 audio-carry fix. FirstBlockCache is
also available as a default-off experimental Turbo option. Its attention default
is Auto: jobs at or above 8K estimated packed
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
sampler. The official LightX2V v1.0 adapters are selectable as a four-step 768p
option and an eight-step 544p option, both at strength 1.0. Loader policy follows
the base model: Original BF16 applies either LoRA in
activation space, avoiding reversible weight-merge copies that exceed 96 GiB;
the compact Speed and Quality profiles merge either LoRA for faster inference
and compatibility with direct-weight quantized kernels. LightX2V's Original
bypass validates the exact official 50-block plus two-refiner adapter layout before installing
any hooks and retains fused modulation. Turbo step defaults are applied by an
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

The first run creates `h3/`, installs ComfyUI and dependencies, and preloads the
Quality profile plus the shared text encoder, VAEs, and Turbo LoRAs. Speed and
Original checkpoints download on demand the first time each workflow variant is
selected. SeedVR2 models are lazy and download only when their gallery
post-processing option is first used. The experimental INT8 ConvRot video VAE
is also lazy and downloads only when its default-off checkbox is enabled.
The gated LTX-2.5 distilled transformers are available as **NVFP4 (default and
recommended for Blackwell)**, **INT8 ConvRot**, and **BF16**. The selected
transformer plus the shared fine-tuned Gemma text encoder and audio/video VAEs
download lazily when the **LTX 2.5** tab is first used. Switching variants later
downloads only the newly selected transformer. Accept the `Lightricks/LTX-2.5`
Hugging Face license and set `HF_TOKEN` before the first run.
Image-to-video accepts a required start keyframe plus optional middle and end
keyframes. The middle position and each keyframe's conditioning strength are
configurable; text-to-video does not load or apply image guides.
Later runs check remote metadata for the preloaded set and refresh only stale
files; lazy checkpoints remain local and are fetched again if missing or incomplete.
On Debian/Ubuntu standalone hosts, `run_h3.sh` also installs the `ffmpeg` system
package through `apt-get` (using `sudo` when needed) if `ffmpeg` or `ffprobe` is
missing.
The UI listens on `http://127.0.0.1:7860` by default. Local launch uses ComfyUI
Dynamic VRAM on GPUs below 64 GiB, allowing current releases to manage model
residency, prefetching, and host-memory pressure. It selects `--gpu-only` on
64 GiB+ GPUs, where keeping the full stack resident avoids unnecessary loading.
The 96 GiB Modal target likewise remains GPU-only.

Generate a video, open **Gallery**, select its thumbnail, and choose a method
under **Post-process selected video**. Each run preserves the source and adds a
new processed video to the gallery. **SeedVR2 2x** is the sole upscale option
and uses ComfyUI's native one-step restoration workflow while preserving the
generated audio and frame rate. SeedVR2 can also be selected under
**Generation post-processing** to run automatically as soon as the base H3
video finishes; both the source and upscaled results remain available. Enable
**Unload resident models first** in Gallery, or **Unload H3 models before
SeedVR2** in MiniMax H3, when
lower peak VRAM is more important than avoiding an H3 model reload on the next
generation. **48 fps interpolation** remains available as a non-upscale option
and requires FFmpeg on the server `PATH`.

SeedVR2 offers **3B NVFP4**, **3B INT8**, **7B NVFP4 (default)**, and
**7B Sharp NVFP4** model choices. Only the selected checkpoint downloads on first
use; all choices share the same lazy FP16 SeedVR2 VAE. The native workflow uses
1024-pixel VAE encode/decode tiles for the RTX PRO 6000 target. SeedVR2 and main
H3 generation run eagerly because full-model compile did not improve measured
performance and conflicts with the active attention and cache optimizations.

The **API** tab includes a copy-ready Python example. Its `/generate_video`
endpoint only requires a prompt and uses the same defaults shown in the MiniMax H3
tab. It returns a public HTTP download URL instead of a client-local temporary
file path. The `/generate_video_advanced` endpoint exposes every generation
control; its current request schema is linked from the API tab.
The LTX tab is also available as `/generate_ltx25_video` and shares the same
single-job ComfyUI queue.

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

The deployment pins ComfyUI to a revision with native LTX 2.5 NVFP4 support.
Changing that pin invalidates the Modal image cache so ComfyUI and its matching
`comfy-kitchen` dependency are rebuilt together.

For local installations, `run_h3.sh` checks the installed ComfyUI revision and
`comfy-kitchen` version at startup. It automatically refreshes the environment
once when either is stale; model files are preserved.

The default LTX 2.5 NVFP4 option uses the official packed weights with the
missing ComfyUI quantization markers added. Its SHA-256 is pinned and the
checkpoint header is validated before generation. The INT8 ConvRot and BF16
options continue to come directly from `Lightricks/LTX-2.5`.

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
