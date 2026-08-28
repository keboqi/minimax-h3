# MiniMax H3 launcher

A standalone Gradio interface and deployment toolkit for MiniMax H3 video
generation on NVIDIA Blackwell GPUs. It provisions ComfyUI, the required H3
models, SLA, Sol-Attn, Comfy Kitchen attention, SageAttention 2, Spectrum, and a
bundled FirstBlockCache node.

## What is included

- Unified H3 video, image-frame, and audio result formats across text,
  first/last-frame, and reference-media conditioning
- Image results expose 1–20 decoded frames in a preview gallery and save only
  the frames selected by the user
- Audio results decode the native H3 stereo soundtrack without creating a video
- A dedicated LTX-2.5 text/image-to-video tab with synchronized audio
- A dedicated MiniMax Music 3 tab for caption-and-lyrics song generation
- All nine official LTX-2.5 ComfyUI workflows for two-stage generation,
  text-to-audio, video editing, reference sheets, motion tracks,
  in/outpainting, and pose/depth/canny control
- LTX-2.5 image-to-video uses a visual start-image input plus optional custom
  middle/end keyframes
- Live queue position, workflow stage, node count, overall work, and sampling schedule
- Resolution-aware thumbnail gallery that loads a video only after it is selected
- Speed and quality NVFP4 profiles plus the official Original BF16 profile
- Selectable official Qwen3-VL 32B NVFP4/AWQ, INT8 ConvRot, and BF16 text encoders
- Optional model offload at every H3 stage boundary, automatically required for BF16
- Default-on reuse of unchanged prompt and image/audio/video conditioning through
  content-addressed ComfyUI input staging
- Per-video SeedVR2 or LTX-2.5 IC-LoRA 2x upscale and frame interpolation
- Selectable SeedVR2 target-frame preprocessing for start/end frames and reference
  images, with downloadable results and no forced downscaling
- Optional generation-stage MiniMax H3 latent 2x upscale, with Balanced BF16,
  Fast FP16, and Quality FP32 model choices
- Selectable Larry v4-600 EMA and official LightX2V 4-step/8-step Turbo LoRAs,
  including the dedicated Ref2V 4-step adapter
- SageAttention 2 as the measured-fastest H3 default, with selectable Comfy
  Kitchen comparison, audio-safe SLA block-sparse attention, and optional
  H3-native zero-copy Sol v0.6.2 sparse attention
- Bit-exact fused H3 modulation projections for LightX2V Turbo
- Two-way feed-forward chunking for ConvRot quality checkpoints
- Optional experimental INT8 ConvRot video VAE, lazy-downloaded on first use
- H.264 NVENC hardware encoding for MiniMax H3 video outputs
- Optional 500K single-frame image decoder, lazy-downloaded only when selected;
  the official FP16 video VAE remains the default
- Hardware-aware ComfyUI memory mode selection
- One-click model unloading and VRAM cache release from the UI
- Spectrum v0.2.15 in legacy mode as the normal-generation default and experimental Turbo option
- FirstBlockCache and native ComfyUI EasyCache alternatives
- Matching local and Modal deployment paths
- Version-aware, resumable Hugging Face model provisioning

## Requirements

- Linux or WSL2 with Bash
- Python 3.12
- An NVIDIA Blackwell GPU and a compatible CUDA 13 driver
- Git, FFmpeg, and enough disk space for ComfyUI and the model set
- An NVIDIA driver/GPU with NVENC support for MiniMax H3 video output
- Hugging Face access to every configured model repository

The installer pins the ABI-sensitive stack to Torch 2.11.0 + CUDA 13.0,
NumPy 1.26.4, and SciPy 1.15.3. The pinned ComfyUI 0.32 stack supplies Comfy
Kitchen attention through its matching `comfy-kitchen` dependency. SageAttention
2.2.0 remains installed from the pinned prebuilt wheel for UI comparisons.
SLA is provided by the pinned PlagueKind node pack at
`6ca3037bd16dc143b6d461c67c87a28ca8074063`. Selecting **SLA** exposes three
quality presets: **Fast** uses validated 0.90 sparsity, **Balanced** uses the
LoRA-distilled 0.85 sparsity, and **Quality** uses 0.85 sparsity plus a dense
final sampling step. In a two-stage latent-upscale workflow the Quality dense
tail applies independently to both sampling stages. Every preset uses 64-token
blocks, protects the audio prefix, and leaves sequences shorter than 8192 tokens
dense. Use SLA with an SLA-distilled H3 LoRA.

The Sol-Attn integration is pinned to the reviewed v0.6.2 commit
`930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf` so its ComfyUI node contract
remains reproducible. v0.6.2 adds MiniMax H3 support for SM86 / RTX 30-series
GPUs without changing the attention math or routing policy. Sol uses the zero-copy H3 path, keeps conditioning KV
exact by default, and leaves its optional INT8 attention approximations
disabled. LightX2V Turbo additionally uses v0.6.0's bit-exact fused modulation
node after its LoRA. Larry Turbo keeps its AdaLN path unfused pending composition
validation. Provisioning validates Larry's pinned node against the upstream
dynamic timestep and AdaLN-row implementation; older compatible source is
patched fail-closed so its E-grid adapter derives the same rows as ComfyUI,
including visual and audio reference-conditioning rows. Quality ConvRot models use
bit-preserving two-way feed-forward chunking above 8K packed tokens.

Spectrum is pinned to v0.2.15 and is applied after LoRA, Sol-Attn, and ConvRot
feed-forward patches. Its default uses system-RAM history and replay archives,
degree-1 forecasting, offline smoothing replay, zero spectral audio blending,
and explicit legacy (`model_aware_mode=off`) scheduling. v0.2.15 adds H3 Continuum
actual-prefix interoperability and closes an ER-SDE solver-space edge case while
preserving the existing scheduling behavior. v0.2.14 added a narrow
native ER-SDE offline-replay guard that avoids KJ preview decode/copy work during
the transformer-free replay while preserving the existing solver behavior. It
also contains
native ER-SDE support and its protected two-actual-step replay tail, while the
current H3 graphs continue to use the reviewed Larry and RES sampler paths.
Spectrum, FirstBlockCache, and EasyCache are mutually exclusive acceleration
choices. Turbo defaults to Spectrum through the reviewed Larry Turbo and
RES multistep sampler paths. EasyCache is also available as an experimental,
default-off Turbo option after ComfyUI's H3 audio-carry fix. FirstBlockCache is
also available as a default-off experimental Turbo option. Attention defaults
to **SLA with the Quality preset**, which uses audio-safe block-sparse attention
and a dense final sampling step. Sage 2 remains available through KJNodes'
per-model override, and Kitchen remains ComfyUI's global backend and a selectable
comparison/fallback. SLA also offers Fast and Balanced presets, while
Auto can
still route jobs at or above 8K estimated packed tokens (and reference-media
jobs) through Sol.
Spectrum exposes one continuous capture-and-replay progress range to ComfyUI,
so the Gradio live progress stream remains active during both passes.

Turbo Spectrum remains approximate. Its conservative policy permits at most one
forecast before a completed native refresh, which limits both acceleration and
trajectory error at four to eight steps. Compare the same prompt and seed with
Acceleration Off before relying on it for quality-critical output.

Turbo defaults to the LightX2V four-step adapter at strength 1.0 (FL2V v1.1
768p or the dedicated Ref2V 544p adapter). Larry v4-600 EMA remains available
at six steps through its pinned custom node, which uses a quantization-aware
bypass loader plus the adaptive H3 Turbo sampler. LightX2V also provides a
mode-specific four-step option (FL2V v1.1 768p or
Ref2V v0.1 544p) and an FL2V v1.0 eight-step 768p option, all at strength 1.0.
The FL2V v1.1 workflow applies LightX2V's recommended video/audio sigma shifts
of 6/3 and uses four Euler steps with the simple scheduler by default.
Loader policy follows the base model: Original BF16 applies either LoRA in
activation space, avoiding reversible weight-merge copies that exceed 96 GiB;
the compact Speed and Quality profiles merge either LoRA for faster inference
and compatibility with direct-weight quantized kernels. LightX2V's Original
bypass validates the exact official 50-block plus two-refiner adapter layout before installing
any hooks and retains fused modulation. Turbo step defaults are applied by an
immediate mode/variant UI update before generation is queued, preventing
The resulting step control remains editable so users can increase any Turbo
variant's count for clips that benefit from additional refinement.

Reference mode automatically selects LightX2V's dedicated Ref2V v0.1 adapter
for the four-step option. The LightX2V eight-step and Larry options still reuse
their FL2VA-trained LoRAs in reference mode and remain experimental because no
dedicated Ref2V counterparts are published. The generated model configuration
keeps separate Ref2VA keys so those shared files can be replaced without
changing workflow construction.

## Run locally

```bash
git clone <repository-url>
cd minimax-h3
bash run_h3.sh
```

The first run creates `h3/`, installs ComfyUI and dependencies, and preloads the
Original FL2VA checkpoint plus the Balanced INT8 text encoder, default FP32 latent-upscaler checkpoint, shared VAEs, and default 4-step
Turbo LoRAs. The Quality Ref2VA checkpoint and selectable 6-step/8-step Turbo
LoRAs download on demand when selected. Speed and Original checkpoints download
on demand the first time each workflow variant is selected. SeedVR2 models and the
LTX-2.5 2x upscaler IC-LoRA are lazy and
download only when their post-processing option is first used. The experimental
INT8 ConvRot video VAE
is also lazy and downloads only when its default-off checkbox is enabled.
The experimental **Single-frame 500K** image VAE is a separate 9.69 GB lazy
download. It is used only for Image results; Video continues to use the
official H3 video VAE regardless of this image setting.
The native H3 latent upscaler is also default-off and lazy-downloads only the
selected checkpoint. **Balanced (BF16)** is the default choice; **Fast (FP16)**
and **Quality (FP32)** remain selectable.
The sampling presets also select the H3 text encoder: **Fast** uses
**NVFP4 / AWQ**, **Balanced** uses **INT8 ConvRot**, and **Quality** uses
**BF16** (51.5 GB). Balanced is the initial preset. The NVFP4/AWQ and INT8
ConvRot (27.1 GB) checkpoints download on first selection. Fast and Balanced
disable model offload by default while leaving the checkbox editable; Quality
automatically enables and locks **Offload models
between H3 stages**. After a fresh encode this keeps the text encoder, diffusion
model, optional latent upscaler, and VAEs from remaining resident together. When
unchanged BF16 conditioning is reused, the encoder never loads and all remaining
stage offloads are skipped for that run. INT8 and NVFP4 keep the current all-VRAM
path by default; stage offload can still be enabled manually for either one.
**Reuse unchanged prompt and media** is enabled by default. Uploaded H3 inputs are
staged under content-derived names, so repeating the same prompt and ordered media
combination reuses the expensive text/media conditioning. The cache identity
contains only the prompt, ordered image/audio/video content, selected text encoder,
and reference-media encoder sizing. Generation-only changes such as seed, steps,
sampler, attention/cache mode, duration, resolution, output format, or latent
upscaling still rebuild the required latent/sampling graph but do not re-run the
text encoder. A changed prompt, media file/order, text encoder, or reference-media
encoder sizing performs a fresh encode and retains normal BF16 stage offloading.
Disable reuse to stage fresh media copies and use unconditional BF16 offloading for
that request.

**Videos per batch** generates one to four variants (one by default). Multi-video
batches assign every video an independent random seed and show all completed videos
in separate players for comparison. The videos run one after another through the
same generation path, like clicking Generate repeatedly with a random seed. The
existing **Reuse unchanged prompt and media** setting is passed through unchanged,
so later variants reuse conditioning only when that setting is enabled.

The gated LTX-2.5 distilled transformers are available as **INT8 ConvRot
(default)** and **BF16**. The selected transformer plus the shared fine-tuned
Gemma text encoder and audio/video VAEs
download lazily when the **LTX 2.5** tab is first used. Switching variants later
downloads only the newly selected transformer. Accept the `Lightricks/LTX-2.5`
Hugging Face license and, before using LTX upscaling, the separate
[`LTX-2.5 2x pixel spatial upscaler`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)
license. For standalone use, authenticate once with `hf auth login`; the server
automatically uses the active CLI credential. `HF_TOKEN` remains supported and
takes priority when set, including for Modal deployment secrets.
Image-to-video accepts a required start keyframe plus optional middle and end
keyframes. The middle position and each keyframe's conditioning strength are
configurable; text-to-video does not load or apply image guides.
The tab's **Official workflows and model downloads** section installs the upstream JSON
templates under **Workflows → Browse → LTX 2.5** in the proxied ComfyUI editor.
Modal re-synchronizes these image-local templates from the pinned LTXVideo node
on every cold start before launching ComfyUI.
Its open **Official workflows and model downloads** panel shows live model
availability and Hugging Face source/license links. It can download every model
for the selected workflow or every missing model in the displayed inventory at
once. The official LTX 2.5 templates intentionally reuse LTX 2.3 IC-LoRAs; this
is not a version mismatch. Each IC-LoRA repository may require its own Hugging
Face license acceptance; accepting the main LTX-2.5 license does not grant
access to all of them.
The **MiniMax Music 3** tab uses the same ComfyUI queue and downloads its selected
INT8 ConvRot or FP16 DiT plus the shared autoregressive encoder and audio decoder
on first use. It supports tagged song sections and a maximum duration of five
minutes, with tiled audio decoding enabled by default for lower peak VRAM.
Later runs check remote metadata for the preloaded set and refresh only stale
files; lazy checkpoints remain local and are fetched again if missing or incomplete.
On Debian/Ubuntu standalone hosts, `run_h3.sh` also installs the `ffmpeg` system
package through `apt-get` (using `sudo` when needed) if `ffmpeg` or `ffprobe` is
missing.
The UI listens on `http://127.0.0.1:7860` by default. Local and Modal launches
use ComfyUI Dynamic VRAM on every GPU size, allowing stage-offload workflows to
move models to system RAM. Without an offload barrier, ComfyUI smart memory can
still retain the compact NVFP4/INT8 stack in VRAM. Do not launch with
`--gpu-only` when using the 51.5 GB BF16 text encoder: under that mode ComfyUI
sets each model's offload device to CUDA, so an unload request cannot release
its VRAM residency.

The **MiniMax H3** tab includes local and Gemini prompt writers. The local writer
uses `lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B` with
`Qwen/Qwen3-VL-8B-Instruct-FP8` by default; the BF16
`Qwen/Qwen3-VL-8B-Instruct` base is selectable. It supports the four tasks used
to train the adapter: T2VA, first-frame I2VA, last-frame L2VA, and first/last
frame FL2VA. The model loads lazily on the first local rewrite and is unloaded
before generation so it does not compete with ComfyUI for VRAM. Local rewriting
uses whole-second durations from 4 through 15 seconds. Override the adapter
repository or local path with `H3_PROMPT_REWRITER_ADAPTER` when needed.

Gemini combines the current text, active first/last-frame or reference
image/video/audio inputs, duration, and resolution with the bundled `prompt.txt`
system instruction. It supports `gemini-3.7-flash`, `gemini-3.6-flash`,
`gemini-3.5-flash`, and `gemini-3.5-flash-lite`. Use Gemini for the separate
Reference media mode, which is not one of the four tasks supported by the local
adapter. Set `GEMINI_API_KEY` in the server environment, or enter a temporary
key in the enhancer panel; a key entered in the UI is passed only to enhancement
requests and is not stored by the server. Uploaded Gemini Files are deleted
after each request. The selected operation is exposed as `/enhance_prompt`.

### H3 result formats

The **Result format** control in the MiniMax H3 tab defaults to **Video** and
does not change conditioning or sampling. H3 still generates its joint visual
and audio latent; the selected format controls the final decode:

For video start frames, **Auto cap** sits beside Width and Height and defaults to
**2 MP**. Select **1 MP**, **2 MP**, **4 MP**, or **8 MP** to choose the maximum automatic
canvas while retaining the uploaded aspect ratio and required model alignment.
Changing the cap recomputes an already-loaded start frame. Manually entered Width
and Height values are not capped.

- **Video** decodes both streams and muxes the existing synchronized MP4.
- **Image** replaces the duration control with a 1–20 frame control (5 by
  default), decodes the requested visual frames, and shows every frame in a
  gallery. Select one or more frames and use **Save selected frames** to copy
  only those PNGs into `ComfyUI/output/h3/images`. With a start frame, Image
  mode uses its native resolution without the video workflow's automatic cap,
  rounded only to H3's required 32-pixel grid (or 64-pixel grid when native
  latent upscale is enabled). **Image VAE** defaults to **Official video VAE**.
  The optional **Single-frame 500K (experimental)** decoder returns exactly one
  image from temporal latent slice 0, matching its published inference recipe.
  It is intended for
  structured graphics, diagrams, documents, UI-like layouts, line art, and
  product contours; the official decoder generally remains preferable for
  natural photographs, fine texture, and small scene text.
- **Audio** decodes the native stereo soundtrack to MP3 and skips video decode
  and muxing. Dialogue, ambience, music, and sound effects continue to come
  from the same H3 prompt and optional audio references. Resolution controls
  are ignored and the visual branch uses the minimum 32×32 canvas.

With the default official VAE, H3's native short temporal packets contain 5 or
22 frames. Image requests up to 5 frames sample the 5-frame packet; requests
from 6 through 20 sample the 22-frame packet and trim the decoded batch to the
exact requested count. The single-frame decoder always samples the shortest
5-frame packet and independently decodes only the first normalized video-latent
slice. This keeps official-versus-500K comparisons on the same denoising
trajectory when both request one image. Use the official VAE for multi-image
results.

The **LTX-2.5** tab and **MiniMax Music 3** tab include their own Gemini prompt
writers. They create or enhance prompts from text plus optional keyframe or
visual-reference images, using `prompt_ltx25.txt` and `prompt_music3.txt`.
Their UI/API operations are exposed as `/enhance_ltx25_prompt` and
`/enhance_music3_prompt`.

### Native H3 latent upscale

Enable **Generate at half resolution, then latent upscale 2x** in the MiniMax H3
generation settings to run the upscaler inside the generation workflow. The UI
width and height always describe the final output: a 1024×1024 request first
finishes a 512×512 H3 generation, upscales its clean video latent 2x, then
lightly re-noises and refines it at 1024×1024. The clean first-pass audio is
preserved for the final output.

This is not a gallery or post-processing option. It is disabled by default,
defaults to two high-resolution refinement steps, and disables cache wrappers
across the two samplers. Enabling it automatically rounds both final dimensions
to the nearest multiple of 64 so the half-resolution pass remains on H3's
32-pixel grid. The selected model is downloaded from
[`LBH-123-AI/Minimax_h3_latent_Upscaler`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
on first use.

### Input image upscale

Open **Upscale input images with SeedVR2** in the MiniMax H3 tab after uploading
first/last frames or reference pictures. Select any populated image slots and run
**Upscale selected inputs to frame**. Choose a **1280×1280**, **1920×1920**, or
**3840×3840** bounding-frame preset, or enter a custom width and height. Each
smaller image is enlarged by the greatest uniform scale that fits inside that
frame, preserving its aspect ratio. Images that cannot be enlarged without
exceeding the frame are left at their original resolution; nothing is downscaled.
For example, a 1920×1920 frame maps 500×700 to 1371×1920, leaves 2048×2048
unchanged, and maps 800×400 to 1920×960. One shared SeedVR2 workflow processes
only the images that need enlargement, replaces those UI inputs automatically,
and exposes every selected result (including unchanged originals) for download.
The SeedVR2 model and VAE remain lazy-downloaded, and the optional resident-model
unload control can reduce peak VRAM before this preprocessing pass.

Generate a video, open **Gallery**, select its thumbnail, and choose a method
under **Post-process selected video**. Each run preserves the source and adds a
new processed video to the gallery. **SeedVR2 2x** uses ComfyUI's native
one-step restoration workflow. **LTX-2.5 IC-LoRA 2x** is a generative alternative
that synthesizes fine detail with the transformer selected in the **LTX 2.5**
tab and the official gated pixel spatial upscaler IC-LoRA. Gallery runs accept
an optional scene prompt; automatic post-processing reuses the H3 generation
prompt. Both upscale methods preserve the source audio and frame rate and can
also run automatically as soon as the base H3 video finishes. Enable
**Unload resident models first** in Gallery, or **Unload H3 models before
upscaling** in MiniMax H3, when
lower peak VRAM is more important than avoiding an H3 model reload on the next
generation. **48 fps interpolation** remains available as a non-upscale option
and requires FFmpeg on the server `PATH`.

LTX-2.5 upscaling remains a single full-video pass by default. If a long or
high-resolution source runs out of VRAM, enable **Split source into clips before
LTX upscaling** in Gallery or the MiniMax H3 post-processing settings. The
default target is 5 seconds per clip; cuts are adjusted to LTX-compatible frame
counts, clips are upscaled sequentially, and their video streams are joined
without an additional video encode. The final file is trimmed to the original
frame count and remuxed with the original source audio. Because clips are
generated independently, a visible detail or motion change can occur at a cut.

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
modal secret create custom-secret HF_TOKEN="$HF_TOKEN"
modal deploy modal_h3.py
```

The deployment attaches the `custom-secret` Modal Secret to both runtime
functions and requires it to contain `HF_TOKEN`. If your existing secret uses a
different name, deploy with `H3_MODAL_HF_SECRET=your-secret-name`. To make the
Gemini enhancer available without entering a key in the UI, also store
`GEMINI_API_KEY` in that Modal Secret.

The deployment pins the immutable ComfyUI v0.34.0 release, which includes
native MiniMax Music 3, its non-dynamic-VRAM fix, LTX 2.5 INT8 support, and
Comfy Kitchen attention.
Changing that pin invalidates the Modal image cache so ComfyUI and its matching
`comfy-kitchen` dependency are rebuilt together.

For local installations, `run_h3.sh` checks the installed ComfyUI revision and
`comfy-kitchen` and frontend-package versions at startup. It also verifies every
static file referenced by the frontend index. It automatically refreshes and
repairs the environment when any check fails; model files are preserved.
Frontend assets are installed in copy mode because aiohttp intentionally rejects
static files that are symlinked outside the package's declared web root.
Both the standalone launcher and Modal deployment also probe the rendered
`/comfyui/` page, its immutable assets, and one encoded nested LTX workflow. A
failed proxy check is reported without withholding the main Gradio UI.
Encoded nested userdata paths are forwarded unchanged so saved workflow folders
load correctly through the proxy, including the bundled LTX 2.5 workflows.

The default LTX 2.5 INT8 option uses the official Comfy ConvRot weights from
`Lightricks/LTX-2.5`. The official workflow inventory also uses
the current LTX-2.5 duration head and spatial/temporal latent upscalers. The
INT8 ConvRot and BF16 options continue to come directly from the same
repository.

Useful environment variables include `H3_MODAL_APP_NAME`,
`H3_MODAL_VOLUME`, `H3_MODAL_MIN_CONTAINERS`,
`H3_MODAL_SCALEDOWN_WINDOW`, and `H3_MODAL_PROXY_AUTH`. Set
`H3_MODAL_PROXY_AUTH=1` for Modal proxy authentication.

The Gradio service also exposes the full ComfyUI interface at `/comfyui/`
on the same public URL. HTTP, uploads, and live WebSocket progress are proxied
to the private ComfyUI backend on port 8188. Its public Uvicorn transport uses
`wsproto` with per-message compression disabled, matching Modal's WebSocket
feature set while remaining compatible with standalone servers.

Runtime-only Python files (`gradio_app.py`, `h3_models.py`, `h3_attention.py`,
`h3_prompt_rewriter.py`, and the bundled H3Acceleration node) are mounted into
Modal containers at startup after the expensive ComfyUI image layer is built. Changes to those files
therefore reuse the cached ComfyUI, CUDA, Torch, and dependency layers. Only
`h3_requirements.py`, which controls build-time package installation and ABI
pins, is copied into an earlier image layer.

## Validation

The fast checks do not download models or require a GPU:

```bash
python3 -m py_compile \
  gradio_app.py h3_attention.py h3_models.py h3_node_patches.py h3_prompt_rewriter.py h3_requirements.py \
  modal_h3.py setup_h3.py custom_nodes/H3Acceleration/__init__.py
python3 h3_requirements.py
python3 h3_models.py
python3 h3_node_patches.py --selftest
python3 h3_attention.py --selftest
python3 h3_prompt_rewriter.py
python3 gradio_app.py --selftest
bash -n run_h3.sh
```

## Repository layout

- `gradio_app.py` — UI, workflow construction, and ComfyUI API client
- `h3_prompt_rewriter.py` — lazy local Qwen3-VL 8B + MiniMax-H3 LoRA writer
- `setup_h3.py` — local environment and model provisioning
- `modal_h3.py` — Modal image, volume, and service lifecycle
- `h3_models.py` — shared model inventory and download logic
- `h3_node_patches.py` — verified compatibility patches for pinned external nodes
- `h3_requirements.py` — shared dependency compatibility policy
- `h3_attention.py` — runtime SageAttention capability probe
- `custom_nodes/H3Acceleration` — bundled H3 acceleration and stage-offload nodes
- Larry's pinned `ComfyUI-MiniMax-H3-Turbo` — quantization-aware Turbo loader/sampler
- Lightricks' pinned `ComfyUI-LTXVideo` plus the pinned KJNodes, ControlNet
  preprocessors, and Video Depth Anything nodes required by the official LTX-2.5
  workflow collection

See [REVIEW.md](REVIEW.md) for review findings and refactor history.
