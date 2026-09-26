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
- A dedicated Qwen Image 2.1 tab for native text-to-image generation and
  multi-reference image editing
- A dedicated LTX-2.5 text/image-to-video tab with synchronized audio
- A dedicated MiniMax Music 3 tab for caption-and-lyrics song generation
- A dedicated YuE2 tab for score-planned or direct lyrics-to-song generation
- All ten official LTX-2.5 ComfyUI workflows for two-stage generation,
  audio-to-video, text-to-audio, video editing, reference sheets, motion tracks,
  in/outpainting, and pose/depth/canny control
- LTX-2.5 image-to-video uses a visual start-image input plus optional custom
  middle/end keyframes
- Live queue position, workflow stage, node count, overall work, and sampling schedule
- Switchable video/image/audio gallery with thumbnails and lazy previews
- Speed and quality NVFP4 profiles plus the official Original BF16 profile
- Selectable official Qwen3-VL 32B NVFP4/AWQ, INT8 ConvRot, and BF16 text encoders
- Optional model offload at every H3 stage boundary, automatically required for BF16
- Default-on reuse of unchanged prompt and image/audio/video conditioning through
  content-addressed ComfyUI input staging
- Per-video SeedVR2 or LTX-2.5 IC-LoRA 2x upscale, LTX-2.5 decompression/deblur, and frame interpolation
- Selectable SeedVR2 target-frame preprocessing for start/end frames and reference
  images, with downloadable results and no forced downscaling
- Optional generation-stage MiniMax H3 latent 2x upscale, with Balanced BF16,
  Fast FP16, and Quality FP32 model choices
- Selectable TaoMate-H3 3-step, Larry v4-600 EMA and official LightX2V 4-step/8-step Turbo LoRAs,
  including dedicated LightX2V Ref2V adapters for 4 and 8 steps
- Audio-safe SLA block-sparse attention by default, with selectable SageAttention 2,
  Comfy Kitchen comparison, and optional
  H3-native zero-copy Sol v0.6.2 sparse attention
- Bit-exact fused H3 modulation projections for LightX2V Turbo
- Two-way feed-forward chunking for ConvRot quality checkpoints
- Optional official INT8 ConvRot video VAE, lazy-downloaded on first use
- H.264 NVENC hardware encoding for MiniMax H3 video outputs
- Optional 500K single-frame image decoder, lazy-downloaded only when selected;
  visual conditioning retains the official FP16 path when TensorRT decoding is selected
- Hardware-aware ComfyUI memory mode selection
- One-click model unloading and VRAM cache release from the UI
- Spectrum v0.2.28 in legacy mode as the normal-generation default and experimental Turbo option
- Optional experimental MMH3 tiled/chunked latent refinement for constrained VRAM
- FirstBlockCache and native ComfyUI EasyCache alternatives
- Matching local and Modal deployment paths
- Version-aware, resumable Hugging Face model provisioning

## FL2VA voice references (experimental)

In **First / last frame**, expand **Optional voice references (experimental)**
under the start/end images. Upload up to three short, clean voice samples, filling
slots in order. Start with 2–3 seconds per voice. For example:

```text
The woman uses the voice timbre of <Audio 1> and says, "We should leave."
The man uses the voice timbre of <Audio 2> and replies, "I am ready."
```

These inputs are separate from Ref2VA uploads. Switching modes keeps each set in
its own section; FL2VA voice samples are ignored in Text to video and Reference
media. A first frame, last frame, or both are still required. The advanced API
appends optional `fl2va_audio_1`, `fl2va_audio_2`, and `fl2va_audio_3` fields;
existing positional calls remain supported. Upload paths are not persisted as
browser preferences.

When samples are present, the graph uses
[T8 Audio Conditioning](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)
with `Hybrid`, `audio_mode=native`, and standalone audio references. It keeps the
selected FL2VA weights and FL2VA Turbo adapter, preserves the start/end keyframes,
and generates a new soundtrack. Both latent-upscale stages receive references.
Audio content participates in staging and conditioning cache identities.
Without voice samples, the original FL2VA graph is used.

Local setup and Modal pin T8 to
`6063fafbd9c3b85c5ff40aef435ae11b2844e558` (v1.85.0 source). Its base nodes require no additional
pip packages. Existing local installs refresh it on the next `run_h3.sh` startup;
Modal deployments need rebuilding/redeploying. Missing nodes produce an explicit
update-and-restart error. No new model weights are required for this option.

Semantic Bridge can be enabled alongside FL2VA voice references for experimental
testing; combined voice fidelity and lip-sync are not yet validated. Prompt enhancement supports FL2VA audio tags and
speaker assignments using the prompt and available voice labels. Voice sample
files stay in the generation workflow; they are not uploaded to prompt writers.
If a writer drops or invents audio labels, the original prompt is preserved with
an explicit retry message.
Result settings record the active reference count and conditioning mode.

Voice fidelity on unchanged FL2VA weights is experimental. The integration has
CPU graph/UI regression coverage; GPU voice-transfer quality, lip-sync, and speed
still require matched-seed comparisons on the deployed checkpoints.

## Semantic Bridge (experimental)

The H3 model settings include a default-on **Semantic Bridge** option for text
and first/last-frame FL2VA generation. Start with strength **0.10**; **0.15** is a
stronger comparison setting. Per-token magnitude matching is fixed. Strength zero
bypasses the adapter without loading or downloading it.

The pinned ~11 MB [v1 adapter](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge)
downloads on first use into `ComfyUI/models/semantic_bridge/`. No SenseNova model
is required. The bundled H3Acceleration node ships through both local setup and
Modal; existing deployments need updated provisioning and a ComfyUI restart.
The adapter runs after native conditioning, preserves the raw encoder cache, and
applies to both stages of latent upscaling. Output settings record its filename,
SHA-256, strength, and magnitude-matching mode.

Reference media (Ref2VA) always disables it, including requests made through the
API. The author reports degraded reference-audio singing/lip-sync. Benefits on
FL2VA are preliminary and may vary with prompt, text encoder, and Turbo preset;
compare matched seeds before adopting it. Adapter weights retain the upstream
[MiniMax H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
and the author's [licensing notes](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge/blob/main/LICENSE.md).

## Requirements

- Linux or WSL2 with Bash
- Python 3.12
- An NVIDIA Blackwell GPU and a compatible CUDA 13 driver
- Git, FFmpeg, and enough disk space for ComfyUI and the model set
- An NVIDIA driver/GPU with NVENC support for MiniMax H3 video output
- Hugging Face access to every configured model repository

The installer pins the ABI-sensitive stack to Torch 2.11.0 + CUDA 13.0,
NumPy 1.26.4, and SciPy 1.15.3, and installs the CUDA 13 TensorRT Python
builder/runtime used by the optional TensorRT VAE. The pinned ComfyUI stack
supplies Comfy Kitchen attention through its matching `comfy-kitchen` dependency. SageAttention
2.2.0 remains installed from the pinned prebuilt wheel for UI comparisons.
SLA v1.5.6 is provided by the pinned PlagueKind node pack at
`d58d006a4ea32c25c06499f2ff104f0852a045a6`. Selecting **SLA** exposes three
quality presets: **Fast** uses validated 0.90 sparsity, **Balanced** uses the
LoRA-distilled 0.85 sparsity, and **Quality** uses 0.85 sparsity plus a dense
final sampling step. In a two-stage latent-upscale workflow the Quality dense
tail applies independently to both sampling stages. The initial generation keeps
its dense first step; low-noise refinement skips that first-step anchor because
it starts from the generated latent. Two-step Quality refinement therefore runs
sparse then dense; Fast and Balanced allow both refinement steps to run sparse.
One-step Quality refinement remains dense. Short sequences and other native SLA
compatibility guards still use the dense backend. Every preset uses 64-token
blocks, protects the audio prefix, and leaves sequences shorter than 8192 tokens
dense. The graph explicitly keeps the Triton sparse engine, disables experimental
INT8 QK and tail correction, and forces step zero dense in the initial generation. This preserves the
previous route instead of inheriting v1.5.1's Kitchen sparse-engine defaults.
Use SLA with an SLA-distilled H3 LoRA.

The Sol-Attn integration is pinned to the reviewed v0.6.2 commit
`930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf` so its ComfyUI node contract
remains reproducible. The original repository is unavailable; both installers
fetch that same commit from a public fork. v0.6.2 adds MiniMax H3 support for SM86 / RTX 30-series
GPUs without changing the attention math or routing policy. Sol uses the zero-copy H3 path, keeps conditioning KV
exact by default, and leaves its optional INT8 attention approximations
disabled. LightX2V Turbo additionally uses v0.6.0's bit-exact fused modulation
node after its LoRA. Larry Turbo keeps its AdaLN path unfused pending composition
validation. Provisioning validates Larry's pinned node against the upstream
dynamic timestep and AdaLN-row implementation; older compatible source is
patched fail-closed so its E-grid adapter derives the same rows as ComfyUI,
including visual and audio reference-conditioning rows. Quality ConvRot models use
bit-preserving two-way feed-forward chunking above 8K packed tokens.

Spectrum is pinned to v0.2.28 and is applied after LoRA, Sol-Attn, and ConvRot
feed-forward patches. Its default uses system-RAM history and replay archives,
degree-1 forecasting, offline smoothing replay, zero spectral audio blending,
and explicit legacy (`model_aware_mode=off`) scheduling. v0.2.28 retains that
input contract, fixes retained CUDA target tensors, and streams forecasts from
system RAM through bounded GPU workspaces for supported FinalLayer implementations.
Unknown FinalLayer wrappers retain one-shot projection. This reduces forecast
memory pressure; it does not guarantee faster generation or bitwise-identical output.
It also keeps reviewed-source auditing valid on Windows CRLF checkouts instead of
conservatively falling back to all-actual execution.
The compiler capture safeguards and attention-backend history checks remain active.
Our one-step tail remains authoritative
subject to Spectrum's exact-evaluation safeguards. The current H3 graphs continue to use the
reviewed Larry and RES sampler paths.
Spectrum, FirstBlockCache, and EasyCache are mutually exclusive acceleration
choices. Turbo defaults to Spectrum through the reviewed Larry Turbo and
RES multistep sampler paths. EasyCache is also available as an experimental,
default-off Turbo option after ComfyUI's H3 audio-carry fix. FirstBlockCache is
also available as a default-off experimental Turbo option. Attention defaults
to **SLA with the Fast preset**, which uses audio-safe block-sparse attention
and a dense final sampling step. Sage 2 remains available through KJNodes'
per-model override, and Kitchen remains ComfyUI's global backend and a selectable
comparison/fallback. SLA also offers Fast and Balanced presets, while
Auto can
still route jobs at or above 8K estimated packed tokens (and reference-media
jobs) through Sol.
Spectrum exposes one continuous capture-and-replay progress range to ComfyUI,
so the Gradio live progress stream remains active during both passes.

**Qwen small input attention** lives under **Model and memory (advanced)** and
is off by default. Off uses the server's configured attention backend (Kitchen in the bundled launchers); turn it on to select upstream PyTorch/basic attention
for both Qwen3-VL 32B text and vision encoding. The diffusion
Sage 2 selector does not select Sage for Qwen. Changing this option invalidates
both ComfyUI's conditioning-node cache and the process-local encoder cache;
the next generation re-encodes, including when switching back to a previously
used route. Keeping the option unchanged retains normal conditioning reuse.
The preference is saved in the browser and recorded with generation settings.
Compare the same prompt, reference media, resolution, and seed on your GPU;
backend compatibility, speed, and conditioning quality depend on the inputs.
Restart the UI and update/restart the bundled H3Acceleration node to use it.

Turbo Spectrum remains approximate. Its conservative policy permits at most one
forecast before a completed native refresh, which limits both acceleration and
trajectory error at four to eight steps. Compare the same prompt and seed with
Acceleration Off before relying on it for quality-critical output.

[TaoMate-H3 3-step](https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI)
is available under **Turbo implementation** and downloads its BF16 LoRA on first
use. Selecting it sets 3 steps and the simple scheduler; generation uses Euler,
LoRA strength 0.7, and the existing unguided sampler (CFG 1.0). It supports
**Text to video** and **First / last frame** with the FL2VA base. The same adapter
is also available for **Reference media** generation with Ref2VA.

Turbo defaults to the LightX2V four-step adapter at strength 1.0 (FL2V v1.2
768p or the dedicated Ref2V 544p adapter). Larry v4-600 EMA remains available
at six steps through its pinned custom node, which uses a quantization-aware
bypass loader plus the adaptive H3 Turbo sampler. LightX2V also provides a
mode-specific four-step option (FL2V v1.2 768p or Ref2V v0.1 544p) and
mode-specific FL2V/Ref2V v1.0 eight-step 768p options, all at strength 1.0.
The 768p workflows apply LightX2V's official video/audio sigma shifts of
6/3 and use Euler sampling (four or eight NFE according to the selected LoRA)
with the simple scheduler by default. The 544p Ref2V adapter uses its official
12/3 shifts and four Euler NFE. Every base profile applies either Turbo LoRA
through its sharper runtime path. On Quality ConvRot INT8, fused FC2 kernels
read weights directly and cannot execute a forward bypass hook, so those FC2
adapters alone use ComfyUI's transient post-dequantization weight-cast patch;
the delta is not requantized into the base. LightX2V validates the exact
official 50-block plus two-refiner adapter layout before installing any hooks
and retains fused modulation. Turbo step defaults are applied by an
immediate mode/variant UI update before generation is queued. The resulting
step control remains editable so users can increase any Turbo
variant's count for clips that benefit from additional refinement.

Reference mode automatically selects LightX2V's dedicated Ref2V adapter for
both the four-step and eight-step options. Larry still reuses its FL2VA-trained
LoRA in reference mode and remains experimental because no dedicated Ref2V
counterpart is published. The generated model configuration keeps separate
Ref2VA keys so workflow construction remains mode-specific.

## Run locally

Local and Modal provisioning and the UI test environment pin Gradio 6.27.0.
Existing local installs refresh an older Gradio on the next `run_h3.sh` startup;
Spectrum also refreshes from its shared source pin. Modal deployments require a
rebuild/redeploy to pick up these dependency changes.

```bash
git clone <repository-url>
cd minimax-h3
bash run_h3.sh
```

The first run creates `h3/`, installs ComfyUI and dependencies, and preloads the
Singularity pruned v1.3 INT8 checkpoint plus the Fast NVFP4/AWQ text encoder, default FP32 latent-upscaler checkpoint, shared VAEs, and default 4-step Turbo LoRAs. The Speed, Quality and Original checkpoints plus the selectable 3-step and 8-step Turbo LoRAs download on demand when selected; the Balanced preset's 6-step Larry LoRA is preloaded. SeedVR2 models, the
LTX-2.5 upscaler and restoration IC-LoRAs, and SwiftVR checkpoints are lazy and download only
when their post-processing option is first used. The installer pins the official
SwiftVR inference source; no SWIFTVR_CHECKPOINT_DIR is required unless you want
to use an existing checkpoint directory. The official INT8 ConvRot video VAE is selected by default for Fast and
Singularity, and downloads on first use. Balanced and Quality select FP16.
Both local and Modal launch ComfyUI with
`--fast fp16_accumulation` for the faster H3 VAE encoder and decoder kernels.
The experimental **TensorRT video VAE** is disabled by default. Local and Modal
setup install TensorRT and sync the pinned ComfyUI-H3VAE_TRT node. On first use,
the app downloads the decoder ONNX source and automatically builds its local
engine; the adjacent **Compile TensorRT VAE engine** button remains available
for an explicit build. TensorRT engines are GPU-specific and are not downloaded
by the app. The quality profile keeps decoder LayerNorm reductions and powers in
FP32 while retaining FP16 matrix operations. After an upgrade, an unversioned or
stale engine is rebuilt automatically before generation.
For first/last-frame and reference-media generation, visual conditioning is
encoded with the regular FP16 VAE because the TensorRT encoder's fixed temporal
profile changes causal conditioning latents. The optional TensorRT path is
decoder-only and accelerates final video decoding, where most VAE runtime is
spent.
The experimental **Single-frame 500K** image VAE is a separate 9.69 GB lazy
download. It is used only for Image results; Video continues to use the
official H3 video VAE regardless of this image setting.
The native H3 latent upscaler starts enabled for video and lazy-downloads the
selected checkpoint. **Quality (FP32)** is the default choice; **Fast (FP16)**
and **Balanced (BF16)** remain selectable.
The sampling presets also select the H3 text encoder: **Fast** uses
**NVFP4 / AWQ**, while **Balanced** and **Quality** use **INT8 ConvRot**.
Singularity is the initial preset: it selects the Singularity base model and
otherwise uses Fast settings. Its NVFP4/AWQ text encoder is preloaded; INT8
ConvRot (27.1 GB) downloads on first selection. All presets leave model offload
disabled by default and editable. **BF16** (51.5 GB) is available only as a
manual text-encoder selection; selecting it automatically enables and locks
**Offload models between H3 stages**. When unchanged BF16 conditioning is
reused, the encoder never loads and all remaining stage offloads are skipped
for that run. INT8 and NVFP4 keep the current all-VRAM path by default; stage
offload can still be enabled manually for either one.

**FastH3 8-Step V2** is available as a lazy base-model download using the official
ComfyUI INT8 ConvRot checkpoint. It shares the selected Qwen3-VL text encoder,
video VAE, audio VAE, decoders, finishing, and output pipeline with the other H3
profiles. The checkpoint supports Text to video only. Selecting it locks the
conditioning mode to Text to video and uses its trained Normal, 8-step,
simple-scheduler path with native Kitchen attention; Turbo adapters are not
applied on top of the distilled checkpoint.
**Reuse unchanged prompt and media** is enabled by default. Uploaded H3 inputs are
staged under content-derived names. Qwen reuse requires matching source media,
prompt tokens, text encoder, attention route, and actual visual tensor geometry.
Seed, steps, sampler, and diffusion attention changes can reuse matching encoding.
Changing the canvas or reference-video duration re-encodes if it changes Qwen's
visual inputs. In a two-stage image-conditioned workflow, the low-resolution and
high-resolution stages therefore have separate entries; later matching jobs can
reuse both. Text-only encoding can still be shared across resolutions.

Disable reuse to force fresh Qwen and native/T8 conditioning on every execution,
including refinement and text-only requests, and use unconditional BF16 stage
offloading. The attention toggle independently selects the Qwen route in either
reuse mode. Logs distinguish `Qwen cache disabled`, `Qwen cache miss`, and
`Qwen cache hit`, with short cache/input identifiers for comparisons.

A Qwen cache hit skips only the text/vision encoder. `MiniMaxH3AudioConditioningT8`
may still run its image/audio VAE work, especially at a different resolution.
ComfyUI can reuse the entire conditioning node when all its inputs and dependencies
match and the result remains cached. Progress reports the actual number of cached
workflow nodes and suppresses empty cache notifications; a few cached loaders do
not imply that the encoder or the whole generation was cached.

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
on every cold start before launching ComfyUI. The LTXVideo HDR dependencies
use NumPy 1.26-compatible Colour Science 0.4.6 and OpenImageIO 3.0.12;
Kornia 0.8.3 works with the node's updated pyramid blending import.
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
The **Qwen Image 2.1** tab uses ComfyUI's native `TextEncodeQwenImage21`
workflow for both generation and editing. The BF16 DiT and text encoder are
selected by default; INT8 ConvRot and W4A8 alternatives are available. The selected
DiT, Qwen3-VL 8B encoder, and BF16 VAE download on first use from
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).
Image edit accepts up to 10 inputs and treats `image1` as the edit target.
Single-image prompts refer to the input naturally; multi-image prompts use
`<image1>`, `<image2>`, and subsequent tags. Generated
images and their settings are saved with the other application outputs. The
official 40-step Euler/simple path is the default, native ~4 MP aspect-ratio
sizes are accepted, and Comfy Kitchen INT8 attention plus Spectrum
hidden-state forecasting are available as experimental opt-in speed settings.
For image edits, **Edit output size** offers three exclusive choices:
**Match first image size** (default), **Max resolution (up to 4 MP)**, or
**Use width and height above**. Max resolution scales the first reference
image's aspect ratio toward 4 MP, rounds the output to multiples of 32, and
keeps both sides within 2752 pixels.

The Qwen tab offers three editable presets: **Fast** selects INT8 ConvRot,
Viggle Turbo v0.2.1, six steps, and accelerator Off; **Normal** selects BF16,
Turbo Off, 25 steps, and Spectrum (Quality); **Quality** selects BF16, Turbo
Off, 40 steps, and Spectrum (Quality). Quality is selected initially. The text
encoder and other controls retain their chosen values when switching presets.

The Qwen tab offers **Viggle Turbo v0.2.1** as a six-step ComfyUI LoRA mode.
It downloads the rank-256 adapter on first use and uses its published six-node
schedule with Qwen's resolution-dependent time shift. A bundled custom loader
applies the adapter without merging it into BF16/INT8 weights. The experimental
**Viggle 3-pass (configurable)** option runs a base-model composition pass,
then two six-step Viggle-LoRA refinements by default (6+6+6). Pass 2 denoise
starts at 0.60 and pass 3 at 0.25; pass step counts and denoise values are
editable. The refinements use standard Euler/simple KSampler scheduling and
are an experimental adaptation of the community recipe, not Viggle's published
single-pass sigma schedule. Turbo editing accepts up to three references. The
base mode retains its existing controls and up to ten references. Viggle Turbo
is released under the Qwen Research License for non-commercial research or
evaluation; commercial use requires a separate licence.

**Alibaba PAI PDD 4-step** is a second Turbo mode. Selecting it downloads
[Qwen-Image-2.1-Fun-Acc-4Step.safetensors](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs)
on first use and sets four steps, Euler, CFG 1, and accelerator Off. The
bundled ComfyUI node loads its PDD output heads, backbone LoRA, and trained
normalization weights and uses the checkpoint's stored sigma schedule.
This mode requires exactly four steps. Image edits turn the Qwen prefix KV
cache off, matching the published examples. Alibaba notes that small dense
text may be less legible and some edits may be softer or darker than the
40-step base model. The adapter inherits Qwen's research license.

**Pruna 8-step** and **Pruna 5-step** are additional Qwen Image 2.1 LoRA
modes. Each downloads its v0.1 adapter on first use and uses the matching
published sigma schedule through the same ComfyUI backend. Select 8 steps for
higher quality or 5 for speed. These modes require their exact step count,
Euler, CFG 1, and accelerator Off; editing accepts up to three references.
Pruna recommends 1K output and detailed prompts. This first release is below
the base model's visual quality, and the 5-step adapter has visibly lower
quality than the 8-step adapter. See the
[Pruna model card](https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1)
for the schedules and research license.

The **YuE2** tab uses ComfyUI's native YuE2 nodes (ComfyUI v0.36.0 or newer).
Its INT8 ConvRot checkpoint (about 4 GB) is selected by default; the BF16
checkpoint is an optional alternative. The selected checkpoint downloads on
first use into ComfyUI/models/checkpoints/. Use **Full score** to plan melody
and chords, **Melody only** for a melody plan, or **Direct generation** to skip
the score planner. Lyrics should use section tags such as [verse] and [chorus].
Tiled audio decode is enabled by default for long songs. YuE2 weights from
[Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) are licensed
CC-BY-NC-4.0. Generation is exposed as /generate_yue2; style, lyrics, and an
optional edited ABC score are accepted with the generation settings.
On Debian/Ubuntu standalone hosts, `run_h3.sh` also installs the `ffmpeg` system
package through `apt-get` (using `sudo` when needed) if `ffmpeg` or `ffprobe` is
missing.
The UI listens on `http://127.0.0.1:7860` by default. Local and Modal launches
use ComfyUI Dynamic VRAM on every GPU size, allowing stage-offload workflows to
move models to system RAM. Without an offload barrier, ComfyUI smart memory can
still retain the compact NVFP4/INT8 stack in VRAM. Do not launch with
`--gpu-only` when using the 51.5 GB BF16 text encoder: under that mode ComfyUI
sets each model's offload device to CUDA, so an unload request cannot release
its VRAM residency. CUDA allocation and the Comfy model compiler stay enabled
globally. Only a standard latent-refinement pass whose video latent exceeds one
and a half million spatiotemporal positions bypasses the compiler; this isolates an AIMDO
malloc-graph incompatibility with SLA's high-resolution Triton specialization
without slowing ordinary generation or tiled refinement.

The **MiniMax H3** tab includes local, Gemini, and Lightning AI prompt writers,
with Lightning AI selected by default. The local writer
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
system instruction. It supports `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`,
`gemini-3.5-flash`, and `gemini-3.5-flash-lite`, with `gemini-3.5-flash-lite`
selected by default. Use Gemini for the separate
Reference media mode, which is not one of the four tasks supported by the local
adapter. Set `GEMINI_API_KEY` in the server environment, or enter a temporary
key in the enhancer panel; a key entered in the UI is passed only to enhancement
requests and is not stored by the server. Uploaded Gemini Files are deleted
after each request. The selected operation is exposed as `/enhance_prompt`.

Lightning AI uses the OpenAI Python SDK with
`https://lightning.ai/api/v1/` and the fixed `openai/gpt-5.6-luna` model. It
supports prompt enhancement from text plus active first/last-frame or reference
images. Select Gemini when the active references include video or audio. Set
`LIGHTNING_API_KEY` in the server environment, or enter a temporary key in the
enhancer panel; UI keys are passed only to the request and are not stored.

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

The **Qwen Image 2.1**, **LTX-2.5**, **MiniMax Music 3**, and **YuE2** tabs
also offer Lightning AI and Gemini prompt writers, with Lightning AI selected
by default. Lightning AI uses the same fixed model, server environment key,
and temporary key behavior as the H3 writer. Gemini remains selectable and
uses `gemini-3.5-flash-lite` by default. Each writer uses its own bundled
system prompt: `prompt_qwen_image21.txt`, `prompt_ltx25.txt`,
`prompt_music3.txt`, or `prompt_yue2.txt`.

LTX-2.5 and Music 3 can use optional keyframe or visual-reference images.
Qwen Image 2.1 understands text-to-image and image-edit modes: it refers to a
single edit input naturally and uses ordered `<image1>`, `<image2>`, ... tags
for multi-image edits. YuE2 jointly creates or enhances the production style
and sectioned lyrics. Their UI/API operations are `/enhance_ltx25_prompt`,
`/enhance_music3_prompt`, `/enhance_qwen_image21_prompt`, and
`/enhance_yue2_prompt`, respectively.

### Native H3 latent upscale

Enable **Generate at half resolution, then latent upscale 2x** in the MiniMax H3
generation settings to run the upscaler inside the generation workflow. The UI
width and height always describe the final output: a 1024×1024 request first
finishes a 512×512 H3 generation, upscales its clean video latent 2x, then
lightly re-noises and refines it at 1024×1024. The clean first-pass audio is
preserved for the final output.

This generation option starts enabled for video,
defaults to two high-resolution refinement steps, and disables cache wrappers
across the two samplers. Enabling it automatically rounds both final dimensions
to the nearest multiple of 64 so the half-resolution pass remains on H3's
32-pixel grid. The selected model is downloaded from
[`LBH-123-AI/Minimax_h3_latent_Upscaler`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
on first use. The pinned node keeps temporal chunking and 32-pixel output-grid
alignment enabled and unloads the learned upscaler after inference before the
high-resolution refinement pass.

The **High-resolution refinement method** control keeps **Full-frame
refinement** as the normal path. Selecting **MMH3 Split Upscale
(experimental)** feeds the learned 2x AV latent into upstream's separate
temporal-parameter, spatial-parameter, and tiled resampling nodes. Its controls
appear only in that mode: tile width/height, spatial overlap and fade, temporal
chunk length and overlap, seam-denoise cap, and seam-polish policy. The route
uses triple temporal anchors and color matching with the upstream defaults.
It lowers the memory needed by the target-resolution refinement pass by trading
more sampling work and complexity; fixed-seed GPU comparisons should check
seams, color, identity, reference media, and audio continuity.

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

Open **Gallery** to browse the video library (the default), or switch **Gallery
type** to **Image** or **Audio**. Image mode includes generated Qwen/H3 stills
and SeedVR2 results; Audio mode includes MiniMax H3, MiniMax Music 3, and YuE2
outputs. All three use the same card, preview, settings, import, and download
layout. **Import local media** accepts a matching file for the active library.
Image mode uses the existing one-step
SeedVR2 workflow to upscale the selected still while preserving its aspect ratio;
the processed image is added back to the image gallery with its settings and a
download link. Audio mode provides playback and downloads. Video mode retains
all existing enhancement methods. Each run
preserves the source and adds a new processed output. Choose a target preset from
**1280 × 1280**, **1920 × 1920**, **2560 × 2560**, or **3840 × 3840**; the source
is fitted inside that square without cropping, so its original aspect ratio is
preserved. **SeedVR2 2x** uses ComfyUI's native one-step restoration workflow.
**SwiftVR 2x** runs the official streaming restoration pipeline directly and
downloads H-oliday/SwiftVR into ComfyUI/models/swiftvr on first use.
SWIFTVR_CHECKPOINT_DIR can override that location and must point to the
checkpoint root containing reae.safetensors, prompt_embedding.safetensors,
and the transformer directory. **LTX-2.5 IC-LoRA 2x** is a generative
alternative that synthesizes fine detail with the transformer selected in the
**LTX 2.5**
tab and the official gated pixel spatial upscaler IC-LoRA. Gallery runs accept
an optional scene prompt; automatic post-processing reuses the H3 generation
prompt. All three upscale methods preserve the source audio and frame rate and can
also run automatically as soon as the base H3 video finishes. Enable
**Unload resident models first** in Gallery, or **Unload H3 models before
upscaling** in MiniMax H3, when
lower peak VRAM is more important than avoiding an H3 model reload on the next
generation. **48 fps interpolation** remains available as a non-upscale option
and requires FFmpeg on the server `PATH`.

Gallery also offers **LTX-2.5 IC-LoRA Decompression** to remove compression
artifacts and **LTX-2.5 IC-LoRA Deblur** to restore defocused footage. These
options preserve the source resolution and audio; the target-resolution selector
is hidden. Describe the source scene in **LTX-2.5 scene prompt**; the appropriate
restoration instructions are added automatically. Both use a single-stage,
1x-reference IC-LoRA workflow with the base model selected in the **LTX 2.5** tab.
**LTX-2.5 CQ Video Enhancer V2** is also available for prompt-free generative
quality enhancement of low-resolution or poor-quality video. It preserves the
source resolution and audio and lazily downloads the V2 video LoRA from
[CQdesign's enhancer repository](https://huggingface.co/CQdesign/LTX-2.5-CQ-Video-and-Image-Enhancer-LoRAs).
The adapters download on first use and require access to their separate gated
[Decompression](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Decompression)
and [Deblur](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Deblur)
repositories. Deblur targets defocus, not motion blur. Use **Split source into
clips before LTX processing** for restoration clips that exceed available VRAM.

LTX-2.5 upscaling remains a single full-video pass by default. If a long or
high-resolution source runs out of VRAM, enable **Split source into clips before
LTX processing** in Gallery, or **Split source into clips before LTX upscaling**
in the MiniMax H3 post-processing settings. The
default target is 5 seconds per clip; cuts are adjusted to LTX-compatible frame
counts, clips are upscaled sequentially, and their video streams are joined
without an additional video encode. The final file is trimmed to the original
frame count and remuxed with the original source audio. Because clips are
generated independently, a visible detail or motion change can occur at a cut.

SeedVR2 offers **7B FP16**, **7B INT8 (default)**, **3B FP16**, **3B INT8**,
**7B Sharp FP16**, experimental **7B MXFP8**, and legacy fast/experimental
**NVFP4** choices. Only the
selected checkpoint downloads on first use; all choices share the same lazy FP16
SeedVR2 VAE. The native workflow uses
1024-pixel VAE encode/decode tiles for the RTX PRO 6000 target. SeedVR2 and main
H3 generation run eagerly because full-model compile did not improve measured
performance and conflicts with the active attention and cache optimizations.

The **API** tab includes a copy-ready Python example. Its `/generate_video`
endpoint only requires a prompt and uses the same defaults shown in the MiniMax H3
tab. It returns a public HTTP download URL instead of a client-local temporary
file path. The `/generate_video_advanced` endpoint exposes every generation
control; its current request schema is linked from the API tab.
The LTX tab is also available as `/generate_ltx25_video` and shares the same
single-job ComfyUI queue. Qwen Image 2.1 is available as
`/generate_qwen_image21` and uses that queue as well.

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

## Deploy on Google Colab

Run MiniMax H3 directly on Google Colab using [`minimax_h3_colab.ipynb`](./minimax_h3_colab.ipynb):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keboqi/minimax-h3/blob/main/minimax_h3_colab.ipynb)

Features in the Colab notebook:
- **Instant startup with Python 3.12**: Automatically provisions an isolated Python 3.12 environment with `uv` matching the pinned PyTorch 2.11 + CUDA 13 and SageAttention cp312 runtime requirements.
- **On-demand model provisioning**: No heavy pre-downloads at startup. Checkpoints, text encoders, VAEs, and LoRAs download automatically on first use.
- **Public access via Gradio Share & Cloudflare Tunnel**: Provides both a public `*.gradio.live` URL and a Cloudflare `*.trycloudflare.com` tunnel for reliable, high-bandwidth remote generation.

## Deploy with Modal

Install and authenticate the Modal CLI, then run:

```bash
modal secret create custom-secret HF_TOKEN="$HF_TOKEN"
modal deploy modal_h3.py
```

The deployment attaches the `custom-secret` Modal Secret to both runtime
functions and requires it to contain `HF_TOKEN`. If your existing secret uses a
different name, deploy with `H3_MODAL_HF_SECRET=your-secret-name`. To make a
hosted prompt enhancer available without entering a key in the UI, also store
`GEMINI_API_KEY` and/or `LIGHTNING_API_KEY` in that Modal Secret.
The Modal runtime mounts every model-specific prompt instruction file, including
the Qwen Image 2.1 and YuE2 writers, without rebuilding the heavy ComfyUI image.

The deployment pins an immutable ComfyUI revision with its required
frontend package 1.53.6, Comfy Kitchen 0.2.35 and upstream aimdo 0.5.5.
The source also pins workflow templates 0.11.68 and embedded docs 0.5.12.
This update includes native Qwen Image 2.1 generation/editing, corrected edit
KV-cache placement, compiled Qwen transformer blocks, sparse attention, Comfy
Compiler, optional H3 reference VAEs and DiffSynth/ModelScope H3 LoRA support.
KJNodes 1.5.1 includes the matching H3 low-memory attention callback fix.
This ComfyUI revision also fixes offloaded H3 VAE normalization and blends tiled
VAE output against composited neighbours.

TensorRT VAE is pinned to `4360e00867eca86ab61b3899216c0ec281367b46`.
Upstream now owns optional encoder loading and single-frame encoding. Our
version-5 patch retains reference single-frame decoding, temporal tail trimming
and explicit FP32 decoder normalization casts after ONNX parsing. Provisioning
also installs ONNX for graph-based quantization detection in the compiler. The engine
quality marker is version 4, so existing engines rebuild lazily on the next
TensorRT decode. Model weights are reused. These source and CPU checks do not
replace matched-seed video/audio and memory validation on the deployment GPU.
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

Runtime-only Python sources (`gradio_app.py`, the `h3_ui` package,
`h3_models.py`, `h3_attention.py`, `h3_prompt_rewriter.py`, and the bundled
H3Acceleration node) are mounted into Modal containers at startup after the
expensive ComfyUI image layer is built. Changes to those sources
therefore reuse the cached ComfyUI, CUDA, Torch, and dependency layers. Only
`h3_requirements.py`, which controls build-time package installation and ABI
pins, is copied into an earlier image layer.

## Validation

The consolidated checks run offline and do not require a GPU:

```bash
python -m tests
python -m tests --browser
```

The first command runs discovery, all standalone service self-tests and 15 baseline
workflow fixtures. The second adds settings migration and voice-reference browser
acceptance using headless Chrome/Chromium. `python gradio_app.py --selftest` remains
available. CPU PyTorch numerical tests and cached upstream contract tests run when
their optional dependencies are present. Supported GPU inference, TensorRT
compilation and deployment remain separate checks.

See [the architecture and validation guide](docs/settings-refactor.md) for setup,
module ownership and cancellation boundaries.

## Repository layout

- `gradio_app.py` — launcher and temporary compatibility API
- `h3_app/` — configuration, model services, workflows, generation, media and execution
- `h3_ui/` — application composition, visible sections, events and UI adapters
- `h3_sources.py` — shared local/Modal source pins
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

## Settings and execution architecture

See [the settings refactor guide](docs/settings-refactor.md) for preset behavior, module boundaries, cancellation ownership, preference migration, result metadata, and test commands.
