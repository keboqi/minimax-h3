# Prism Studio

A creative workspace for video, images, music, and audio production, with a
standalone Gradio interface and deployment toolkit for NVIDIA Blackwell GPUs.
Prism Studio brings multiple generation engines together with shared media,
projects, and jobs. It provisions ComfyUI, the required models, and acceleration
tools for local and Modal deployments.

Formerly the MiniMax H3 launcher. Existing launch commands, configuration keys,
storage paths, and APIs remain compatible.

## Creative workspace

`bash run_h3.sh` and `python gradio_app.py` launch the workspace. Local and
Modal launches use the same UI.

The studio layout separates composition, output settings, and preview into
clearly marked sections, with a shared teal palette in light and dark mode.
The H3 composer keeps inputs and the prompt beside results and generation
actions. Recipes and output essentials stay visible; every other setting sits
below in **Fine-tune**, one named tab per group (Model & sampling; Size, seed &
canvas; Prompt writer; Speed & attention; Memory & decoding; Upscale & finish;
Input upscaling). Generate actions sit directly under the result, and
Ctrl/⌘+Enter generates with the visible engine.
Mobile section links jump to compose, generate, preview, and settings.
Create supports H3,
LTX 2.5, Qwen Image 2.1, FLUX 3, MOSS-Image, Fibo 2.0, MiniMax Music 3, YuE2,
[Soundstage](docs/features/drama3.md), the audio production workspace powered by Drama 3, and
[Dub & Translate](docs/features/dubbing.md), which transcribes, translates and re-voices a video
with each speaker's cloned voice, with every line editable before dubbing. Prompt suggestions require
acceptance. Media includes indexed search, shared tags/favorites, lineage,
image comparison, synchronized video playback, and downloads from video URLs.
**Image** defaults to **Qwen Image 2.1**, and **Song** defaults to **MiniMax Music 3**.
Both tasks remember explicit engine choices.

Song walks through four steps shown as a step strip: **Idea**, **Lyrics**,
**Generate**, and **Listen & refine**. Any step can be opened at any time, and
buttons on each step lead to the next one.

**Idea.** With MiniMax Music 3, start with a plain-language idea or a starter
example. Three optional choices replace music jargon: **Who sings** (Auto,
female, male, duet, group, or Instrumental), **Lyric language**, and **Made for**
(full song, short hook or jingle, or background music), plus the length limit
with 30-second, 1-, 2-, or 3-minute buttons. **Write the song** then makes the
producer's decisions: it sizes a song form to the length limit (a hook for 30
seconds, verse and chorus for a minute, a bridge and final chorus for three
minutes), passes a lyric-line budget per section, and writes a caption with
tempo, energy map, section-by-section arrangement, and ending, plus singable
original lyrics. Conventions follow the genre named in the idea: rap gets a
hook-first form with dense one-bar verse lines, dance gets builds and drops,
ballads and folk grow through restraint, and ambient cues evolve around a motif.
**Write & generate a take** is the one-click alternative: it applies the
writer's draft without review (Undo still restores the previous draft) and
starts a take through the normal Generate button and queue; a writer failure
stops before any GPU job. **I have my own lyrics** skips straight to the
lyrics. The writing service and optional visual inspiration sit in collapsed
panels on this step.

**Lyrics.** The writer's suggestion appears here for review, with Undo after
acceptance. Existing lyrics are kept exactly unless **Polish my lyrics** is
selected. A live **song check** beside the lyrics shows the section timeline and
estimated length, and flags drafts that would be cut off, missing tags or a
chorus, overlong lines, and lyrics in an instrumental.

**Generate.** A recap shows the sound, voice, and song check the next take will
use. Generate needs a music description; with lyrics alone, Write drafts one.
After each take, the status reports its length and warns when it ran to the
limit (likely cut off) or ended far earlier than planned. Advanced generation
controls stay collapsed.

**Listen & refine.** **Your takes** lists every take from the browser session
with its length and seed. Pick any two as A and B to play them back to back; the
panel names what changed between their drafts (description, lyrics, length,
voice, seed, and so on). **Back to take A's draft** restores the draft that
produced a take, and **Use take A's seed** keeps its seed for the next take.
Takes are kept for the session only; every file also stays in **Media**.
**What should change?** takes a listener's words, or one-click quick fixes such
as *Stronger chorus* or *The voice gets lost*; **Revise the draft** sends them to
the writer, which makes the smallest musical change while keeping everything
else, and opens the Lyrics step to review it. **Generate another take** records
the current draft again.

Song with YuE2 follows the same four steps and choices. Its writer turns the
idea into a short comma-separated style tag line, which YuE2 reads best, in
the order YuE2's authors use: the sung language first and the BPM last. Its
**Lyrics & score** step has **Lyrics** and **Score** tabs. After a take that
planned or used a score, the **Score** tab shows it; **Reuse this score for the
next take** keeps the melody while the style or sound changes; **Reuse take A's
score** in **Your takes** does the same for any earlier take. While a reused
score is in use, the song check compares the lyrics with the words that take
sang: it flags a changed line count and any line that gained or lost more than
one syllable, since the melody has notes for the old words. Scores are saved
beside the song as `<song>.score.abc` and deleted with it. The **Generate** step
holds the composition mode (plan melody + chords, plan melody only, or no
score); the Idea step offers 1-, 2-, 3-, or 5-minute duration presets. YuE2
needs both a style and lyrics, so Generate waits for both; an instrumental gets
section tags that follow the planned form.

Jobs capture requests before the GPU queue, retain seeds for failed-variant
retries, and support separate H3 finishing retries. SQLite stores technical job
history across restarts. A signed browser cookie and recovery key preserve
ownership. Explicitly saved projects retain prompts/settings and copied sources;
unsaved request content remains temporary. Recovery observes recorded backend
work and never automatically resubmits an ambiguous request.

See the [workspace operating guide](docs/workspace.md) for retention,
backup/restore and comparison limits. Automated tests run on CPU with synthetic
backends; GPU inference and deployment are validated separately.

## What is included

- Unified H3 video, image-frame, and audio result formats across text,
  first/last-frame, and reference-media conditioning
- Image results expose 1–20 decoded frames in a preview gallery and save only
  the frames selected by the user
- Audio results decode the native H3 stereo soundtrack without creating a video
- A dedicated Qwen Image 2.1 tab for native text-to-image generation and
  multi-reference image editing
- A MOSS-Image tab for OpenMOSS MOSS-Image SFT and Base text-to-image generation with
  Fast (8-step Turbo LoRA), Balanced and Quality presets, plus default-on Spectrum acceleration
- Fibo 2.0 Dev generation/editing and Edit Turbo, with ordered reference images
- A dedicated LTX-2.5 text/image-to-video tab with synchronized audio
- A dedicated MiniMax Music 3 tab for caption-and-lyrics song generation
- A dedicated YuE2 tab for score-planned or direct lyrics-to-song generation
- Soundstage for script writing and direction, voice casting, sound effects,
  music, reusable takes, and mixing
- Dub & Translate: local (OmniVAD + Qwen3-ASR + ForcedAligner + Hy-MT) or Gemini
  transcription and translation, vocal separation, a line editor, and Drama 3
  voice-cloned dubbing muxed back into the video with subtitle tracks
- Media downloads from video URLs with yt-dlp, including saved per-site cookies
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
- Selectable Larry v4-600 EMA, official LightX2V 4-step/8-step, and experimental DMAD 4-step Turbo LoRAs,
  including dedicated LightX2V Ref2V adapters for 4 and 8 steps
- Audio-safe SLA block-sparse attention by default, with selectable SageAttention 2,
  Comfy Kitchen comparison, experimental Veda learned sparse attention, and optional
  H3-native zero-copy Sol v0.6.2 sparse attention
- Bit-exact fused H3 modulation projections for LightX2V Turbo
- Two-way feed-forward chunking for ConvRot quality checkpoints
- Optional official INT8 ConvRot video VAE, lazy-downloaded on first use
- Optional LynnReal Light INT8 video VAE, lazy-downloaded on first use
- Independent Original / LightVAE Lite video encoding and optional LightVAE Lite
  or LightVAE Lite INT8 ConvRot decoding, lazy-downloaded on first use
- H.264 video output with NVENC hardware encoding and automatic CPU libx264 fallback
- Optional 500K single-frame image decoder, lazy-downloaded only when selected;
  visual conditioning retains the official FP16 path when TensorRT decoding is selected
- Hardware-aware ComfyUI memory mode selection
- One-click model unloading and VRAM cache release from the UI
- Spectrum v0.2.29 in legacy mode as the normal-generation default and experimental Turbo option
- Optional experimental MMH3 tiled/chunked latent refinement for constrained VRAM
- FirstBlockCache and native ComfyUI EasyCache alternatives
- Matching local and Modal deployment paths
- Version-aware, resumable Hugging Face model provisioning

## Feature guides

- [MiniMax H3](docs/features/h3.md): models and presets, LightVAE, Veda, FL2VA voice
  references, Semantic Bridge, attention and Turbo, prompt writers, result formats,
  latent upscale and input image upscale
- [Other engines](docs/features/engines.md): LTX 2.5, MiniMax Music 3, Qwen Image 2.1,
  MOSS-Image, Fibo 2.0 and YuE2
- [Media and finishing](docs/features/media.md): the library, comparison, SeedVR2, LTX
  upscaling, restoration, SDR to HDR and interpolation
- [Soundstage](docs/features/drama3.md) and [Dub & Translate](docs/features/dubbing.md)
- [Workspace operating guide](docs/workspace.md): storage, ownership, jobs and backups

## Requirements

- Linux or WSL2 with Bash
- Python 3.12
- An NVIDIA Blackwell GPU and a compatible CUDA 13 driver
- Git, FFmpeg, and enough disk space for ComfyUI and the model set
- An NVIDIA driver/GPU for generation; NVENC is optional for faster video encoding
- Hugging Face access to every configured model repository

The installer pins the ABI-sensitive stack to Torch 2.11.0 + CUDA 13.0,
NumPy 1.26.4, and SciPy 1.15.3, and installs the CUDA 13 TensorRT Python
builder/runtime used by the optional TensorRT VAE. The pinned ComfyUI stack
supplies Comfy Kitchen attention through its matching `comfy-kitchen` dependency. SageAttention
2.2.0 remains installed from the pinned prebuilt wheel for UI comparisons.

## Run locally

Local and Modal provisioning and the UI test environment pin Gradio 6.30.0.
Existing local installs refresh an older Gradio on the next `run_h3.sh` startup;
Spectrum also refreshes from its shared source pin. Modal deployments require a
rebuild/redeploy to pick up these dependency changes.

```bash
git clone <repository-url>
cd minimax-h3
bash run_h3.sh
```

The launcher reuses an active Python 3.12 environment. On hosts whose default
Python has another version (including Lightning/Conda environments on Python
3.14), it uses `uv` to provision an isolated Python 3.12 environment at `h3/.venv`
and reuses it on later launches. Missing `uv` is bootstrapped into `h3/.bootstrap`.
Setup, ComfyUI, and Gradio use that interpreter; the host Conda environment is
left intact. Direct `setup_h3.py` invocations require Python 3.12 and fail early
with a launcher hint if the interpreter is incompatible.

Video saving opens an actual encoder session before choosing NVENC. GPUs without
NVENC, including [H100/H200](https://docs.nvidia.com/dynamo/v1.4.0/multimodal/video-decode-gpu-requirements),
automatically use CPU `libx264` for H.264 MP4 output;
generation stays on the GPU. Audio, frame rate, frame count, and metadata are
preserved. The quality value maps to NVENC CQ or software CRF, with `medium` as
the software preset. 48 fps interpolation also checks whether its encoder can
actually run, rather than relying on FFmpeg's codec list.

The first run creates `h3/`, installs ComfyUI and dependencies, and preloads the
default Singularity models. Other checkpoints, encoders, VAEs and LoRAs download on
first use; see the [H3 guide](docs/features/h3.md) for the model set and presets.
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
its VRAM residency. CUDA allocation and the Comfy model compiler stay enabled
globally. Only a standard latent-refinement pass whose video latent exceeds one
and a half million spatiotemporal positions bypasses the compiler; this isolates an AIMDO
malloc-graph incompatibility with SLA's high-resolution Triton specialization
without slowing ordinary generation or tiled refinement.

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

The standalone launcher creates a Cloudflare `*.trycloudflare.com` public URL by
default and prints the ComfyUI proxy URL alongside it. It uses `cloudflared` from
`PATH`, or downloads it into `.cache/cloudflared` on Linux and Windows.
Set `ENABLE_CLOUDFLARE_TUNNEL=false` for local access only. To use Gradio Share,
set `GRADIO_SHARE=true`; this disables Cloudflare unless explicitly enabled.
Modal continues to use its own public endpoint.

Cloudflare's [Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations)
do not support Server-Sent Events (SSE), which Gradio uses for queue progress.
The workspace browser carries queue updates and session heartbeats over
WebSockets, so generation progress and intermediate results reach the UI through
Cloudflare without relying on public SSE support. External Gradio API clients
still use HTTP/SSE; use a named Cloudflare tunnel when those clients require SSE.
Disable the automatic Quick Tunnel with `ENABLE_CLOUDFLARE_TUNNEL=false` when
running your own tunnel.

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
- **Public access via Cloudflare Tunnel**: Generates a `*.trycloudflare.com` URL by default. Enable the independent `ENABLE_GRADIO_SHARE` checkbox in launch cell 6 to add a Gradio public URL alongside Cloudflare; it defaults to off.

## Drama 3 speech studio

Compose dialogue, reuse a saved cast, and generate speech on SGLang-Omni
(Linux). Auto can rewrite a draft, prepare voices and sounds, and render it.
See the [Drama 3 guide](docs/features/drama3.md) for the workflow, speaker library,
backend configuration and deployment requirements.

## Deploy with Modal

Install and authenticate the Modal CLI, then run:

```bash
modal secret create custom-secret HF_TOKEN="$HF_TOKEN"
modal deploy modal_h3.py
```

Deployment exposes four independent URLs: `storage` for the CPU-only
**H3 Storage Console**, and three GPU endpoints that serve the same app: `h3`,
`h3-video` and `h3-soundstage` (see **Preloaded endpoints** below). Open the **storage** URL
printed by Modal first. The console uses a separate lightweight image with
Modal's default CPU and RAM; opening it does not start the expensive GPU
endpoint or ComfyUI. Its **Models** tab prepares checkpoints and LoRAs; its
**Media library** tab is the main app's library without GPU enhancement.

The GPU endpoint scales to zero by default (`H3_MODAL_MIN_CONTAINERS=0`), with a
300-second idle scale-down window (`H3_MODAL_SCALEDOWN_WINDOW`). It uses
Modal CPU and experimental GPU memory snapshots to capture ComfyUI
after startup and readiness checks. Restored containers reuse ComfyUI and
start Gradio after restoration so the workspace database, browser ownership and
media catalog are initialized from the current shared volume. The initialized
workspace is committed before the UI accepts requests. Snapshots are created only
for deployed apps; the initial snapshot creation still performs a full startup.
Model weights continue to load on demand during generation. Use the Modal
Containers tab to confirm snapshot restoration and measure cold-start latency.
Changes to the shared volume do not invalidate snapshots, so redeploy after
changing the installed model inventory to capture fresh startup state.

### Preloaded endpoints

`h3-video` and `h3-soundstage` also hold a workflow's default models on the GPU
in their snapshot, so a restored container's first job skips loading them:

- `h3-video` runs the MiniMax H3 tab's default Video request once (the
  `/generate_video` defaults: Singularity, Turbo, 5 s at 1376×768 with latent
  upscale) and keeps ComfyUI's models loaded. The warmup clip is deleted.
- `h3-soundstage` loads what Soundstage's Auto uses by default: Qwen
  VoiceDesign, Stable Audio 3 Medium (Advanced scripts), then Drama 3 on
  SGLang-Omni. The snapshot is refused if any of them does not load.

The models load in the Gradio process before it opens the workspace; after each
restore it builds the UI against the current volume, as `h3` does. The first
snapshot downloads any missing default models, so it takes longer than `h3`'s.
On `h3-soundstage`, SGLang-Omni's kernel caches and logs and the VoiceDesign/Stable
Audio worker log stay in the container while it runs (open files must not be on
the volume when a snapshot is taken); they are copied to the volume when the
snapshot is taken and when the container stops. The captured SGLang-Omni server
stops instead of sleeping when another engine needs the GPU; a server started
after that sleeps as usual. Other engines still work on both endpoints, loading
their models on demand.

All three GPU endpoints share the volume and its workspace database, so use one
at a time, as with the storage console below. `H3_MODAL_MIN_CONTAINERS` applies
to each of them.

The **Media library** tab browses videos, images and audio from the shared
volume with the same search, favorites, tags, lineage, settings, A/B
comparison, import, download, historical scan and safe deletion as the main
app. Enhancement (SeedVR2, LTX restoration and upscaling) needs the GPU
endpoint and is not offered there. **Reload shared volume** picks up outputs
the GPU endpoint saved after the page opened.

Select one or more presets (MiniMax H3, Qwen Image 2.1, LTX 2.5, Music 3, or
YuE2), click **Select preset models**, and adjust the individual model/LoRA
selection. **Download / update selected** checks remote versions and downloads
the batch sequentially with resumable Hugging Face downloads. Optional presets
include Qwen Turbo LoRAs and LTX IC-LoRAs. The inventory shows installed,
missing, and unverified files with their disk sizes and source repositories,
grouped into MiniMax H3, Qwen Image 2.1, LTX 2.5, Music 3, YuE2, and SeedVR2
catalog tabs. Each tab can add its entire family to the batch selection.
**Select installed files** and **Remove selected** support batch cleanup;
removal requires the confirmation checkbox. Failed downloads are listed for
retry, and completed files remain saved to the shared volume.

Stop the GPU endpoints before downloading or removing models, or before
importing, annotating or deleting media, then start it after you finish: both
services write the same workspace database on the volume. Modal volumes require a reload to see another
container's commits, and their file locks do not coordinate separate containers.
GPU startup now loads the catalog without provisioning models or checking remote
versions. Generation still downloads missing assets on demand. The old
`manage_models` function (URL label `models`) is now `storage_console`
(label `storage`).

The deployment attaches the `custom-secret` Modal Secret to both runtime
services and requires it to contain `HF_TOKEN`. If your existing secret uses a
different name, deploy with `H3_MODAL_HF_SECRET=your-secret-name`. To make a
hosted prompt enhancer available without entering a key in the UI, also store
`GEMINI_API_KEY` and/or `OPENAI_API_KEY` (plus `OPENAI_BASE_URL` for a
non-OpenAI endpoint) in that Modal Secret.
The Modal runtime mounts every model-specific prompt instruction file, including
the Qwen Image 2.1 and YuE2 writers, without rebuilding the heavy ComfyUI image.

The deployment pins ComfyUI at immutable official-master revision
`bc5f880dbdd43e426077f42ccec53bb577ffee9f`, which includes kijai's merged native Fibo 2.0
support ([Comfy-Org/ComfyUI#17240](https://github.com/Comfy-Org/ComfyUI/pull/17240)).
It is v0.42.1 plus development commits: MOSS RoPE/color-depth controls, H3 PDD LoRA fixes,
the Fibo model, text encoder and node, a top-p nucleus fix for native text generation, a
Pad Image node and an opt-in agent host (off unless `--enable-agent`). It also encodes
structured Fibo 2.0 negatives as written instead of cleaning them like captions
([Comfy-Org/ComfyUI#17285](https://github.com/Comfy-Org/ComfyUI/pull/17285)), so our
text-to-image negative applies as intended, and adds aimdo 0.6.0 dynamic-VRAM KV caches for
Qwen Image 2.1. Its requirements: frontend package 1.57.1, Comfy Kitchen 0.2.40, upstream
aimdo 0.6.0, workflow templates 0.11.96 and embedded docs 0.5.14. Return to an official
release once one includes Fibo 2.0; the v0.42.x tags through v0.42.5 do not.
v0.42 adds native MOSS-Image, Qwen 3.x decode speedups, TextGenerate cleanup,
baked-in H3 PDD heads and synchronized CUDA-graph teardown.
v0.40.0 fixed Qwen Image 2.1 caching under low VRAM and updates shared
Linear/fusion and bypass-LoRA dispatch. Our optional 500K decoder supplies the
direct single-latent metadata required by the new native H3 still-image path.
The asset system stays off by default for ordinary Linux/Modal launches;
`--enable-assets` opts in. Existing deployments need updated setup and a ComfyUI
restart; rebuild/redeploy Modal images. GPU quality and memory validation remain
separate from the CPU/browser checks.
Version 0.39.0 also reduces asset scan/database contention, pauses scans during
generation, fixes cross-volume uploads, and updates Qwen Image attention support.
This update includes native Qwen Image 2.1 generation/editing, corrected edit
KV-cache placement, compiled Qwen transformer blocks, sparse attention, Comfy
Compiler, optional H3 reference VAEs and DiffSynth/ModelScope H3 LoRA support.
KJNodes 1.5.1 includes the matching H3 low-memory attention callback fix.
This ComfyUI revision also fixes offloaded H3 VAE normalization and blends tiled
VAE output against composited neighbours. It handles RGBA Qwen-VL reference
images and clamps Qwen Image 2.1 FP16 activations in place. Version 0.38.0 adds
SeedVR2 and YuE2 autoregressive optimizations, Qwen Image 2.1 Union ControlNet
support, and asset database startup fixes.

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

Runtime-only Python sources (`gradio_app.py`, the `h3_app` and `h3_ui` packages,
`h3_models.py`, `h3_attention.py`, `h3_prompt_rewriter.py`, and the bundled
H3Acceleration node) are mounted into Modal containers at startup after the
expensive ComfyUI image layer is built. Changes to those sources
therefore reuse the cached ComfyUI, CUDA, Torch, and dependency layers.
`h3_requirements.py` and `h3_sources.py`, which control build-time package
installation, ABI policy and source pins, are copied into an earlier image layer.

## Validation

The consolidated checks run offline and do not require a GPU:

```bash
python -m tests
python -m tests --browser
```

The first command runs discovery, standalone provisioning checks, focused service
contracts and 15 baseline workflow fixtures. Graph fixtures call pure builders
directly, without importing the UI. The second adds browser acceptance using
headless Chrome/Chromium. `python gradio_app.py --selftest` runs the same CPU suite.
CPU PyTorch numerical tests and cached upstream contract tests run when
their optional dependencies are present. Supported GPU inference, TensorRT
compilation and deployment remain separate checks.

See [the architecture guide](docs/architecture.md) for test setup, module
ownership and cancellation boundaries.

## Repository layout

- `gradio_app.py` — workspace launcher
- `h3_app/` — configuration, model services, workflows, generation, media and execution
- `h3_ui/` — application composition, visible sections, events and UI adapters
- `h3_sources.py` — shared local/Modal source pins
- `h3_prompt_rewriter.py` — lazy local Qwen3-VL 8B + MiniMax-H3 LoRA writer
- `prompts/` — system prompts for each engine's prompt writer
- `docs/` — feature guides, architecture, workspace operations and upstream status
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

See [the architecture guide](docs/architecture.md) for preset behavior, module
boundaries, cancellation ownership, preference migration and result metadata.
[Upstream status](docs/upstream-status.md) lists open component and model
upgrade items.
