# Other generation engines

LTX 2.5, MiniMax Music 3, Qwen Image 2.1, MOSS-Image, Fibo 2.0 and YuE2. Each
shares the Create workspace, job queue, Media library and output settings with
MiniMax H3.

## LTX 2.5

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

## MiniMax Music 3

The **MiniMax Music 3** tab uses the same ComfyUI queue and downloads its selected
INT8 ConvRot or FP16 DiT plus the shared autoregressive encoder and audio decoder
on first use. It supports tagged song sections and a maximum duration of five
minutes, with tiled audio decoding enabled by default for lower peak VRAM.

## Qwen Image 2.1

The **Qwen Image 2.1** tab uses ComfyUI's native `TextEncodeQwenImage21`
workflow for both generation and editing. The INT8 ConvRot DiT and BF16 text encoder are
selected by default; BF16 DiT and W4A8 encoder alternatives are available. The selected
DiT, Qwen3-VL 8B encoder, and BF16 VAE download on first use from
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).
Image edit accepts up to 10 inputs and treats `image1` as the edit target.
Single-image prompts refer to the input naturally; multi-image prompts use
`<image1>`, `<image2>`, and subsequent tags. Generated
images and their settings are saved with the other application outputs. The
**Images per batch** control generates one to four images from the same inputs.
Each image uses a distinct seed (successive seeds when a seed is specified),
appears in the gallery when complete, and keeps its own settings snapshot. The
default Fast preset uses Official Turbo with eight Euler steps. The base
40-step Quality path uses **Qwen 2.1 (resolution-aware)** scheduling,
following the official dynamic shift and 0.02 terminal setting; `simple`,
`normal`, and `beta` remain selectable. Native ~4 MP aspect-ratio
sizes are accepted, and Comfy Kitchen INT8 attention plus Spectrum
hidden-state forecasting are available as experimental opt-in speed settings.
For image edits, **Edit output size** offers three exclusive choices:
**Match first image size** (default), **Max resolution (up to 4 MP)**, or
**Use width and height above**. Max resolution scales the first reference
image's aspect ratio toward 4 MP, rounds the output to multiples of 32, and
keeps both sides within 2752 pixels.

Reference PNGs retain their alpha channels through native RGBA reconstruction
before Qwen encoding. The scheduler counts the output latent's native 16× grid
for both editing and text-to-image; reference token counts do not affect it.

The Qwen tab offers three editable presets: **Fast** selects INT8 ConvRot,
Official Turbo, eight steps, and accelerator Off; **Normal** selects BF16,
Turbo Off, 25 steps, and Spectrum (Quality); **Quality** selects BF16, Turbo
Off, 40 steps, and Spectrum (Quality). Fast is selected initially. The text
encoder and other controls retain their chosen values when switching presets.

**Official Turbo (8-step)** uses Kijai's LoRA extraction of the official
Qwen-Image-2.1-Turbo checkpoint from
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1#turbo).
Only the approximately 913 MB adapter downloads on first use; the selected
base BF16/INT8 model, text encoder, and VAE are reused. It uses ComfyUI's native
LoRA loader at strength 1, eight Euler steps, CFG 1, and accelerator Off.
The checkpoint's fixed sigma schedule is supplied explicitly, without the
base model's resolution shift or terminal stretch, regardless of the scheduler
control. Generation and editing with up to ten references are supported.

**LightX2V Lightning (4-step)** adds the approximately 67 MB
[Qwen-Image-2.1-Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning)
as an on-demand download, reusing the selected base model, encoder, and VAE.
It uses the native LoRA loader at strength 0.50, four Euler steps, CFG 1, and
accelerator Off for generation and editing. Full strength shifts subjects to the
left; 0.50 is the
[reported fix](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/discussions/7),
though a slight shift can remain. **Lightning LoRA strength** (0.05–1.5) appears under
**Model and encoding** for both Lightning modes, including the 6-step community mode. Its raw sigma nodes
`[1, 0.75, 0.5, 0.25]` receive the author's resolution-dependent exponential
shift (256/8192 tokens, 0.5/0.9 shifts), followed by zero, with no terminal
stretch. This schedule overrides the scheduler control. Edit KV caching is
disabled to match the published example. GPU image quality has not been
validated locally; the author notes that BF16 inference can differ from the
FP32 latents used during training and LightX2V inference.

Checked 2026-10-15: the official eight-step adapter still has SHA-256
`208dd43250e1e01467ba190572ae2e7107a7870ec1ff026bc65f791fa1e80e95`, matching
our existing inventory. The upstream Turbo checkpoint's latest changes are
model-card examples; its fixed eight-step sigma schedule is unchanged.

The Qwen tab also offers **Viggle Turbo v0.2.1** as a six-step ComfyUI LoRA mode.
It downloads the rank-256 adapter on first use and uses its published six-node
schedule with Qwen's resolution-dependent time shift. A bundled custom loader
applies the adapter without merging it into BF16/INT8 weights. Turbo editing
accepts up to three references. The
base mode retains its existing controls and up to ten references. Viggle Turbo
is released under the Qwen Research License for non-commercial research or
evaluation; commercial use requires a separate licence.

**Viggle Turbo v0.3 (6-step)** uses the unmerged v0.3 rank-256 adapter for
all six steps with the published six-step schedule. Compared with v0.2.1,
Viggle reports cleaner surfaces and less grain, with slightly softer fine
texture. It shares the adapter download with the nine-step mode.

**Viggle Turbo v0.3 (9-step)** is an optional hybrid mode: seven Euler steps
with the unmerged v0.3 rank-256 adapter, then two steps with the base model.
It uses the published nine-step sigma schedule, CFG 1, and accelerator Off.
The base phase continues the intermediate noisy latent without adding noise;
ComfyUI resets its text/reference KV cache between the two sampling runs.
The adapter downloads on first use; no second base checkpoint is required.
Viggle reports finer detail and improved small text at about 1.4–1.5 times
the six-step runtime, though results and timing depend on the request and
hardware. The Fast, Normal, and Quality presets retain their existing settings.

## MOSS-Image

The **MOSS-Image** tab runs [MOSS-Image SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
on ComfyUI's native MOSS-Image support. Every file in
[Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) is selectable and
downloads on first use: the SFT or pretrained Base DiT in BF16 or INT8 ConvRot, a Qwen3.5 9B
text encoder (INT8 ConvRot by default, W4A8 or BF16 optional) and the full or small FLUX.2
VAE. The full VAE is the same file as FLUX 3's and is shared with it.
**Fast** (default) uses INT8 SFT, the small VAE and the
[Lodestones 8-step DMD LoRA](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd)
at 8 Euler steps and CFG 1; the LoRA requires an SFT model. **Balanced** and **Quality** run
BF16 SFT with the full VAE and no LoRA for 25 or 50 steps at CFG 3, skipping layer 22 in the empty negative pass as in OpenMOSS's
reference sampler. Other step-distilled LoRAs can be added to the **Turbo LoRA** list.
**Accelerate with Spectrum** is on by default. The pinned
[Spectrum-MOSS](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/tree/3fde0674b29342565e7f42a0674af48d65aabc4a)
adaptive node uses the upstream defaults and Euler + `sgm_uniform` for non-distilled runs
longer than 10 steps. Turn it off to restore the original Euler + `simple` graph.
Step-distilled Turbo LoRAs, including the default Fast preset, bypass Spectrum and keep
their original schedule. Until setup has installed the node and ComfyUI has restarted,
generation reports that and uses the original graph. The output settings record both the
preference and whether Spectrum was actually applied. Forecasting can affect details; its GPU speed/quality
has not been benchmarked here.
**1M** and **2M resolution presets**, OpenMOSS's recommended range, set common aspect
ratios on the 16-pixel grid. The FLUX 3 tab has matching **1K** and **2K** presets up to its
4 MP limit, and a FLUX 3 preset overrides edit size matching, as in Qwen Image 2.1.
Optional layout boxes use the same JSON format as FLUX 3. **Rewrite with Lightning AI or
Gemini** follows OpenMOSS's
[prompt-enhancer skill](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer):
the selected LLM returns a caption or a pixel Layout draft for the current width and height,
and the app validates and compiles it with a port of the skill's `compile_prompt.py`.

## Fibo 2.0

**Fibo 2.0** offers **Dev** for text-to-image and editing (30 steps, CFG 5 by default),
and **Edit Turbo** for editing only (4 steps, CFG 1). Add up to four ordered reference
images; refer to them as `<image_1>`, `<image_2>`, etc. Enter a plain short caption, an
edit instruction or structured VGL JSON. No prompt-conversion API key is required.
Bria trained Fibo 2.0 at about 1 MP. The **1M** presets are its trained sizes for each
aspect ratio (1024×1024, 1424×800 and so on); the **2M** presets, up to 1456×1456 or
1920×1088, run above the trained size and their quality is unverified. **Edit output size**
keeps the first reference's aspect ratio at the trained ~1 MP size (default), at its own
size capped at 2 MP, or scaled up to 2 MP (**Max resolution**), or uses the width and
height. A preset overrides it, as in FLUX 3. A single reference is center-cropped to the
output size.
The native nodes have no mask input, so masked edits are not offered.

Fibo runs on ComfyUI's native Fibo 2.0 support (`TextEncodeFibo2`, CLIPLoader type
`fibo2`) with the ungated [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0)
repack; no Hugging Face access request or token is needed. **Precision** selects BF16
(16.5 GB) or INT8 ConvRot (8.4 GB, lower VRAM) diffusion weights. Text-to-image uses the
INT8 Qwen3-VL 4B encoder and the FLUX.2 VAE, both byte-identical to FLUX 3's and shared
with it. Edits use the BF16 encoder (8.9 GB), since the INT8 encoder fails on image inputs
([Comfy-Org/ComfyUI#17231](https://github.com/Comfy-Org/ComfyUI/issues/17231)).
Selected weights download on first use. The weights are CC BY-NC 4.0 (noncommercial).
Rerun `setup_h3.py` and restart ComfyUI for the new Core pin; Modal users should redeploy.
The UI shares reference-library selection, generation progress, cancellation and output
metadata with the other image engines.

## YuE2

The **YuE2** tab uses ComfyUI's native YuE2 nodes (ComfyUI v0.36.0 or newer).
Its INT8 ConvRot checkpoint (about 4 GB) is selected by default; the BF16
checkpoint is an optional alternative. The selected checkpoint downloads on
first use into ComfyUI/models/checkpoints/. Use **Full score** to plan melody
and chords, **Melody** for a melody plan, or **Direct** to skip the score
planner. Lyrics should use section tags such as [verse] and [chorus].
Tiled audio decode is enabled by default for long songs. YuE2 weights from
[Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) are licensed
CC-BY-NC-4.0. Generation is exposed as /generate_yue2; style, lyrics, and an
optional edited ABC score are accepted with the generation settings.

## Prompt writers

The **Qwen Image 2.1**, **LTX-2.5**, **MiniMax Music 3**, and **YuE2** tabs
also offer Gemini and Lightning AI prompt writers, with Gemini
(`gemini-3.8-flash`) selected by default. Lightning AI remains selectable and
uses the same fixed model, server environment key, and temporary key behavior
as the H3 writer. Each writer uses its own bundled
system prompt in `prompts/`: `prompt_qwen_image21.txt`, `prompt_ltx25.txt`,
`prompt_music3.txt`, or `prompt_yue2.txt`.

LTX-2.5 and Music 3 can use optional keyframe or visual-reference images.
Qwen Image 2.1 understands text-to-image and image-edit modes: it refers to a
single edit input naturally and uses ordered `<image1>`, `<image2>`, ... tags
for multi-image edits. YuE2 jointly creates or enhances the production style
and sectioned lyrics. Their UI/API operations are `/enhance_ltx25_prompt`,
`/enhance_music3_prompt`, `/enhance_qwen_image21_prompt`, and
`/enhance_yue2_prompt`, respectively.
