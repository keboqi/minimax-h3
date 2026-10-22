# Media and finishing

The Media library and the upscale, restoration, HDR and interpolation methods
available from Media and as automatic H3 finishing.

Open **Media** to browse the video library (the default), or switch **Media
type** to **Image** or **Audio**. Image mode includes generated Qwen/H3 stills
and SeedVR2 results; Audio mode includes MiniMax H3, MiniMax Music 3, and YuE2
outputs. All three use the same card, preview, settings, import, and download
layout. **Import local media** accepts a matching file for the active library.
New and imported images, videos, and audio create their thumbnails and catalog
metadata when they are saved. The persistent SQLite catalog lives at
`outputs/.h3-workspace/workspace.sqlite3`, and thumbnails live in
`outputs/.gallery_thumbnails`. Browsing, switching media types, and restarting
the server read the catalog without scanning the output folders. This workflow
is the same for local storage and mounted Google Drive.
Use **Scan historical media** once to add older or externally created files,
or to reconcile files changed outside the app. This explicit scan preserves
existing tags and favorites. Clicking an image thumbnail displays the full
original image; the grid uses the smaller saved thumbnails.
To compare outputs, select a library thumbnail and choose **Add to compare A**
or **Add to compare B**. The slots show thumbnails and keep their picks while
you filter the library. Choose two different images or two different videos;
adding another type starts a new pair. **Clear A** and **Clear B** remove picks.
In **Jobs**, click a history row (or focus it and press Enter/Space) to inspect
that request and its available actions.
Image mode offers the one-step **SeedVR2 2x** workflow and
**LTX-2.5 CQ Image Enhancer**, using the same target-resolution presets.
CQ resizes the selected still to fit the target, then generatively restores
detail using the dedicated image LoRA, the LTX development INT8 model, a
0.5-strength distillation adapter, and the convolutional VAE. It needs no prompt
and can also enhance at the same resolution or resize downward. Its required
models download on first use. Reconstructed detail may differ from the source.
The methods preserve the image's aspect ratio;
the processed image is added back to the image gallery with its settings and a
download link. Audio mode provides playback and downloads. Video mode retains
all existing enhancement methods. Each run
preserves the source and adds a new processed output. Choose a target preset from
**1280 × 1280**, **1920 × 1920**, **2560 × 2560**, or **3840 × 3840**; the source
is fitted inside that square without cropping, so its original aspect ratio is
preserved. **SeedVR2 2x** uses ComfyUI's native one-step restoration workflow.
**LTX-2.5 IC-LoRA 2x** is a generative
alternative that synthesizes fine detail with the transformer selected in the
**LTX 2.5**
tab and the official gated pixel spatial upscaler IC-LoRA. Gallery runs accept
an optional scene prompt; automatic post-processing reuses the H3 generation
prompt. **LTX-2.5 Tiled Upscale** adds the upstream Tiled Fusion sampler with
overlapping spatial tiles and 97-frame temporal windows on one shared canvas.
Use it for large target resolutions without enabling independent clip splits.
It uses the gated [Refine Details IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details),
downloaded on first use, and the selected LTX base model. Streaming guides are
encoded freshly for each temporal window. Media accepts an optional texture or
lighting prompt; automatic H3 finishing uses a neutral detail prompt. Avoid
named subjects in tiled prompts because each spatial tile receives that prompt.
Inputs are padded to LTX-compatible frame counts and trimmed back after decoding.
All four upscale methods preserve the source audio and frame rate and can
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
repositories. Deblur targets defocus, not motion blur. Use **Advanced: split source
into clips before LTX processing** for restoration clips that exceed available VRAM.

**LTX-2.5 IC-LoRA SDR to HDR** converts an SDR gallery video into a 10-bit
BT.2020/HLG HEVC MP4 while preserving source resolution, frame rate, and audio.
It uses the LTX base selected in the **LTX 2.5** tab, precomputed scene embeddings
(no prompt or runtime text encoder), ACEScct input conversion, and the full
LTX video VAE in float32. The adapter, embeddings, and required base/VAE weights
download on first use. Accept access to the separate gated
[SDR-to-HDR repository](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR)
and authenticate with `hf auth login` or `HF_TOKEN`. HDR playback requires an
HLG/HEVC-capable player and display. Optional split processing joins HDR streams
without re-encoding them. This uses plain IC-LoRA conditioning; the upstream
pipeline's optional seam-keyframe/high-quality mode and EXR export are not enabled.

Independent clip splitting remains off by default. Tiled Fusion streams temporal
windows internally, but source frames, output frames, and the shared latent canvas
still require memory. If a long or high-resolution source exceeds available memory,
enable **Advanced: split source into clips before LTX processing** in Media, or
**Advanced: split source into clips before LTX upscaling** in the MiniMax H3
post-processing settings. The
default target is 5 seconds per clip; cuts are adjusted to LTX-compatible frame
counts, clips are upscaled sequentially, and their video streams are joined
without an additional video encode. The final file is trimmed to the original
frame count and remuxed with the original source audio. Because clips are
generated independently, a visible detail or motion change can occur at a cut.

Existing installations need setup and a ComfyUI restart to load the new pinned
LTX nodes and local frame-padding node; rebuild/redeploy Modal images as applicable.
The bundled upstream workflow inventory includes ten classic workflows and the two
Tiled Fusion examples. Tiled upscaling is exposed through Media and H3 finishing.

SeedVR2 offers **7B FP16**, **7B INT8 (default)**, **3B FP16**, **3B INT8**,
**7B Sharp FP16**, experimental **7B MXFP8**, and legacy fast/experimental
**NVFP4** choices. Only the
selected checkpoint downloads on first use; all choices share the same lazy FP16
SeedVR2 VAE. The native workflow uses
1024-pixel VAE encode/decode tiles for the RTX PRO 6000 target. SeedVR2 and main
H3 generation run eagerly because full-model compile did not improve measured
performance and conflicts with the active attention and cache optimizations.
