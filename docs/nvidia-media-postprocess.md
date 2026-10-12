# NVIDIA enhancement in Media

Select an image or video in **Media → Enhance selected media** and choose **NVIDIA RTX Video Super Resolution** or **DLSS 5 Neural Rendering (experimental)**. Both save a new copy and keep the original. Imported videos work too. The controls and runtime checks run on the server that processes media, not the browser's machine.

VSR provides quality, enhancement strength, compressed/high-bitrate source profiles, and optional denoise/deblur preprocessing. To clean a video without upscaling, enable **Keep original resolution** and choose Denoise or Deblur. At original resolution the preprocessing strength is used. DLSS 5 provides Default/Natural/Cinematic styles and intensity. It defaults to original resolution; disable **Keep original resolution** for DLSS Super Resolution followed by neural rendering.

The initial integration supports SDR, constant-frame-rate videos with square pixels, zero video start timestamp, and normalized rotation. HDR/linear/wide-gamut BT.2020 sources and irregular timestamps are rejected explicitly. The frame-timing check is conservative: some container timestamp jitter can require conversion to CFR even when the video is nominally CFR. It scans the whole video before enhancement. Targets preserve aspect ratio with encoder/SDK alignment; VSR supports 1–4× and the DLSS adapter limits upscaling to 1–3×. No downscaling is performed.

Video output is 8-bit HEVC MP4. The encoder is tested before enhancement: HEVC NVENC is preferred, with CPU libx265 if hardware encoding cannot initialize. An encoding failure after processing starts fails the job instead of silently recomputing it. Compatible audio streams are copied; other audio is converted to AAC. Videos without audio are supported. Frame count, rate, dimensions, and audio-stream presence are checked before publication.

Image output is lossless 8-bit PNG with an embedded sRGB profile. EXIF orientation is applied before enhancement; valid embedded color profiles are converted to sRGB. Transparency is retained separately and resized to the target using Lanczos, then restored after enhancement. Each image uses one RGB/RGBA worker frame; DLSS motion estimation is disabled for stills. The image path does not encode an intermediate video or require a video encoder. Animated images, high-bit-depth/linear PNGs, unsupported pixel modes, invalid color profiles, and images over 8192 pixels per axis are rejected. Untagged 8-bit images are treated as sRGB. Native-size images may have odd dimensions; video output still requires even dimensions.

## VSR installation

The app starts without NVIDIA's optional Python package. To install it into the existing H3 GPU environment, run setup using that environment's Python:

```bash
python setup_h3.py --install-dir h3 --skip-env --enable-nvidia-vsr
```

The pinned requirement is `nvidia-vfx==0.2.0.0`. Alternatively set `H3_ENABLE_NVIDIA_VSR=1` when running normal setup. This does not add a third-party ComfyUI node. A worker uses `nvvfx.VideoSuperRes` directly.

For a separate GPU Python, set `H3_VSR_PYTHON` to its absolute executable path before starting the app. That environment needs the pinned VFX package, NumPy, and a CUDA/Blackwell-compatible PyTorch build. The worker imports these only when run. The browser/UI process does not load the GPU runtime while browsing Media.

NVIDIA documents Linux VSR driver requirements as 570.190+, 580.82+, or 590.44+ on the corresponding driver branches. Effect initialization remains the authoritative runtime check. See the [SDK requirements](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html).

## DLSS 5 helper installation

Use the operator-supplied [video2dlssnr](https://github.com/DaniilSokolyuk/video2dlssnr) Windows helper and its matching libraries. The supported interface is its raw RGBA `--nr-video` pipeline, `--nr-in`, target size, style/intensity, and automatic optical-flow options. Keep `nvngx.dll_dlssnr.dll` beside the executable. Supply `nvngx_dlssnr.dll` and any additional required NGX/runtime libraries from an authorized installation; these proprietary files are not included in this repository or downloaded by the application.

On Windows:

```powershell
$env:H3_DLSSNR_EXE = 'D:\tools\video2dlssnr\video2dlssnr.exe'
$env:H3_DLSSNR_DLL_DIR = 'D:\tools\video2dlssnr'
```

On Linux, install a compatible Wine runner and its D3D12/Vulkan/NVAPI dependencies, and initialize a **dedicated** prefix for this helper before launching H3. Set:

```bash
export H3_DLSSNR_EXE=/opt/video2dlssnr/video2dlssnr.exe
export H3_DLSSNR_DLL_DIR=/opt/video2dlssnr
export H3_DLSSNR_WINE=/absolute/path/to/wine
export H3_DLSSNR_WINEPREFIX=/absolute/path/to/dedicated-dlss-prefix
```

`H3_DLSSNR_WINE` is the Wine executable, not a shell command or Proton launch string. Use a runner with a sibling `wineserver` executable. Configure its prefix's DXVK/VKD3D/NVAPI stack according to that runner and the helper's requirements. The prefix must already contain `system.reg`. A standard Wine installation alone does not establish DLSS compatibility. No runner version is claimed validated on Linux RTX PRO 6000 yet.

The prefix is exclusively for this feature: the application locks it while using it and stops that prefix's wineserver on completion, cancellation, or failure. Never point it at a shared application or game prefix. Other Wine prefixes are not stopped.

Set `H3_DLSSNR_MODEL_SHA256` to the expected model hash to reject mismatched runtime files. Hash verification runs inside processing, not while browsing. Each output records the model and executable hashes. Keep the exact helper/model/runner combination fixed when reproducing results.

Choose **Check NVIDIA runtime** to evaluate two textured frames on the GPU. The DLSS probe runs at native resolution, requires changed RGB pixels and the helper's neural-feature initialization confirmation, and rejects pass-through output (including alpha-only changes). Enhancement repeats the probe before processing the selected video; a passing file/prerequisite check alone never establishes neural execution. The underlying helper must return nonzero on a failed NGX evaluation, as the inspected `video2dlssnr` implementation does. **Stop runtime check** cancels the probe; **Interrupt** cancels enhancement. Both share the existing GPU queue with generation jobs.

This integration does not reimplement NVIDIA's model and does not establish official DLSS support for RTX PRO cards. Linux/Wine behavior and actual neural output still require acceptance on the intended processing host. A runtime failure is shown in the Media status and recorded as a failed job.

## Modal and headless hosts

Set `H3_ENABLE_NVIDIA_VSR=1` in the deployment environment when building the Modal image to include the optional VFX runtime. Rebuild/redeploy after changing this build flag. Validate the effect on the allocated GPU/host driver; the application cannot upgrade the provider's driver.

DLSS 5 is not bundled into the standard Modal image. An operator must provide the helper files, dedicated initialized Wine environment, dependencies, and environment configuration in a custom image/mount, then pass **Check NVIDIA runtime** in that container. Keep proprietary binaries out of public images. Existing generation and enhancement options work without these dependencies. Installing Wine or deploying GPU infrastructure is not done automatically by Media actions.

## Output records and verification

Successful output settings include the method parameters, requested/actual dimensions, rational frame rate, processed frame count, encoder and fallback status, audio policy, and source asset lineage. VSR also reports its runtime, PyTorch version, GPU, and selected effect mode; DLSS records executable/model hashes, requested motion mode, and the actual optical-flow backend when reported by the helper. Runtime evidence is retained even for long videos. Working files are temporary and never appear as completed library assets.

CPU tests use synthetic workers, still images, and small FFmpeg clips, covering raw binary transport, transparency, color/orientation, failures, timeouts, cancellation, metadata validation, lineage, and request capture. They do not load either GPU runtime. Before claiming hardware support, run a real runtime probe, still-image comparisons, a short clip with motion/cuts, a representative long clip, cancellation followed by another job, and original/enhanced comparisons on the actual deployment. Record warm-up separately from steady-state speed and memory. No GPU speed or quality claim has been validated by this change.
