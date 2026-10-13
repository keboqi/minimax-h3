# NVIDIA enhancement in Media

Select an image or video in **Media → Enhance selected media** and choose **NVIDIA RTX Video Super Resolution** or **DLSS 5 Neural Rendering (experimental)**. Both save a new copy and keep the original. Imported videos work too. The controls and runtime checks run on the server that processes media, not the browser's machine.

VSR provides quality, enhancement strength, compressed/high-bitrate source profiles, and optional denoise/deblur preprocessing. To clean a video without upscaling, enable **Keep original resolution** and choose Denoise or Deblur. At original resolution the preprocessing strength is used. DLSS 5 provides Default/Natural/Cinematic styles and intensity. It defaults to original resolution; disable **Keep original resolution** for DLSS Super Resolution followed by neural rendering.

The initial integration supports SDR, constant-frame-rate videos with square pixels, zero video start timestamp, and normalized rotation. HDR/linear/wide-gamut BT.2020 sources and irregular timestamps are rejected explicitly. The frame-timing check is conservative: some container timestamp jitter can require conversion to CFR even when the video is nominally CFR. It scans the whole video before enhancement. Targets preserve aspect ratio with encoder/SDK alignment; VSR supports 1–4× and the DLSS adapter limits upscaling to 1–3×. No downscaling is performed.

Video output is 8-bit HEVC MP4. The encoder is tested before enhancement: HEVC NVENC is preferred, with CPU libx265 if hardware encoding cannot initialize. An encoding failure after processing starts fails the job instead of silently recomputing it. Compatible audio streams are copied; other audio is converted to AAC. Videos without audio are supported. Frame count, rate, dimensions, and audio-stream presence are checked before publication.

Image output is lossless 8-bit PNG with an embedded sRGB profile. EXIF orientation is applied before enhancement; valid embedded color profiles are converted to sRGB. Transparency is retained separately and resized to the target using Lanczos, then restored after enhancement. Each image uses one RGB/RGBA worker frame; DLSS motion estimation is disabled for stills. The image path does not encode an intermediate video or require a video encoder. Animated images, high-bit-depth/linear PNGs, unsupported pixel modes, invalid color profiles, and images over 8192 pixels per axis are rejected. Untagged 8-bit images are treated as sRGB. Native-size images may have odd dimensions; video output still requires even dimensions.

## Automatic installation

Run the normal launcher, `bash run_h3.sh`, including from the Colab notebook. Setup installs `nvidia-vfx==0.2.0.0` by default. The launcher downloads and verifies the pinned video2dlssnr v1.4.1 release, including its executable, forwarder, Super Resolution library, and neural rendering model. Linux setup also installs its display/Vulkan dependencies and GE-Proton11-7 graphics stack. Older glibc hosts, including Ubuntu 22.04 Colab sessions, automatically use the pinned Kron4ek Wine 11.0 runner with the same graphics stack.

Files are cached under `h3/nvidia` (`H3_NVIDIA_RUNTIME_DIR` overrides this location). Linux receives a dedicated prefix, native DXVK/VKD3D/NVAPI DLLs, and a headless Xvfb display when no desktop display exists. Startup locates the host driver's NGX Wine bridges or downloads that exact NVIDIA driver archive and **extracts only the bridge DLLs**. It never installs or upgrades the host driver. Missing libraries are repaired on restart; successful installations reuse their files. The manifest records the helper, runner, neural model hash, and driver version. Downloads remain outside source control.

No manual helper paths, model downloads, Wine installation, or prefix configuration are required. The VSR worker uses `nvvfx.VideoSuperRes` directly, without a third-party ComfyUI node. For an intentional opt-out, set `H3_ENABLE_NVIDIA_VSR=0` or `H3_ENABLE_DLSSNR=0` before setup/deployment. Linux package installation uses apt-get with root or sudo access; the Colab and Modal environments already run as root.

For a separate GPU Python, set `H3_VSR_PYTHON` to its absolute executable path before starting the app. That environment needs the pinned VFX package, NumPy, and a CUDA/Blackwell-compatible PyTorch build. The worker imports these only when run. The browser/UI process does not load the GPU runtime while browsing Media.

NVIDIA documents Linux VSR driver requirements as 570.190+, 580.82+, or 590.44+ on the corresponding driver branches. Effect initialization remains the authoritative runtime check. See the [SDK requirements](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html).

## Runtime checks and advanced overrides

The [video2dlssnr](https://github.com/DaniilSokolyuk/video2dlssnr) helper uses its raw RGBA `--nr-video` pipeline, target size, style/intensity, and automatic optical flow. Its libraries retain their upstream licenses/terms; the installer keeps the upstream README beside the executable. Automatic installation establishes prerequisites, not GPU compatibility. NVIDIA's driver, Vulkan capabilities, and neural feature initialization remain authoritative. A Colab/Modal provider's incompatible GPU or driver cannot be fixed by installing these files; the Media runtime check reports that failure. Linux RTX PRO 6000 neural execution still needs verification on the actual host.

Advanced users can bypass managed DLSS installation by setting `H3_DLSSNR_EXE` before launch. On Windows:

```powershell
$env:H3_DLSSNR_EXE = 'D:\tools\video2dlssnr\video2dlssnr.exe'
$env:H3_DLSSNR_DLL_DIR = 'D:\tools\video2dlssnr'
```

For a custom Linux installation, supply a compatible runner and **dedicated** initialized prefix:

```bash
export H3_DLSSNR_EXE=/opt/video2dlssnr/video2dlssnr.exe
export H3_DLSSNR_DLL_DIR=/opt/video2dlssnr
export H3_DLSSNR_WINE=/absolute/path/to/wine
export H3_DLSSNR_WINEPREFIX=/absolute/path/to/dedicated-dlss-prefix
```

`H3_DLSSNR_WINE` is the Wine executable, not a shell command or Proton launch string. A custom runner needs a sibling `wineserver`, DXVK/VKD3D/NVAPI, matching NGX bridges, and a prefix containing `system.reg`. Explicit custom paths take precedence over the managed configuration.

The prefix is exclusively for this feature: the application locks it while using it and stops that prefix's wineserver on completion, cancellation, or failure. Never point it at a shared application or game prefix. Other Wine prefixes are not stopped.

Managed setup saves the model SHA-256 automatically. For custom installations, `H3_DLSSNR_MODEL_SHA256` provides the same verification. Hash verification runs inside processing, not while browsing. Each output records the model and executable hashes. Keep the exact helper/model/runner combination fixed when reproducing results.

Choose **Check NVIDIA runtime** to evaluate two textured frames on the GPU. The DLSS probe runs at native resolution, requires changed RGB pixels and the helper's neural-feature initialization confirmation, and rejects pass-through output (including alpha-only changes). Enhancement repeats the probe before processing the selected video; a passing file/prerequisite check alone never establishes neural execution. The underlying helper must return nonzero on a failed NGX evaluation, as the inspected `video2dlssnr` implementation does. **Stop runtime check** cancels the probe; **Interrupt** cancels enhancement. Both share the existing GPU queue with generation jobs.

This integration does not reimplement NVIDIA's model and does not establish official DLSS support for RTX PRO cards. Linux/Wine behavior and actual neural output still require acceptance on the intended processing host. A runtime failure is shown in the Media status and recorded as a failed job.

## Modal and headless hosts

The standard Modal image installs VSR and builds the same pinned DLSS runtime automatically. It initializes its dedicated Wine prefix and matching driver bridges after snapshot restore, before launching Gradio. Rebuild/redeploy to receive these image dependencies. No GPU inference runs during image construction.

Colab's `run_h3_colab.sh` delegates to `run_h3.sh`, so notebook and terminal launches receive the same setup. First launch needs extra download time and disk space for the helper/model and Linux runner. Existing generation options remain available if experimental DLSS host preparation fails; the precise error appears in Media. Runtime browsing and Media actions never install system packages or download software.

## Output records and verification

Successful output settings include the method parameters, requested/actual dimensions, rational frame rate, processed frame count, encoder and fallback status, audio policy, and source asset lineage. VSR also reports its runtime, PyTorch version, GPU, and selected effect mode; DLSS records executable/model hashes, requested motion mode, and the actual optical-flow backend when reported by the helper. Runtime evidence is retained even for long videos. Working files are temporary and never appear as completed library assets.

CPU tests use synthetic workers, still images, and small FFmpeg clips, covering raw binary transport, transparency, color/orientation, failures, timeouts, cancellation, metadata validation, lineage, and request capture. They do not load either GPU runtime. Before claiming hardware support, run a real runtime probe, still-image comparisons, a short clip with motion/cuts, a representative long clip, cancellation followed by another job, and original/enhanced comparisons on the actual deployment. Record warm-up separately from steady-state speed and memory. No GPU speed or quality claim has been validated by this change.
