# Experimental DLSS 5 Media enhancement

Select **DLSS 5 Neural Rendering (experimental)** in Media for images or videos. Default/Natural/Cinematic styles and intensity are available. Keep original resolution is the default. Disabling it selects the largest supported DLSS factor fitting the target box: 1×, 1.5×, 1.724×, 2×, or 3×. The actual resolution is recorded in output metadata. Output must fit 7680 × 4320 in either orientation; no downscaling occurs.

**Upscaler (above 1×).** The default, *Lanczos resize, then DLSS 5 at 1×*, enlarges images (PIL) and video frames (ffmpeg) to the target and runs the neural pass at native size; metadata records `stage_order: Lanczos → NR`. Under Wine the ordinary DLSS Super Resolution carrier fails: NGX's signed load of `nvngx_dlss.dll` (`nvLoadSignedLibraryW`) is rejected with the Windows 616 core, giving `CreateFeature(1) -> 0xBAD0000B`. *DLSS Super Resolution (experimental)* uses that carrier (`DLSS SR → NR`). On Linux it runs from a separate host directory whose `_nvngx.dll` is extracted from a Windows R580-branch driver, which matches the host's Linux driver branch. Each job's runtime check uses the same NGX core as the job. The first use downloads that package (~800 MB). It tries 581.57, 581.42, 581.29, 580.97 and 580.88 in order; set `H3_DLSSNR_R580_DRIVER` to choose another. These packages are not checksum-pinned; the downloaded archive and core hashes are recorded in output metadata (`ngx_core`). The default host directory and core are never modified.

The implementation uses the MIT-licensed RunningHub host and bridge, pinned to commit `71ab1060865ea781c7f4a8b7c0c79600f51b6617`. The Python adapter converts bounded RGBA frames into its DNR2 protocol: RGB float32 pixels, FP16 motion guides, indexed reply headers, and END1 completion. Feature readiness is recorded only after a successful frame evaluation. Images use zero motion and retain their original alpha separately. Videos use OpenCV DIS motion and scene-cut resets when OpenCV is available; otherwise every frame resets history with zero motion.

## Automatic processing-host setup

`run_h3.sh` invokes the shared `h3_nvidia_runtime.py` installer. Colab uses that same launcher. Modal builds immutable runtime files into its image, then initializes its dedicated Wine prefix against the live host after restore. Installation and runtime checks must run on the processing host, not on a developer's desktop.

Setup obtains:

- Wine 11 amd64-wow64, which avoids a dependency on the 32-bit Linux loader.
- DXVK 3.1, VKD3D-Proton 3.0.1 and DXVK-NVAPI 0.9.2, with release SHA-256 checks.
- RunningHub's host, bridge and caller shim, checked against pinned Git blob identities.
- The existing SHA-256-pinned model bundle; its previous video processing executable is not used.
- `_nvngx.dll` extracted from the pinned official NVIDIA **Windows** 616.56 package. The driver installer is never executed and the host driver is never replaced. This first setup downloads a large driver archive solely for extraction.
- A server-only build of pinned DXVK-NVAPI with a narrow headless fix: when Vulkan has no valid device LUID, use the same DXGI adapter's LUID. The build uses distribution MinGW/Meson packages and pinned submodules, with tests disabled, and caches the resulting DLL. This is a project patch, not an upstream released fix.

Linux setup requires apt-get with root or sudo access. Modal/Colab install the same prerequisites. Set `H3_ENABLE_DLSSNR=0` to opt out. No optional enhancement failure prevents ordinary generation from starting.

The new managed Wine prefix is isolated from prior prefixes. Wine inherits no application `LD_LIBRARY_PATH`. NR is enabled through the NVAPI DRS override. A separately managed Xvfb display keeps display wrappers away from binary stdout. CUDA GPU assignments are translated to a Vulkan UUID; `H3_DLSSNR_GPU_UUID` can explicitly identify the assigned GPU.

## Processing-host requirements and diagnostics

CUDA or `nvidia-smi` alone is insufficient. The assigned NVIDIA GPU must appear in `vulkaninfo --summary` under the display used by Wine. Container creation must expose the NVIDIA graphics capabilities, ICD, and matching userspace driver libraries, including ray-tracing components used by VKD3D. Setting a capability variable after container creation cannot mount missing host libraries. The launcher cannot repair provider-controlled device access. When the provider mounts `libGLX_nvidia.so.0` but omits the Vulkan ICD manifest (loader error `Found no drivers`), the launcher generates `h3/nvidia/vulkan/nvidia_icd.json` and points `VK_DRIVER_FILES` at it. Failure messages include host diagnostics: `NVIDIA_DRIVER_CAPABILITIES`, ICD manifests found, and whether `libGLX_nvidia`, `libnvidia-glcore`, `libnvidia-rtcore` and `libnvidia-glvkspirv` are present.

Under Wine, NGX cannot locate its driver store (`QueryAdapterInfo` fails with `0xD0000002`), so it falls back to `C:\ProgramData\NVIDIA\NGX`. Before each run the launcher makes sure `C:\ProgramData` exists in the dedicated prefix and removes `NVIDIA\NGX`, whose access control list NGX rewrites when it creates the folder. Otherwise `NGXGetPath` fails and `Init_ProjectID` returns `0xBAD00002`. New prefixes wait for `wineserver` to exit so that the registry is saved.

Startup checks Wine prefix creation and NVIDIA Vulkan visibility; it does not run GPU inference. **Check NVIDIA runtime** evaluates two textured native-size frames and rejects bypassed, malformed or all-black output. Image/video enhancement runs this probe before opening the main processing pipeline. Actual model evaluation on the deployed GPU remains the compatibility gate; successful provisioning is not proof of inference support.

Videos must be SDR with constant frame rate, square pixels, zero start timestamp and normalized rotation. HDR/linear/BT.2020 sources are rejected. A known frame count from 1 to 1,000,000 is required by the host protocol; if container metadata omits it, ffprobe counts it first. Timing and frame counts are validated before publication. Audio is copied when MP4-compatible, otherwise encoded as AAC. Images are normalized to sRGB and saved losslessly as PNG with original transparency restored.

Cancellation stops job-owned processes and only the dedicated Wine prefix's server. Source files are retained. Partial outputs are never published. Output provenance records the model/helper hashes, actual dimensions, motion backend, stage order, frame rate/count, encoder and source lineage.

## References

- [RunningHub Linux implementation](https://github.com/RH-RunningHub/ComfyUI-RH-DLSS5)
- [NGX runtime requirements](https://github.com/RH-RunningHub/ComfyUI-RH-DLSS5/blob/main/runtime/README.txt)
- [NVAPI source](https://github.com/jp7677/dxvk-nvapi/tree/v0.9.2)
- [NVIDIA container driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)
- MIT notices for the adapted motion code and host: `docs/licenses/RunningHub-DLSS5-MIT.txt`.
