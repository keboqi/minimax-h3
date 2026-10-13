# DLSS 5 processing-host acceptance plan

Implementation details and deployment setup are maintained in [the enhancement guide](nvidia-media-postprocess.md).

Acceptance must run on the assigned processing host only. No local installation, Wine experiments, GPU probes or test execution are authorized for this change.

1. Launch through the normal standalone script, or rebuild the Modal image / launch the Colab notebook. Record provisioning failures separately from inference failures.
2. Confirm the assigned NVIDIA Vulkan GPU and initialized Wine prefix. Report missing provider driver components without replacing the host driver.
3. Run Media's explicit runtime check and retain NGX/NVAPI diagnostics.
4. Enhance a still image at native size, including transparency, then a supported upscale factor.
5. Enhance a short SDR video with motion, a scene cut and audio; verify actual resolution, frame count, frame rate and audio.
6. Cancel processing, then run again to check dedicated-prefix cleanup. Check a longer video for bounded memory and sustained throughput.

The host/bridge, Windows core, graphics stack, headless NVAPI patch and model have not been validated together on the user's GPU by local checks. Compatibility and image quality remain deployment acceptance results, not assumptions from downloaded files or mock tests.
