# Media post-processing: NVIDIA VSR and experimental DLSS 5

Status: integration implemented, 2026-10-13. CPU contracts and Media browser checks pass. Actual NVIDIA runtime execution and Linux RTX PRO 6000 compatibility remain unvalidated. See [installation and current limitations](nvidia-media-postprocess.md).

## Outcome and scope

Add two methods to **Media → Enhance selected media → Method**:

- **NVIDIA RTX Video Super Resolution**
- **DLSS 5 Neural Rendering (experimental)**

The initial integration processes selected videos, including imported videos, and now also selected still images. Each method creates a new managed output, retains source lineage, and uses the existing gallery GPU queue, progress, cancellation, and result preview. A user can process the resulting copy again with another method. Automatic multi-method chaining, generation-stage finishing, frame interpolation, and HDR processing are outside this release. Image support uses one worker frame, PNG output, sRGB normalization, and preserved transparency.

VSR uses NVIDIA's native Python VFX runtime. DLSS 5 uses a persistent Windows processing executable, directly on Windows or through a dedicated Wine environment on Linux. Availability belongs to the GPU processing host, including a Modal container, rather than to the browser or desktop showing the UI.

## Existing integration points

| File | Current role | Planned change |
| --- | --- | --- |
| `h3_app/catalog.py` | Gallery and generation enhancement choices | Add the two gallery choices and a separate NVIDIA-method set; keep generation choices unchanged |
| `h3_ui/media_view.py` | Enhancement controls | Add method-specific settings and runtime availability text |
| `h3_ui/media_bindings.py` | Conditional controls and GPU job submission | Show relevant controls and capture their values at submission |
| `h3_ui/media_controller.py` | Gallery video dispatch and result presentation | Dispatch NVIDIA requests before the existing generic FFmpeg branch |
| `h3_ui/application.py`, `h3_ui/bootstrap.py` | Service injection and component assembly | Wire the new service and controls through the existing boundaries |
| `h3_ui/job_admission.py`, `h3_ui/job_bindings.py` | Immutable accepted requests and GPU ownership | Preserve request capture, source references, and gallery job behavior |
| `h3_app/processes.py` | Cancellation, timeouts, bounded diagnostics | Add or share narrowly scoped support for streaming worker pipelines and child cleanup |
| `h3_app/provenance.py`, `h3_app/media_catalog.py` | Settings sidecars, lineage, catalog registration | Publish method settings and actual execution details through existing APIs |
| `h3_requirements.py`, `setup_h3.py`, `modal_h3.py` | Pinned dependencies and deployment | Provision optional runtimes consistently, subject to host capability |

## User controls

| Method | Default controls | Advanced controls |
| --- | --- | --- |
| VSR | Target resolution; quality (default High); strength (default 1.0) | Source profile: compressed/high-bitrate; optional denoise or deblur preprocessing, with separate strength |
| DLSS 5 | Original size or target resolution; model-supported style; intensity | Structure/tone/skin controls only where supported by the pinned runtime; motion estimation enabled by default |

Use the existing target-resolution fit policy, with a new original-size option limited to these methods. For VSR, original size must map to a supported denoise/deblur operation; reject an incompatible upscale-mode request. Check scale and alignment against the actual pinned SDK instead of silently resizing beyond its range. DLSS 5 defaults to original size to let users evaluate neural appearance changes independently of upscaling. An explicit larger target enables the helper's supported DLSS SR plus NR path and records the stage order.

Hide LTX model/prompt/split controls, SeedVR2 model controls, and random seed for both methods. Keep **Unload resident models first** available and honor it for these external GPU workers. Use a separate NVIDIA settings group so adding these methods to a shared AI set does not expose unrelated controls elsewhere.

Show a concise availability explanation next to the selected method. Examples: missing VFX runtime, incompatible driver, missing DLSS model, or helper initialization failed. Keep unavailable methods discoverable but disable Enhance for the selected unavailable method. Revalidate on submission; UI state is not the authority. Switching libraries must restore the correct controls, expose both methods for Image and Video, and keep enhancement unavailable for Audio.

## Backend design

Create a focused `h3_app/nvidia_postprocess.py` service with validated settings, capability reports, and dispatch. Keep processing implementations in separate `h3_app/vsr_worker.py` and `h3_app/dlssnr.py` modules. Avoid refactoring existing SeedVR2/LTX/SwiftVR implementations for this feature.

Define a versioned request containing source, method, sizing, settings, and output policy. Validate it before GPU execution, then validate source metadata and runtime capabilities on the execution host. Append optional settings to existing gallery entry points with legacy defaults; update wrappers and job input names together. Existing saved gallery jobs must still replay with default settings.

Capability inspection is lazy and inexpensive: imports/version checks, paths, and cached driver information. An actual model initialization/evaluation probe runs as an explicit queued runtime check or inside the accepted enhancement job, never while browsing thumbnails or rendering a tab. Cache probe results by GPU, driver, runtime/model fingerprint, and runner configuration; provide a refresh action.

### VSR worker

Use `nvvfx.VideoSuperRes` directly rather than adding a dependency on the third-party ComfyUI node. Run it in a separate Python worker using the configured GPU environment so SDK failures and GPU allocations are isolated from the UI process.

1. Probe source dimensions, rational frame rate, duration, color metadata, and audio streams.
2. Decode progressively with FFmpeg; transfer a bounded number of frames to CUDA.
3. Optionally run same-resolution denoise/deblur, then VSR at the validated target.
4. Preserve DLPack output ownership and synchronize the SDK work and copies correctly before buffer reuse.
5. Stream output to one encoder. Keep RAM use independent of video duration.
6. Mux source audio, publish the completed file, and release worker resources.

Pin a tested `nvidia-vfx` version and its compatible Python/PyTorch environment. The project's original node is a behavioral reference for modes and frame handling, not proof of compatibility on the deployed GPU.

### DLSS 5 worker: feasibility gate first

Start from a pinned `video2dlssnr` helper because it already offers offline video processing, raw-frame pipes, ComfyUI integration, motion estimation, and scene-cut handling. Its published core targets Windows/D3D12; Linux execution is an experiment until demonstrated on the actual host.

Before implementing the full UI path, establish these gates:

1. The executable starts under the chosen Wine runner, finds its runtime libraries, and selects the intended RTX PRO 6000.
2. NGX feature 18 initializes and evaluates on representative frames; initialization alone is insufficient.
3. Enabled/disabled output and diagnostics confirm a neural pass ran rather than a bypass or SR-only path.
4. A short clip with motion and a scene cut completes with bounded memory and correct frame count.
5. Cancellation stops the worker and its descendants, releases GPU resources, and permits a clean next run.

Use an isolated, pre-initialized Wine prefix and a known compatible Wine/VKD3D/DXVK-NVAPI combination as required by the chosen helper. Account for Wine paths, working directory, binary stdin/stdout, and stderr diagnostics. Do not assume the helper's Windows driver minimum maps directly to Linux driver numbering. Run one helper for the clip to retain temporal history and amortize initialization.

Configure the helper, runner, prefix, and model directory on the server; do not expose executable paths as per-user enhancement inputs. Record the model hash and helper revision. Keep NVIDIA DLLs separately supplied from an authorized installation, outside source control and public deployment images.

If the D3D12 helper fails the Linux gates, evaluate adapting the Windows Vulkan NGX helper from `DLSS5VKLayer` for offline frame input. Its existing presentation layer is not an offline video processor. This is a separate engineering decision, not an automatic runtime fallback. Keep DLSS 5 unavailable with the specific failed requirement until one route is validated.

## Shared output and job guarantees

- Start with SDR inputs and outputs. Extend metadata probing to detect transfer functions, primaries, and bit depth. Reject PQ/HLG/HDR inputs with a clear message until an end-to-end HDR path is validated; never silently treat them as SDR.
- Preserve source dimensions for original-size operations, aspect ratio for resizing, frame count, frame order, timestamps, and audio synchronization. Preserve rational rates such as 24000/1001. Detect VFR and either preserve its timestamps or explicitly reject it in the first release.
- Use HEVC for the initial output policy and validate dimensions against the encoder. Test actual encoder initialization. If NVENC cannot initialize, use an explicit supported CPU encoder fallback and record it. Do not restart enhancement after a partially encoded failure without user-visible failure/retry behavior.
- Copy compatible audio streams; use a documented audio transcode when the output container requires it. Handle videos without audio and reject inputs without a usable video stream.
- Drain worker/encoder diagnostics continuously and keep them bounded. Apply backpressure and handle truncated frames, broken pipes, timeouts, disk exhaustion, and worker crashes.
- Track the complete FFmpeg/worker/Wine process tree. Cancel only the job-owned processes or isolated prefix; do not stop unrelated Wine applications. Clean incomplete outputs on failure, cancellation, and worker shutdown.
- Write to a unique per-job processing directory and final partial file. Validate the completed media, check cancellation, then atomically publish. Register only successful outputs in the library.
- Write a settings snapshot with source asset ID, requested and actual dimensions, method parameters, GPU, driver, runtime/helper/model versions, stage order, motion backend, actual encoder, and any encoding fallback. Reuse existing catalog, tags/favorites, preview, and comparison behavior.
- Report preparation, processing frames/total, encoding/muxing, and completion. Record failures in the job system as well as returning a readable gallery message; do not let a caught worker exception look like a successful job.

## Installation and deployment

VSR is an optional pinned runtime installed by a dedicated setup option or documented provisioning path. Missing packages must not prevent normal app startup. Reuse the project's GPU environment rather than replacing PyTorch globally.

DLSS 5 is a separately enabled experimental runtime. Provision its open-source helper and runner dependencies ahead of processing; do not install Wine or download arbitrary binaries during a gallery job. Supply proprietary runtime files through an operator-managed mounted directory.

Apply the same capability checks on local Linux and Modal. VSR should be the first container integration. DLSS 5 needs an additional headless container smoke test with the actual GPU and host driver before being enabled there. The application cannot upgrade the provider's host driver. An unsupported Modal host should report DLSS 5 unavailable while keeping VSR and existing methods usable.

## Implementation sequence

1. **Contracts and checks:** add method identifiers, settings validation, runtime configuration, metadata extensions, and capability reporting. Specify defaults for old jobs.
2. **VSR end to end:** implement the bounded-memory worker, output publication, provenance, and cancellation; wire it into the existing Media flow.
3. **DLSS Linux spike:** run the gates above independently of the UI and record the exact hardware/runtime combination and observed limitations. Select one helper route.
4. **DLSS integration:** add its worker adapter, temporal processing, method-specific controls, availability behavior, and job snapshots. Keep the experimental label.
5. **Provisioning and verification:** finish local/Modal setup, regression coverage, operator instructions, and representative media checks. Publish support claims only for configurations actually tested.

## Validation and completion criteria

CPU tests use fake workers and small FFmpeg fixtures. Cover missing dependencies, capability changes, settings validation, job capture/replay, queue ownership, stop/timeout cleanup including descendants, worker crashes, partial frame reads, encoder fallback, audio/no-audio, timing/frame count, HDR/VFR rejection, lineage, and no catalog entries for incomplete files. UI checks cover method switching, library switching, relevant controls, unavailable states, and selected-result refresh. Run existing gallery, job, provenance, encoder, settings, and UI contract tests affected by the integration.

GPU validation is an explicit deployment acceptance task, separate from ordinary unit tests: one VSR frame, one DLSS feature-18 evaluation, a short motion/scene-cut clip, a representative long clip, cancellation followed by another run, and quality comparison at original size and an upscale target. Measure warm-up separately from steady-state throughput and record peak VRAM/RAM. No specific frame rate or quality improvement is promised before these measurements. Do not automatically run inference or deploy as part of this planning task.

Completion means both methods can produce verified new videos on the validated Linux RTX PRO 6000 configuration, preserve the stated media properties, appear correctly in Media, and fail clearly on unsupported hosts. An unavailable DLSS placeholder alone does not complete the DLSS integration.

## References

- NVIDIA VFX Python package and runtime requirements: https://pypi.org/project/nvidia-vfx/
- NVIDIA Linux SDK requirements: https://docs.nvidia.com/maxine/vfx/latest/LinuxVFXSDK/GetStartedonLinux.html
- NVIDIA VSR modes and driver requirements: https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html
- Original VSR node reference: https://github.com/tritant/ComfyUI_NVIDIA_RTX_VSR_Long_Video
- Offline DLSS helper and ComfyUI wrappers: https://github.com/DaniilSokolyuk/video2dlssnr
- Linux Vulkan/Wine NGX helper reference: https://github.com/bmitch87/DLSS5VKLayer
- Community Linux port measurements and integration limitations: https://github.com/paulkoan/NeuralScreen-Linux/blob/main/docs/M0-FINDINGS.md

Community documentation establishes candidate approaches, not independent verification on this application's host. Pin revisions when implementation begins.
