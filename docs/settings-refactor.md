# Settings and execution architecture

The UI describes the **next run**. Each completed output has an independent **settings used** record.

## User behavior

- Generation presets apply sampling, encoder, offload, attention, start-frame cap and refinement defaults. Singularity also selects its base model; the other presets preserve the selected model. All preserve prompt, media, output format, dimensions, duration and seed.
- Editing an active preset field shows **Modified**. The details list actual differences, and **Restore preset settings** reapplies only the preset-owned fields.
- Normal and Turbo maintain separate session preferences. The first visit to a mode starts from the selected preset.
- Required adjustments are separate from manual differences. Native refinement forces acceleration Off; BF16 requires stage offload; the 500K decoder produces one frame.
- Inactive preferences remain stored when switching output formats or attention implementations. They do not affect execution or the active modification count.
- Readiness and execution details use the shared resolved settings. Prompt and conditioning requirements are checked separately.
- Browser preferences use schema version 4 inside the original v3 BrowserState transport key and secret, allowing existing encrypted preferences to migrate in place. Each field is validated; presets are not reapplied during restoration.
- Prompts, uploaded paths, credentials, outputs and transient confirmations are excluded from browser preferences.
- Existing public generation API names and positional contracts remain available. H3 UI preset context is carried through a separate internal adapter.

## Boundaries

| Module | Owns |
|---|---|
| h3_app/config.py | Immutable startup paths, backend mode and transport limits |
| h3_sources.py | The 21 repository, revision and wheel pins shared by local and Modal setup |
| h3_app/settings.py, resources.py | Requested/effective settings, presets and output-specific decoder dependencies |
| h3_app/contracts.py | Stable positional API ordering |
| h3_app/model_types.py, model_service.py | Model selection, lazy provisioning and TensorRT engine lifecycle |
| h3_models.py | Model inventory and manifest transactions locked across threads and processes |
| h3_app/jobs.py | Session/tab ownership, cancellation and the application GPU lease |
| h3_app/comfy.py, execution.py | HTTP transport and one submission lifecycle, deadline and websocket owner |
| h3_app/workflows/ | Pure H3, LTX, Music and upscale graph builders with staged input identities |
| h3_app/generation/ | Named requests and typed services; preparation, graph construction, execution and finishing |
| h3_app/staging.py, media_tools.py, swiftvr.py | Input staging, media inspection, controlled subprocesses and post-processing |
| h3_app/media.py, outputs.py, gallery_store.py | Contained discovery, explicit output context and complete managed gallery inventory |
| h3_app/provenance.py | Atomic sidecars, copied-media provenance and escaped metadata rendering |
| h3_app/prompt_service.py | Local, Gemini and Lightning writer adapters with transient credentials |
| h3_app/server.py | Configured FastAPI routes and HTTP/WebSocket proxy |
| h3_ui/application.py | Application composition and temporary public/UI compatibility adapters |
| h3_ui/sections/, events/ | Existing visible sections and event registration, preserving component order |
| h3_ui/job_bindings.py | Gradio request injection and shared GPU-action binding |
| h3_ui/settings_controller.py, settings_presentation.py | Serialized settings actions and resolved-plan rendering |
| h3_ui/persistence.py | Explicit browser preference allowlist and migration |
| gradio_app.py | Launcher and temporary import compatibility shim |
| tests/__main__.py, test_legacy_services.py, test_workflow_fixtures.py | Consolidated validation, legacy suites and 15 baseline graph fixtures |

The H3/LTX/Music graph algorithms and model patch ordering remain intact. This refactor does not replace the inference stack or introduce a plugin framework.

## Ownership and cancellation

Application GPU actions register through bind_gpu_action, including manual TensorRT compilation. Standalone compilation, local prompt enhancement and unloading acquire the same application lease; maintenance inside a generation reuses its existing ownership. Managed generation also acquires a process-local lease and records its browser session, tab family and Comfy prompt ID. Settings updates use their own short queue.

Stop first requests cancellation for the current session and tab, then cancels its Gradio events. Pending Comfy prompts are deleted by ID. A backend-wide interrupt is issued only after checking that the running prompt belongs to that session/tab. Submission and cancellation share a registry lock.

ComfyUI's interrupt endpoint remains global. External clients using ComfyUI directly are outside the application scheduler; the running-ID check narrows that boundary but cannot make a remote global endpoint atomic with externally submitted jobs. Multiple application processes also require a shared coordinator before they can share one GPU lease.

ExecutionRunner owns submission identity, save-prefix scope, websocket cleanup, progress fallback and final history. Its HTTP calls and websocket receives use the same monotonic deadline; a disconnected socket does not restart the time budget. H3, LTX, Music and gallery jobs all use this lifecycle. The string-compatible prompt handle retains the submission for legacy callers.

FFmpeg subprocesses check cancellation and a deadline, drain bounded diagnostics, terminate and reap on interruption, and remove partial output. Interpolation and concatenation publish their final media name only after success. SwiftVR's intermediate video stays in a processing directory excluded from gallery discovery. Cancellation during optional finishing leaves the completed source available.

Every submission receives a unique save prefix. Fallback output discovery requires that submission token and a contained path; an unowned timestamp-only scan returns no output.

## Result provenance

H3, LTX and Music write versioned JSON sidecars next to generated media. Records contain effective execution facts and actual seeds; H3 also records preset differences and automatic adjustments. They omit prompts, media inputs and credentials. Selected image-frame copies, gallery imports, interpolation and concatenation retain their records, and gallery deletion removes the corresponding sidecar. Gallery inventory remains complete for deletion and lookup even when the display limit shows only recent results.

H3 metadata is attached before Gradio copies media into its cache. Gallery metadata uses the managed source path. Cached LTX/Music output metadata is recovered only from a unique exact managed basename containing the submission UUID. Older or ambiguous outputs report settings unavailable.

## Compatibility and deployment

Importing gradio_app resolves to the composition module during migration, preserving existing names, signatures and callback patching. New domain code imports h3_app modules and receives runtime configuration and named services explicitly. H3Request groups conditioning, sampling, output and finishing inputs. Resolved settings remain the policy authority; capability and model checks take place during preparation.

Graph builders accept staged paths and explicit output naming tokens. Importing them or generation, media and model services does not construct Gradio components, contact a backend or download models. Graph fixture normalization changes only volatile save-prefix timestamp/UUID suffixes.

Local and Modal setup consume h3_sources.py. Modal copies that small build dependency into its image and mounts h3_app and h3_ui at runtime, so ordinary application changes do not invalidate expensive image builds. The refactor preserves the pins and inference algorithms.

## Validation

Pure policy and ownership tests require only Python:

```bash
python -m unittest tests.test_settings
```

UI and integration tests use the lightweight test dependencies:

```bash
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-test.txt
.venv/Scripts/python.exe -m tests
.venv/Scripts/python.exe -m tests --browser
```

The consolidated command runs discovery, all remaining standalone service self-tests and the legacy workflow contracts, with offline Hugging Face mode. The gradio_app.py --selftest entry point remains available. CPU PyTorch numerical tests run only when PyTorch is installed; pinned upstream contract tests run only when their .cache/upstream-upgrade sources exist. Neither gate fetches dependencies. The --browser option adds settings and voice-reference acceptance.

On Linux use .venv/bin/python. Browser checks use installed Chrome on Windows, or Playwright Chromium elsewhere; H3_BROWSER_EXECUTABLE can select a Chromium executable. The browser fixture mocks only backend health and never loads models or generates media.

Browser checks cover preset application, overrides/reset, mode memory, reload, session isolation, conditional controls and horizontal overflow at a narrow viewport. Screenshots are written to .cache/ui-review.

A real supported GPU and provisioned ComfyUI installation are still needed for inference smoke tests and performance/quality comparisons.
