# Architecture

The UI describes the **next run**. Each completed output has an independent **settings used** record.

## User behavior

- Generation presets apply sampling, encoder, offload, attention, start-frame cap and refinement defaults. Singularity also selects its base model; the other presets preserve the selected model. All preserve prompt, media, output format, dimensions, duration and seed.
- Editing an active preset field shows **Modified**. The details list actual differences, and **Restore preset settings** reapplies only the preset-owned fields.
- Normal and Turbo maintain separate session preferences. The first visit to a mode starts from the selected preset.
- Required adjustments are separate from manual differences. Native refinement forces acceleration Off; BF16 requires stage offload; the 500K decoder produces one frame.
- Inactive preferences remain stored when switching output formats or attention implementations. They do not affect execution or the active modification count.
- Readiness and execution details use the shared resolved settings. Prompt and conditioning requirements are checked separately.
- Browser preferences use schema version 6 inside the original v3 BrowserState transport key and secret, allowing existing encrypted preferences to migrate in place. Each field is validated; presets are not reapplied during restoration. Qwen technical settings, independent finishing models and workspace task/engine selection are included.
- Prompts, uploaded paths, credentials, outputs and transient confirmations are excluded from browser preferences.
- Page load restores preferences, resolves settings and restores navigation in one response. Values equal to their defaults are not rewritten, so their change listeners stay idle.
- Each settings edit returns its presentation and its preference save in the same direct (unqueued) response. The session's preference snapshot sends a save only when the allowlisted values changed. Controls outside H3 settings and navigation save through one direct request.
- Existing public generation API names and positional contracts remain available. H3 UI preset context is carried through a separate internal adapter.

## Boundaries

| Module | Owns |
|---|---|
| setup_h3.py, run_h3.sh, modal_h3.py, Colab notebook | Environment provisioning and deployment lifecycle |
| h3_requirements.py | Dependency compatibility policy shared by setup and Modal |
| h3_node_patches.py, h3_attention.py, custom_nodes/H3Acceleration/ | Pinned upstream compatibility patches and GPU runtime optimizations |
| h3_app/config.py | Immutable startup paths, backend mode and transport limits |
| h3_sources.py | Repository, revision and wheel pins shared by local and Modal setup |
| h3_app/settings.py, resources.py | Requested/effective settings, presets and output-specific decoder dependencies |
| h3_app/contracts.py | Stable positional API ordering |
| h3_app/model_types.py, model_service.py | Model selection, lazy provisioning through `ensure_models`, and TensorRT engine lifecycle |
| h3_models.py | Model inventory and manifest transactions locked across threads and processes |
| h3_app/jobs.py | Bounded admission, immutable requests, input leases, durable technical ledgers, exact retries, browser ownership, cancellation and the application GPU lease |
| h3_app/workspace_store.py, workspace_admin.py | SQLite ownership, technical history, explicit projects, asset annotations, backup/restore |
| h3_ui/prompt_controller.py, model_controller.py, media_controller.py | The H3 prompt writer, model downloads, and the Media library with its finishing jobs and H3 input-image tools. Their service records hold only the runtime and late-bound ComfyUI, graph and file effects; `application.py` builds one of each (`PROMPTS`, `MODELS`, `MEDIA`) |
| h3_app/comfy.py, execution.py | HTTP transport and one submission lifecycle, deadline and websocket owner |
| h3_app/capabilities.py | Create tasks, engines and their navigation labels |
| h3_app/workflows/ | Pure H3, LTX, Qwen, Music, YuE2, FLUX 3, MOSS-Image, Fibo 2.0, Drama 3 and finishing graph builders with staged input identities |
| h3_app/generation/ | Named requests and typed services; preparation, graph construction, execution and finishing. One module per engine; image_jobs.py is the shared ComfyUI job for FLUX 3, MOSS-Image and Fibo 2.0 |
| h3_app/staging.py, media_tools.py | Input staging, media inspection, controlled subprocesses and post-processing |
| h3_app/media.py, outputs.py, gallery_store.py | Contained discovery, explicit output context and complete managed gallery inventory |
| h3_app/provenance.py | Atomic sidecars, copied-media provenance and escaped metadata rendering. Listeners in `SNAPSHOT_LISTENERS` hear of each new sidecar; the media catalog registers there, so provenance imports neither the catalog nor the gallery |
| h3_app/prompt_writers/ | One prompt writer module per engine (h3, ltx, qwen, flux3, moss_image, song, drama3) over `common.py`'s uploaded-media handling and Gemini/OpenAI-compatible request |
| h3_app/llm/ | Gemini and OpenAI-compatible clients: transient keys, temporary Gemini file uploads, generation |
| h3_app/workers/ | Isolated uv environments, setup steps with cancellation, and resident JSON-line worker processes |
| prompts/ | Writer system prompts for each engine, read through h3_app/config.py |
| h3_app/song_brief.py, h3_ui/song_controls.py, song_takes.py | Song planning and checks, Song-specific controls and the session take history for Music 3 and YuE2 |
| h3_ui/step_cards.py | The guided step strip shared by Soundstage and Dub & Translate, and the intro and card headings Song uses too |
| h3_app/drama3/ | Soundstage cast, script lint, speaker and sound libraries, voice design, and Drama 3 on SGLang-Omni in serving/ (environment, planner, takes, mixing, checkpointed sleep). serving/speech.py is the line-speaking interface other features use |
| h3_app/audio_tools.py, audio_tools_worker.py, voice_design.py, stable_audio.py | The isolated resident worker for Qwen VoiceDesign and Stable Audio, and its clients |
| h3_app/server.py | Configured FastAPI routes and HTTP/WebSocket proxy |
| h3_ui/application.py | Startup and runtime configuration, the ComfyUI and file effects controllers and engines receive, the full H3 `generate`, and the controller and handler singletons. Controllers never import it; tests patch its effects and find a name's owner with `tests/app_targets.py` |
| h3_ui/snapshot_preload.py | Modal's `h3-video` and `h3-soundstage` endpoints: loads the default H3 Video or Soundstage models before the workspace opens, then waits for the snapshot restore to start the UI |
| h3_ui/bootstrap.py, contracts.py | UI composition; contracts.py holds the catalog and service records passed to event registration |
| h3_ui/drama3/ | Soundstage view, bindings, slot controls and saved-speaker library. handlers.py builds `Drama3Handlers` from explicit `Drama3Services`; its generation, writer, voices, speakers and sounds methods live in one module each |
| h3_app/dubbing/ | Dub & Translate sessions, line helpers, Gemini transcription, the isolated tools worker (separation, OmniVAD + Qwen3-ASR, Hy-MT) and the Drama 3 dub/mix pipeline |
| h3_app/video_download.py | yt-dlp downloads into Media and the private per-site cookie store |
| h3_ui/dubbing/ | Dub & Translate tab and Media's Download from a URL panel; handlers.py builds `DubbingHandlers` from explicit `DubbingServices` |
| h3_ui/engines/ | Create engines that own their view, bindings and handlers (FLUX 3, MOSS-Image, Fibo 2.0). Each registers an `EngineFeature` in `FEATURES` and receives `EngineDeps` explicitly; controls.py holds the shared Generate/Interrupt/progress, prompt-writer, resolution-preset and saved-settings helpers |
| h3_ui/h3/ | The H3 composer: view.py, services.py and sections/ (one dataclass per visible section); layout.py computes resolution, result-format and frame control updates; handlers.py holds `H3Handlers` (batched generation, the default-settings API, settings summaries and result lookup) |
| h3_ui/events/ | H3 and workspace-wide event registration (generation, prompt, resolution, input upscale, gallery, system, layout) |
| h3_ui/job_bindings.py, job_admission.py, jobs_view.py | Gradio request injection, acceptance before queueing, shared GPU binding, owner-scoped job presentation, and the Projects tab of saved requests |
| h3_app/reference_bindings.py, h3_ui/prompt_preview.py, prompt_review.py | Stable reference translation and guarded accept/keep/undo prompt previews |
| h3_ui/media_actions.py | Selection- and file-specific deletion confirmation |
| h3_ui/settings_controller.py, settings_presentation.py | Serialized settings actions and resolved-plan rendering |
| h3_ui/persistence.py | Explicit browser preference allowlist, migration and single-response restoration |
| gradio_app.py | Workspace launcher |
| tests/__main__.py, test_service_contracts.py | Consolidated validation and standalone provisioning checks |
| tests/test_application_adapters.py, test_workflow_contracts.py, test_server_proxy.py | Focused application, workflow and proxy behavior |
| tests/workflow_cases.py, test_workflow_fixtures.py | Shared fixed inputs and 15 unchanged baseline graph fixtures, independent of Gradio |

The large H3 workflow and acceleration modules encode runtime-sensitive ComfyUI node ordering and optimization behavior. Change their numerical paths only with GPU evidence; the graph fixtures are the CPU regression boundary.

## Ownership and cancellation

Application GPU actions register through bind_gpu_action, including manual TensorRT compilation. Standalone compilation, local prompt enhancement and unloading acquire the same application lease; maintenance inside a generation reuses its existing ownership. Managed generation also acquires a process-local lease and records its browser session, tab family and Comfy prompt ID. Settings updates use their own short queue.

Stop first requests cancellation for the current session and tab, then cancels its Gradio events. Pending Comfy prompts are deleted by ID. A backend-wide interrupt is issued only after checking that the running prompt belongs to that session/tab. Submission and cancellation share a registry lock.

ComfyUI's interrupt endpoint remains global. External clients using ComfyUI directly are outside the application scheduler; the running-ID check narrows that boundary but cannot make a remote global endpoint atomic with externally submitted jobs. Multiple application processes also require a shared coordinator before they can share one GPU lease.

ExecutionRunner owns submission identity, save-prefix scope, websocket cleanup, progress fallback and final history. Its HTTP calls and websocket receives use the same monotonic deadline; a disconnected socket does not restart the time budget. H3, LTX, Music and gallery jobs all use this lifecycle. The string-compatible prompt handle retains the submission for legacy callers.

FFmpeg subprocesses check cancellation and a deadline, drain bounded diagnostics, terminate and reap on interruption, and remove partial output. Interpolation and concatenation publish their final media name only after success. Cancellation during optional finishing leaves the completed source available.

Every submission receives a unique save prefix. Fallback output discovery requires that submission token and a contained path; an unowned timestamp-only scan returns no output.

## Result provenance

H3, LTX and Music write versioned JSON sidecars next to generated media. Records contain effective execution facts and actual seeds; H3 also records preset differences and automatic adjustments. They omit prompts, media inputs and credentials. Selected image-frame copies, gallery imports, interpolation and concatenation retain their records, and gallery deletion removes the corresponding sidecar. Gallery inventory remains complete for deletion and lookup even when the display limit shows only recent results.

H3 metadata is attached before Gradio copies media into its cache. Gallery metadata uses the managed source path. Cached LTX/Music output metadata is recovered only from a unique exact managed basename containing the submission UUID. Older or ambiguous outputs report settings unavailable.

## Compatibility and deployment

gradio_app is a launcher only. Prompt, model and media controllers receive explicit service records; the workspace is the only UI. Application exports are runtime wiring and public API adapters. Tests import pure policies, catalogs and workflow helpers from their owning modules. New domain code imports h3_app modules and receives runtime configuration and named services explicitly. H3Request groups conditioning, sampling, output and finishing inputs. Resolved settings remain the policy authority; capability and model checks take place during preparation.

Graph builders accept staged paths and explicit output naming tokens. Importing them or generation, media and model services does not construct Gradio components, contact a backend or download models. Graph fixture normalization changes only volatile save-prefix timestamp/UUID suffixes.

Local and Modal setup consume h3_sources.py. Modal copies that small build dependency into its image and mounts h3_app and h3_ui at runtime, so ordinary application changes do not invalidate expensive image builds.

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

The consolidated command runs discovery, standalone provisioning checks and baseline workflow contracts, with offline Hugging Face mode. The gradio_app.py --selftest entry point delegates to that same CPU suite. CPU PyTorch numerical tests run only when PyTorch is installed; pinned upstream contract tests run only when their .cache/upstream-upgrade sources exist. Neither gate fetches dependencies. Cached upstream files must match the current pins in h3_requirements.py and h3_sources.py; refresh them after upgrades or select a matching directory with H3_UPSTREAM_SOURCE_DIR. The --browser option adds queue transport, settings, voice-reference, first-frame resolution, image-library, workspace, media-performance, media-deletion and UI request-budget acceptance. `python -m tests.browser_ui_requests --report` prints the round trips each common action costs.

On Linux use .venv/bin/python. Browser checks use installed Chrome on Windows, or Playwright Chromium elsewhere; H3_BROWSER_EXECUTABLE can select a Chromium executable. Fixtures mock backend health and never load models. Workspace job checks use a synthetic generation callback and test-only queue controls; they do not measure inference.

Browser checks cover preset application, overrides/reset, mode memory, reload, session isolation, conditional controls and horizontal overflow at a narrow viewport. Screenshots are written to .cache/ui-review.

A real supported GPU and provisioned ComfyUI installation are still needed for inference smoke tests and performance/quality comparisons.

## Keeping the structure small

An H3 setting that reaches the graph is one `H3GraphOptions` field in
`h3_app/workflows/h3.py` and one line in `construct_graph.graph_options`; both
builders accept it by name and `tests/test_h3_graph_options.py` checks the
wiring. Its control is one field in its `h3_ui/h3/sections/` dataclass; `H3View`
collects section fields by name.

A new Create engine is one module in `h3_ui/engines/` plus its entry in
`FEATURES`, its `h3_app/capabilities.py` navigation entry, its layout tab and
its saved-settings allowlist; `tests/test_engine_features.py` checks they agree.
Its handlers take `EngineDeps` rather than module-level state. Features import
each other's public names only; `tests/test_import_boundaries.py` rejects a
private name taken from another feature.

Keep one response per user action. A settings edit that changes dimensions
or another policy input uses `SettingsController.bind_refresh` instead of a
`.then(refresh)` chain, and presentation the settings transition already owns
gets no second listener. Labelling that needs only static catalogs runs as a
browser-side `js` callback.

Add behavior checks beside the owning feature, not to a launcher self-test, and
do not export a domain helper through `application.py` only to reach it from
tests. Event registration imports feature binding functions directly; pass only
runtime callbacks through `BootstrapServices`. Preserve behavioral regressions
even when their names mention an older feature: preference migrations, retired
model selections, legacy media and positional API calls still have consumers.
Git history holds earlier audits and plans; keep `docs/` describing the current
repository.
