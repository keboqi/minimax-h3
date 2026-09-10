# Repository review and refactoring plan — 2026-09-10

Reviewed baseline: `f1648d9` (`Auto-recompile TensorRT VAE on version mismatch and force manual rebuilds`). The working tree was clean before this review. The review below records the original baseline. The approved implementation was completed on 2026-09-11; see the implementation record at the end.

The review covers application orchestration, settings and API boundaries, GPU job ownership, output handling, deployment configuration, and existing regression coverage. It builds on `docs/settings-refactor.md`. It is a source and CPU contract review, not an inference-quality or performance validation.

## Findings to fix first

### R1 — P1: Manual TensorRT compilation bypasses GPU serialization

**Location:** `h3_ui/app_bindings.py:70–74`; compiler entry at `gradio_app.py:2219`.

The compile button registers a regular Gradio event without `concurrency_id="h3-gpu"`. Generation, local prompt enhancement, and model unloading use that shared group. The compiler also does not acquire the application GPU lease. Its `_TRT_VAE_COMPILE_LOCK` only serializes other engine compilations.

Consequently, clicking Compile during H3/LTX/Music generation or local prompt enhancement can start another GPU workload concurrently. This defeats the application's serialization policy and can cause memory exhaustion. The compiler also replaces an engine that another request may be preparing to use.

**Evidence:** Constructing the real UI with backend health mocked produces a function-specific concurrency ID and `limit=default` for `compile_trt_video_vae`; H3 generation, LTX generation, and unloading all produce `h3-gpu` with limit 1. This confirms independent queues without running any GPU code.

**Fix:** Put manual compilation in the shared GPU group and introduce a lease-owning boundary for standalone GPU maintenance. Keep the low-level compiler callable by generation while its lease is already held; reacquiring the existing non-reentrant lock inside that path would deadlock. Review backend memory release separately: the compiler imports Comfy model management into the UI process, while inference runs in another process. Local cache clearing cannot establish that the backend has released its resident models.

**Acceptance:** A UI contract test checks the compiler's group. A coordinator test proves standalone compilation waits for another owned GPU operation and automatic compilation does not acquire the lease twice. On supported hardware, compile after a completed generation with models still resident and verify memory release and subsequent decoding.

### R2 — P2: Unused video decoders can block audio and 500K image generation

**Location:** `gradio_app.py:7564–7569`; output branches at `gradio_app.py:4130–4170`.

Preparation unconditionally honors `use_int8_vae` and `use_trt_vae` before determining the resources required by the output branch. TensorRT is enabled in the UI defaults. Audio output only decodes audio; an Image result using the 500K decoder uses a separate image VAE. Both still download/validate/compile the optional TensorRT video decoder when its remembered checkbox is enabled. An unavailable compiler therefore prevents a request that does not need it.

**Evidence:** Run the actual `generate()` iterator with model lookup and downloads mocked and make `ensure_trt_video_vae_engine()` raise a sentinel error. Both Audio and Image with the 500K decoder invoke it and return that error before reaching output generation. This does not imply that every Image request should skip the video VAE: images using the normal decoder still need their selected decode path, and visual conditioning can need the reference encoder.

**Fix:** Resolve conditioning encoders and output decoders explicitly before provisioning. Derive required model keys, required nodes, graph inputs, and output metadata from that resource decision. Preserve inactive browser preferences without treating them as active execution dependencies.

**Acceptance:** Test Video, Audio, normal Image, and 500K Image with the optional decoder flags on and off. Audio and 500K output must proceed without unused decoder provisioning; normal images and video must retain their required decoder. Include first/last-frame and reference conditioning so the normal visual encoder remains available when needed.

### R3 — P2: Stop cannot interrupt FFmpeg interpolation

**Location:** `gradio_app.py:6404`; cancellation at `h3_app/jobs.py:55`.

Interpolation calls `subprocess.run()` with no timeout or cancellation handling. Stop marks the application job as cancelled and handles Comfy prompt IDs, but it cannot terminate this local child process. Once interpolation is running, an expensive or stalled FFmpeg invocation continues holding the generation worker and GPU lease. The helper can return a successful output after cancellation.

**Evidence:** Under a real `JobCoordinator` lease, replace `subprocess.run()` with a controllable blocking fake and request cancellation after the helper enters it. The cancellation flag is set, but the helper remains blocked, the lease remains held, and no timeout was passed. Releasing the fake lets the helper return an output despite cancellation. No real media process was launched for this reproduction.

**Fix:** Add a subprocess runner that observes cancellation and a deadline, terminates and reaps the child, drains bounded diagnostics, and removes partial output. Use it for interpolation first, then audit staging, split/concat, and other FFmpeg calls. Keep a completed source video available when cancellation happens during optional post-processing.

**Acceptance:** A controlled child-process test verifies cancellation, timeout, exit cleanup, partial-output cleanup, and eventual lease release. A cancelled post-processing operation must not report success or discard an already completed source video.

## Architecture assessment

Existing boundaries are useful: pure settings, positional API adaptation, job ownership, contained media discovery, sidecars, and the FastAPI proxy already have separate modules. Preserve these boundaries and their tests.

The remaining concentration is measurable:

| Area | Current size or coupling | Refactoring consequence |
|---|---|---|
| `gradio_app.py` | 9,402 physical lines | Configuration, providers, model preparation, graph policy, progress, media tools, and UI adapters change together. |
| `generate()` | 995 lines; 81 parameters including progress | One function resolves policy, provisions models, builds graphs, executes, post-processes, records metadata, and renders statuses. |
| `build_h3_view()` | 1,104 lines | Settings sections are difficult to change independently. |
| `bind_app()` | 543 lines | Scheduling rules are repeated manually; R1 is a concrete omission. |
| Local/Modal provisioning | 21 duplicated repository/ref/wheel constants | The copies currently agree; drift is a maintenance risk, not an observed pin mismatch. |
| Model-specific runtime logic | Broad direct use of globals and Gradio updates | Moving functions alone would preserve hidden dependencies and make isolated tests harder. |

Line counts indicate useful seams, not targets for arbitrary file-size limits. In particular, the graph primitive is already small; it does not need another abstraction layer.

## Implementation sequence

Each numbered item is a reviewable change. Keep correctness changes separate from mechanical extraction so graph differences remain attributable.

### 1. Close the three behavior gaps

Deliver R1, R2, and R3 as separate fixes with their focused regressions. Introduce only the resource-selection and process-control primitives needed by those fixes. Preserve current presets, model pins, sampler parameters, output formats, and public API argument ordering.

### 2. Extract configuration and model preparation

- Introduce an immutable runtime configuration for paths, Comfy URL, timeouts, and backend mode, constructed at application startup. Keep the pure settings module free of environment reads.
- Consolidate the 21 duplicated deployment constants into a dependency-free source catalog consumed by local and Modal setup. Keep their different installation and lifecycle operations explicit.
- Move `ModelConfig`, model configuration loading, lazy provisioning, and TensorRT engine lifecycle into a model-service boundary. Reuse `h3_models.py` for catalog and manifest operations.
- Review manifest ownership as part of this extraction: `sync_models()` currently reads and rewrites the whole manifest, and `write_json_atomic()` uses a shared `.partial` name. Choose explicit synchronization before allowing multiple preparation callers; unique temporary names alone would not prevent lost read/modify/write updates.

**Done when:** model preparation can be tested without importing `gradio_app` or constructing Gradio components; local and Modal resolve identical source pins; Modal build-time and runtime mounts include the new modules. Preserve the separation that keeps ordinary runtime changes from invalidating expensive image builds.

### 3. Extract the execution runner

- Extend `h3_app/comfy.py` for the transport operations needed by submission, queue inspection, history, and progress. Inject it into the runner rather than calling application globals.
- Give one execution runner ownership of prompt submission, prompt identity, output token, progress fallback, deadline, websocket lifetime, and cleanup.
- Use one deadline across websocket, polling fallback, and final-history retrieval. Those helpers currently start fresh full-duration budgets independently.
- Keep Gradio request injection and `ContextVar` setup in the UI boundary. Pass explicit execution context inside the runner and media lookup.
- Return named progress and result records. Render status strings and Gradio tuples at the adapter boundary.

**Done when:** H3, LTX, Music, and gallery workflows use the same submission lifecycle; existing session/tab cancellation isolation remains intact; transport fallback and cleanup are independently testable. Direct external Comfy clients remain outside the process-local scheduler, as the existing architecture guide documents.

### 4. Extract workflows one family at a time

Start with Music, then LTX, then H3, then post-processing workflows. Stage local input files before graph construction and pass staged references, effective settings, prepared model names, and available node capabilities into the builders.

H3 can keep explicit functions for model patches, FL2VA/Ref2VA conditioning, sampling/refinement, and decode/save. Do not replace those sequences with a generic plugin framework. Preserve LoRA and attention/cache patch order, stage-specific SLA behavior, BF16 offload and conditioning reuse, voice references, Semantic Bridge routing, and decoder-only TensorRT conditioning behavior.

Retain the existing public function names as temporary compatibility adapters where needed. The stable positional API remains at `h3_app/contracts.py`; internal execution should take named grouped inputs. Avoid retaining a second large settings resolver inside `generate()`.

**Done when:** graph builders import without Gradio, network, or model downloads. Representative graph fixtures match the baseline after normalizing only submission UUIDs and fixture paths. Keep semantic assertions for patch order and model/output branches alongside fixtures.

### 5. Extract media and prompt services

Move staging, probing, thumbnailing, managed gallery operations, and post-processing into focused media modules. Share containment and sidecar handling with `h3_app/media.py` and `h3_app/provenance.py`. Keep the complete gallery independent of the display limit, and retain submission-scoped fallback discovery.

Move Gemini, Lightning, and local prompt-writer adapters out of the application entry point. Keep API credentials transient and preserve the existing separation between voice sample files and prompt-writer inputs, including audio-label validation.

**Done when:** media and provider tests use temporary files or mocked transports without the UI; copied outputs retain provenance; interrupted operations do not leave publishable partial files.

### 6. Simplify UI assembly and consolidate validation

Split the H3 view and event binding by existing visible sections. Introduce a shared GPU-action binding helper so compiler, generation, local enhancement, and unload actions receive consistent scheduling. Give service interfaces meaningful parameter and return types instead of broad `Callable[..., Any]` contracts where practical.

Move legacy self-test groups into discoverable suites as the modules they cover move. Keep the `--selftest` entry point during migration. Add one documented command that runs discovery and the remaining standalone self-tests; keep optional CPU PyTorch, cached-upstream, browser, and GPU checks explicit. Update the architecture guide and contradictory historical default descriptions only after behavior is settled.

**Done when:** the entry point constructs configuration and services, assembles the UI/server, and manages process lifecycle. UI adapters own component ordering; domain code produces named values. Browser settings migration, public endpoints, and positional arity remain compatible.

## Verification baseline and gates

Completed during this review with `.tmp-ui-venv/Scripts/python.exe`:

| Check | Result |
|---|---|
| `-m unittest discover -s tests` | 67 tests run; OK, with 3 CPU PyTorch adapter tests skipped. |
| `gradio_app.py --selftest` | Passed workflow, API, gallery, and proxy compatibility assertions. |
| `h3_requirements.selftest()` | Passed. |
| `h3_models.selftest()` | Passed. |
| `h3_node_patches.selftest()` | Passed. |
| `h3_attention.selftest()` | Passed. |
| `h3_prompt_rewriter.selftest()` | Passed. |
| Review-only mocked reproductions | Confirmed R1 event grouping, R2 unused preparation, and R3 blocked cancellation. |

No live GPU generation, TensorRT build, browser suite, or deployment was run. The upstream contract tests use the existing local source cache; this review did not independently refresh or verify upstream releases.

For subsequent changes, run focused regressions and the applicable existing suites. Run browser settings and voice-reference checks when bindings or request adapters change. Before accepting graph or engine-lifecycle changes for deployment, smoke-test supported hardware with H3 text, first/last-frame plus voice, Ref2VA, native refinement, Video/Image/Audio, LTX, Music, and cancelled post-processing. Use matched seeds for changes that could affect sampling or conditioning; CPU graph checks cannot establish audiovisual equivalence.

## Implementation record — 2026-09-11

All six implementation stages are implemented:

1. Compiler scheduling and lease ownership, output-aware decoder selection, and cancellable media subprocesses have focused regressions.
2. Immutable runtime configuration, model services, a shared 21-pin source catalog, and manifest locking across threads and processes are in place.
3. One execution runner owns the submission identity, deadline, transport fallback and cleanup. Managed output lookup receives explicit context.
4. Pure Music, LTX, H3 and upscale workflows are extracted. Fifteen baseline graphs match with only volatile save prefixes normalized.
5. Prompt providers, staging, media tools and gallery storage are extracted. Completed copies retain provenance; media processing uses unpublished partial files and preserves a completed source on cancellation.
6. The launcher is a compatibility shim over UI composition. H3 generation uses grouped requests and typed effect interfaces with separate preparation, construction and finishing phases. Visible UI sections and event bindings are split, and GPU event registration uses one helper.

Validation uses `python -m tests`, which includes discovery and all remaining standalone self-tests; `--browser` adds settings and voice-reference acceptance. The implementation CPU run completed 91 tests successfully, with three numerical tests skipped because CPU PyTorch is unavailable. Both browser suites passed: settings/preset migration, overrides/reset, mode memory, reload, session isolation, conditional controls, voice upload/isolation/retention and narrow layout. Browser expectations were corrected to the pre-existing Singularity and default-on Semantic Bridge settings.

Syntax compilation, static undefined-name checks, shell syntax validation and git whitespace checks passed. The compatibility --selftest entry point also passed.

No model pins, sampling algorithms, presets, output formats or positional API ordering were changed. Documentation now reflects the existing SLA Fast, Quality FP32 latent-upscaler and default-on refinement settings.

Hardware acceptance remains a deployment gate: real H3/LTX/Music inference, TensorRT compilation after resident-model unloading, matched-seed quality comparisons and GPU cancellation were not run in this CPU/browser environment. Backend `/free` is asynchronous; memory release before local compilation still needs the supported-hardware check described in R1.

