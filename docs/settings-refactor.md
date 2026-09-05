# Settings and execution architecture

The UI describes the **next run**. Each completed output has an independent **settings used** record.

## User behavior

- Generation presets apply sampling, encoder, offload, attention, start-frame cap and refinement defaults. They preserve the base model, prompt, media, output format, dimensions, duration and seed.
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
| h3_app/settings.py | Typed requests, preset definitions, effective settings, adjustments, mode transitions and validation |
| h3_app/contracts.py | Stable positional API adapter; named values and grouped reference inputs internally |
| h3_app/jobs.py | Session/tab job ownership, submission identity, cancellation and the application GPU lease |
| h3_app/comfy.py | ComfyUI HTTP transport |
| h3_app/server.py | Configured FastAPI routes and HTTP/WebSocket proxy |
| h3_app/media.py | Contained history lookup and submission-scoped output fallback |
| h3_app/provenance.py | Atomic versioned output sidecars and escaped metadata rendering |
| h3_app/graph.py | Dependency-free graph construction primitive |
| h3_ui/settings_controller.py | Explicit serialized user actions and conditional component updates |
| h3_ui/settings_presentation.py | Rendering the resolved plan |
| h3_ui/contracts.py, h3_ui/h3_view.py | Typed components and named dependency contracts |
| h3_ui/persistence.py | Explicit browser preference allowlist and migration |
| gradio_app.py | Model-specific orchestration and graph builders, configuration and application assembly |
| tests/legacy_selftest.py | Existing workflow and compatibility regressions, loaded only on request |

The H3/LTX/Music graph algorithms and model patch ordering remain intact. This refactor does not replace the inference stack or introduce a plugin framework.

## Ownership and cancellation

Application GPU work shares the h3-gpu Gradio concurrency group. Managed generation also acquires a process-local lease and records its browser session, tab family and Comfy prompt ID. Settings updates use their own short queue.

Stop first requests cancellation for the current session and tab, then cancels its Gradio events. Pending Comfy prompts are deleted by ID. A backend-wide interrupt is issued only after checking that the running prompt belongs to that session/tab. Submission and cancellation share a registry lock.

ComfyUI's interrupt endpoint remains global. External clients using ComfyUI directly are outside the application scheduler; the running-ID check narrows that boundary but cannot make a remote global endpoint atomic with externally submitted jobs. Multiple application processes also require a shared coordinator before they can share one GPU lease.

Every submission receives a unique save prefix. Fallback output discovery requires that submission token and a contained path; an unowned timestamp-only scan returns no output.

## Result provenance

H3, LTX and Music write versioned JSON sidecars next to generated media. Records contain effective execution facts and actual seeds; H3 also records preset differences and automatic adjustments. They omit prompts, media inputs and credentials. Selected image-frame copies retain their records, and gallery deletion removes the corresponding sidecar.

H3 metadata is attached before Gradio copies media into its cache. Gallery metadata uses the managed source path. Cached LTX/Music output metadata is recovered only from a unique exact managed basename containing the submission UUID. Older or ambiguous outputs report settings unavailable.

## Validation

Pure policy and ownership tests require only Python:

```bash
python -m unittest tests.test_settings
```

UI and integration tests use the lightweight test dependencies:

```bash
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-test.txt
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe gradio_app.py --selftest
.venv/Scripts/python.exe tests/browser_settings.py
```

On Linux use .venv/bin/python. Browser checks use installed Chrome on Windows, or Playwright Chromium elsewhere; H3_BROWSER_EXECUTABLE can select a Chromium executable. The browser fixture mocks only backend health and never loads models or generates media.

Browser checks cover preset application, overrides/reset, mode memory, reload, session isolation, conditional controls and horizontal overflow at a narrow viewport. Screenshots are written to .cache/ui-review.

A real supported GPU and provisioned ComfyUI installation are still needed for inference smoke tests and performance/quality comparisons.
