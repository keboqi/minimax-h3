# Repository structure review — 2026-10-07

The repository already has useful deployment, domain and presentation boundaries.
The accumulated maintenance burden was primarily obsolete adapters and validation
that still depended on the former application monolith. This cleanup removes
those layers while retaining the published generation contracts and reviewed
workflow graphs.

## Ownership

| Area | Responsibility |
|---|---|
| `setup_h3.py`, `run_h3.sh`, `modal_h3.py`, Colab notebook | Environment provisioning and deployment lifecycle |
| `h3_sources.py`, `h3_requirements.py`, `h3_models.py` | Shared source pins, dependency policy, model inventory and downloads |
| `h3_node_patches.py`, `h3_attention.py`, `custom_nodes/H3Acceleration/` | Pinned upstream compatibility and GPU runtime optimizations |
| `h3_app/settings.py`, `policy.py`, `resources.py`, `model_types.py` | Requested/effective settings, resource selection and model contracts |
| `h3_app/workflows/` | Pure H3, LTX, Qwen, Music, YuE2 and finishing graph construction |
| `h3_app/generation/`, `comfy.py`, `execution.py`, `jobs.py` | Generation orchestration, backend transport, ownership and cancellation |
| `h3_app/media*`, `gallery_store.py`, `outputs.py`, `provenance.py` | Managed media, discovery, previews and technical output records |
| `h3_app/workspace*`, `h3_ui/asset_index.py` | Durable workspace data, annotations and indexed library queries |
| `h3_ui/application.py`, `bootstrap.py`, controllers | Runtime wiring, published callbacks and UI composition |
| `h3_ui/*_view.py`, `*_bindings.py`, `sections/`, `events/` | Feature views, reusable controls and event registration |
| `tests/test_*.py`, `workflow_cases.py`, `fixtures/` | Discoverable CPU contracts and fixed graph/API baselines |
| `tests/browser_*.py` and fixture servers | Optional browser acceptance with synthetic backend behavior |

Deployment paths already share source pins and provisioning policy. Graph builders
already accept explicit inputs and services; consolidating these into the UI or
into one deployment script would erase useful boundaries. The large H3 workflow
and acceleration modules encode distinct inference and optimization behavior.
Their numerical paths require GPU evidence before substantive changes.

## Changes

- Removed 38 unused application functions, including retired preset/mode and
  resolution callbacks, private prompt/model forwarding methods and unused
  staging/output convenience wrappers. Current settings actions continue through
  the settings controller and resolved-settings policy.
- Removed 21 unused methods from the prompt, model and media controllers. Service tests
  call the implementing domain service directly with explicit runtime context.
- Removed the application module's test-only helper re-exports and oversized
  `__all__` list. Tests import policies, graph helpers, catalogs and transport
  utilities from their owners. Unused injected service fields were also removed.
- Replaced the 2,460-line `service_selftest.py` with named application, workflow
  and proxy tests. Repeated graph assertions now rely on the existing complete
  snapshots. Constant-only checks for retired UI adapters were removed; current
  settings, preference migration, model retirement and public API checks remain.
- Recorded fixed inputs for all 15 graph snapshots. Snapshot tests invoke pure
  builders without importing Gradio, running unrelated service checks, patching
  application globals or tracking event loops. The expected graph JSON is
  unchanged. Model-stack checks share fixed setup and override the behavior under
  test.
- Converted retained assertions to unittest assertions, which remain active under
  optimized Python. Removed a voice-reference mock that targeted an unused
  application alias instead of the policy implementation.
- Unified validation entry points: `python gradio_app.py --selftest` delegates to
  the same CPU suite as `python -m tests`; browser acceptance stays opt-in.

The application module shrank from 3,929 to 3,282 lines. The prompt controller
shrank from 458 to 266 lines. Its remaining size, and that of the application,
includes signature-preserving public callbacks and runtime service wiring.
Replacing those with dynamic argument forwarding would obscure the API contract.

## Validation and cache hygiene

The initial CPU baseline had one failure: ignored cached upstream requirements
still specified `comfy-kitchen==0.2.36`, while the repository pinned 0.2.37. The
cache was refreshed from the exact `COMFY_REF` in `h3_requirements.py`; no
production source or dependency pin changed. Cached upstream checks intentionally
fail when supplied source disagrees with the configured contract.

The consolidated CPU suite now discovers 388 tests: 384 pass and four optional
checks are skipped on this host. All 15 graph snapshots remain unchanged, and
the existing UI suite checks the published generation parameters and outputs.
All seven browser suites pass: queue transport, settings, references, resolution,
image library, workspace and media performance. The settings suite had one
transient mode-edit failure and passed on an independent rerun.

Validation uses CPU fixtures and synthetic generation callbacks. GPU inference,
TensorRT compilation, model quality and deployment are separate checks.

## Cleanup recheck

A second audit compared removed adapters and the original self-test against their
current callers and tests. Retained application and controller functions are
unchanged apart from unused dependency fields and the unified self-test runner.
No active runtime caller requires the deleted exports; the implementing domain
services remain available. The launcher still exits before server initialization
and propagates test failure codes.

The audit identified useful assertions that graph snapshots alone did not retain.
Feature-owned tests now cover LTX first/middle/end upload staging and forwarding,
readiness messages that never echo private prompts, layout updates that preserve
inactive preferences, preflight node declarations versus generated LTX/Music/
upscale graphs, and the SeedVR2 temporal-chunk progress label. The expected graph
JSON remains unchanged. One remaining unused video-import controller method was
removed after comparing it with the active generic media-import path.

The 384-test suite passed in reverse order before these restored checks. The
final launcher CPU suite passed all 388 discovered tests (four optional skips).
The test suite's remaining wildcard import was replaced with explicit settings
imports, and an unused workspace-test import was removed. Ruff's F checks now
pass across the whole test directory and the changed runtime modules.

## Keeping the structure small

Add behavior checks beside the owning feature. Avoid adding new assertions to a
launcher self-test, or exporting a domain helper through `application.py` solely
to make it accessible to tests. Preserve current behavioral regressions even when
their names mention an older feature: migrations, retired model selections,
legacy media and positional API calls still have active consumers.

`README.md` and `docs/settings-refactor.md` describe the current repository.
`REVIEW.md` and dated upstream audits preserve historical decisions and may
describe superseded defaults. Refer to the current guides and source catalogs
when evaluating a new change.
