# Workspace implementation and operating guide

Updated 3 October 2026 against the [revised plan](ui-ux-redesign-plan-2026-10-03.md).
The workspace is the sole UI. Milestone A and B application features are
implemented. The user excluded real GPU inference from this work. The checks
below validate CPU contracts and simulated browser execution; they do not
establish inference quality or production throughput.

## Running and storage

Start `bash run_h3.sh`, or `python gradio_app.py` in the provisioned environment.
Modal builds the same interface. `H3_UI_LAYOUT`, `--ui-layout`, the old UI builder,
legacy styles and the `gradio_app` import shim have been removed. Rollback now
requires checking out a prior code version; there is no runtime layout switch.
Public named generation APIs retain their parameter ordering and return shape.
Workflow scheduling modes that happen to be called legacy remain supported.

The database is `<GRADIO_OUTPUT_DIR>/.h3-workspace/workspace.sqlite3` by default,
so it persists with the outputs. An existing database in the previous location
is copied on first startup, retaining ownership, history and annotations.
Private inputs and project files remain in
`<GRADIO_OUTPUT_DIR parent>/h3-workspace`. `H3_WORKSPACE_DIR` explicitly overrides
both locations. Modal places workspace state at `/data/h3-workspace` on the
persisted data volume. SQLite schema 1 stores ownership, jobs, projects, assets
and annotations. A newer schema is rejected rather than overwritten. Generated
media and schema-1 technical sidecars stay in their existing directories.
The database directory is blocked from Gradio file serving. New media register
their thumbnails and catalog records when saved; ordinary browsing and server
restarts do not discover files. **Scan historical media** explicitly discovers
older or externally added images, videos and audio and reconciles missing files.

Browser preferences use schema 6 inside the existing v3 transport key. Values
migrate through field validation. Prompts, uploaded paths, provider credentials,
job tickets and confirmations are excluded. Task/engine navigation restores
through a validated engine value, including engines outside the initial task.

## Implemented behavior

| Area | Behavior |
|---|---|
| Shell | Create, Media, Jobs, System and API/workflows; task-filtered choices and remembered engines |
| H3 | Compact composer/results; collapsed Output & recipe and tabbed Advanced settings; mobile Compose/Preview links |
| Engines | H3, LTX, Qwen, Music and YuE2 share the shell, ownership, GPU queue and guarded prompt review |
| References | Stable slots/tags, shared sparse-to-dense translation, repair after replacement/removal |
| Prompt review | Preview, Accept, Keep and Undo; stale suggestions reject changed drafts or conditioning |
| Jobs | Immutable UI acceptance, owner-scoped idempotency, pinned sources, bounded admission, stage/seed ledger, inspect/cancel/download |
| Retry | Failed subsets reuse recorded seeds/configuration; unknown submission outcomes block replay; H3 finishing retries use completed raw sources |
| Recovery | Signed browser ownership and recovery keys; technical records survive restart; reconnect observes exact backend history/queue and collects existing outputs without automatic replay |
| Projects | Explicit save of private prompts/settings and source copies; inspect, run a saved request, delete retained content; saved finishing state supports restart recovery |
| Media | Typed pagination, search by filename/tags/technical settings, tags/favorites, stable IDs and lineage; bounded index batches; safe scoped deletion |
| Comparison | Image reveal slider; two videos on elapsed time at native frame rates; shorter-duration timeline, drift correction, buffering pause, mute/A/B audio |
| Canvas | Existing input-derived default; optional H3 conditioned Image fit/pad or fill/crop, effective dimensions and preview; transforms derived copies |
| System | Cached local model inventory/readiness; execution retains model/node/TensorRT checks |
| Refactor | Explicit bootstrap services and prompt/model/media controllers; launcher only starts the application |

## Ownership and retention

The server issues an HttpOnly, SameSite=Lax signed browser cookie. Its validity
is 30 days, with renewal near expiry. It survives page reloads and process
restart when workspace state is retained. Jobs and projects are scoped to this
owner. A recovery key restores access in a different browser; it is a bearer
secret and should be kept privately. API session identities occupy a separate
namespace and cannot claim a cookie owner's records by supplying its session ID.
This ownership mechanism does not provide login or protect the server's shared
Media library from other people with server access.

Unsaved prompts, private replay graphs and request snapshots stay in memory.
Technical SQLite records omit prompt/graph text, credentials and backend error
or progress messages. Normal generation sidecars also omit prompts. Only the
explicit Save project action writes full request content, replay graphs and
copied sources to durable storage. Project deletion removes that content and
invalidates loaded snapshots pointing into its source directory. Generated
outputs retain the existing shared-library policy. Tags/favorites are shared
library metadata, while projects and Jobs are private to their browser owner.

Admission allows eight nonterminal jobs per owner, 128 process-local records,
2 MiB per request and 8 GiB of retained temporary inputs by default.
`H3_JOB_MAX_INPUT_BYTES` changes the temporary-copy quota. Expired terminal
requests are pruned every 30 seconds, 30 minutes after completion, and can be
evicted earlier at capacity. Waiting/active sources do not expire. Explicit saved
projects remain until deletion and are outside the temporary input quota.
Technical database history remains durable; the displayed working set is bounded.

File copies happen under a separate admission lock, leaving monitoring and
cancellation accessible. Copies reject sources that changed during copying.
Temporary leases use cross-process file locks. Normal shutdown releases them;
startup removes verified orphan leases while preserving another live process's
copies. Each retry takes its own copy. Multiple application processes still need
a shared scheduler before sharing a GPU/backend. ComfyUI's global interrupt is
checked against owned prompt IDs; external clients remain outside this scheduler.

After restart, unfinished jobs are marked Recovering. Reconnect to backend
history checks recorded prompt IDs. Pending work stays observed; interrupted
application delivery/finishing is exposed for explicit recovery. A submission
without a confirmed ID requires manual review. Nothing is automatically
resubmitted. Exact replay after restart requires an explicitly saved project;
technical history alone cannot reconstruct a private request. Missing sources
and changed models/workflows reject replay.

## Backup, restore and rollback

Back up state separately from generated media:

```bash
python -m h3_app.workspace_admin --state /path/to/h3-workspace backup /backups/h3-state.zip
```

The command uses a consistent SQLite snapshot and includes explicitly saved
project sources. Write the archive outside the state directory. It contains
private project content and ownership signing state. Generated outputs, models
and ComfyUI inputs are not included; back them up separately.

Stop H3, then restore into an empty directory:

```bash
python -m h3_app.workspace_admin --state /path/to/restored-state restore /backups/h3-state.zip
```

Set `H3_WORKSPACE_DIR` to that directory before restarting. Restore relocates
saved project source paths and preserves ownership/recovery keys. Generated
output paths require the original media paths or a separate media relocation.
Unsafe archive paths, incompatible schemas and corrupt databases are rejected.

Index rebuild marks unavailable files and refreshes derived technical metadata;
it preserves identities, annotations, projects, ownership and idempotency.
Schema-1 sidecars accept missing optional asset/job/lineage fields from old files.
Rolling back application code must preserve the state directory and media.
Older code can leave new SQLite state dormant. Back up before a future schema
migration; never replace authoritative state with a media rescan.

## Canvas and comparison limits

Custom canvas applies only to H3 Image output with First / last frame
conditioning and a supplied first frame. Width/height are 256–4096, aligned to
32 pixels. Fit/pad centers the source on black; fill/crop centers the crop.
Both use EXIF orientation and Lanczos RGB resampling, including upscaling.
The first and optional last frame become derived PNGs in the job input lease;
the saved/original inputs remain intact. Video, Reference media and the default
Input-derived mode retain their previous behavior.

Video comparison uses browser codec support and corrects drift above 80 ms.
It compares elapsed seconds, not matched frame indices, and is not frame-accurate
editing. Playback uses the shorter duration and one audio source at a time.

## Validation

Install `requirements-test.txt`, then run:

```bash
python -m tests
python -m tests --browser
python gradio_app.py --selftest
```

Tests use offline Hugging Face mode. Fixtures start the production server with
mock backend health and generation callbacks; test-only routes exist only in
fixtures. CPU FFmpeg creates comparison clips with different frame rates and
durations. Coverage includes settings/default graph contracts, request pinning,
idempotency across restart/eviction, ownership/recovery, source retention and
orphan cleanup, saved replay, ambiguous submissions, interrupted finishing,
backup/restore, schema guards, annotation-preserving rebuild and custom canvas.
Browser suites cover settings, voice references, first-frame resolution, all five
engine actions, prompt review, queued snapshots, project recovery, annotations,
image/video comparison, narrow widths, dark theme, keyboard focus and 200% zoom.

Final verification on this host: 260 tests passed with three optional PyTorch
checks skipped; all four browser suites passed. The 13 published API contracts
match a fixed baseline from commit `2dbf18d`, and the 15 normalized workflow
fixtures remain unchanged. Standalone self-test, lint, Python compilation,
launcher Bash syntax and Git whitespace checks passed. Storage/privacy regression
checks were rerun after the final project-deletion correction.

Local evidence is written under ignored `.cache/ui-redesign/`, including
`final-acceptance2.log`, `final-storage-check.log`, `final-selftest.log`,
`workspace/results.json` and screenshots. The latest mock workspace became ready
in 1.91 seconds; ten prompt readiness updates took
142–171 ms, satisfying the local 250 ms p95 gate for that path.
Browser timing includes Playwright round trips. Other technical-setting actions
remain serialized and were not separately benchmarked. These measurements are
not inference or production benchmarks. Basic automated accessibility checks
are not a full WCAG certification.

Real GPU inference was deliberately excluded. Actual Modal deployment, a full
screen-reader/usability audit and production storage/performance measurements
were not performed on this host. Modal environment/volume wiring and local HTTP
startup are covered by CPU checks, with inference graphs preserved by fixtures.

## Gradio and interaction follow-up — 4 October 2026

The shared Gradio pin is now 6.29.1. H3's readiness message, generation actions
and progress sit in the right results column directly below Next run. Other
engines keep their actions with the composer. Mobile navigation provides a
direct Generate anchor, and the native main tabs wrap. Technical disclosures
stay collapsed, Qwen controls follow mode
and CFG, and all engines show composition prerequisites before submission.
Idle Interrupt is disabled until owned work is accepted. Selected Jobs details,
outputs and actions refresh together, including terminal cancellation state.

H3 ordinary settings handlers use at most 24 inputs; scalar edits return four
outputs while coupled policy changes retain the full resolver presentation.
Authoritative transition memory and its lock are per session. Numeric handlers
listen to user input/commit events, avoiding delayed preset-change echoes.
Preference saves merge seven namespaces into a session snapshot, with explicit
saves after unqueued resolution refreshes. Existing privacy and migrations remain.

Media search and Favorites filter the paged thumbnail grid. One selection feeds
preview, metadata, tags, lineage and comparison A. New/changed managed files and
sidecars are indexed during browsing; rebuild lives in System maintenance and
preserves annotations. Empty results clear the inspector. Comparison inventory
loads when its panel opens; scan results are reused for pages, database updates
are batched, and video posters use four shared workers. The local 1,060-asset
benchmark and remaining indexing costs are recorded in
[the gallery investigation](gallery-performance-2026-10-04.md).

Final pre-push verification runs 293 CPU tests (three optional PyTorch checks skipped),
the four browser suites and standalone self-test. The 13 published API contracts
and 15 normalized workflow fixtures remain unchanged. Measurements, validation
details and limitations are recorded in
[the Gradio/UI audit](gradio-ui-audit-2026-10-04.md). GPU inference and deployment
remain excluded from local acceptance.
