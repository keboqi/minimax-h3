# Prism Studio operating guide

## Running and storage

Start `bash run_h3.sh`, or `python gradio_app.py` in the provisioned environment.
Modal builds the same interface. Public named generation APIs retain their
parameter ordering and return shape.

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

## Features

| Area | Behavior |
|---|---|
| Shell | Create, Projects, Media, Jobs, System and API/workflows; task-filtered choices and remembered engines |
| H3 | Compose (inputs, prompt, recipe, output essentials) beside Preview & generate; Fine-tune tabs for every other setting; mobile Compose/Generate/Preview/Settings links |
| Engines | H3, LTX, Qwen, Music and YuE2 share the shell, ownership, GPU queue and guarded prompt review |
| References | Stable slots/tags, shared sparse-to-dense translation, repair after replacement/removal |
| Prompt review | Preview, Accept, Keep and Undo; stale suggestions reject changed drafts or conditioning |
| Jobs | Immutable UI acceptance, owner-scoped idempotency, pinned sources, bounded admission, stage/seed ledger, inspect/cancel/download |
| Retry | Failed subsets reuse recorded seeds/configuration; unknown submission outcomes block replay; H3 finishing retries use completed raw sources |
| Recovery | Signed browser ownership and recovery keys; technical records survive restart; reconnect observes exact backend history/queue and collects existing outputs without automatic replay |
| Projects | Save a job's private prompt/settings and source copies from **Save as project** in Jobs; the **Projects** tab inspects, runs and deletes saved requests; saved finishing state supports restart recovery |
| Media | Typed pagination, search by filename/tags/technical settings, tags/favorites, stable IDs and lineage; bounded index batches; safe scoped deletion |
| Comparison | Image reveal slider; two videos on elapsed time at native frame rates; shorter-duration timeline, drift correction, buffering pause, mute/A/B audio |
| Canvas | Existing input-derived default; optional H3 conditioned Image fit/pad or fill/crop, effective dimensions and preview; transforms derived copies |
| System | Cached local model inventory/readiness; execution retains model/node/TensorRT checks |

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

## Interaction notes

H3's readiness message, generation actions and progress sit in the right results
column directly below Next run; other engines keep their actions with the
composer. Mobile navigation provides a direct Generate anchor, and the main tabs
wrap. Idle Interrupt is disabled until owned work is accepted. Selected Jobs
details, outputs and actions refresh together, including terminal cancellation.

H3 ordinary settings handlers use at most 24 inputs; scalar edits return four
outputs while coupled policy changes retain the full resolver presentation.
Transition memory and its lock are per session. Numeric handlers listen to user
input/commit events, avoiding delayed preset-change echoes.

Media search and Favorites filter the paged thumbnail grid from the asset
index; one selection feeds preview, metadata, tags, lineage and comparison A.
Rebuild lives in System maintenance and preserves annotations. Comparison
inventory loads when its panel opens, index updates are batched in one
transaction per scan, and video posters use four shared workers with a
10-second timeout.

## Validation

Install `requirements-test.txt`, then run:

```bash
python -m tests
python -m tests --browser
```

Tests use offline Hugging Face mode. Fixtures start the production server with
mock backend health and generation callbacks; test-only routes exist only in
fixtures. They validate CPU contracts and simulated browser execution, not
inference quality, production throughput or a full accessibility audit.
