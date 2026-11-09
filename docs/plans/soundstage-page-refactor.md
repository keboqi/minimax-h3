# Soundstage page review and refactor plan

Date: 2026-11-09. Reviewed at `02879d3`, following the standalone-page redesign,
Workshop integration, save locking, and browser still playback. Scope: the page,
its API, persistence and adapters; no change to synthesis, writer policy or mix quality.

## Implementation status

Implemented on 2026-11-09:

- Project-owned save coordination serializes writes, drains in-flight requests,
  retains failed patches, and blocks dependent navigation/actions on failure.
  Response scope tokens and load generations reject answers from earlier projects.
- Writer and cast results check their source fields/slot versions before applying.
  Conflicting answers and clips remain available through explicit review/apply controls.
  SQLite transactions protect project edits across connections and processes;
  project creation uses an insert that cannot replace an existing project.
- Native browser modules separate Script, Cast, Library, Mix, Video, transport,
  and project/save reconciliation. `app.js` is the small bootstrap, and
  `workspace.js` assembles the views and owns playback/navigation orchestration.
- `page_adapter.py` isolates positional writer/speech contracts and decodes named
  writer, speech and cast progress. Existing synthesis handlers and published
  Gradio API signatures remain compatible.
- Shared page runs live in `h3_ui/page_runs.py`, with compatibility imports for
  Storyboard. Auto/Cast admission checks share one lock. Cancellation before job
  admission is explicit. Patch models validate types, finite numbers, nulls and
  choices while preserving the existing slider clamping behavior.

Verification: 108 focused Python tests passed across Soundstage page modules,
published batch/library APIs, new conflict/admission regressions, Storyboard page,
and workspace persistence. Six deterministic JavaScript tests passed. A headless
Chrome smoke passed across all page steps, library management view, still playback,
subtitle changes across project switches, save failure and retry. JavaScript syntax,
Python compilation and tracked-diff whitespace checks passed.

No full suite, hosted writer or GPU generation was run. Page run registries remain
process-local; durable project transactions do not turn them into a distributed
scheduler. The compatibility adapter preserves the existing synthesis handlers
rather than replacing their public interfaces.

Review follow-up: same-project loads now refetch when a save completes during
their read, so a clean edit queue cannot allow an older snapshot to replace the
editor. Writer/cast proposals carry result IDs; applying requires the reviewed ID,
and cast audio previews use that ID too. Replaced proposals are rejected without
consuming the newer result. Legacy proposals receive stable content-derived IDs.
Verification for these fixes: 15 focused Python tests, six JavaScript tests, and
the browser smoke including a delayed same-project refresh passed.

## Assessment

Keep the redesign and refactor incrementally. Script, Cast, Library and Video already
have separate route installers. The iframe shell and published Gradio API are explicit,
and the page uses the existing job admission path. Project edit locks prevent concurrent
read-modify-write operations from erasing unrelated fields. Drawing stills separately
from rendering an export is a useful boundary to preserve.

The largest remaining issue is asynchronous state ownership, rather than file size.
The approximately 2,200-line browser controller combines rendering, persistence,
polling, playback and orchestration. Splitting it without fixing these contracts would
spread the same races across several files.

## Review findings

### 1. High priority: save completion and failure are not reliable barriers

Evidence in `h3_ui/drama3/page/app.js`:

- `flushScript` (1359) clears its timer and pending body before awaiting PATCH.
  A second flush returns immediately while that PATCH is in flight. Two debounced
  saves can overlap, so server lock acquisition order can persist an older edit last.
- `flushSlots` (1903) clears its pending map before awaiting its requests and does
  not expose an in-flight promise. Both flush functions catch failures without
  restoring the unsaved data or rejecting the caller.
- `flushEdits` (2076) consequently cannot guarantee Speak or Auto uses what the
  editor displays. A failed save can be followed by generation using old data.
- Video has another independent queue (`flushVideo`, 606), omitted from
  `loadProject` (1138) and `flushEdits`. A pending video timer resolves the project
  name when it fires, so switching projects before it fires can save A's choices to B.
- `saveSettings` (2061) serializes requests, but also resolves `named(...)` inside
  the queued callback rather than capturing the project at edit time.

These are code-path findings; no delayed-network browser reproduction was run in
this review. Existing server locks serialize writes but do not establish client order.

### 2. High priority: late answers can overwrite newer work

`rewrite_project` in `page_script.py:153` reads a snapshot, releases the project lock
for the hosted writer, then applies script, title and cast metadata to the latest
project without checking whether those source fields changed. This preserves
unrelated clips/settings, but can overwrite a script edited while the writer ran.
The existing concurrent-writer test checks unrelated changes only.

On the client, `writerAnswered` (1398), `castAnswered` (1930), video replies and
settings replies do not consistently check the originating project or request
generation. A writer reply for A can replace the current project's state after
switching to B. Whole-project response assignment also competes with unsaved local
edits in other panels.

Cast completion (`page_cast.py:424`) writes generated audio back to a slot index
without checking whether the slot's source was edited or replaced during generation.
Treat long-running results as proposals against a particular input version.

### 3. Medium priority: the page depends on Gradio presentation contracts

`generation_values`, `writer_slots`, `_apply_cast`, and cast `apply_outputs` encode
long positional input/output lists, component visibility offsets and Gradio update
sentinels. Status text prefixes also determine errors in speech, cast and alignment.
The public API must stay compatible, but these presentation details should end at
an adapter rather than define the page's application contract.

### 4. Medium priority: run coordination and request contracts need clearer homes

Soundstage imports `Run`, `Runs` and `guarded` from `storyboard/page.py`, bringing
another feature's route module into its infrastructure dependency graph. Separate
speech/cast/video registries make exclusion checks such as Auto versus Cast a
check-then-start operation across registries (`page.py:325`, `page_cast.py:568`).
Those checks are not atomic under simultaneous requests.

`apply_fields` casts arbitrary dictionaries using `bool`, `str`, `int`, and `float`.
This permits surprising values (for example `bool("false")`), incomplete enum
validation and inconsistent error handling. Typed patch contracts would also clarify
the difference between omitted fields, explicit nulls and defaults.

## Ordered implementation

### PR 1: make browser saves dependable

Introduce one project-scoped save coordinator with independent field/slot channels.
Capture project identity when accepting an edit; serialize writes in each channel;
merge newer pending edits; retain failed edits for retry. Expose `flush(project)`
that awaits queued and in-flight saves and rejects on failure. Derive save status
from all channels, including video. Runs and project switches must await it.
Use a project load generation token to ignore stale loads and responses.

Acceptance: controlled delayed responses cannot reverse edits or apply them to
another project; failed saves remain dirty and prevent dependent actions; a second
flush waits for an already-running save. Navigation on failure stays on the draft
and explains the save failure.

### PR 2: protect long-running results from stale inputs

Add repository-level conditional updates with field-scoped input revisions or
fingerprints. A writer result checks script/mode/cast inputs; a generated voice
checks its slot inputs. Preserve unrelated changes. Return a recoverable conflict
with the generated proposal retained for explicit application. Initially this can
use the existing process lock if deployment is single-process; use a transactional
SQLite compare-and-update when multiple processes can write. Do not hold locks
while hosted writers or GPU jobs run.

Acceptance: editing the script, replacing a voice, or resizing/reusing a slot while
work runs never silently applies an old answer to the new input. Unrelated settings
and clips still merge successfully. Keep persisted project keys compatible.

### PR 3: split the browser along existing behavior boundaries

Use native ES modules: transport, project/save controller, run polling, Script,
Cast, Library, and Mix/player with Video. Keep local UI selections separate from
server snapshots and dirty edits. Centralize response reconciliation; update only
the returned feature's authoritative fields and preserve pending local values.
Give pollers explicit start/stop and project identity. Keep one small bootstrap.

Extend the explicit static asset allowlist and script loading for the new modules.
Preserve playback position, take selection, subtitles, light/dark behavior and
CPU Workshop restrictions. A framework or build-system migration is unnecessary.

### PR 4: establish named application results behind both interfaces

First isolate existing positional conversions in one compatibility adapter with
contract tests. Then introduce named writer/cast/speech result and progress types,
including structured success, warning, failure and interruption states. The page
uses these directly; Gradio adapters produce the existing positional outputs and
human-readable statuses. Move project operations out of route closures as touched,
leaving routes responsible for validation, identity and response mapping.

Acceptance: existing published API parameter order and streamed output shapes stay
compatible. Page control flow no longer parses English error prefixes or component
visibility updates. Avoid rewriting every handler in one change.

### PR 5: extract shared run infrastructure and validate patches

Move the existing Run/Runs/error-wrapper primitives into a neutral UI infrastructure
module with compatibility imports as needed. Add atomic project operation admission
for incompatible operations (particularly Auto/Cast), explicit cancellation state,
and documented process-local lifetime. Continue using JOBS for GPU admission and
history; do not introduce a second scheduler.

Introduce typed patch/request models with deliberate coercion, ranges, enums and
null semantics. Reuse field metadata for boot ranges/defaults where it reduces
duplication. Route every project mutation through the repository boundary.

Acceptance: simultaneous Auto/Cast starts cannot both pass; validation rejects
invalid choices consistently; CPU Workshop and legacy saved projects still load.

## Validation and rollout

For each PR, run the affected page modules and meaningful new regressions, not the
full suite on every edit. For async client changes, add deterministic delayed/failing
fetch tests and a focused browser smoke covering switch-during-save, rewrite-during-
edit, failure-and-retry, and job polling after reopening the page. API TestClient
coverage does not execute these browser paths.

Before merging the module split and adapter changes, exercise Script -> Cast ->
Speak -> line regeneration -> word repair -> still playback -> Save to Media,
plus CPU Workshop preparation and existing scripted API contracts. Reuse the
recent successful listening/GPU validation unless synthesis inputs or behavior change.

Start with PRs 1 and 2. They address data integrity and supply regression coverage
for later structural work. Preserve the current UI and media formats throughout.

## Review verification

Read the redesigned page, route modules, shared helpers, project persistence,
Gradio shell/bindings, recent commit history and the four page test modules.
Attempted `.tmp-ui-venv/Scripts/python.exe -m unittest tests.test_drama3_page
tests.test_drama3_page_script tests.test_drama3_page_cast tests.test_drama3_page_video`.
The command produced no output and was interrupted; there is no fresh pass/fail
result from this review. No production code was changed; no hosted writer, GPU
generation, browser session or full suite was run.
