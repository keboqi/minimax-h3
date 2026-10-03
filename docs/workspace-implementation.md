# Workspace implementation status

Updated 3 October 2026 against the [revised UI/UX plan](ui-ux-redesign-plan-2026-10-03.md).

The first-release workspace features are implemented behind a server selection
flag. Workspace is the default at the user's request on 3 October 2026. This remains a preview implementation;
Milestone A5 release acceptance is incomplete.

## Running and rollback

Start the existing local launcher with `bash run_h3.sh`, or run
`python gradio_app.py` in the provisioned environment. Both default to workspace.
Modal also defaults to workspace; the image captures the deployment environment's
`H3_UI_LAYOUT` override. Use `H3_UI_LAYOUT=legacy bash run_h3.sh`,
`python gradio_app.py --ui-layout legacy`, or redeploy Modal with
`H3_UI_LAYOUT=legacy` to return to the old layout.
An invalid value fails explicitly. Switching the layout does not migrate media
directories or replace the inference stack.

Browser preferences use schema 6 in the original v3 transport key and secret.
Existing preferences migrate through field validation. Prompts, paths, credentials,
job tickets and confirmations remain excluded. The old allowlist ignores new
fields on rollback. Workspace task selection is authoritative for H3 output
format and commits through the serialized settings reducer after restoration.

Technical sidecars remain schema 1. Optional `application_job_id`, `variant` and
`retry_of` fields add process-local linkage; the existing `job_id` continues to
mean the backend Comfy prompt ID. No request prompt or replay graph is written
to these sidecars. Existing filenames, records and legacy APIs remain usable.

## Implemented behavior

| Area | Behavior |
|---|---|
| Shell | Create, Media, Jobs, System and API/workflows; task-filtered engine choices and remembered per-task engine selection |
| H3 composer | Separate result preview; collapsed expert controls; output essentials; recipe changes/reset; exclusive decoder selection; mobile Compose/Preview links |
| Canvas | Aspect/size presets reuse existing resolution policy; exact dimensions remain available; conditioned Image output keeps the first frame's aligned native size |
| References | Stable slot bindings, visible tag insertion, sparse-to-dense translation shared by enhancement and generation, explicit repair after replacement/removal, bounded local metadata inspection |
| Prompt writers | Preview, Accept, Keep and Undo across H3, LTX, Qwen, Music and YuE2; late suggestions reject changes to the draft or conditioning context |
| Jobs | Request capture and owner-scoped idempotency before the existing GPU queue; copied inputs; stable IDs; progress/stage ledger; owner-scoped inspect/cancel/download |
| Recovery | Failed-variant selection with original seeds, model/workflow drift rejection, indeterminate submission rejection, separate finishing-only retry from preserved H3 sources |
| Media | Typed pagination advances past missing thumbnails; finishing models are independent of hidden engine settings; deletion names the scope and revalidates selection and file identity |
| Preferences | Qwen technical settings and separate generation/gallery finishing models; validated migration; private composition and credentials excluded |
| System | Cached local model-file inventory and capabilities; execution retains authoritative node/model/TensorRT checks |
| Refactor | Explicit bootstrap catalog/services; feature-owned views/bindings; compatibility exports and wrappers preserve existing callers |

`application.py` still contains legacy compatibility and media service wrappers.
They remain available for callers that patch those names. Durable project storage,
cross-restart identity/reconciliation, asset search/index/lineage, comparisons and
custom conditioned-image canvas policy are the separately scoped Milestone B.

## Process-local retention and privacy

Each accepted request has an immutable in-memory JSON snapshot. Media inputs are
copied into a unique OS temporary directory before queueing; edits and Gradio
cache expiry cannot change a waiting request. Jobs are authorized by the live
Gradio session hash. This is not durable authentication or cross-session recovery.
Reloading into a different session does not grant access to earlier jobs.

Limits are eight nonterminal jobs per owner, at most 128 process-wide records,
2 MiB per request snapshot, and 8 GiB of retained copied inputs by default.
`H3_JOB_MAX_INPUT_BYTES` changes the process-wide input-copy limit in bytes.
Admission rejects excess capacity without creating a phantom job. Expired
terminal jobs and their input copies are pruned every 30 seconds; finished jobs
can be evicted earlier to admit new work. Waiting/active inputs do not expire.
The retry window is 30 minutes from completion, subject to capacity eviction.
Normal process shutdown removes leases. Each retry takes its own source copy.

Prompts and private replay graphs are held in process memory for this window,
and copied input bytes are temporary local files. Credentials are not generation
inputs and are not captured in accepted jobs. There is no automatic saved project.
Generated outputs and their technical sidecars retain the existing library policy.
An abnormal process kill can leave `h3-job-*` directories in the OS temporary
directory; these are not replayable after restart. Automatic cleanup of such
orphaned directories is not implemented. Inspect and remove confirmed stale
directories through normal host maintenance after the process has stopped.

Acceptance copies files while holding the registry admission lock. Large uploads
can delay concurrent admission/inspection; measure this on the deployment storage
before promotion. Multiple application processes must not share a GPU/backend
without a shared coordinator. External Comfy clients remain outside the local
lease; backend-wide interrupt is guarded by an owned active prompt check but
cannot be atomic with unrelated remote submissions.

## Validation and remaining release gates

Run `python -m tests --browser` with `requirements-test.txt` installed. Tests run
offline and include existing workflow fixtures and service self-tests. The
workspace fixture simulates generation and queue rejection without loading
models. Its `/test/*` routes exist only in the test fixture.

Coverage includes immutable concurrent acceptance, idempotency, source pinning
and expiry, ownership, failed subsets with actual recorded seeds, submission
ambiguity, graph/model drift, finishing-only recovery and schema-1 sidecars.
Both layouts preserve all 13 explicitly published API contracts. Browser checks
exercise presets, mode memory, preference reload and isolation, first-frame
resolution, voice conditioning, stale suggestion rejection, accept/undo, queue
handoff rejection and job isolation. Workspace screenshots cover 390, 768, 1280
and 1440 pixels, light/dark themes and reduced motion; basic keyboard focus and
horizontal overflow are checked.

On this host, discovery ran 244 tests successfully with three optional PyTorch
tests skipped. All four browser suites passed. Undefined-name checks, Python
compilation and `git diff --check` passed. The final workspace upload checks also
cover input-derived Image canvas locking and sparse reference replacement/repair.

Evidence is generated under `.cache/ui-redesign/`: `final-tests.log`,
`default-workspace-tests.log`, baseline
screenshots, workspace screenshots, `workspace/results.json`, and server logs.
These local artifacts are ignored by Git. Browser timing includes Playwright
round trips and server updates; it is not an inference or production benchmark.
The latest synthetic run became ready in approximately 1.74 seconds. Ten
individual prompt/readiness updates took approximately 0.23–0.53 seconds; this
does not satisfy the plan's 200 ms target. Baseline comparison and tuning remain
release work.

Remaining validation gates (the default switch does not imply these passed):

1. Provisioned supported-GPU inference for H3 Text/Frames/References, all outputs,
   other engines, partial batches, finishing and interruption. The checked host
   has an RTX 3080 Ti; ComfyUI at `127.0.0.1:8188` was unreachable. No GPU smoke
   test or quality/performance claim is made.
2. Local and Modal deployment smoke and rollback rehearsal, including model/node
   readiness and shared finishing model selection. Modal configuration wiring is
   implemented, but no deployment was performed.
3. Full keyboard/screen-reader, contrast, focus, zoom and error-announcement audit,
   with fixes from real user testing. Basic browser checks do not certify WCAG.
4. Matched baseline/workspace performance measurements on deployment hardware,
   especially the plan's 200 ms settings-response target, large-file admission,
   cold startup, gallery size and GPU queue responsiveness.
5. Broad reference replacement/removal and rapid mode/settings browser scenarios
   with real uploaded media across engines; pure policy tests cover the core map.

The user requested the workspace default before these acceptance gates were
completed. The explicit legacy override remains available. Milestone B is not
delivered by this change.
