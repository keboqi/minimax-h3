# H3 Continuum Gradio integration plan

Date: 2026-10-18. Status: proposed implementation plan; no runtime changes or GPU validation performed.

## Decision and scope

Add a **Clip / Sequence (experimental)** selector within H3. Clip is the default and preserves today's independent generation and existing functions, with up to 15 seconds per clip. Sequence uses Continuum from its first chunk and lets users stack chunks with their own prompts. Keep conditioning choices independent of this selector. Sequence has its own internal request, workflow and orchestration components while sharing provisioning, jobs, GPU scheduling, staging, outputs and ownership.

Use upstream Continuum for sampling, continuity, chunk compatibility, revision selection and assembly. Gradio owns the user experience. Do not port the sampler, reproduce its signature algorithm, or depend on ComfyUI browser JavaScript.

The first complete release includes text-to-video, first-image generation and image-based Ref2VA with nine stable reference slots, fixed and per-chunk prompts, full generation, review/continue/retry, take selection, extension, saved-project restart recovery, and final export. Reference-only generation is supported upstream; delaying all Ref2VA was an initial staging choice, not an upstream limitation. Existing single-clip reference audio/video functionality remains available. Matching all reference audio/video combinations in continued sequences and latent refinement requires separate validation gates.

### Mode and conversion semantics

- Clip always uses the current pipeline. Its output is an independent video, not a saved Continuum chunk.
- Sequence always uses Continuum, even with only one chunk. A saved compatible first chunk can therefore be continued later.
- Add chunk is a Sequence action. Before generation it appends a draft prompt row; afterward it extends the target and generates the suffix while reusing compatible accepted chunks.
- An optional **Use settings in Sequence** action copies available prompts, inputs and compatible settings into a new draft. It does not migrate sampling history. Show: **Starts a new sequence; the first chunk will be regenerated.** Preserve the original clip; switching modes never starts generation.
- No current-pipeline latent capture/import bridge is required. An MP4 is not native continuation state.
- Keep separate drafts/preferences. One-chunk Sequence output is not promised to be numerically identical to Clip output.
- Each Sequence row can override the shared prompt. Initially use one common chunk duration: upstream exposes a scalar `chunk_seconds`, not arbitrary per-chunk durations.

Preserve all existing Clip functions. Sequence exposes validated combinations and explains unavailable options before generation.

## Evidence and upstream pin

- Reviewed upstream main: `7e14fde33f8e8fa5804fa409777d33afee2427f4`.
- Proposed deployment pin: v3.9.8 commit `58fbd855feb13fa67a0613614df04e2b6fe808e2`.
- GitHub comparison shows documentation, packaging, example and test changes between those revisions, with no runtime Python/JavaScript changes in that comparison. Verify the exact shipped package during implementation.
- The public node registry exposes `H3ContinuumSamplerV39`, `H3ContinuumReferenceImagesV39`, `H3ContinuumReferenceAudios`, `H3ContinuumAssembleSeamV35` (Finalize), and `H3DecodeCacheHelper`. Older internal sampler classes are not all publicly registered; do not build against historical node names.
- V3.9 inherits backend review inputs: `generation_mode`, `review_action`, `take_group`, `take_revision_id`, and `take_action`. Continue, Regenerate Current and Finish Remaining are server-side actions.
- Supply an explicit UUID `project_id` and leave legacy `run_name` empty. Upstream otherwise derives identity from graph topology, which is unsuitable as the app's ownership identity.
- Sampler outputs include lists of video/audio latents, an assembly plan, status, driving audio, and refinement context. List mapping and physical-group boundaries must survive decode and assembly.
- Current source permits 1–16 chunks and 4–30 seconds per chunk. Those are schema limits, not validated application resource limits.
- Runs are stored under ComfyUI output `h3_continuum/runs/run_<uuid>`. Scratch-storage settings affect assembly/cache, not this saved history.
- Upstream storage HTTP routes inspected manage scratch settings and cleanup; they are not a complete owner-scoped application run API.
- This app pins Spectrum 0.2.28, while upstream documents finite audited Spectrum profiles including 0.2.29. Fresh generation and safe saved-history reuse are distinct capabilities.

Primary sources:

- [Release](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/releases/tag/v3.9.8)
- [Public node registry](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/blob/7e14fde33f8e8fa5804fa409777d33afee2427f4/nodes.py)
- [V3.9 and review node schemas](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/blob/7e14fde33f8e8fa5804fa409777d33afee2427f4/v3/driving_nodes.py)
- [Run storage and compatibility](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/blob/7e14fde33f8e8fa5804fa409777d33afee2427f4/run_storage.py)
- [Review state machine](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/blob/7e14fde33f8e8fa5804fa409777d33afee2427f4/v3/review_control.py)
- [Reuse signatures](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/blob/main/docs/ISSUE31_SAMPLING_SIGNATURES.md)

## Gradio experience

Keep Clip and Sequence drafts/preferences separate. The selector changes visible controls, never an active job. Clip retains its composer. Sequence shows the chunk list, Add chunk, total duration and review actions even with one chunk. Draft edits never change an accepted run in place.

Both modes use the same UI shell and shared component groups: conditioning/reference inputs, prompt area, model/resolution/sampling settings, result preview, and finishing controls. Sequence expands the prompt area into a shared scene/style prompt plus editable chunk prompts. Selecting a chunk updates the existing preview and shows its takes. The main action changes with state (Generate next chunk, Accept and continue, or Regenerate from here); Retry and Finish remaining appear only where meaningful. Keep advanced reference assignment and take management collapsed until needed. Incompatible Sequence options show a reason instead of silently changing the user's Clip preferences. Retrying an earlier chunk preserves earlier accepted chunks but can invalidate later dependent chunks; show the affected range before submission.

Composition controls:

- Sequence prompt; Fixed or Per-chunk mode initially. A table provides chunk number, editable prompt and derived duration preview. Duration is shared across chunks initially. Timeline syntax comes after parser/validation coverage.
- Chunk count, seconds per chunk, fixed sequence seed, output dimensions and continuity preset. Display requested duration and resolved frame/duration facts separately.
- Sequence starts with one 5-second chunk; added chunks inherit the common duration and shared prompt/references. Clip retains current defaults. Initial multi-chunk application limits: at most four chunks and 30 seconds total, subject to reduction after GPU/memory measurement. Expanding limits requires a measured configuration, not just upstream schema acceptance.
- Review each chunk or Generate full video. Review is the default for saved runs.
- Explicit **Save resumable project** choice with a short explanation that prompts, source copies and latent takes are retained. Unsaved full generation uses storage Off; review/restart resume requires saved storage. This is normal product behavior, not a recurring confirmation dialog.

Review controls:

- Show the current review unit, its take, accepted prefix, final target, and resumability status.
- Provide Continue, Retry this section, Finish remaining, Use selected take, Continue from selected take, Export current sequence, and Stop.
- Selecting a take for preview does not silently change the active branch. Applying it is an explicit action.
- Extension increases the final chunk target. It does not interpret the count as additional chunks.
- Later settings edits create a proposed revision; they do not mutate an accepted request. Show which suffix must be regenerated before executing it.
- When upstream merges the terminal pair, display e.g. **Chunks 3–4** as one review unit and disable impossible individual edits.
- Review previews are labelled partial. Completed exports enter Media; intermediate latent files and internal cache artifacts do not.

State model: draft → queued → sampling → review-ready or sampled → decoding/exporting → complete. Interrupted/failed states retain the last committed prefix. Model sampling completion, preview availability, export completion and durable checkpoint status separately: a decode failure must not erase successfully sampled chunks.

## Application and backend boundaries

Introduce `ContinuumRequest`, `ContinuumAction`, and `ContinuumRunSnapshot` instead of expanding the existing positional H3 API. Snapshot fields include run UUID, owner/project association, request revision, expected active revision, physical review unit, selected take, completed prefix, reuse capability/reason, actual seed, output references and checkpoint state.

Proposed modules:

| Area | Proposed files / existing integration points |
|---|---|
| Pins and install | `h3_sources.py`, `setup_h3.py`, `modal_h3.py` |
| Requests and policy | `h3_app/generation/continuum_requests.py`, `h3_app/continuum_policy.py` |
| Pure graph builder | `h3_app/workflows/continuum.py` |
| Orchestration | `h3_app/generation/continuum.py` |
| Run metadata adapter | `h3_app/continuum_store.py` |
| Optional backend bridge | `custom_nodes/H3ContinuumBridge/` |
| Gradio | `h3_ui/continuum_view.py`, `h3_ui/continuum_bindings.py`, bootstrap/contracts/application wiring |
| Shared lifecycle | `jobs.py`, `execution.py`, `job_admission.py`, `workspace_store.py`, `workspace_admin.py` |
| Results and status | `status.py`, `provenance.py`, media discovery and settings presentation |

Construct the workflow from the registered node schema and our graph builder, not by loading the interactive template wholesale. Reuse loader/LoRA/scheduler helpers only where their numerical and wrapper behavior is compatible; the current H3 patch chain must not automatically transfer to Continuum.

Graph path: loaders and approved model patches → sampler/sigmas → V3.9 sampler → per-group video/audio decode → upstream Finalize → video creation/save. Evaluate Decode Cache Helper after baseline correctness; it is a performance optimization, not run persistence. Connect status to an output-visible report adapter if needed, because a STRING output alone does not guarantee structured history delivery.

Keep all torch objects in ComfyUI. The app receives JSON metadata and managed media paths only. Prefer a read-only, versioned manifest projection for run/take presentation. If upstream lacks a usable structured contract, add a narrow backend bridge that uses pinned upstream validation and emits a bounded JSON snapshot. It must not reimplement sampling or write upstream manifests directly. All sampling/selection mutations go through upstream node inputs.

Implementation phase 1 must establish the precise bridge necessity, output schema, caching behavior and export-only path before freezing the app API. Do not parse human-readable status text as the authority for continuation or take validity.

## Job admission, retries and concurrency

Register a separate `h3_continuum` job family through existing acceptance and GPU binding. Capture action, request revision, source leases, expected run revision and selected take before queueing. Preserve existing single-clip API names and positional contracts.

Use one application mutation lock per run in addition to the shared GPU lease and upstream filesystem lock. Check expected revision again at execution; reject a stale second-tab action rather than applying it to a newer take. Idempotency belongs to an operation UUID: transport retry reuses it; a deliberate Retry click creates a new one.

Start with one backend submission per review action. Full generation may use upstream Full Run; reconcile intermediate checkpoints after cancellation or failure. Never infer that a timed-out HTTP request failed to submit. Use the existing prompt identity/history reconciliation and never automatically resubmit ambiguous GPU work.

Keep upstream and app commits separate. Record operation intent first, submit once, then reconcile the actual upstream revision into app metadata. On a crash between commits, recover from prompt history plus validated run manifests. Only committed complete physical groups are reusable.

Verify that repeated identical Continue/Retry inputs execute despite ComfyUI caching. Use the upstream change-detection mechanism where available; any bridge execution token must invalidate orchestration without contaminating the sampling identity or reroll seed contract.

Stop follows current owner/tab/prompt cancellation rules. After stopping, refresh committed history and release leases. Export or finishing failures offer a decode/export retry without new sampling; take selection itself may still require decode work.

## Persistence, ownership and retention

SQLite remains authoritative for owner, project, operation ledger and the mapping to an upstream run UUID. Upstream manifests/tensors remain authoritative for chunk/take validity and sampling identity. Add a forward migration; do not change the meaning of existing project records.

Saved projects retain the prompt plan, effective settings, immutable source copies/hashes and run identity before the first resumable action. Resolve a random seed once and persist it. BrowserState stores only safe UI preferences; prompts, paths and run authority stay server-side.

Use upstream's normal run root initially, with an application-owned UUID mapping. Do not rely on UUID secrecy for authorization. Run reads, previews, selection and deletion validate the authenticated owner and contained paths. Raw filenames and revision IDs supplied by clients are never trusted.

The existing general ComfyUI editor/proxy is a separate access boundary. Audit routes that expose output JSON/tensors or global storage cleanup; app ownership does not automatically secure direct backend/editor access. Keep the backend private and avoid claiming tenant isolation for unrestricted editor access. Any managed run endpoints must perform ownership checks.

Track saved inputs, raw latent takes, decode cache, scratch buffers and final media separately. Delete cache independently; project deletion removes only that project's unshared saved state after leases end. Exclude internal previews and run artifacts from normal gallery discovery. Enforce configurable disk quotas and a free-space reserve before new work, and retain valid history on disk-full errors.

Current workspace backup excludes external generated output. Extend backup/restore with an explicit resumable-project bundle containing run manifests/tensors and retained sources; label metadata-only backups as non-resumable. Backup requires a consistent run snapshot. Restore verifies containment and schema compatibility and relocates paths without rewriting sampling contracts to pretend compatibility.

Modal already persists ComfyUI input/output on a volume. Add explicit checkpoint synchronization at successful review boundaries and terminal outcomes, with a durable-status acknowledgement. A local fsync is not a volume commit. Wire this through the existing service lifecycle rather than importing Modal in domain modules. Do not reload a mounted volume while active handles/writes exist. Validate a real cold restart; record the maximum loss window if mid-Full-Run volume commits are not yet available.

## Compatibility policy

Long video has a separate capability resolver and conservative effective defaults. Existing Singularity/SLA/Spectrum/LightVAE defaults must not silently carry into it.

| Feature | Initial policy |
|---|---|
| Attention/cache | Establish one native Core PyTorch/Kitchen configuration with cache Off first |
| Spectrum 0.2.28 | Unverified for resume; disabled for initial saved-run profile |
| Spectrum 0.2.29 | Candidate later; require provider-contract and GPU tests, with single-clip regressions if shared pin changes |
| SLA, Veda, Sol, feed-forward and fused modulation patches | Individually gated; no assumption of reuse compatibility |
| LightX2V Turbo | Admit one task-matched profile after baseline/new-run/resume tests; no blanket support for every Turbo LoRA |
| Video VAE | Native official path first; certify encoding and decoding choices separately |
| LightVAE/LynnReal/TRT | Deferred until shape, continuity and reuse tests pass |
| First image / image-based Ref2VA / nine references | Initial release; validate task-matched model/LoRA selection, reference-only generation and per-chunk routing |
| Last image | Validate terminal merge before admitting multi-chunk support; existing single-clip support remains |
| Reference/driving audio, video guides | Later, with distinct semantics and dedicated tests |
| Semantic Bridge / T8 voice conditioning | Deferred; not implicitly composed with upstream conditioning |
| Native latent upscale / Second Pass | Deferred until continuation and export are stable |
| Existing video finishing | Apply to a completed exported source; retain source and separate retry |
| Image-only / audio-only / batch variants | Keep in existing single-clip mode initially |

Unknown wrappers or changed signatures must never be forced reusable. Surface the upstream reason and offer a new run. Reconcile both application settings and loaded-runtime evidence: a UI fingerprint is not a substitute for upstream's signature checks.

## Delivery sequence and exit gates

1. **Pin and contract fixture.** Install the same revision locally/Modal; capture actual `/object_info` schemas, required sockets, status/take metadata, frame rules and cache behavior. Produce text and Ref2VA two-chunk headless API graphs and decide the narrow adapter contract. Gate: graphs run without browser interaction, and a saved one-chunk Sequence extends without resampling its compatible first chunk. No Clip-state migration is required.
2. **Pure requests, policy and graph.** Implement isolated long-video types, effective defaults, model checks and graph construction. Gate: focused graph/policy tests and unchanged single-clip fixtures.
3. **Durable runs and operation ledger.** Add owner mapping, explicit project retention, schema migration, locks, idempotency, crash reconciliation, safe inspection and deletion. Gate: duplicate/stale/cross-owner actions and interrupted writes are handled correctly in synthetic tests.
4. **Generation and export.** Connect submission/progress/cancellation, list decoding, Finalize, provenance and export-only retries. Gate: GPU two-chunk output, AV timing, reuse counters and unchanged accepted-prefix hashes; sampling is not repeated after export failure.
5. **Gradio review workflow.** Build the Clip/Sequence selector, chunk editor, Add chunk controls, nine-slot image references, take review, extension and restoration. Gate: browser tests cover unchanged Clip behavior, separate drafts, explicit settings-copy conversion without automatic generation, and Sequence create → review → retry → choose → continue → finish → reload for text and Ref2VA; double-click and two-tab cases are deterministic.
6. **Deployment and resources.** Add Modal checkpoint commits, quotas, cleanup and resumable backup. Gate: service cold restart preserves owner, inputs, run and accepted prefix; measured RAM/VRAM/disk fit published launch limits.
7. **Conditioning expansion.** Add multi-chunk first/last frame and timeline prompts with targeted GPU validation. Admit acceleration, alternate VAEs, audio/video conditioning and refinement only as separately proven combinations. Nine-image Ref2VA is part of the initial release rather than deferred expansion.

Milestones 1–6 constitute the initial Gradio release. Milestone 7 is planned expansion, not a claim that all existing H3 options work on launch. If a GPU gate fails, repair or narrow the exposed profile before continuing with that dependent feature.

## Validation and acceptance

Run focused tests per change, not the full suite every time. Proposed test groups: `test_continuum_contracts`, `test_continuum_workflow`, `test_continuum_store`, `test_continuum_jobs`, and browser review/recovery scenarios. Reuse existing ownership, execution and fixture infrastructure. A single broader relevant regression pass belongs at integration completion.

Critical cases:

- Resume after browser refresh, application restart, backend restart and Modal cold start.
- Crash after backend submission, after chunk commit, and before app metadata commit.
- Repeating an operation does not accidentally produce a new take; intentional retry does.
- Changing seed/model/LoRA/dimensions/prefix prompts invalidates the appropriate reuse. Extending an unchanged compatible run preserves its accepted prefix.
- Take selection preserves alternatives and rejects invalid revision/group combinations.
- Terminal-merged pair cannot be split by retry/selection/export.
- Cancellation, OOM, disk-full, corrupt/missing chunk, decode failure and export failure preserve earlier valid state.
- Requested duration, retained frame count and audio samples match the assembly contract, including non-integer chunk durations.
- Stable reference slots do not renumber when earlier slots are empty.
- Saved-project deletion and backup handle all owned files without touching another run.
- No prompts or source paths leak into ordinary technical provenance/browser preferences.
- GPU quality review compares movement, identity, audio continuity and seams; numerical reuse tests alone do not establish visual quality.

Before implementation locks the public contract, resolve: exact structured history projection, guaranteed export-only invocation, signature support for our pinned Core and minimal model wrappers, memory cost of downstream encoding, and Modal checkpoint commit delivery. These are engineering investigations, not user preference blockers.
