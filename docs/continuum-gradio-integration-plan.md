# H3 Continuum Gradio integration plan

## Implementation status (2026-10-18)

Implemented on CPU, without GPU acceptance:

- **Pin and install:** `CONTINUUM_REF` v3.9.8 (`58fbd85…`) in `h3_sources.py`; cloned by `setup_h3.py` and `modal_h3.py`. Graph inputs were built from the pinned source's V3.9 sampler, Reference Images V3.9 and Finalize V3.5 schemas.
- **Requests and policy:** `h3_app/generation/continuum_requests.py` (`ContinuumRequest`, `ContinuumAction`) and `h3_app/continuum_policy.py` (limits, prompt compiler v1, issues/notes, Run Storage report parser).
- **Graph:** `h3_app/workflows/continuum.py` builds a single upstream review operation. The graph uses fixed semantic node IDs, explicit List prompt mode, Review Each Chunk + Save + Auto Resume, an explicit `project_id`, fixed reference slots, Core decoding, upstream Finalize and save. Every one-shot review field is compiled for each action.
- **Store:** `h3_app/continuum_store.py` with workspace schema v2 tables `continuum_runs`, `continuum_requests`, `continuum_operations`. Sources are retained per run, requests are immutable versions, and reservations use CAS on the run epoch. The store also provides idempotent request keys, a per-run dispatch lock, startup reconciliation that never resubmits, and a bounded `project.json` projection. Owner-scoped deletion covers the upstream run and preview directories.
- **Orchestration:** `h3_app/generation/continuum.py` makes one submission per physical group. Finish loops single-group Continue under one GPU job, with a checkpoint after each group (Modal: `H3_MODAL_COMMIT_VOLUME` commit). Each operation parses upstream's `N reused, M generated` line. A regenerated prefix that should have been reused produces a warning and stops Finish. Partial previews go to `h3_continuum/previews/<run>/`, which the gallery skips; completed sequences go to `h3/sequence/` with a snapshot.
- **UI:** `h3_ui/continuum_view.py` and `h3_ui/continuum_bindings.py` add a Clip/Sequence switch in the shared composer. Sequence provides a scene prompt, four chunk rows, size/steps/seed/continuity, a live summary with the affected range, and saved-sequence load/delete. Review actions are Generate/Accept, Retry, Generate all remaining, Use/Continue from take, Rebuild video, Regenerate from chunk and Interrupt. Admission gained a generic `job_admission` hook for click-time validation and media selection.
- **Tests:** `tests/test_continuum.py` (32 tests: prompts, policy, graph, store CAS/idempotency/reconcile/ownership, orchestrator with a fake Continuum, admission). `tests/workspace_fixture.py` runs Sequence against the same fake, for browser checks.

Deliberate deviations from the design below:

- **Sequence owns its recipe.** It uses Normal sampling, its own steps (default 20, scheduler `simple`), seed and conservative sizes (768×448, 1024×576, 512×512, 448×768) instead of inheriting Clip's Turbo/1376×768 defaults. Turbo is a note, not a blocker.
- **Locked request fields.** Only the scene prompt, chunk prompts and chunk count are editable on a saved sequence. Size, seed, model, steps, continuity, chunk length and source images are fixed, so retained sources need no re-upload and the reuse contract stays stable.
- **Separate saved-sequence list.** Saved sequences live in the Sequence panel's own list, not in the existing project list, which keeps Clip project behaviour untouched.
- **Read-only export bridge not built.** "Rebuild video" uses upstream **Use This Take** on the selected take. This path is designed to select and decode without sampling, and the app flags any generated chunks it reports. A custom bridge is still needed if GPU tests show Use This Take resampling.
- **Prefix enforcement happens after the operation, not before.** Upstream offers no pre-sampling "fail if prefix incompatible" input (`strict_compatibility` is part of the contract, not a gate). The app detects shortfalls from the reuse report afterwards and stops automatic continuation.
- **No backup bundle or quotas yet.** Resumable-project backup/restore and disk quotas (milestone 6) are not implemented.

Remaining GPU gates: two-chunk text and Ref2VA runs on the exact Core/model profile; reuse counters across restart; Use This Take performing zero sampling; RAM/VRAM at four chunks and 1024×576; Modal cold-start recovery after the per-group commit.

Date: 2026-10-18. Status: reviewed implementation design; runtime adapters and GPU acceptance remain to be proven. Review baseline: application checkpoint `ee99558`.

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
- Each Sequence row adds its own action prompt to the shared scene/style prompt. Initially use one common chunk duration: upstream exposes a scalar `chunk_seconds`, not arbitrary per-chunk durations.
- Sequence is explicitly saved before its first generation. Use **Save sequence & generate** for the first action, with a project-name field and concise retention explanation. Subsequent actions use ordinary Generate/Continue labels. Omit an unsaved Sequence branch in the initial release: it would contradict the promise that its first chunk can be extended. Clip retains existing temporary-request behavior.

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
- Application Core pin is `c9ea36b9d2140c89b2ef84f93f428df474e72ba5` in `h3_requirements.py`; upstream's reported GPU acceptance does not certify this exact runtime.
- Upstream `web/project_id.js` reads `project.json` through ComfyUI `/view`. This supplies a presentation catalog, not authoritative validation against the currently loaded model. We can project that data server-side without copying its frontend code.
- `tests/test_v38_review_cache.py` verifies that V3.9 inherits always-changed review behavior. Use that existing contract; no custom cache-busting seed or operation token belongs in sampler inputs.
- `v2/prompts.py` accepts a JSON string array in List mode. Use it instead of assembling delimiter-based prompt scripts.

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
- Sequence generation starts with **Save sequence & generate**, retaining prompts, source copies and latent takes under the owner. All Sequence operations use saved storage; Clip remains the path for temporary independent generation.

Review controls:

- Show the current review unit, its take, accepted prefix, final target, and resumability status.
- Provide Continue, Retry this section, Finish remaining, Use selected take, Continue from selected take, Export current sequence, and Stop.
- Selecting a take for preview does not silently change the active branch. Applying it is an explicit action.
- Extension increases the final chunk target. It does not interpret the count as additional chunks.
- Later settings edits create a proposed revision; they do not mutate an accepted request. Show which suffix must be regenerated before executing it.
- When upstream merges the terminal pair, display e.g. **Chunks 3–4** as one review unit and disable impossible individual edits.
- Review previews are labelled partial. Completed exports enter Media; intermediate latent files and internal cache artifacts do not.

State model: draft → queued → sampling → review-ready or sampled → decoding/exporting → complete. Interrupted/failed states retain the last committed prefix. Model sampling completion, preview availability, export completion and durable checkpoint status separately: a decode failure must not erase successfully sampled chunks.

### Concrete UI and prompt contracts

Use existing Gradio components and a fixed maximum set of four chunk editor groups for the launch limit; hide unused groups. Each row has a stable application UUID, prompt and reference assignment. Add/remove/reorder only unsampled draft rows initially. Changing a sampled row uses the regeneration action, not an in-place list edit. Keep the adapter capable of a higher configured limit without adding dynamic `gr.render` dependencies in the first implementation.

Normalize line endings once at admission. The effective prompt for chunk i is the nonempty shared scene/style followed by two newlines and the nonempty chunk action; empty actions inherit the shared prompt. Reject an empty effective prompt. Serialize the full array with JSON and explicitly set upstream List mode, even when all rows are identical. Store the exact resulting strings and compiler version in the saved request. Do not switch Fixed/List automatically as rows are added: changing prompt mode can change reuse contracts. Editing shared text affects every chunk; editing row i affects that row and its continuation suffix.

Require the serialized prompt-array length to equal the target chunk count. Upstream can repeat the last entry or ignore excess entries; the application should reject that mismatch instead of silently changing the user's plan. Add a preview of the effective prompt for each row so composition and reference-tag translation are inspectable.

Use fixed `@R1`–`@R9` identities in Sequence. Do not call the current Clip admission path's `ReferenceMap.dense_values()`: it compacts slots. Any existing Picture-tag conversion must be performed once against the selected chunk's effective routing and verified, not inherited blindly. Initially references apply to all chunks; the advanced per-chunk matrix stores explicit assignments, and Add chunk copies the last row's assignments. Task-matched Ref2VA model/LoRA selection remains mandatory.

Share view components, not mutable execution state. The visible mode selects a separate request compiler at admission, then snapshots it. Switching modes while a job runs cannot redirect its callback, overwrite another draft, or publish a Sequence result as a Clip result. The Generate keyboard shortcut must dispatch through the same mode-aware admission path. Extend the existing admission abstraction with an explicit compiler/media-selector hook for `h3_continuum`; currently settings resolution and active-media filtering only run inside the `family == 'h3'` branch. Do not rename Sequence to `h3` to obtain those behaviors.

### Action mapping

| App action | Backend behavior and preconditions |
|---|---|
| Generate first / next | Review Each Chunk + Continue / Next; take action Automatic, Regenerate From Auto. One new physical group at most. |
| Accept & continue | Same backend action on the current validated review head; no independent app-only acceptance flag authorizes reuse. |
| Retry current review unit | Regenerate Current with unchanged settings; upstream manages its new variation. Never use this for an arbitrarily selected historical row. |
| Regenerate from chunk i | Explicit Regenerate From at the physical-group boundary, using the edited saved request and a new operation identity. Show the affected suffix; clear the one-shot intent after completion. Verify nonce lifecycle in the contract fixture. |
| Preview a take | Read a published preview keyed by immutable revision/group; selecting it does not change the canonical run. |
| Use selected take | Use This Take with the exact group/revision; selection/decode work may execute but no new sampling is allowed. |
| Continue from selected take | Continue From Here with the exact group/revision, producing a new compatible branch. |
| Generate all / Finish remaining | App loops ordinary one-group Continue operations under one admitted job until the captured final target is reached; checkpoint after each group. No upstream Full Run/Finish Remaining call in the initial implementation. |
| Export / Retry export | Dedicated validated read/decode/assembly graph with no sampler node. Export the selected compatible prefix without changing the branch. |

Give app operations stable ids but never substitute those ids for upstream variation/nonces. After an ambiguous submission, reconcile first; a repeated Retry operation must not accidentally ask upstream for another variation.

Compile all one-shot fields explicitly on every action: manual regeneration uses Continue / Next plus its explicit boundary; take actions use Auto boundary and the selected immutable ids; ordinary Continue clears stale take selection and reroll intent. Merely previewing a historical row must never redirect Retry Current or Continue. Disable those buttons or label them against the actual review head; historical-row regeneration uses the separate Regenerate from here action.

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
| Backend inspection/export bridge | `custom_nodes/H3ContinuumBridge/` |
| Gradio | `h3_ui/continuum_view.py`, `h3_ui/continuum_bindings.py`, bootstrap/contracts/application wiring |
| Shared lifecycle | `jobs.py`, `execution.py`, `job_admission.py`, `workspace_store.py`, `workspace_admin.py` |
| Results and status | `status.py`, `provenance.py`, media discovery and settings presentation |

Construct the workflow from the registered node schema and our graph builder, not by loading the interactive template wholesale. Reuse loader/LoRA/scheduler helpers only where their numerical and wrapper behavior is compatible; the current H3 patch chain must not automatically transfer to Continuum.

Sampling graph: loaders and approved model patches → sampler/sigmas → V3.9 sampler → per-group video/audio decode → upstream Finalize → managed review-preview save. Initial completion export uses the separate read/decode graph below. Evaluate Decode Cache Helper after correctness; it is not run persistence. Keep semantic node IDs and loader topology stable across extension, retry and export settings changes. Output save tokens must not enter the upstream sampling dependency graph.

Keep torch objects in ComfyUI. For the supported local/Modal deployments, the app and ComfyUI share storage; read bounded, schema-checked `project.json`/manifest projections in `continuum_store.py` for display. Record that validation is pending until the backend validates the selected prefix. Do not expose raw `/view` history reads to the browser as an app API. An arbitrary remote ComfyUI server without shared storage is outside initial scope.

Add a narrow pinned backend bridge for validated inspection and read-only export. Its proposed outputs are the upstream-compatible latent lists, assembly plan and a bounded machine-readable report. It may adapt upstream internal helpers behind version/hash checks; it must not reimplement sampling, forge contracts, enter a storage context that mutates the canonical branch during export, or write manifests directly. Sampling and take-selection mutations remain upstream node inputs. Persist an atomic operation receipt outside upstream manifests with operation id, selected revision/group, validation result and generated/reused group counts. Reports survive decode failure and contain no tensor data.

Export graph: validated prefix loader → Core video/audio decode → upstream Finalize → video creation/save. Pin revision, group end, request snapshot and decoder configuration. A cached preview can satisfy a preview request, but concatenating encoded previews is not the default final export. Preserve upstream trimming/audio alignment by rebuilding its assembly plan with its helpers. Unknown schemas, missing tensors or invalid ancestry fail before decoding. Retrying export never runs a sampler and never changes the accepted branch. This bridge is a phase-1 proof, not a capability already established by source inspection; if the helper contract cannot be adapted safely, stop that implementation path and revise it before UI work.

Resume/extend also needs a backend enforcement check before sampling: the requested accepted prefix must validate and be reused, or the action fails. Upstream can otherwise fall back to starting fresh when a prefix is incompatible. The fixture must demonstrate the pinned hook/adapter enforcing this without weakening signatures. A post-execution reuse count is useful evidence but is too late to prevent silently regenerating chunk 1.

## Job admission, retries and concurrency

Register a separate `h3_continuum` job family through existing acceptance and GPU binding. Capture action, request revision, source leases, expected run revision and selected take before queueing. Preserve existing single-clip API names and positional contracts.

Use one application mutation lock per run in addition to the shared GPU lease and upstream filesystem lock. Check expected revision again at execution; reject a stale second-tab action rather than applying it to a newer take. Idempotency belongs to an operation UUID: transport retry reuses it; a deliberate Retry click creates a new one.

Use one backend submission per physical group. Generate all keeps one app job/GPU lease while submitting successive review operations, checking cancellation and committing a checkpoint between them. The captured target and prompts are immutable for that job. This avoids a separate Full Run recovery path and permits explicit Modal commits per group. Never infer that a timed-out HTTP request failed to submit; reconcile prompt identity/history and backend receipts before further work.

Keep upstream and app commits separate. Record operation intent first, submit once, then reconcile the actual upstream revision into app metadata. On a crash between commits, recover from prompt history plus validated run manifests. Only committed complete physical groups are reusable.

Verify inherited `IS_CHANGED` behavior in a real Core fixture; review is already always-changed upstream. Read-only export cache keys include immutable revision/group and decoder identity. No mutation is retried by blindly replaying the generic job variant-retry mechanism in `jobs.py`/`execution.replay_stage`; Sequence recovery resumes its operation ledger instead. Preserve generic job cancellation and submission transport.

Stop follows current owner/tab/prompt cancellation rules. After stopping, refresh committed history and release leases. Export or finishing failures offer a decode/export retry without new sampling; take selection itself may still require decode work.

## Persistence, ownership and retention

SQLite remains authoritative for ownership and operation intent; upstream remains authoritative for chunk/take validity. Add migration version 2 with `continuum_runs` (id, owner, project id, upstream UUID, request version, observed head, mutation epoch, state), `continuum_requests` (run id, immutable version, payload/hash), and `continuum_operations` (id, run id, owner idempotency key, expected epoch/head, action, request version, state, prompt id, resulting head, receipt/checkpoint metadata). Add owner-scoped indexes and unique operation keys. A Generate all job has a parent intent and ordered group operations; do not overload the existing variant index.

Existing `save_project()` retains one job's positional values; it cannot serve as the mutable history of a multi-job sequence. Add explicit sequence-project creation and a project payload discriminator linking to `continuum_runs`, preserving existing Clip project behavior. Create retained source copies and the immutable request before admitting GPU work. Use an atomic compare-and-set on the mutation epoch to reserve an operation; never hold a SQLite transaction across network/GPU waits. Run locks protect dispatch, and upstream filesystem locks protect its files. Initial deployment remains one app coordinator per backend; multi-process scheduling is not promised.

Saved projects retain the prompt plan, effective settings, immutable source copies/hashes and run identity before the first resumable action. Resolve a random seed once and persist it. BrowserState stores only safe UI preferences; prompts, paths and run authority stay server-side.

Use upstream's normal run root initially, with an application-owned UUID mapping. Do not rely on UUID secrecy for authorization. Run reads, previews, selection and deletion validate the authenticated owner and contained paths. Raw filenames and revision IDs supplied by clients are never trusted.

The existing general ComfyUI editor/proxy is a separate access boundary. Audit routes that expose output JSON/tensors or global storage cleanup; app ownership does not automatically secure direct backend/editor access. Keep the backend private and avoid claiming tenant isolation for unrestricted editor access. Any managed run endpoints must perform ownership checks.

Track saved inputs, raw latent takes, decode cache, scratch buffers and final media separately. Delete cache independently; project deletion removes only that project's unshared saved state after leases end. Exclude internal previews and run artifacts from normal gallery discovery. Enforce configurable disk quotas and a free-space reserve before new work, and retain valid history on disk-full errors.

Current workspace backup excludes external generated output. Extend backup/restore with an explicit resumable-project bundle containing run manifests/tensors and retained sources; label metadata-only backups as non-resumable. Backup requires a consistent run snapshot. Restore verifies containment and schema compatibility and relocates paths without rewriting sampling contracts to pretend compatibility.

Modal already persists ComfyUI input/output on a volume, but Gradio is a subprocess and cannot call the service object's `volume.commit()` directly. Add an injected checkpoint service with a local implementation and a deployment adapter. For Modal, the parent service runs a loopback-only authenticated helper accepting operation/checkpoint ids (never arbitrary paths), calls `volume.commit()` after app writes and backend group writes finish, and returns an acknowledgement. Keep Modal imports in deployment code and inject a per-container helper address/token through the child environment. Confirm the helper survives the existing snapshot/start lifecycle.

Do not submit the next automatic group until the previous checkpoint is acknowledged. On commit failure, pause with the local prefix intact and allow retry of persistence without rerunning sampling. Do not call volume reload during active work. Test the acknowledgement on actual cold restart; SQLite and tensor fsync alone do not establish remote durability. The single-group dispatch design removes the need for mid-sampler volume callbacks.

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

Start validation at 512×512, 24fps, 2×5 seconds with official model/encoder/VAE files and cache Off, separately for text/first-image and Ref2VA. Select an actually runnable task-matched checkpoint on the target GPU; if disabling existing ConvRot feed-forward patches causes OOM, that is a failed launch profile, not permission to silently reinstall those wrappers. Record the approved exact model, Core/Python/Torch/Kitchen versions, wrappers, LoRA and decoder path as a capability profile. UI readiness consumes the profile rather than assuming every entry in the existing model catalog is approved.

Treat four chunks/30 seconds as provisional launch limits, not a claim of resource safety. Model peak decoded storage from frames × height × width × channels × element size, then measure simultaneous decode lists, Finalize and saver allocations. Record peak host RAM, VRAM, disk, latency and output duration for 1/2/4 chunks. Either constrain resolution/duration to measured bounds or implement a separately reviewed streaming export path before raising them. Disk-backed Finalize alone is insufficient. A saved sequence retains its generation profile; decoder-only changes may permit re-export without enabling new sampling under a different profile.

## Delivery sequence and exit gates

1. **Pin and contract fixture.** Install the same revision locally/Modal; capture actual `/object_info` schemas, required sockets, status/take metadata, frame rules and cache behavior. Produce text and Ref2VA two-chunk headless API graphs and prove the inspection/export bridge. Gate: graphs run without browser interaction; a saved one-chunk Sequence extends without resampling its compatible first chunk; incompatible-prefix continuation fails before sampling; read-only export contains no sampler and preserves manifests. No Clip-state migration is required. Do this before database/UI expansion.
2. **Pure requests, policy and graph.** Implement isolated long-video types, effective defaults, model checks and graph construction. Gate: focused graph/policy tests and unchanged single-clip fixtures.
3. **Durable runs and operation ledger.** Add owner mapping, explicit project retention, schema migration, locks, idempotency, crash reconciliation, safe inspection and deletion. Gate: duplicate/stale/cross-owner actions and interrupted writes are handled correctly in synthetic tests.
4. **Generation and export.** Connect submission/progress/cancellation, list decoding, Finalize, provenance and export-only retries. Gate: GPU two-chunk output, AV timing, reuse counters and unchanged accepted-prefix hashes; sampling is not repeated after export failure.
5. **Gradio review workflow.** Build the Clip/Sequence selector, chunk editor, Add chunk controls, nine-slot image references, take review, extension and restoration. Gate: browser tests cover unchanged Clip behavior, separate drafts, explicit settings-copy conversion without automatic generation, and Sequence create → review → retry → choose → continue → finish → reload for text and Ref2VA; double-click and two-tab cases are deterministic.
6. **Deployment and resources.** Add Modal checkpoint commits, quotas, cleanup and resumable backup. Gate: service cold restart preserves owner, inputs, run and accepted prefix; measured RAM/VRAM/disk fit published launch limits.
7. **Conditioning expansion.** Add multi-chunk first/last frame and timeline prompts with targeted GPU validation. Admit acceleration, alternate VAEs, audio/video conditioning and refinement only as separately proven combinations. Nine-image Ref2VA is part of the initial release rather than deferred expansion.

Milestones 1–6 constitute the initial Gradio release. Milestone 7 is planned expansion, not a claim that all existing H3 options work on launch. If a GPU gate fails, repair or narrow the exposed profile before continuing with that dependent feature.

## Validation and acceptance

Run focused tests per change, not the full suite every time. Proposed test groups: `test_continuum_contracts`, `test_continuum_workflow`, `test_continuum_store`, `test_continuum_jobs`, and browser review/recovery scenarios. Reuse existing ownership, execution and fixture infrastructure. A single broader relevant regression pass belongs at integration completion.

Baseline checkpoint `ee99558` has four known failing UI-contract assertions from the focused 84-test run: encoder-toggle API position, FL2VA audio API positions, refinement-LoRA API position, and published advanced-API parameters/defaults. Record and resolve those against the intended current API before claiming unchanged Clip behavior. Do not simply regenerate snapshots or count those failures as acceptable Sequence regressions. No tests were run for this documentation-only design review.

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
- A forged or stale selected-take id is rejected; previewing a take never mutates the branch.
- Shared prompt edits, JSON quotes/newlines/delimiter-like text, prompt-count mismatch and sparse reference slots have deterministic compiled requests.
- Switching Clip/Sequence while another job is queued preserves the captured mode, inputs and result ownership; keyboard generation follows the selected mode at admission.
- Generate all checks cancellation and persistence between groups, stops on a failed commit, and can retry checkpoint persistence without sampling.
- Export after backend restart performs zero sampler calls, produces aligned AV output, and leaves original manifests/tensor hashes unchanged.
- GPU quality review compares movement, identity, audio continuity and seams; numerical reuse tests alone do not establish visual quality.

## Review conclusion and remaining proof obligations

The shared Clip/Sequence UI and separate execution paths are implementable with the existing Gradio/job architecture. The design now specifies prompt composition, source retention, operation identity, action mapping, storage authority and incremental dispatch rather than leaving these to UI callbacks.

Three runtime gates remain intentionally unproven: (1) safe read-only export and pre-sampling prefix enforcement through the pinned backend adapter; (2) generation and restart reuse with our exact Core/model profile for text and Ref2VA; (3) actual Modal checkpoint acknowledgement/cold-start behavior and measured resource limits. Source review cannot certify those. The first gate precedes substantial implementation; the other two precede release. If a gate fails, revise the adapter or supported profile explicitly rather than silently regenerate earlier chunks or disable the safety checks.
