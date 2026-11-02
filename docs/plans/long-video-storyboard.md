# Long video storyboard plan

Date: 2026-11-02. Status: proposed; no runtime changes or GPU validation performed.

## Decision and scope

Add a **Storyboard** task to Create, placed before Soundstage: Video, Image, Song, Storyboard, Soundstage, Dub & Translate. Users build a storyboard of scenes, generate them one by one or all at once, keep every take, regenerate any scene, and export one video. The task hides H3 model settings: each storyboard copies a settings profile from MiniMax H3 once and keeps it fixed.

The old saved-request **Projects** tab has been removed (2026-11-02), so storyboards are the only saved projects in the workspace.

The MiniMax H3 composer is unchanged. This is not the reverted Clip/Sequence switch (commit `224fbfb`): it does not live inside the H3 composer and does not depend on H3 Continuum.

Continuation uses native ComfyUI per-stream AV noise masks, as in [ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream) (MIT, reviewed at `775be90`) and [Herrgott's H3 Infinite Continuation Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite). We take only the small masking logic and do not install either node suite. Their Clip Bin, gallery, lineage tree and timeline web UI duplicate what the Storyboard task and Media already do.

Design mockups: [Long Video Storyboard canvas](https://claude.ai/artifact/KtmFpdQ22k14SyvGHyijox) (storyboard screen, new storyboard, take change / queue / export; drawn as a Video engine tab before the placement moved to its own task).

## Concepts

- **Storyboard**: a named, owner-scoped document with a fixed settings profile, a cast, ordered scenes and saved versions. The UI may call it a project.
- Creating a storyboard is the user's explicit choice to keep its prompts and cast pictures on the server. The workspace otherwise never stores request content, and since the Projects removal jobs no longer keep it across a restart, so the storyboard itself must hold every scene's full request for regeneration.
- **Scene**: prompt, length (5, 10 or 15 s), cast selection, link to the previous scene, optional seed/overlap/takes-per-run, and its takes.
- **Link**: `flow` continues from the previous scene's take in use through the masked overlap. `cut` starts a fresh shot that only shares the cast references. The first scene has no link.
- **Take**: one generated result for a scene, with its MP4, its saved latents and the exact inputs it was made from. The **take in use** is the one later scenes continue from and the one export uses.
- **Outdated**: a done scene whose inputs no longer match what its take in use was made from. Outdated scenes keep their takes and can still be played or exported.

### Outdated rule

A scene's take records the take it continued from (for flow), its prompt, length, cast and settings profile. A scene is outdated when any of those differ from the scene's current values. In practice:

- Changing the take in use, link, prompt, length or cast of scene *n* outdates *n* (when the change touches *n*'s own inputs) and every following scene connected by an unbroken run of `flow` links. The run stops at the first `cut`.
- Applying a new settings profile outdates every scene.
- Computing this from recorded inputs, not from events, keeps it correct after undo, reorder and restart.

## Settings profile

At creation the user picks **Current MiniMax H3 settings** or a saved H3 recipe. The profile is a frozen copy of the H3 request fields that affect generation: base model, Normal/Turbo and Turbo LoRA, steps and scheduler, size, text encoder, video encoder/decoder, audio VAE, latent upscale model/precision/refine steps, attention backend, Spectrum and sampling acceleration, conditioning reuse, offload.

- Store the profile as the `H3SamplingInputs`, `H3OutputInputs` and `H3FinishingInputs` groups of `H3Request` (`h3_app/generation/requests.py`), without the per-scene fields (duration, seed, result format, image frames). Each scene supplies its own `H3MediaInputs` and duration, and `H3Request.from_values` combines them, so the profile and an H3 job agree field for field. Do not duplicate defaults.
- Fields that change latent geometry (model, size, VAEs, latent upscale on/off) are what make flow possible. Speed-only fields (attention, Spectrum, offload) could change per run, but v1 keeps the whole profile fixed for simplicity.
- **Apply to storyboard** replaces the profile with the current H3 settings and outdates all scenes after a confirmation that names how many scenes are affected.
- Excluded from the profile: conditioning mode, first/last frames, reference video/audio, prompt, seconds and variations. The tab owns those per scene.

## Continuation engine

### Masking node

Add `H3ContinuationPrefix` to `custom_nodes/H3Acceleration/__init__.py`, based on PrefixStream's `pipeline/native_masked_av.py` with attribution in the module docstring:

- Inputs: target AV latent, source AV latent (the previous take's), overlap frames (22/39/56, snapped to the H3 `17k+5` grid), audio handover mode.
- Picks the phase-aligned source tail, copies video steps and 40 Hz audio ticks into the head of the target, and sets `noise_mask` as a nested `(video, audio)` mask with zeros on the protected head. This is the same mask form `H3RegeneratePrepareLatent` already uses.
- Records `prefix_frames` on the latent for trimming.
- Batch size 1 only; videos per batch is not exposed in Storyboard.

The pinned ComfyUI (`0df1646`) already contains the per-token video/audio mask support from Comfy-Org/ComfyUI#15375 (merged as `ff6c8a8`). No ComfyUI or model patches are needed. Keyframes inside the protected head are dropped from conditioning, as PrefixStream does.

Leave out PrefixStream's experimental soft context denoise, reference noise and continuation-only LoRA for v1.

### Saved latents

Every take saves its final AV latent as safetensors next to its MP4 (about 13 MB at 720p for 5 s, about 37 MB for 15 s, in fp32; audio is negligible). When latent upscale is on, also save the stage-one low-resolution latent. Takes made without saved latents cannot be continued from.

### Graph changes

Build the scene graph with the existing H3 workflow builder (`h3_app/workflows/h3.py`) and the frozen profile, adding the prefix step only for `flow` scenes:

- **Single stage**: prefix node between the empty AV latent and `SamplerCustomAdvanced`.
- **Latent upscale**: prefix node on the low-resolution latent with the previous take's stage-one latent. Then apply the prefix node again on the combined latent, with the previous take's full-resolution latent, before refinement. Without this, refinement re-noises the overlap and the seam drifts. It is also required for correctness: the stage-one sampler output keeps its low-resolution `noise_mask`, `H3SeparateAVLatent` and `H3CombineAVLatent` copy the latent dict, and a stale mask of the wrong size would reach refinement. The second prefix node always replaces `noise_mask`. The stage-one context comes from the previous take before refinement, so the low-resolution head can differ slightly from the full-resolution head pasted later; the refinement pass covers that.
- Decode, then trim the overlap from the new take's frames and audio before saving the take's playable MP4 section. Keep the untrimmed latent.
- Turbo LoRAs (including Larry's `MiniMaxH3TurboSampler`, a standard `KSAMPLER`), Qwen conditioning reuse, SLA/Veda/Sol attention, VAE choices and stage offload need no change. Spectrum v0.2.29 has masked-sampling tests upstream but must be checked on a real run.

`cut` scenes and the first scene run the normal H3 graph with the cast as reference images.

## Generation and jobs

- Each take is one job in the existing job system, visible in Jobs and owned like other jobs. Takes per run (1–4) submits that many jobs with independent seeds.
- `JobCoordinator` runs one GPU job at a time (`with self.gpu`), and an accepted job resolves its inputs when it starts. A `flow` job submitted early would start without its predecessor's latent. So a storyboard runner submits the next scene only after the previous one's take is saved and in use. It doesn't queue the whole storyboard as jobs up front.
- **Generate remaining** walks draft and outdated scenes in order. Scenes after a `cut` don't wait for an earlier flow run to finish, so the runner can move a ready cut scene ahead while a flow scene waits for the user to pick a take. Show waiting reasons in the queue panel.
- The runner's queue is saved in the storyboard. After a restart it is paused, not resumed, matching the workspace rule that nothing is resubmitted automatically. The user presses Generate remaining again.
- **Stop after this scene** lets the running job finish and clears the rest of the queue.
- When a take finishes for a scene that has no take in use, it becomes the take in use. Otherwise the user picks.
- Changing the take in use asks once: **Use and regenerate the following flow scenes**, **Use, regenerate later**, or **Cancel**. The dialog lists which scenes go outdated and which stay because of a cut.

## Storage

Bump `WorkspaceStore` `SCHEMA_VERSION` to 2 with a migration that only adds tables:

- `storyboards(id, owner, name, created, updated, payload)`: profile, cast, scene order and scene fields as JSON.
- `storyboard_takes(id, storyboard_id, scene_id, created, job_id, payload)`: media paths, latent paths, recorded inputs and seed.
- `storyboard_versions(id, storyboard_id, created, label, payload)`: snapshot of scene order, links and the take in use per scene. Export and **Save version** create one; opening one branches into a new storyboard.

Deleting a storyboard deletes its takes and latents. Takes stay out of Media; exports go into Media with a link back to the storyboard. H3 outputs currently enter Media automatically: `write_snapshot` calls `SNAPSHOT_LISTENERS`, which includes `register_created_media`. Take snapshots therefore need a marker, such as `storyboard_take: true`, that the Media listener skips. They also need their own output folder so **Scan historical media** doesn't pick them up.

## UI

New `h3_ui/storyboard/` package following the `h3_ui/drama3/` layout (`view.py`, `bindings.py`, `handlers.py`). Register it as a new Create task, placed before Soundstage:

- `h3_app/capabilities.py`: add `"Storyboard"` to `TASKS` before `"TTS"`, and an `EngineCapability("storyboard", "Storyboard", ("Storyboard",), …)` before `drama3`.
- `h3_ui/layout.py`: an engine tab with id `storyboard` before the Soundstage tab, and `FORMAT_FOR_TASK["Storyboard"] = "Video"`. The task label is already the plain name, so `TASK_LABELS` needs no entry.
- Saved task preferences keep working, because tasks are stored by value and the new one is only inserted.

- **Header**: storyboard dropdown, New, Duplicate, and a one-line settings profile summary with Change.
- **Storyboard strip**: scene cards in a horizontally scrolling row with thumbnail of the take in use, number, length and state; a Flow/Cut toggle between cards; Add scene at the end. Built with `gr.HTML` plus small JS bindings for selection, following the existing queue/transport JS pattern. Plain Gradio components hold all state.
- **Scene editor**: prompt `gr.Textbox`, length `gr.Slider` (5/10/15), Flow/Cut `gr.Radio`, cast `gr.CheckboxGroup`, a Scene options accordion (overlap, seed, takes per run), and Generate / Duplicate / Move / Delete.
- **Takes**: `gr.Video` for the selected take and a `gr.Gallery` of takes with Use this take.
- **Full video**: a timeline strip with one segment per scene sized by length and marked at cuts, plus Play all (concatenated preview built on demand), Generate remaining and Export.
- **New storyboard**: name, start mode (split a story, write scenes, continue a video from Media), story text, target and usual scene length, prompt writer (defaults to the H3 one), cast pictures, settings profile table, drafted scene list, Create storyboard.
- **Export**: flow joins (seamless trim or short crossfade), cut joins (hard cut or dip to black), audio at cuts, optional SeedVR2 finish, include outdated scenes, Export to Media.

"Story to scenes" reuses the existing prompt writer services with a new prompt in `prompts/` that returns scene title, prompt, length and link. "Continue a video from Media" makes a one-scene storyboard from a Media H3 item that has a saved latent, and only works if its settings can become the profile.

## Phases

1. **Engine**: prefix node, latent saving, graph builder for single-stage and latent-upscale `flow` scenes, trim. Unit tests for tail selection, grid snapping, audio tick alignment, the nested mask, and replacing a stale mask after upscale, using CPU tensors with the real H3 shapes.
2. **Data**: schema v2 tables, storyboard service with the outdated rule, versions and the runner queue. Unit tests for outdated propagation across flow runs and cuts, reorder, profile apply, branching, runner ordering, and the queue pausing after a restart.
3. **Task**: Storyboard task and tab, scene editor, takes, queue, export, Media skip for takes. Wire to jobs. UI contract tests for the task order and tab ids, in the style of the existing Gradio tests.
4. **Writer and extras**: story to scenes, continue from Media, versions panel.
5. **Validation on Modal (only when asked)**: a three-scene storyboard (flow, flow, cut) with Turbo + latent upscale + Spectrum. Check the seams frame by frame, check audio continuity, and compare flow scenes against the same scene generated as a cut. Record texture drift across at least five chained flow scenes, the known risk PrefixStream's own notes describe.

## Open questions and risks

- **Drift over long flow chains**: copied latents carry earlier artifacts forward. Measure before promising long unbroken flows. Encouraging cuts every few scenes, which storyboards do anyway, is the mitigation.
- **Overlap default**: 39 frames (about 1.6 s) is PrefixStream's default. 22 frames leaves more new content per scene but may weaken continuity. Decide after phase 5.
- **Scene length limit**: H3's 15 s maximum per clip applies per scene; the overlap counts against it, so a 10 s flow scene with a 39-frame overlap adds about 8.4 s of new footage. The UI should show new seconds, not raw clip length, in totals.
- **Disk use**: dozens of takes at 1080p with saved latents reach gigabytes. Show storyboard size and offer to delete unused takes.
- **Profile capture**: if H3 request construction can't produce a profile without a full composer state, add a small adapter rather than copying defaults.
