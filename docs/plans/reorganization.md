# Reorganization plan

Status: phases 0–6 are done; Later lists what remains. Combines two code reviews.
One review covered feature ownership and the workspace, the other the cost of
adding a feature. Remove this file when the plan is finished.

## Why

The foundations are sound. Generation requests, graph builders, execution,
jobs and storage already live in separate modules, and almost no dead code
exists. The cost comes from how features are wired together:

- **One H3 option is spelled out in 11–15 files.** `build_fl2va_graph` and
  `build_ref2va_graph` take 60 parameters each. Their signatures are copied in
  `h3_ui/application.py` and `h3_app/generation/services.py`, called twice
  with 56 nearly identical lines in `construct_graph.py`, and passed through
  an 86-parameter `generate()`. The UI repeats every control name in a section
  dataclass, the ~325-field `H3View` and a hand-written name map in
  `h3_view.py`.
- **`application.py` (3,517 lines) is mostly a test seam.** The
  Prompt/Model/Media controllers receive service records filled with
  `application.py`'s own wrappers (the media record has ~75 entries) and are
  rebuilt on every call. A call goes `app.X` → `controller.X` →
  `services.Y` (= `app.Y`) → a new controller's `Y`, so `patch.object(app, …)`
  in 32 test files keeps working.
- **Copied code.** Model downloads repeat the check, download and verify steps
  in about 25 `ensure_*` functions, although a generic `ensure_model_keys`
  exists. Each one is wrapped again in `ModelController` and `application.py`.
  The Gemini generateContent call exists three times. The H3 Gemini/Lightning
  writer paths copy the general media paths.
- **Features reach into each other's private helpers.** Dub uses four private
  Gemini helpers from `prompt_service`. `audio_tools` and `dubbing/tools`
  import `_setup_lock` from `drama3_serving/environment`. Dub imports 7 Drama 3
  serving modules plus `_with_heartbeat` and `_blocking`. `dubbing_view`
  borrows step cards from `song_controls`. `fibo2_view` imports the private
  `_actions` and `_bind_generation`.
- **Mixed ownership.** `generation/optional_engines.py` runs FLUX, MOSS,
  Fibo and Drama 3 through one `isinstance` dispatcher. `prompt_service.py`
  (1,752 lines) mixes provider transport with every engine's writer and
  Drama 3 script parsing. `drama3_handlers` and `dubbing_handlers` receive
  their dependencies through a module-level `install()`.
- **Inconsistent UI building blocks.** FLUX/MOSS/Fibo share a run, stop and
  progress helper. The other seven views build their own, with differing
  labels ("Interrupt", "Interrupt voice generation", "Interrupt auto"). Saved
  projects sit in a collapsed accordion inside Jobs (`h3_ui/jobs_view.py`).

The Create task labels already read Video · Image · Song · Soundstage ·
Dub & Translate (`TASK_LABELS` in `h3_ui/layout.py`). The internal task IDs
(`TTS`, `Audio / Music`, `Dub`) are stored in browser preferences and keep
their values.

## Guardrails for every phase

- Unchanged: public `api_name` endpoints, the parameter order in
  `h3_app/contracts.py`, browser preference schema 6 with its task and engine
  IDs, the 15 baseline graph fixtures, and GPU-tuned code (`H3Acceleration`,
  the graph-building code inside `workflows/h3.py`).
- One commit per batch. Each batch passes the CPU suite (`python -m tests`),
  plus `--browser` when navigation or layout changes. Structural work gets no
  GPU validation.

## Phase 0: Cleanup (no behavior change)

- Delete unused definitions: `mix_parallel_tracks`,
  `DEFAULT_LOOP_DURATION_SECONDS`, `DRAMA3_SOUND_META_INPUT_NAMES`.
- Replace the 16 `"""Extracted X boundary."""` docstrings with real
  descriptions.
- `docs/`: fold the open items from the dated upstream audits into one
  current `upstream-status.md` and delete the audits. Delete the implemented
  FLUX and engine-update notes. Drop the date from the open Qwen Lightning
  drift investigation. Move the Continuum plan to `docs/plans/`. Update the
  boundary table in `architecture.md`.
- Move the `prompt_*.txt` files into `prompts/`. Modal mounts them from one
  `PROMPT_FILES` list, and a test keeps that list equal to the folder.
- Move the H3, engine and Media documentation out of the README into
  `docs/features/` (`h3.md`, `engines.md`, `media.md`), unchanged. The
  Soundstage and Dub guides move there too. The README keeps the overview,
  run, deploy and validation instructions.

## Phase 1: Shared services

- `h3_app/llm/gemini.py` and `lightning.py`: one upload, generateContent and
  error path. Dub stops using `prompt_service` private helpers.
- `h3_app/workers/env.py`: one module that provisions isolated environments
  and runs resident workers, used by SGLang, VoiceDesign/Stable Audio and Dub
  tools.
- Model provisioning: one `ensure(keys, label)` plus one "which keys does
  this choice need" function per engine.
- A small public Drama 3 serving interface for Dub.

## Phase 2: Feature pattern, proven on FLUX / MOSS / Fibo

- Split `optional_engines.py` into `flux3`, `moss_image` and `fibo2`. Drama 3
  generation moves to its own module.
- Extend each engine's entry in `h3_app/capabilities.py` instead of adding a
  second registry. Each entry records how to build its view with explicit
  dependencies, how to bind its events, which controls are saved, and its API
  names.
- Promote `_actions` to a public shared run/stop/progress component.
- Construct these engines' dependencies explicitly instead of through
  `install()` or `application.py`. Their tests patch the owning module.
- Done when adding an image engine takes one folder and one registry entry.

## Phase 3: H3 settings objects (independent, may run beside phase 2)

- Group the graph-builder parameters into frozen dataclasses (sampling, cache,
  VAE/decoder, refinement/upscale, memory), built once from `H3Request`. The
  baseline graph fixtures must stay identical.
- Derive the H3 view's control-name map from the section dataclasses.
- Split the 503-line `prepare_h3`.
- Done when a new H3 option touches at most 4 files.

## Phase 4: Move the remaining engines to the pattern

- LTX, Qwen, Music and YuE2 move into `h3_ui/engines/<engine>/`. Their
  `generate_*`/`enhance_*` adapters leave `application.py` and their writers
  leave `prompt_service.py`.
- Soundstage moves into `h3_app/drama3/` (with `serving/`) and
  `h3_ui/drama3/`. Split `drama3_handlers` into generation/takes,
  writer/cast, voice design, speaker library and sound library. Explicit
  construction replaces `install()`. Step cards move into a shared module.
- Dub & Translate gets the same treatment.
- H3 `sections/` and `events/` join the pattern, removing two import cycles.
- Done when `prompt_service.py` is gone and a boundary test rejects
  cross-feature private imports.

## Phase 5: `application.py` keeps only startup wiring and compatibility

- Tests build the Model, Media and Prompt controllers with fake services.
  Then remove the wrapper loop.
- Merge Media's legacy video-only functions into the media-typed ones and
  drop the tuple-reshaping adapters.
- Keep remaining public Python names in a small `h3_ui/compat.py`.
- Break the gallery_store / media_catalog / provenance cycle by moving the
  misplaced functions.
- Done when `application.py` is under ~800 lines and no controller refers
  back into it.

Outcome: the wrapper loop, the Media video-only adapters and the
gallery_store / media_catalog / provenance cycle are gone (provenance now
notifies `SNAPSHOT_LISTENERS`, which the media catalog joins). Tests import
from owning modules, so no `compat.py` was needed. `application.py` is 1,234
lines, not ~800: what remains is the ComfyUI transport, staging and graph
wrappers that tests patch as the effect seam, the full `generate`, backend
status and `build_ui`/`main`. Moving those means giving tests a fake runtime
instead of `patch.object(app, …)`, listed under Later.

## Phase 6: Workspace experience (separate UI track, after phase 2)

- Make Projects a top-level tab between Create and Media. It shows the
  existing saved requests, with no behavior change.
- Give every workspace the same layout: inputs, creation controls, results,
  refinement. Generate, Stop, progress and result actions share placement
  and wording. Song and Dub keep their guided steps; Video and Image keep the
  compact composer.
- Run focused browser checks for navigation and narrow viewports.

Outcome: Projects sits between Create and Media; Jobs keeps **Save as
project** beside the selected job, and a project run reports in Projects.
LTX, Qwen, Music 3 and YuE2 build their Generate/Interrupt/progress row with
`generation_actions`, as FLUX 3, MOSS-Image, Fibo 2.0 and Soundstage do, and
every stop button reads **Interrupt**. H3's composer keeps its own row (it
adds **Refresh status** and the shortcut hint), and Dub keeps its guided
steps with Markdown status.

## Later

- Move the ComfyUI transport, staging and graph wrappers out of
  `application.py` behind one runtime-effects object that tests replace,
  instead of patching module functions.
- One project model. Song takes (browser session), Dub sessions (private
  `session.json` folders), Drama 3 takes and saved requests (SQLite) persist
  differently. Making them editable multi-step projects needs its own
  design.
- Split `test_ui_contract`, `test_refactor_boundaries` and
  `test_engine_updates` by feature as each feature moves.

## Order

Phase 0 has no risk. Phase 1 builds what phases 2–5 depend on, so features
move once. Phase 2 proves the pattern on the newest, smallest engines. Phase 3
is the largest single saving and has no dependencies. Phase 5 becomes easy
only after phases 2–4 drain `application.py`. Phase 6 reuses phase 2's shared
action component.
