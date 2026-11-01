# Soundstage sound design: optimization and refactor plan

Date: 2026-11-01. Scope: scene-based sound and music design in Advanced mode, for both Single-pass and Guided writers and the shared renderer. Builds on `soundstage-production-research.md` and `soundstage-writer-audit.md`; those documents' implemented items (anchors, roles/perspective, beds with spans and resume, holds, sidechain ducking) are assumed.

## Implementation status (2026-11-01)

Implemented: Phase 0, Phase 1, Phase 2 items 1–4, Phase 3, Phase 4, and Phase 6 items 1, 3 and 4. Tests are in `tests/test_drama3_sound_design.py`, and the user docs are in `docs/features/drama3.md` § Scene sound design.

- **Phase 0:** `h3_app/drama3/sound_report.py` builds the density report. Every Advanced rewrite appends its one-line summary to the writer status. Writer drop notes keep appearing in the existing status notes. The baseline numbers are not recorded yet, because they need real writer runs.
- **Phase 2 item 5 (during without a phrase):** only the prompt changed; it now requires a phrase for `during`. The compiler keeps its midpoint fallback rather than downgrading to `with`, because the fallback still lands inside the line.
- **Phase 6 item 2:** implemented as a test that both prompts document the shared vocabulary (duration classes, roles, levels, transitions, `lead_in`/`tail`), not as prompt templating.

Deferred:
- **Phase 2 item 6 (spatial-wording lint):** waits for fresh output to show whether it is needed.
- **Phase 5 (density UI):** comes after the density gains are measured.

Not measured: real model behavior and audio. The next step is to run one dialogue story and one action story through both writers, and compare the status density lines with a listening pass.

## Problem summary

The prompts are well structured and the renderer is mature. The remaining gaps are where the writer cannot express a choice, where information is not shared between stages, and an overall bias toward too few sounds compared with a human editor.

| # | Problem | Evidence | Where |
| --- | --- | --- | --- |
| 1 | Too few effects compared with a human editor | About a dozen restraint clauses and almost no positive pressure; "a short scene usually needs zero to four"; the quiet-plan example is shown first; "do not invent actions" is read as "only explicitly narrated sounds" | `prompts/prompt_drama3_guided_sound.txt`, `prompts/prompt_drama3.txt` §Scene design |
| 2 | Event cap truncates dense scenes | `event_budget = min(16, max(4, (lines + 3) // 4))` allows 5 events in a 20-line scene | `h3_app/drama3/guided/compile.py:220` |
| 3 | Dialogue scenes often play dry | "room is null for ordinary conversations" and optional ambience; no room-tone concept | both sound prompts |
| 4 | Guided has no level control | Fixed volumes: ambience 0.4, music 0.2, effect 1.0, stinger 0.4. A blizzard and a hushed chamber play at the same level | `compile.py:33` `SOUND_SETTINGS` |
| 5 | Score has no story-wide identity | Sound batches run in parallel and cannot see each other; each one invents its own music | `drama3_guided.py:404` |
| 6 | Every scene transition is identical | Fixed 0.8 s `SCENE_PAUSE`, 1 s bed tail and 0.8 s lead-in; no establishing moment or sound bridge | `compile.py:26`, `serving/audio.py:654-656` |
| 7 | Durations are coarse | Guided effects default to 4 s; only accent/reaction/texture classes exist | `compile.py:33`, `sound_policy.py` |
| 8 | Sound knowledge is duplicated and drifts | Vocabulary, ranges and rules are written separately in two prompts, in `compile.py` and in `sound_policy.py` | prompts, compile, sound_policy |
| 9 | Unverified: space baked into prompts | Older outputs showed "echoing in a vast cavern" on effects that also got a room and `distance="far"`. Needs a fresh run to confirm | — |

## Principles

- Keep the hard guardrails: no invented plot events or intelligible dialogue, no laugh-track audience, no duplicate of a sound a character already performs, dry foreground assets, one coherent acoustic space.
- Change behavior through controls the renderer consumes, not more descriptive prose.
- Both writers target the same controls. Single-pass stays one call; Guided keeps its stages.
- Basic mode and legacy scripts render unchanged.
- No local GPU runs. Verification is text-level stats, unit tests, and user-run listening checks.

## Phase 0: Measurement first

Add `sound_density_report(script) -> dict` beside `h3_app/drama3/lint.py`. It is pure text analysis of a compiled Advanced script and reports, per scene:
- speech lines and events
- events per line, and events per 100 spoken words (per 200 characters for CJK), because a long Chinese narrator line is not comparable to a short English reply
- whether a bed or room is present
- counts by role and perspective
- unique assets to generate, the cost that rises with density

The script alone can't show what the writer dropped, so count drop and truncation notes separately from the writer's notes list. Surface the totals in the writer's status notes (one line) so every run shows its density. Before changing prompts, record a baseline on two or three stories, including one dialogue-heavy story and one action story, through both writers.

Done when the report runs in tests on fixture scripts and baseline numbers are recorded in this document.

## Phase 1: Sound density rebalance (problems 1–3)

Prompt changes in both writers. Wording must stay parallel between the two.

1. **Plot events vs. implied sounds.** Replace the "Do not invent actions…" line with a two-part rule. Inventing plot events, people or consequential actions is forbidden. Rendering the physical sounds the scene already implies is expected: entrances and exits, footsteps on movement, objects handled or mentioned in use, sitting and standing, doors on arrival, weather the setting establishes, weapons drawn when the dialogue implies it.
2. **Editor's pass.** For each scene, consider in order:
   - bed or room tone
   - entrances, exits and movement
   - handled objects
   - environmental change
   - impacts and reveals
   - transition in and out
   
   Then cut whatever masks speech. This replaces the restraint-first framing. "One foreground focus at a time" stays as a mixing rule, not a count limit.
3. **Density by scene type** replaces "zero to four":
   - intimate or emotional dialogue: a bed plus about 1 cue per 4–6 lines
   - ordinary dialogue with activity: about 1 per 2–4 lines
   - action, chase or fight: about 1 per 1–2 lines, with layered impacts allowed
   - abstract, narration-only or dream: a deliberate choice, may be sparse

   Present these as typical editor practice, not quotas. Risk: smaller models take numbers literally and may decorate every line. The audit removed earlier quotas for this reason, so watch the Phase 0 upper end as well as the lower end. If action scenes exceed about 1 event per line, or intimate scenes exceed their range, drop the numbers and keep only the qualitative scene-type contrast.
4. **Example order.** Move the quiet-plan sentence after the main schema, and add one dense action example next to the existing timing examples. Keep the existing examples.

Code changes:
- `compile.py` event budget: `min(32, max(6, (lines + 1) // 2))`. Move the constants into `sound_policy.py` first (Phase 6.1), as part of this phase.
- Keep the one-comic-accent rule and `_drop_performed_sounds` unchanged.

Tests: update the `tests/test_drama3_writer_guidance.py` assertions that pin prompt wording. Add a compile test where a 20-line scene keeps 10 events.

Done when Phase 0 stats on the baseline stories show higher events per line in action and ordinary scenes, intimate scenes stay sparse, and no guardrail regressions appear in the notes (performed duplicates, invented crowds).

## Phase 2: Levels and asset hygiene (problems 4, 7, 9)

1. **Level control (Guided).** Add optional `level: low|mid|high` on the scene's `ambience` and `music` selections (object form `{"name": …, "level": …}`, string form still accepted) and on events. Map levels in `sound_policy.py`:

   | Kind | low | mid | high |
   | --- | --- | --- | --- |
   | ambience | 0.2 | 0.4 | 0.55 |
   | music | 0.12 | 0.2 | 0.28 |
   | effect | 0.6 | 1.0 | 1.3 |
   | stinger | 0.25 | 0.4 | 0.55 |

   These are relative to the mixer's speech-leveled gain, not absolute loudness; the parser already clamps volume to 0–2. Effect `high` raises gain only; the existing one-shot ducking still applies. Level belongs to the use, not the asset: the same asset can be quiet in one scene and loud in another, so `sound_line` takes the level as a parameter. This touches `Staging`/`SoundEvent` in `model.py`, `parse_staging` in `schema.py` and `sound_line` in `compile.py`. Single-pass already has `volume`; align its documented ranges with this table.
2. **Room tone by default (moved from Phase 1, because it needs `level: low`).** Any scene with a physical setting gets a low ambience or room-tone bed unless it is deliberately a void; a dry scene remains valid when deliberate. Reuse one room-tone asset per location so asset count doesn't grow with scene count.
3. **Durations.** Add a duration class `event` (6 s) for sustained actions such as a swarm, a passing cart or a scuffle, and change the Guided stinger default to 5 s. Both prompts get one line on when to use it.
4. **Loop hygiene.** Prompt line for both writers: loop beds are steady textures without distinctive one-off events, because a roar inside a 25-second loop repeats audibly. Recurring sounds such as a howl or a bell become separate events.
5. **During without a phrase.** Guided `during` without `phrase` currently uses `expected_seconds × 0.5`. Change the prompt to require a phrase for `during`; in the compiler, downgrade a phrase-less `during` to `with` and add a note.
6. **Spatial-wording lint (conditional on problem 9).** After Phase 1, run the stats on fresh output. If in-world effects still carry room or distance words while the code applies a room, `distance` or a moving pan, add a lint in `lint.py`. It flags those phrases (echoing, reverberant, in a vast …, distant, far away, approaching, receding) and either strips them or writes a note. Skip this item if fresh output is clean.

## Phase 3: Score palette and story-wide sound identity (problem 5)

1. **Palette from the content stage.** Extend the Guided content answer (both preserve and brief) with optional `sound_palette`: genre/era, score instrument family and mood range, and an overall density lean, plus two to four story-wide music assets, each with a complete `name`, `kind` and `prompt` (for example `main_theme`, `tension_bed`). Parse it in `parse_content_preserve`/`parse_content_brief`, and inject it into every sound batch request next to `sound_context`.
2. **Recurring assets.** Each batch may select palette assets by name without redefining them. Naming alone is not enough: `_merge_batch_sounds` keeps a name only when the whole `Sound` is identical, and renames it to `main_theme_2` when a batch writes a slightly different prompt, which breaks continuity. So palette assets are authoritative. They are pre-seeded into the merged sounds, and a batch's redefinition of a palette name is replaced by the palette version with a note rather than renamed. The renderer then resumes the reused loop name across scenes.
3. **Prefer music spans.** Prompt guidance: dialogue-heavy scenes use anchored music spans at turns rather than a full-scene drone. Full-scene music beds are for montage, narration-led or action scenes.
4. **Single-pass.** It already sees the whole story. Add one line asking it to define a consistent score palette privately and reuse music asset names across scenes.

Done when a multi-batch run produces a shared music naming and instrumentation across batches; this is visible in the merged sounds and in the absence of rename notes for music.

## Phase 4: Scene transitions (problem 6)

1. **Renderer.**
   - `<parallel>` accepts optional `lead_in="0.8–4"` and `tail="1–4"` attributes, used in `mix_parallel` instead of the constants when present. They need parsing on `ParallelNode` and serializing back in `serving/tree.py`, which round-trips scripts, so that saved and edited scripts keep them. The writer formatting and lint must also accept them.
   - A top-level pause between scenes stays optional.
   - The outgoing tail already overlaps what follows through `advance`, so a longer tail plus a short or absent gap produces a bridge.
2. **Guided.** Scene staging gets optional `transition: cut|establish|bridge`, mapped in the compiler:
   - `cut`: today's behavior.
   - `establish`: `lead_in` about 2.5 s, for a new location or the opening scene.
   - `bridge`: the previous scene's `tail` about 3 s, scene pause reduced to 0.3 s, and an optional transition-role sound placed at the boundary.
3. **Single-pass.** Document `lead_in`/`tail` in the scene-design section and the transition example.

Tests: `tests/test_drama3_mix_details.py` covers lead-in and tail lengths. A compile test covers each transition type. A tree round-trip test keeps the new attributes, and `tests/test_drama3_examples.py` still parses the updated Single-pass example.

Done when the opening scene gets an audible establishing bed before the first line, and a bridge transition carries the outgoing bed under the next scene's opening without a gap or click.

## Phase 5: Optional density setting

A Sparse / Standard / Rich selector in Advanced mode, next to the script mode in `h3_ui/drama3/writer.py`. It is passed through `sound_context` as one instruction line that scales Phase 1's density table (about 0.5×, 1×, 1.5×) and the event budget. Standard is the default and behaves as Phase 1. This touches the UI layout, the snapshot preload contract and `tests/test_ui_contract.py`, so it is a separate change after Phases 1–4 prove out.

## Phase 6: Refactor (problem 8)

Do these alongside the phases above, not as one large refactor at the end.

1. **One source of sound policy.** Move into `h3_app/drama3/sound_policy.py`:
   - `SOUND_SETTINGS`, the level table, duration classes (with `event`) and the event-budget formula
   - transition presets
   - the scene-type density table

   `compile.py`, the Single-pass parser and lint all import from there.
2. **Shared prompt vocabulary.** Both prompts restate roles, perspectives, rooms, vocal effects, duration classes and levels. Add a small placeholder substitution at prompt load (`{sound_vocabulary}`, rendered from `sound_policy.py`, using `str.replace`, not `str.format`, because the prompts contain literal JSON braces) so the vocabulary list is generated once. Each prompt keeps its own syntax: JSON fields for Guided, attributes for Single-pass. If templating is judged too invasive, add a test instead that every vocabulary item in `sound_policy.py` appears in both prompts.
3. **Structured beats.** Beats are currently JSON folded into `Scene.change`. Give `Scene` a `beats` tuple, render it explicitly in `_scene_listing`, and let the sound prompt reference beat intents directly.
4. **Thin `compile_script`.** The Advanced branch is about 150 lines. Extract `_scene_events` (budget, matching, local cues) and `_scene_master` (loops, room, stage, transition) so the Phase 2 and Phase 4 additions don't grow one function further.

## Order and size

| Order | Phase | Size | Risk |
| --- | --- | --- | --- |
| 1 | Phase 0 stats | S | none |
| 2 | Phase 6.1 constants move, then Phase 1 density | M | prompt behavior; verify with stats |
| 3 | Phase 2 levels, room tone, durations, during | M | schema change, backward-compatible |
| 4 | Phase 3 palette | M | content-stage schema; optional field; merge change |
| 5 | Phase 4 transitions | M | renderer and tree round-trip; covered by tests |
| 6 | Phase 6.2–6.4 refactors | M | mechanical; full test suite |
| 7 | Phase 2.6 lint (if needed), Phase 5 density UI | S / M | — |

## Cost and runtime

Higher density, room tone in every scene and palette assets all mean more generated sound assets, and more phrase anchors mean more alignment work. Reuse keeps this bounded: one room tone per location, palette music shared across scenes, and generic foley (footsteps, door) reused by name. The Phase 0 report tracks unique assets per story, so a density gain that doubles generation time is visible before it ships. The existing skip-existing-sounds option still avoids regenerating saved assets.

## Verification limits

Unit tests cover parsing, compilation and mixing. Density and palette effects need real writer runs (user-triggered LLM calls), measured with the Phase 0 report. Audible quality (levels, transitions, loop repetition) needs a short listening pass on one dialogue story and one action story after Phases 2 and 4. No local GPU generation is part of this plan.
