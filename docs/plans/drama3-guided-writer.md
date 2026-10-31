# Drama 3 Guided writer: multi-step scripts for small models

Date: 2026-10-31. Status: proposed design; nothing implemented.

## Why

The Single-pass writer sends `prompts/prompt_drama3.txt` (about 6,800 words) and asks one call to read the source, cast it, write or preserve every line, direct each line, design the sound, schedule it in nested `<parallel>`/`<sequential>` blocks and escape the whole script inside JSON. Strong models manage it. On gemini-3.5-flash-lite, 25–45% of runs failed per batch of 16 across 48 test runs, and the failures were structural, not stylistic: narration lines with no speaker, sounds closed with `</sequential>`, invented `<span>`/`<shivering>` tags, unescaped JSON, empty voice previews, narration given to characters, English briefs written in Chinese.

The Guided writer splits that job into small stages. Each stage gets a short prompt with only the rules it needs, answers in a narrow JSON shape that refers to IDs instead of repeating text, and leaves every number, tag and nesting decision to code. Code compiles the answers into today's script format, so playback, the cast and sound slots, and the lint stay unchanged.

Single-pass stays the default and keeps `prompt_drama3.txt`. Guided is a second workflow beside it, not a replacement.

## Principles

1. **The model decides, code writes.** No stage outputs Drama 3 markup. Models pick speakers, words, cue wording, sound categories and anchor IDs; code places cues, chooses volumes, durations and delays, and writes every tag.
2. **Refer, do not repeat.** After text is fixed, later stages cite utterance IDs and exact phrases; they never copy a line, so they cannot drop or reword it.
3. **One concern per call.** Attribution, writing, voices, direction and sound are separate prompts of roughly 1–1.5k tokens each.
4. **Validate every stage, repair narrowly.** Each answer is checked in code before the next stage. A failure gets one retry that names the exact problem and the valid options; optional work (cues, sounds) that still fails is dropped with a note, required work (cast, attribution) fails the run with the user's text kept.
5. **Same output contract.** The result is the script, cast, sounds and title that `enhance_drama3_text` returns today, so the UI, the generator and saved drafts need no changes beyond the workflow switch.

## Workflow

```
User text + voice slots + speaker mode + script mode + saved sounds
        │
  S0  Analyze (code)        language, input kind, numbered source units
        │
  S1  Content (LLM)         preserve: who speaks each unit, cast, scenes
        │                   brief:    outline, cast, plain lines per scene
     gate: cast, coverage, speaker mode          ── text is now locked
        │
  ┌─────┼──────────────┐
  S2 Voices (LLM)   S3 Direction (LLM)   S4 Sound staging (LLM, Advanced)
  per new speaker   per scene batch      per scene batch
  └─────┼──────────────┘
        │
  S5  Compile + validate (code)   script, cast, sounds, title, notes
```

S2, S3 and S4 depend only on the locked S1 result, so they run concurrently. Typical cost: preserved prose in Basic mode is 3 calls in 2 rounds; Advanced adds S4 in the same round; a brief adds an outline call for longer pieces. Long inputs batch S3/S4 by scene groups of about 40 utterances.

Speaker mode Single skips S1's attribution (everything is speaker 0 or the one slot) and S2 when the slot has a reference voice.

## S0 Analyze (code)

- **Language**: `drama3_language_instruction` (already added) names the language; every prompt states it.
- **Input kind**: labelled script (most non-empty lines look like `Name: …` or `Name（cue）：…`), prose (quotes or several sentences of narrative), or brief (a short description: "two friends arguing about pizza, 30 seconds"). Code computes a guess; S1 confirms it with one field, because brief vs. short prose is ambiguous.
- **Source units** for labelled scripts and prose. Split into paragraphs, then at quotation boundaries (`“”`, `""`, `「」`, `『』`), then long narration at sentence ends so each unit is at most two sentences. Each unit gets `u1…uN`, its exact text, and a kind:
  - `quote`: words inside quotation marks (marks removed when compiled);
  - `narration`: everything else in prose, defaulting to the narrator;
  - `line`: a labelled script line, with its label and any parenthesized cue kept as a hint.

  `“Chen,” the teacher said, waving, “come out.”` becomes `u4 quote "Chen,"`, `u5 narration "the teacher said, waving,"`, `u6 quote "come out."`. Code, not the model, decides the boundaries, so source coverage holds by construction.
- Headings at the top become the title and the narrator's first unit, as the Single-pass rules require.

## S1 Content

### S1-P: preserved text (labelled script or prose)

The model only answers who speaks what and where scenes change. Input: the numbered units (kind shown), voice slots, speaker mode. Output:

```json
{
  "kind": "preserve",
  "cast": [
    {"name": "Tom", "aliases": ["he"], "slot": null},
    {"name": "Nora", "aliases": [], "slot": 1}
  ],
  "speakers": {"u2": "Tom", "u4": "Nora", "u6": "Nora", "u9": "Tom"},
  "scenes": [
    {"from": "u1", "place": "small flat kitchen at night", "change": "a calm evening turns to fear when the bell rings twice"}
  ],
  "fixes": [{"unit": "u7", "from": "以经", "to": "已经"}]
}
```

- `speakers` lists only units not spoken by the narrator: every `quote` unit, plus any narration the model judges to be unquoted speech. Units it omits are the narrator's.
- Labelled `line` units are pre-filled from their labels; the model only merges aliases into canonical names.
- `fixes` are optional typo and homophone corrections. Code accepts a fix only when `from` occurs exactly once in that unit and the change is small (at most 2 characters for CJK, one word otherwise).
- `scenes` start at a unit ID, give the place and one line on what changes. S3 and S4 use them; they never add text.

Gate: every name exists in `cast`, slots obey the speaker mode and reserved IDs, every `quote` unit has a speaker, fixes validate. Retry once with the specific unit IDs that are missing or wrong.

### S1-B: briefs

The model writes the piece as plain lines. For a target up to about two minutes, one call; longer pieces first get an outline call (scenes with place, who wants what, and the turn), then one call per scene with the outline and the previous scene's last lines.

```json
{
  "kind": "brief",
  "title": "Museum Night Shift",
  "cast": [{"name": "Arthur", "slot": null}, {"name": "Maya", "slot": null}],
  "scenes": [{
    "place": "Egyptian wing of a museum after closing",
    "change": "routine patrol turns to dread when something moves behind the glass",
    "lines": [
      {"speaker": "narrator", "text": "The last visitor left an hour ago."},
      {"speaker": "Maya", "text": "Did you hear that?"},
      {"speaker": "Arthur", "text": "Probably the vents. They do that at night."},
      {"speaker": "Maya", "text": "Wait.", "type": "reaction"},
      {"speaker": "Arthur", "text": "I said it's probably—", "type": "line"},
      {"speaker": "Maya", "text": "Look at the sarcophagus.", "type": "interrupt"}
    ]
  }]
}
```

- Lines contain only spoken words: no brackets, tags, quotes or stage directions (code rejects lines that do).
- `type` is `line` (default), `reaction` (one to three words heard under the previous speaker's line) or `interrupt` (cuts in as the previous line ends; that line ends with an em dash). Reactions and interruptions are content, so they are decided here, not added later.
- The prompt carries the dramatic guidance from the immersion discussion, in short form: a listener who cannot see the scene, each scene has a turn, characters react to what they hear before explaining it, quiet before reveals. It names the target language explicitly.

Gate: speakers exist, lines are non-empty plain text, reactions are short and follow a different speaker, interrupts follow a line ending in a dash. After the gate, utterances are numbered `u1…` and locked.

## S2 Voices

One call for all speakers without a reference voice; speakers with a reference keep their slot name and description. Input: name, role summary from S1, a few of the character's lines, language. Output per speaker: `description` (must open with "A female voice" / "A male voice" / …, in English) and `preview_text` (two to four calm sentences in the script's language, no controls). Code checks the sex phrase, length limits and forbidden characters; failures retry once for that speaker only.

## S3 Direction

Per scene batch. Input: scene place and change, the cast, and the locked utterances with IDs. Output:

```json
{"lines": [
  {"id": "u2", "cue": "to her, half joking", "stress": "vents"},
  {"id": "u4", "cue": "frozen, barely breathing", "event": "gasps"},
  {"id": "u7", "cue": "hushed, urgent", "phrase": {"text": "behind me", "tag": "slow"}}
]}
```

- `cue`: at most five plain-English words, briefing feeling plus intent or addressee. Code inserts it with `insert_cue`, so the model never chooses a position. A cue that is a bare repeat of the previous line's cue is dropped with a note.
- `stress` / `phrase`: an exact substring of that utterance to wrap in one of the seven scoped tags (`stress` defaults to `<emphasis>`). Skipped if it occurs zero or several times.
- `event`: one of the audible events in `planner.AUDIBLE_EVENTS`, appended at the end of the line.
- Lines left out stay plain. Narrator lines get restrained cues; the prompt says so.

This replaces the existing direction pass in Guided mode.

## S4 Sound staging (Advanced only)

Per scene batch. Input: scene place and change, the utterances (IDs and text), saved sound names. The model chooses categories and anchors; code chooses every number.

```json
{
  "sounds": [
    {"name": "museum_hum", "kind": "ambience", "prompt": "Low ventilation hum in a vast empty stone gallery at night"},
    {"name": "stone_scrape", "kind": "effect", "prompt": "Heavy stone lid scraping slowly across stone, close"},
    {"name": "reveal_hit", "kind": "stinger", "prompt": "Single low orchestral hit with shimmering bells, no vocals"}
  ],
  "scene": {"ambience": "museum_hum", "music": null, "room": "large_hall", "positions": {"Maya": -0.4, "Arthur": 0.4}},
  "events": [
    {"sound": "stone_scrape", "at": "u3", "when": "during", "place": "far"},
    {"sound": "reveal_hit", "at": "u9", "when": "with"}
  ],
  "voices": [{"id": "u12", "effect": "walkie_talkie"}],
  "pause_before": [{"id": "u9", "length": "long"}]
}
```

- `kind` is `ambience`, `music`, `effect` or `stinger`; code sets loop, volume and duration from it (the Single-pass ranges: ambience loop 0.3–0.5 / 20–30 s, music loop 0.15–0.25 / 30–47 s, effect 2–6 s, stinger 0.3–0.5 / 3–8 s).
- `when` is one of `before`, `after`, `with` (starts with the line) or `during` (lands inside it; code computes the delay from `planner.expected_seconds`). `place` is `left`, `right`, `near`, `far`, `moving_left_to_right` or omitted; code maps it to `pan`/`distance`.
- `room` must be one of `rooms.SCENE_ROOMS` or null; `effect` one of the speaker effect tags; `length` is `short`/`long` (code: 0.6 s / 1.2 s).
- Limits enforced in code: one ambience and one music per scene, at most four events per scene, names reused across scenes when the same sound recurs, saved sound names reused verbatim, character effects applied to every line of that character.

## S5 Compile and validate (code)

Builds the Drama 3 script from the locked utterances, S3 and S4:

- **Basic**: one `Name: text` line per utterance with cues, scoped tags and events applied.
- **Advanced**, per scene: when the scene has loops, a room or positions, a master `<parallel room=… stage=…>` holds the loop `<sound>` lines and a `<sequential>` timeline; scenes are separated by `<#0.8#>`.
- Timeline rules: `before`/`after` sounds become standalone `<sound>` lines; `with` wraps the sound and that utterance in a `<parallel>`; `during` wraps the utterance with `<sequential><#delay#><sound/></sequential>`; a `reaction` becomes the side-remark block under the previous utterance with a computed delay; an `interrupt` follows a `<#-0.3#>`; `pause_before` becomes a pause tag.
- A speaker's continuous passage is never broken by a pause, and a `with`/`during` block is only opened on the utterance an event is anchored to, so Drama 3 still performs passages as one take.
- Pronunciation notes: a later optional stage or a lookup can add `{pinyin}`/Arpabet notes; v1 keeps notes that were already in the source.

The compiled result then goes through the existing checks: `parse_script_cast_and_sounds`, `lint_drama3_script`, `coverage_note` (expected empty for preserved text) and `voices_without_sex`. A compiler bug therefore surfaces as a test failure, never as a broken script handed to the user.

## Errors, retries and cost

| Stage | Required | On invalid answer |
|---|---|---|
| S1 | yes | one retry naming the bad unit IDs and the valid names; then fail, keep the user's text |
| S2 | yes for new voices | one retry per failing speaker; then fail |
| S3 | no | one retry for the invalid entries; then drop them, note "N lines left undirected" |
| S4 | no | one retry for invalid events or sounds; then drop them, note it |

No silent fallback to Single-pass or to another model. Cancelling stops later stages. Each stage reports its model, call count and time in the status, so cost is visible.

## Integration

- `h3_app/prompt_writers/drama3_guided.py`: orchestrator `write_drama3_guided(...)` returning what `enhance_drama3_text` returns; calls `common.write_prompt` per stage with the stage's system prompt.
- `h3_app/drama3/guided/`: `segment.py` (S0), `schema.py` (stage answer validation), `compile.py` (S5).
- `prompts/drama3_guided/`: `content_preserve.txt`, `outline.txt`, `content_brief.txt`, `voices.txt`, `direction.txt`, `sound.txt`.
- `enhance_drama3_text` gains `workflow="Single-pass" | "Guided"`; the Drama 3 writer UI gets a **Writer workflow** selector next to Script mode, defaulting to Single-pass. "Copy full prompt" and "Apply pasted reply" stay Single-pass only.

## Build order

1. **S0 and S5 without any LLM.** Segmenter and compiler, tested with hand-written stage answers for parts of Blackthorn Reef and 昆仑墟底的回响 plus the prose cases from the flash-lite runs. Exit: compiled scripts pass the existing parser and lint with no notes, preserve every source unit, and keep passages whole.
2. **Stage prompts and orchestrator** with mocked model replies in tests, including each retry path.
3. **UI switch.**
4. **Evaluation** with the flash-lite harness: Guided on flash-lite vs. Single-pass on flash-lite and on gemini-3.8-flash, 4+ runs per case, English and Chinese, brief and preserved, Basic and Advanced. Measure valid-output rate, source fidelity, calls, tokens and time, then listen to rendered audio on Colab for a subset. Only then tune prompts or add stages.

## Not in v1

Regenerating a single stage or scene from the UI, pronunciation-note generation, music cues that change mid-scene, more than the listed staging patterns, and arbitrary mid-sentence timing. The IR leaves room for all of them.
