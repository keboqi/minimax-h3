# Soundstage script writer audit

Date: 2026-11-01. Scope: Single-pass and Guided rewrite/enhance paths, including stage requests, source segmentation, schema parsing, compilation and shared final validation. The findings record the pre-change behavior; implementation status follows.

## Implementation update

Review follow-up: source labels now distinguish English attribution clauses from
long character names; only complete recognized expansion directives at request
edges can be removed. Approximate spelling/anagram acceptance was replaced by a
bounded repair table. Duplicate quote handling requires explicit joint attribution
and distinct performers. Numeric symbols and selected abbreviations use deterministic
English spoken-form equivalences rather than word-similarity exceptions. Legacy cue
localization leaves ambiguous overlapping phrases in their original scope, and its
container parser accepts mixed case while rejecting mismatched tags with H3Error.
Focused regressions exercise all seven review findings, including the overlap cue
that previously moved from 20 ms to 700 ms; it now remains at 20 ms.

The implementation now protects short supplied text from brief routing and rejects a model's attempted preserve-to-brief handoff. It retains existing supported Advanced controls while enhancing delivery, checks ordered source words and known labels in both workflows, preserves repeated cues, recognizes attribution verbs instead of using length alone, validates stage envelopes and scene coverage, and enforces practical English preview length.

Guided now carries authoritative voice descriptions into stage context and skips redesign for referenced voices. Source-linked spelling repairs, validated pronunciation mappings and explicitly authorized after-unit insertions are supported. Single-speaker preservation retains scene analysis. Conflicting cross-batch sound definitions are renamed and reported instead of silently replaced; unanchored with-effects layer simultaneously, and unmistakable actor-event duplicates are reconciled. Prompt quotas and contradictory cue rules are removed, the brief example is corrected, supplied-heading metadata permits 100 characters, and the sound prompt consolidates its supported controls and distinct timing examples.

Boundaries: existing structured scripts retain their sounds rather than receiving a second generated sound plan. Expansion directives must be on a separate line and match the documented explicit instruction forms. Spelling validation permits small repairs rather than general sentence rewriting. The ordered fidelity check is conservative and may reject a legitimate larger normalization; it does not establish semantic intent. Reference previews use supplied preview text where available, otherwise neutral English/Chinese/Japanese/Korean synthesis text or a source-language excerpt; they are never transcripts inferred from uploaded audio. Cross-batch asset conflicts preserve each batch's requested asset but still merit listening review for continuity. No actual model A/B or generated-audio evaluation has been performed in this implementation session.

## Conclusion

Optimize for faithful, performable scripts before optimizing prompt length. Guided has a useful architecture: lock source text, assign speakers, share character profiles and scene beats, then compile constrained stage answers. However, deterministic routing and compilation can undermine the model's choices. Several differences from Single-pass are capability gaps rather than redundant wording.

The recent Single-pass cleanup is not the origin of these Guided problems. The latest Guided prompt change in a6ba365 primarily extended the sound prompt; the older cue and reaction policies remain. Do not attribute all current shortcomings to the recent cleanup.

## Confirmed findings, ordered by impact

### 1. Finished short text can enter creative writing

Location: `h3_app/drama3/guided/segment.py:98`; `h3_app/prompt_writers/drama3_guided.py:240`.

Local probes:

- `The door was locked. Nobody answered.` -> `brief`.
- `Mia: Do not open the door.` -> `brief`.
- A two-line script whose labels each contain four words also routes to `brief`: `_label` rejects names longer than three words, and the remaining prose heuristic falls back to brief.

These inputs go directly to original writing, without the preserve stage. The result declares `source_mode=brief`, which disables the final preservation check. Whether a particular model actually rewrites a given input was not measured, but the route explicitly permits it.

Fix direction: recognize a valid one-line labelled script; use explicit creation intent rather than short length alone to authorize new words. Resolve ambiguous short prose conservatively. Keep source classification authoritative instead of allowing the output's own mode to bypass a known preservation requirement.

### 2. Existing Advanced scripts are not parsed as structured source

Location: `h3_app/drama3/guided/segment.py:98,140`.

A valid input with `<parallel room="large_hall">`, a named sound, a sequential timeline and two labelled speakers routes to prose. Segmentation creates quotation units for XML attribute values such as `large_hall`, `wind` and `true`, and narration units for container tags and labelled dialogue. The casting model therefore receives broken source semantics before it starts.

Fix direction: parse existing supported script structure before prose quotation splitting. Preserve or explicitly normalize staging and sound identities; do not read markup as story prose. Re-enhancement needs a round-trip contract for supported controls.

### 3. Shared preservation validation does not establish fidelity

Location: `h3_app/prompt_writers/drama3.py:228,269`; `h3_app/drama3/lint.py:361`.

Confirmed locally:

- A reply that drops an entire source sentence is accepted by `parse_drama3_reply`; the missing sentence produces a status note only.
- Reordering two source lines and swapping their speakers yields no missing-phrase report. Coverage compares global word/character bigram sets, not ordered speaker assignments.
- `source_mode=brief` skips coverage entirely.

Guided's locked-source design prevents many omissions after correct routing, but it does not protect against incorrect classification, attribution, source preprocessing or final shared transformations. Single-pass relies more heavily on this same weak check.

Fix direction: validate ordered source-unit coverage and speaker assignments where known. Separate permitted punctuation, typo and annotation conversions from content edits. Retain a compact edit/source map for Guided. Treat actual source loss as a failed rewrite or targeted repair, not a successful enhancement with a warning. Do not enforce emotional cue quotas as a substitute for fidelity checks.

### 4. Important narrated actions can lose acting direction silently

Location: `h3_app/drama3/guided/compile.py:77,84`.

`is_attribution` uses narrator identity, previous speaker and text length, without checking that the text is an attribution. Compiling `Mia: We are safe now.` followed by `narrator: The gun fired.` with a requested `sudden, stark` cue removes that cue, with no note. A reveal or action can therefore be flattened even when the model directed it correctly.

Fix direction: use actual attribution evidence or source metadata. Preserve deliberate restraint for `she said` while allowing short actions, realizations and reveals to carry delivery. Avoid replacing this heuristic with mandatory narrator direction.

### 5. Legitimate repeated cues are deleted across actors

Location: `prompts/prompt_drama3_guided_direction.txt:17`; `h3_app/drama3/guided/compile.py:102`.

The prompt demands cue variation, and the compiler drops a cue equal to the previous accepted cue. A local two-actor example gives both `quiet, reassuring`; only the first actor keeps it. The comparison is not scoped to a speaker and carries through undirected lines.

Fix direction: align with Single-pass: a continuing state may keep its cue or need none. Change both prompt and compiler; editing the prompt alone leaves the deletion in place.

### 6. Missing optional-stage schemas are accepted as intentional restraint

Location: `h3_app/drama3/guided/schema.py:313,329`; `h3_app/prompt_writers/drama3_guided.py:280,291`.

Both `parse_direction('{}', ...)` and `parse_staging('{}', ...)` succeed with empty plans. In a mocked end-to-end Single-speaker preserve run, voices succeeds and direction/sound return `{}`: the writer returns a successful result in three calls without a retry or a degraded-stage note.

Fix direction: distinguish explicit valid emptiness (`lines: []`; a complete scene with no sound choices) from missing required schema. Validate stage envelopes and scene coverage, retry malformed answers, and report optional-stage degradation. Empty sound plans and restrained passages must remain valid creative decisions.

### 7. Existing voice context is omitted from the voice-design request

Location: `h3_app/prompt_writers/drama3_guided.py:269,307,336`.

All speakers require voice-design results, including speakers with references. The request includes source profiles and short text excerpts, but not their slot descriptions or `has_reference`. A local run with a saved elderly, gravelly reference sends neither fact to the voice stage. The authored description is restored only after design, while the generated preview can reflect a different baseline. A voice-stage failure can also fail a run even when reference voices already exist.

Fix direction: pass authoritative descriptions and reference state to the relevant stages. Separate new voice design from calm preview preparation for existing voices; do not redesign a known identity. Never infer uploaded audio properties. Preserve the existing metadata behavior.

### 8. English preview validation permits unusably short references

Location: `h3_app/drama3/guided/schema.py:282`.

The prompt asks for 2–4 calm sentences and roughly 10–20 seconds. The parser accepts `preview_text: "Okay."` for an English voice. There is a CJK minimum, but no comparable Latin-language adequacy check. Existing tests also use short English previews, so passing tests do not establish the intended reference quality.

Fix direction: validate practical reference length without imposing English word counts on every language. Use targeted retries for clearly inadequate previews. Keep calm baseline, script language and clean recording constraints.

### 9. Preservation feature parity requires more than prompt changes

Locations: `h3_app/drama3/guided/model.py`; `h3_app/drama3/guided/segment.py`; `prompts/prompt_drama3_guided_preserve.txt`.

- `(sighs)` remains literal locked speech in a labelled source probe. Guided has no general stage converting performance-only parentheticals into supported events while retaining prose asides. Single-pass explicitly distinguishes these cases.
- Guided has no structured correction or pronunciation-generation result for the typo and risky-reading instructions in Single-pass. Existing pronunciation notes may pass through source text; generating consistent new notes is a different capability.
- Preserve mode locks source units and never enters the brief-writing stage to insert explicitly requested expansions. A dedicated mixed preserve/expansion contract is absent.

Fix direction: introduce narrow, validated annotations/edits tied to source spans. Preserve literal prose, facts, names and order; allow only certain corrections, supported performance conversions and explicitly requested insertions. Pronunciation needs its own consistent word-to-reading mapping. Do not ask downstream direction or sound stages to rewrite locked words.

## Additional design risks and policy inconsistencies

- **Single speaker loses scene analysis:** preserved Single mode bypasses the content stage and calls `scenes_from_units(..., [])`. This produces one scene without extracted profiles or shared beats, even for a multi-scene story. Saving an attribution call should not erase place/time analysis needed for sound design.
- **Independent stages cannot inspect each other's choices:** direction and sound share the content-stage beats, which is useful, but sound does not see the final actor events. A rule against duplicating a named character's gasp/laugh cannot guarantee coordination. A deterministic reconciliation step can address duplicates without adding a mandatory planning call.
- **Cross-batch sound continuity is under-specified:** sound batches receive the supplied sound context, but not a shared registry of new assets selected by other batches. Merging keeps the first definition for a reused name. Distinct later definitions can therefore be replaced by that first asset. Validate conflicting names and plan recurring assets once where appropriate.
- **Unanchored layering differs from the guidance:** multiple `with` effects on one line compile into a sequential sound chain. Phrase-anchored effects sharing a phrase compile as simultaneous scene cues. Make the intended distinction explicit and test simultaneous versus ordered layers.
- **Forced policies remain:** the brief prompt requires 1–3 reactions per dialogue scene; direction asks for stress on roughly half the directed lines; ambiguous attribution is forced to a likely character rather than narrator. These differ from the revised Single-pass policies.
- **Invalid brief example:** Arthur speaks in the example but is absent from its cast list.
- **Title contract differs:** Guided truncates output titles to 60 characters, while Single-pass now allows up to 100 for supplied headings. Keep full spoken headings, and align metadata limits deliberately.
- **The design document is stale:** `docs/plans/drama3-guided-writer.md` still says nothing is implemented and describes planned fixes/outline behavior that should not be treated as proof of runtime support.

## Optimization boundaries

Keep separate stage prompts: repeated language/source/reference constraints across independent calls can be necessary. Remove duplicate rules within a stage, integrate appended extensions into the original schema, and retain examples that teach distinct decisions.

Keep the sound prompt's causal anchoring, exact phrases, duration classes, editorial/in-world distinction, sparse density, dry assets, holds and score withdrawal. Its repeated paragraphs can be consolidated; these capabilities should not be cut to reach a token target.

The desired output is neither maximal tagging nor minimal tagging. Preserve the story, make meaningful delivery choices executable, and leave neutral material and intentional silence alone.

## Recommended implementation order

1. Protect input intent and structured-source round trips; strengthen final source fidelity checks.
2. Fix destructive direction heuristics and validate optional-stage schemas without outlawing deliberate empty plans.
3. Preserve authoritative voice context and validate reference previews.
4. Add explicit preservation parity for performance annotations, certain corrections, pronunciations and requested expansion, with narrow source-linked contracts.
5. Reconcile scene, performance and asset continuity, including Single mode and batch boundaries.
6. Consolidate prompts around the implemented schemas, align creative policies and correct examples/documentation.

For each phase, use focused behavioral regressions. Then evaluate old/new actual model outputs on English and Chinese prose, labelled and Advanced scripts, short and long inputs, briefs and requested expansion, references and new voices, Single/Auto/Adapt, and Basic/Advanced. Measure routing, source/order/speaker fidelity, surviving direction choices, valid timing references, degraded stages and call cost. Listen to a representative rendered subset for continuity, intention, intelligibility, quiet moments and causal sound timing. JSON validity and cue counts alone cannot establish acting quality.

## Verification performed

Ran local deterministic probes against current functions, including one mocked end-to-end Guided run. No external model calls or generated-audio listening comparison was performed. Findings about model behavior beyond these routes and compiler transformations are risks, not measured failure rates.

Focused baseline: `python -m unittest tests.test_drama3_guided tests.test_drama3_writer_guidance tests.test_drama3_examples tests.test_drama3_immersive` using `.tmp-ui-venv/Scripts/python.exe`: 58 tests passed. Passing tests coexist with the reproduced failures above. No full test suite was run.

Implementation verification: 124 focused tests passed across writer integrity regressions, Guided, prompt guidance, examples, immersive staging, lint, cast, provider mocks, Single-pass writer calls and Basic/Advanced stripping. Temporary test files were directed into the workspace. The new regressions cover source routing, mode bypass, order/known speakers/numbers, repeated turns and cues, short narrated actions, missing stage schemas, preview length, structured-script retention, authorized expansion, Chinese repairs, pronunciation mappings, Single-speaker scene analysis, simultaneous layers and cross-batch asset conflicts. The sound prompt is approximately 1,160 whitespace-delimited words, down from 1,922, while retaining distinct timing examples and supported production fields.
