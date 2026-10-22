# A producer-led music workflow for Prism Studio

Design proposal · 2026-10-22. Builds on `music-experience-review.md`. The rules below are proposed product behavior, not tested model guarantees. No runtime integration or GPU listening evaluation accompanies this proposal. A companion prompt kit is in `music-producer-prompts.md`.

Working-tree note: during this review, concurrent runtime edits appeared in `h3_app/song_brief.py`, the Song views, prompt service, and active prompt files. They already begin duration-aware planning, plain-language choices, lyric locking, and draft checks. This review did not author or modify those edits. Treat the earlier issue list as a baseline; integrate this design with that work instead of implementing the same fixes twice. Its behavior and tests were not fully reviewed here.

## Product decision

Let the user describe a result and respond to music. Let the application handle the musical preparation: a focused brief, appropriate form, singable words, a recognizable hook, arrangement contrast, and a deliberate ending. Keep intermediate work available without making the user operate it.

The main journey is **Describe → Create → Listen → Refine**. Internally it is **brief → composition plan → lyrics → arrangement → engine instructions → generation → checks → revision**. These are logical responsibilities; they do not require eight LLM calls, eight screens, or multiple autonomous agents. Start with one structured planning/writing call, deterministic validation, and at most one repair call.

The professional standard is intentional musical choices, reliable boundaries around revisions, and honest assessment of the result. Complexity, dense orchestration, and a long prompt are not quality criteria.

## The musical decisions to automate

These are proposed defaults conditioned on the brief and genre, not universal songwriting rules. Preserve deliberate repetition, irregular phrasing, dissonance, abrupt endings, or sparse form when the user's idiom calls for them.

The in-progress prompts and planner need particular care here: a two-bar lyric line, 4/4 timing, a fixed syllable range across several languages, a rising pre-chorus, a chorus that is always the loudest/highest point, and an ever-larger final chorus are useful conventions for some pop arrangements. They should not become universal constraints. Likewise, a key's major/minor quality does not establish the intended emotion. Make these optional genre/form recipes with a supported scope; outside that scope, use broader timing guidance and allow the writer to preserve the requested idiom. This avoids turning professional assistance into formulaic output.

| Production responsibility | Internal decision | Simple user expression |
| --- | --- | --- |
| Purpose | Foreground song, unobtrusive background, dance track, short cue, or audiovisual performance | “A birthday song,” “music under my video,” “something to study to” |
| Emotional direction | Starting feeling, change, and landing; sadness need not imply minor harmony | “Nostalgic but hopeful” |
| Song identity | One central lyric idea or instrumental motif, with recognizable returns | “A line people will remember” |
| Form | Section roles, repetitions, transition space, and first payoff position | “Get to the good part sooner” |
| Prosody | Natural language stress, melodic emphasis, phrase length, breathing, and meaning work together | “Make the words easier to sing” |
| Melody | Range, contour, repetition, rest space, and contrast between sections | “More sing-along” |
| Harmony | Harmonic rhythm, tension/release, and voicing support the intended feeling | “Warmer,” “more unsettled,” “a satisfying ending” |
| Groove | Pulse, subdivision, accents, swing/straight feel, bass/kick relationship | “More bounce, same pace” |
| Arrangement | Give instruments distinct roles, registers, and entry/exit points | “Less busy,” “a bigger chorus” |
| Vocal direction | Delivery, intelligibility, phrase density, doubling, and backing-vocal placement | “Closer and more personal” |
| Ending | Resolve, decay, fade, loop candidate, or intentionally sudden stop | “Finish naturally” |
| Delivery | Preserve a lossless source; derive sharing copies and requested edits | “Download” or “Use in a video” |

Prosody provides the organizing principle: lyrical, melodic, harmonic, and rhythmic decisions should serve the song's intended meaning. [Berklee: prosody](https://online.berklee.edu/takenote/prosody-in-music-and-songwriting/). Repetition, contrasting phrase lengths, and rest space are useful compositional tools rather than defects to remove automatically. [Berklee: melody writing](https://online.berklee.edu/takenote/simple-tools-for-better-melodies/). Arrangement can leave room for the lead through timing and register choices as well as instrument selection. [iZotope: arranging around vocals](https://www.izotope.com/community/blog/how-to-write-better-song-arrangements-around-vocals).

## Screen 1: one idea, very little setup

Use **“What would you like to make?”** above a generous text box. Example: “A hopeful Mandarin song for my graduation video, about leaving home without leaving my friends behind. Around a minute. No choir.”

Below it, show three compact optional controls: **Voice: Singing / Instrumental / Auto**, **Language: Auto**, **Length: About 1 minute**. Keep **Use my lyrics** as an obvious secondary entry. Pasted lyrics are locked by default. If a user supplies a score, expose it through advanced input rather than asking everyone about notation.

Genre, vocal gender, musical key, exact tempo, meter, and technical settings are optional. If the user writes them, respect them. Do not infer singer gender or language from a person's name or demographic context. Do not make an instrumental user choose a lyric language.

Infer use case from the text when reasonable. Ask a single question only for a consequential conflict that has no faithful resolution: for example, instrumental selected with an explicit request for sung lyrics. Ordinary uncertainty should produce an editable assumption, not a questionnaire.

Distinguish an approximate composition length from the engine's generation ceiling. “About 1 minute” plans a compact song; “Must end at 60.0 seconds” is an exact delivery requirement that may need an edit. Exact duration should not be promised through a slider alone.

**Review first** is the initial default, retaining the app's existing suggestion acceptance behavior. **Just create** is an explicit optional mode: its button says **“Write and generate 1 take.”** It authorizes using newly generated lyrics and an inferred production plan in that run, without approving changes to locked supplied material. Keep an expandable “What I chose” summary. The user can always stop before the next stage. If the writing service is unavailable, preserve the input and offer a manual draft; do not submit an empty or half-compiled job.

Selecting an engine is secondary. Display the chosen engine and a short reason under details. Respect remembered explicit choices. Automatic selection should use an installed capability table and validated use-case presets, not unsupported model-quality rankings. Pin the engine for revisions unless the user requests a switch; switching changes the musical realization and may require another model download.

## The invisible production pass

### Brief and identity

Keep three separate records: **user intent**, **proposed musical plan**, and **rendered result**. Never replace the user's idea with an expanded caption. Mark fields as explicit, inferred, or derived; inferred values are changeable decisions, not facts about the user.

Identify the song's anchor. For a vocal song this may be the central hook phrase and the emotional point of view. For an instrumental it may be a returning motif and sound. A short logo sting, drone, ambient work, and narrative ballad need different anchors; not all need a pop chorus.

Use a small number of compatible sounds with defined jobs: lead, harmonic support, low end, pulse, and optional color. One instrument can fill several jobs. Do not require every job or add an instrument merely to make the caption appear complete.

### Form and time

Plan phrase groups and musical breathing space before filling the lyric box. Use meter/tempo/bar counts when useful and supported, with a confidence label. For constant tempo, duration in seconds is bars × quarter-note beats per bar × 60 / quarter-note BPM. Compound meters require the pulse unit to be explicit; do not apply a 4/4 formula to every song.

For example, at 96 quarter-note BPM in 4/4, a 24-bar sketch is 60 seconds before pickup, tempo variation, and release tail. A possible compact form is 2 bars intro, 8 verse, 8 chorus, 4 hook return, 2 ending. This is one coherent plan, not a guarantee that Music 3 or YuE2 renders exactly those boundaries. Reserve space for the ending rather than treating the maximum as usable lyric time.

Select form from purpose. A short social clip may introduce its hook early; a narrative song may earn its payoff later. Background music may deliberately avoid strong foreground hooks. Loop requests need an actual rendered-seam check; a prompt saying “seamless” is insufficient. A preview generated separately from a full song is a separate sketch, not a guaranteed prefix of the finished song.

### Words that can be sung

Draft the central idea and hook before elaborating verses. Give verses new information; let recurring hooks remain recognizable. Avoid forced rhyme, unexplained changes of narrator, generic filler, and dense sentences whose grammatical stresses fight the likely phrasing. Permit stylistic exceptions deliberately.

Estimate syllabic load and phrase length in the intended language. English word count is not a multilingual timing model. Mandarin needs phrase segmentation and natural lexical grouping; characters, syllables, and melodic notes are not interchangeable. Do not impose English rhyme/stress rules or mechanically force speech tones onto every Mandarin melody note. Singing pronunciation and melody/text fit still need listening evaluation.

For a supplied lyric, retain its exact text. If it is too dense for the planned duration, offer “Allow more time” or “Suggest a shorter version.” Never make it fit by silently deleting lines. For an existing score, check changed lyric underlay rather than assuming any new text fits unchanged notes.

### Arrangement and performance

Treat section contrast as a small set of purposeful changes: register, rhythmic activity, texture width, instrumental participation, harmonic pace, and vocal delivery. “Bigger” need not mean faster, louder, more instruments, a higher key, and a choir all at once.

For each section, specify the focal sound and what supports it. Leave gaps for fills rather than continuous competition with the vocal. Plan transitions by musical function: a pickup, brief removal, held note, fill, or change in texture. State the ending explicitly.

For “intimate,” consider close delivery, fewer doubles, restrained accompaniment, and less spacious ambience. For “danceable,” first address pulse and accent placement. Do not translate emotional adjectives directly into diffusion CFG, sampling temperature, or mastering loudness.

### Validate and compile

Check constraints, locked content, section order, lyric density, timing plausibility, vocal intent, required inputs, and unsupported promises. Exact schema or lock failures block submission. Musical heuristics produce limited warnings, not arbitrary quality scores. Conflicting explicit constraints need a resolution; the writer must not choose secretly.

Produce a concise human summary: **“Warm Mandarin indie pop; intimate opening; memorable chorus; a hopeful finish; around one minute.”** Expose meaningful assumptions beside it. Store structured decisions, not hidden chains of reasoning.

Compile from the accepted brief into native engine inputs. Validate the compiler output again. Keep prompt schema/version and model/backend versions in the technical record; keep personal text in the existing explicitly saved project boundary.

## Screen 2: review only what matters

The review should have one short musical-direction sentence, the hook or motif, and a readable song journey. Plain labels can show **Opening → Story → Main hook → Finish**, with Verse/Chorus available as secondary labels. These are proposed sections, not detected audio timestamps.

Show lyrics when the user wants to edit them. For first-time generated lyrics, offer a complete preview; do not hide meaningful words behind a score panel. Allow **Accept and generate** or a small correction. Keep one prominent next action rather than competing “write,” “accept,” and “generate” primary buttons.

Useful corrections include **Start sooner**, **More personal**, **Less busy**, **Keep my words**, and **Change the musical direction**. Three context-relevant suggestions are enough. Do not put ten expressive sliders beside ten technical sliders.

Advanced users can inspect the full brief, tempo/meter, section plan, caption, ABC, and sampling controls. Progressive disclosure should keep a single project model underneath both simple and detailed views.

## Screen 3: listen, then direct a revision

Make the current take central. Keep earlier takes intact and readily playable. A successful revision becomes a new take linked to its parent, not an overwrite. Keep user favorites separate from any machine recommendation.

Ask **“What would you change?”** with examples tied to the current take. Convert feedback into a bounded edit and display its effect before spending another generation: **“Bring the drums and bass forward in the chorus; keep the words. This creates a new full take.”**

| Listener request | Producer interpretation | Keep stable by default |
| --- | --- | --- |
| “Make the chorus hit harder” | Increase contrast from the preceding section; strengthen groove or register, with selective widening | Lyrics, central hook idea, tempo unless requested |
| “Less cheesy” | Reduce stock lyric phrases if unlocked, remove excessive backing parts, use restrained delivery; expose this interpretation for correction | Story, requested genre, supplied words |
| “More bounce, not faster” | Adjust subdivision/accent placement and bass/drum interaction | Tempo, lyric content |
| “Easier to sing along” | More recognizable repetition, less phrase density, simpler contour and manageable register | Theme; supplied words stay locked |
| “The voice gets lost” | Reduce competing instruments and fills, clarify lead prominence | Lyric and harmonic intent |
| “Keep the tune, change the instruments” | Reuse YuE2 score with a revised arrangement prompt where supported | Source score pitches/rhythm; generated interpretation remains approximate |
| “Fix that word” | Distinguish wrong lyric text from pronunciation; correct the relevant source or suggest an alternate wording if unlocked | All unrelated words and intended meaning |
| “Shorter, but keep the good bit” | Identify the favored section; propose a new compact form or an explicit audio edit | Selected hook/section material where available |

No generic **Improve** button should launch an unbounded sequence of rewrites. A single revision should address one main dimension unless the request clearly combines changes. Feedback marked at a playback timestamp targets the comment; it does not imply the engine can inpaint only those seconds.

### Three different kinds of preservation

1. **Exact source preservation:** original lyrics, user settings, accepted score file, and previous audio can be kept byte-for-byte in application state.
2. **Conditioning preservation:** reusing the same score or constraints can encourage musical continuity in a new performance. Rendering may still change timing, phrasing, voice, and mix.
3. **Exact audio preservation:** unchanged regions require an audio-edit or regional-generation path. The current full-song graphs do not provide one.

Label locks accordingly. “Keep lyrics” can be enforced on the request but cannot guarantee every word is sung correctly. “Reuse score” should not be advertised as an identical melody performance. No current voice-lock promise should be inferred from a fixed seed. Prompted instrument changes are a new render, not isolated stem mixing.

## Engine-specific professional automation

**Music 3:** retain its native caption/lyrics separation. Translate the section plan into the three structured caption sections, emphasizing audible relationships instead of adjective lists. Keep lyric text separate from arrangement directions. Revisions currently mean a new full render; describe their intended scope without promising that unrelated audio stays identical. The official caption guide supports section-aware, constraint-preserving prompting. [MiniMax caption guide](https://github.com/MiniMax-AI/MiniMax-Music3/blob/main/skills/music-caption-rewriter/SKILL.md).

**YuE2:** capture generated ABC first. Provide a score-only planning stage for detailed control, with automatic progression in Just create mode. Reuse the accepted score for arrangement revisions. For melody-only input, detect chord symbols rather than assuming the mode strips them. If preserving melody during reharmonization, compare pitch/rhythm and inspect note/chord clashes; formal checks do not replace hearing the result. These opportunities follow upstream's separate planning/synthesis and editable-score interfaces. [YuE2 model card](https://huggingface.co/m-a-p/YuE2-3B).

**H3:** plan a compact audible event or cue at the chosen short duration: entry, development or hit, and release. For audiovisual work, prioritize synchronization and clear reference assignments. Avoid forcing the full-song form or raw ABC controls into H3. Its official model targets short audiovisual generation. [H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

An automatic route should never convert a user-selected Music 3 project into YuE2 merely because “keep the tune” was requested. Explain the current limitation and offer a separate score-based project path when available.

## Checks after generation

The current code does not establish that a song sounds good. Add narrowly scoped checks with evidence and confidence:

- **File validity:** decode succeeds; duration/channels/sample rate are known; no empty or nonfinite samples.
- **Technical candidates for review:** long unexpected silence, near-ceiling samples, sudden ending energy, or unexpectedly short duration. These are flags, not proof of distortion or a bad ending.
- **Vocal/lyric candidates for review:** transcription alignment and voice activity can suggest missing/repeated words or unwanted vocals. Singing, rap, choir, and multilingual material require cautious confidence handling.
- **Musical candidates for review:** detected pulse, section changes, and repetition may support comparison. A long repeated texture can be correct for ambient music; divergence from a requested BPM may be half-time ambiguity.
- **Human preference:** hook appeal, emotional conviction, taste, and whether the take fulfills the intended use remain listening decisions. Never show an invented “professional quality: 96%” badge.

Initially automate only deterministic failures and suggestions. A valid but aesthetically uncertain take remains playable. Default generation budget is one take, with zero automatic creative rerenders. Offer a clearly priced two-take exploration or user-selected retry budget later. Never automatically resubmit an ambiguous interrupted backend job; preserve the existing recovery policy.

For two candidates, first exclude technical failures, then compare explicit requirement adherence, then invite preference. Level-match the audition copies, preserve masters, and use neutral A/B ordering. Do not rank solely by loudness, transcription accuracy, or a generic quality model.

Export a source-quality lossless file and a convenient compressed copy. “Mastering” is a separate processing feature and should only be offered if it actually processes and verifies the signal. Do not imply true stems exist; separated stems, if later added, must be labeled as estimates.

## State and implementation boundaries

Proposed objects: `SongIntent`, `SongPlan`, `SongDraft`, `GenerationTake`, and `RevisionRequest`. Give sections stable IDs. Keep source idea, accepted lyrics, derived caption, score, and actual audio distinct. A revision stores parent IDs, requested change, proposed field changes, locks, generation scope, and a finite budget.

On input changes, invalidate only dependent draft artifacts: lyric language affects lyrics and underlay; duration affects form and density; arrangement feedback may leave words intact; score changes invalidate derived melody summaries. Do not erase old takes. Any change relevant to a pending suggestion invalidates its acceptance fingerprint.

Start by extending the in-progress `h3_app/song_brief.py` rather than creating another brief system. If responsibilities grow, proposed modules under `h3_app/music/` could separate planning, validation, compilation, and revision, with a shared `h3_ui/song_workspace.py`. Those split-out paths are proposals, not existing modules. Use the existing queue, request adapters, prompt-review mechanics, projects, and media ownership rules. Do not create a second persistence or GPU-queue system.

Keep public generation schemas compatible. Introduce versioned planner/revision endpoints separately from existing positional generation calls. Draft changes and generation submission must be distinct recorded operations even when one user action authorizes both.

## Delivery plan and acceptance cases

**Foundation:** complete and verify the in-progress duration/readiness/lyric-preservation fixes, then extend the brief with constraint provenance, separate source idea, schema validation, and a concise review summary. Make form recipes genre-aware. Keep current generation defaults. Test lyric-only entry, instrumental conflicts, writer failure, stale acceptance, and legacy API calls.

**Professional planning:** add form/density planning, section-aware compilation, explicit ending, and the two automation modes. Add YuE2 score capture and reuse. Test the 30/60/180-second forms, locked long lyrics, multilingual inputs, malformed or conflicting ABC, and cancellation between planning and generation.

**Iteration:** add parented takes and bounded natural-language revisions. Test that “more bounce, same pace” does not change tempo, that supplied lyrics remain exact, that unsupported regional edits are labeled full rerenders, and that switching engines is explicit.

**Listening and delivery:** add technical checks, optional A/B generation, lossless export, and cautious audio-analysis helpers. Calibrate heuristics on genres and languages before displaying them to users.

Evaluate each layer separately: planning correctness without GPU; prompt adherence with fixed seeds/settings; blind musical preference with multiple seeds; and usability with people who do not know musical terms. In the existing six-brief pilot, add two fixed user revisions per selected take rather than increasing all settings at once. Report acceptable takes per total generation cost, time to first acceptable song, failed starts, and unintended changes during revisions. Include background/ambient and non-pop forms so the system is not rewarded for imposing a chorus everywhere.
