# Music workflow, UX, and prompt review

Reviewed 2026-10-22 against the current repository and primary upstream documentation. This is an implementation proposal, not a claim of measured audio improvement. Review covered source, prompt templates, workflow graphs, and existing test coverage; it did not include a live browser session or GPU listening tests. No runtime code was changed.

## Recommendation

Keep the existing idea → draft → track layout. Improve the information passed between those stages, make vocal intent explicit, and add a reliable listen → revise → compare loop. These changes offer a better starting point than adding more sampling sliders.

Treat MiniMax H3 and MiniMax Music 3 as separate engines. Both are present here, alongside YuE2. Suggested product positioning:

| User intent | Suggested route | Reason |
| --- | --- | --- |
| Create a complete song from an idea or lyrics | MiniMax Music 3 | Already the Song default; dedicated caption-and-lyrics workflow. |
| Develop a song through an editable melody/harmony plan | YuE2 | Expose the score that the engine already generates. |
| Create a short musical cue, soundscape, or audiovisual performance | H3 | Present it as a short audio/AV workflow rather than an interchangeable full-song engine. |

H3's official published output range is 4–15 seconds; Music 3 is designed for songs up to five minutes. These describe upstream capabilities, not comparative quality measurements. [H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), [Music 3 model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3).

## Concrete issues to fix first

1. **Music 3's writer does not receive the duration.** `h3_ui/music_bindings.py` omits duration from the writer inputs, and `enhance_music3_prompt` in `h3_app/prompt_service.py:430` only supplies existing lyrics as context. Yet `prompt_music3.txt` asks for enough lyrics for the requested duration. Pass the selected maximum and, preferably, a separate approximate composition target. Include these in proposal invalidation so changing the duration cannot leave a stale suggestion looking current. Preserve existing positional API calls when adding fields.

2. **Lyrics-only input is a dead end in Music 3.** `h3_ui/engine_readiness.py:29` enables Generate if caption OR lyrics exists. `h3_app/generation/music.py:55` requires a caption. The writer also passes an empty caption to a helper that rejects requests with neither prompt text nor images, even when lyrics exist. YuE2 already supplies a fallback request for lyrics-only input. Bring Music 3 into alignment: enable writing from lyrics, require a caption before generation, and offer “Suggest a musical style.” Do not silently manufacture a caption while submitting a GPU job.

3. **YuE2 does not enforce the promised lyric preservation.** Its UI says the writer keeps existing words, but `h3_app/prompt_service.py:602` replaces supplied lyrics with the returned candidate. Music 3 only fills blank lyrics. Preview/accept prevents an immediate overwrite, but accepting still applies changes contrary to that promise. Default to a code-enforced lyric lock; offer an explicit rewrite action with a word-level diff.

4. **YuE2's score can be supplied, but the generated score cannot be retrieved through the current flow.** `h3_app/workflows/yue2.py:61` connects generated ABC directly to synthesis, with no score output/save node. The generation handler resolves only audio; the UI only offers a manual override. Return the generated score to the draft and save it with an explicitly saved project. A score-only “Plan music” action should allow inspection before audio generation.

5. **Instrumental intent has inconsistent representations.** Music 3's writer is instructed to return `N/A`, which becomes blank lyrics, while the UI recommends instrumental section tags. YuE2 needs nonempty section tags. Add a visible Vocal / Instrumental choice, carry it through writing and validation, and use an engine-specific representation. Do not assume any one tag sequence guarantees vocal-free audio; compare blank lyrics and tags with actual generation before standardizing Music 3 behavior. Warn about conflicts with pasted vocal lyrics without deleting them.

6. **The music workflows save MP3 only.** Both graphs end in `SaveAudioMP3` at V0 quality; H3 audio output does too. Offer a lossless master plus an MP3 sharing copy. This preserves the generated signal for further editing; it does not repair model artifacts or improve the underlying composition. Check installed node support, media indexing, and the primary-output resolver before emitting two files.

## A simpler interaction

The first screen should ask for an idea, with a small row of optional choices: Vocal / Instrumental, lyric language, approximate length, and broad energy. Let these be inferred from text when absent; expose conflicts when text and controls disagree. Avoid requiring a genre taxonomy, musical key, or BPM before starting.

Keep the original idea separate from the generated engine prompt. Today Music 3 uses one field for both, and acceptance replaces the idea with the production caption. A durable brief makes later feedback such as “keep the chorus but make the verses more intimate” easier to interpret.

Make the writing action depend on what exists:

- Idea only: **Draft song**.
- Existing lyrics: **Build arrangement**, with lyrics locked.
- Finished draft: **Revise draft**, applying a specific change request.

Keep the current acceptance and Undo mechanics. Improve review with separate style/arrangement and lyrics previews, visible changed sections, and independent acceptance where practical. Hide service configuration after successful setup; indicate writer availability near the action so the first click does not become a credential surprise. Direct manual generation should remain available.

In the draft, add lightweight section controls: insert verse/chorus/bridge, duplicate chorus, reorder sections, and inspect an approximate duration fit. Estimate fit from tempo, section bars, language, and vocal density when available. Label it as an estimate; do not turn a word count into a promise of exact playback length. A 30-second request needs a compact form, not a full multi-verse lyric cut off at 30 seconds.

For YuE2, use plain labels such as “Plan melody + chords,” “Plan melody,” and “Generate without a score.” Put raw ABC and detailed sampling under advanced controls. When Direct is selected, hide irrelevant score-sampling controls and clearly mark any retained score as unused.

After generation, show take cards with seed, actual duration, draft version, and the action that created the take. Add **New variation**, **Reuse this draft**, **Compare**, and, for YuE2, **Reuse this score**. Audio comparison should be level-matched for audition, without modifying source files. Musical events may not line up between takes, so synchronized elapsed time alone is insufficient.

Start with an optional two-take generation action, displaying that it runs two jobs. Keep the default single-take flow. Use the shared queue and per-take failure/retry records. A new seed alone should not be described as preserving a singer, melody, or arrangement.

## Better prompt handling

Use a small shared song brief internally: original idea, vocal mode, language, explicit constraints, exclusions, approximate target, maximum duration, section plan, lyrics, and locks. Compile it differently for each engine rather than sharing one generic “make this better” prompt.

For Music 3, retain the three caption sections already in `prompt_music3.txt`. Improve section-aware arrangement: translate production directions attached to lyric tags into the matching arrangement section, distinguish explicit requirements from inferred defaults, and avoid inventing precise BPM/key/vocal identity. The official rewriter describes a constraint-first, section-aware approach and a compact genre-reference routing system. Evaluate a small curated set of references before adopting a large template library. [Official Music 3 caption rewriter](https://github.com/MiniMax-AI/MiniMax-Music3/blob/main/skills/music-caption-rewriter/SKILL.md).

For YuE2, keep the style concise and make the score a first-class revision artifact. Separate “change arrangement while reusing this score” from “compose a new melody.” Validate supplied ABC and show a readable score summary before submission; a MIDI audition, if added, must be labeled as a score preview rather than a preview of the final generated vocal. Upstream exposes separate planning and synthesis stages and demonstrates score-based edits. [YuE2 model card](https://huggingface.co/m-a-p/YuE2-3B).

For H3 audio, add a focused prompt branch covering audible events, musical development, voice intent, timing, and ending. The remote writer already receives `result_format=Audio`, but `prompt.txt` remains heavily audiovisual. Retain necessary native H3 conventions while avoiding irrelevant camera choreography. The local H3 writer currently rejects audio-only mode; hide or disable it for that mode with a clear alternative instead of letting the request fail. Treat quality benefits from the new branch as a listening-test hypothesis.

Validate writer output before proposing it: required fields, nonempty style/caption, preserved locked lyrics, instrumental/vocal consistency, section agreement, and duration-fit warnings. Use structured output when supported by the provider, with explicit schema validation and a compatible parser fallback. A malformed response should not be silently treated as a finished caption. Keep personal draft content within the app's existing explicit-save retention policy.

Example shared brief: “A roughly one-minute Mandarin indie-pop song about the last train home; warm intimate lead, electric piano and restrained drums; hopeful chorus; no backing choir; short instrumental intro, one verse, chorus, resolving outro.” Compile this to Music 3's caption sections or YuE2's style and score plan. Do not give an H3 short cue the same full-song structure.

## Generation improvements worth evaluating

- Persist YuE2's ABC and, later, intermediate semantic/acoustic artifacts. This would let a synthesis comparison hold the musical plan fixed. A single seed reused across all stages does not establish the same composition when prompts or planning settings change.
- Distinguish planning, acoustic-token generation, diffusion, and decode in progress messages. Existing stage timings provide a starting point; measured timings can later support cautious estimates.
- Keep existing defaults initially. Introduce named speed/quality presets only after matched listening comparisons; higher CFG, more steps, or higher precision are not automatic quality wins.
- Check the selected YuE2 checkpoint's decoder against upstream's current recommended VAE before offering decoder choices. This audit does not establish which decoder is bundled in the deployed checkpoint.
- Consider reference-audio-to-score via SheetSage2 after score persistence works. Comfy's upstream YuE2 page already links a cover workflow, but the local integration does not implement it. Scope this as a separate feature with additional provisioning and testing. [Comfy YuE2 workflows](https://huggingface.co/Comfy-Org/YuE2).

The current upstream Comfy node deliberately switches to Direct mode when ABC is empty. The local graph's `mode="full"` placeholder on that path is therefore not itself a confirmed bug. Optional autoregressive guidance exists upstream; capability-check the deployed version before exposing it. [ComfyUI YuE2 node](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_yue2.py).

## Delivery order and validation

**First patch:** duration propagation and stale-proposal handling; lyrics-only readiness/writer support; YuE2 lyric preservation; explicit vocal intent; writer-response validation. Extend focused prompt and UI tests, including legacy API calls. Avoid a broad layout rewrite.

**Second patch:** generated-score capture, project persistence, score reuse, and a score-only planning action. Verify fresh planning, supplied ABC, Direct mode, cancellation, retry, project reload, and missing-node handling.

**Third patch:** take comparison, optional two-take generation, lossless export, and targeted revision controls. Then evaluate quality presets and reference-audio workflows.

Use a small listening pilot: six briefs covering an English pop vocal, Mandarin ballad, instrumental cue, rap, short jingle, and a structured longer song; two fixed seeds per brief; compare current and proposed prompting on the same engine/settings. That is 24 generations per engine, run deliberately as a pilot rather than on every patch. Evaluate the H3 cue branch separately at appropriate short durations.

Record lyric completeness/pronunciation, unwanted vocals, adherence to explicit constraints, repetition, section progression, ending quality, audible artifacts, generation time, and number of user edits before an acceptable take. Blind and level-match audition order where possible. Preserve engine/model versions and resolved settings. Use automatic transcription only as supporting evidence, especially for multilingual singing. Expand the pilot only if results warrant it.

Success should mean fewer failed starts, fewer edits to get a usable draft, and more acceptable takes within a fixed generation budget—not merely longer prompts or more expensive sampling.
