# Soundstage immersion: concrete implementation gaps

Audit: 2026-11-01. Scope: Single-pass and Guided with Advanced sound rendering. User interaction remains unchanged. This replaces the earlier broad recommendations with observed limitations and specific proposed changes. Implementation now covers shared sound metadata/routing, duration classes, phrase anchors and holds, anchored/resumable beds, measured reaction entries, clean dialogue sidechains, and both writers’ creative guidance. Guided also preserves character profiles and shared beat context. See docs/features/drama3.md and tests/test_drama3_immersive.py.

## Conclusion

The prompts already ask for scene planning, distinct voices, meaningful sounds, restrained rooms and clear mixes. Repeating those instructions is unlikely to solve the main limitations. The useful work is to preserve information between stages, let the output format express the intended timing and automation, and retain sound roles through rendering.

Single-pass and Guided are writer choices; Advanced is a rendering/script mode. Improvements must cover both writers in Advanced mode while preserving Basic behavior. Keep Single-pass as one writer call and keep Guided's existing staged calls. Both should target the same supported rendering controls.

## 1. Character evidence disappears in Guided preserve mode

Evidence: `parse_content_preserve` returns only title, speakers and scenes. Unlike brief mode, it does not return character roles. The orchestrator initializes `roles = {}`; `voices_stage` receives each character's first 300 characters of dialogue and only 240 characters from the first six utterances as story context.

Consequence: an original chapter may explicitly describe an elderly character, a distinctive voice or a relationship in narration, but the voice designer never receives that evidence. The existing instruction to design appropriate voices cannot recover missing context reliably.

Proposed fix:

- Guided: extract a compact character profile in the existing content call and preserve it through schema parsing. Include source-supported vocal traits, age/register when stated, usual manner and important relationships. Pass the profiles to voice design and direction.
- Single-pass: it already sees the source, so do not add this extraction stage. Require its existing speaker descriptions to preserve explicit source vocal traits and use contrast only where traits are unspecified. Include a short contrasting example where narration reveals a trait after a character's first line.
- Both: reuse chosen voices; separate permanent identity from momentary emotion. Do not add auditions or a new voice-selection workflow.

Acceptance: a character introduced with neutral dialogue but described later as having a low, aged voice receives that trait in the generated voice brief in both paths. This checks the brief; actual voice realization still needs a small listening comparison.

## 2. Scene planning has no executable representation of a turning point

Evidence: Guided Scene contains place/change but no beat boundaries. Direction and sound batches receive these summaries independently. Single-pass explicitly plans privately and emits script, so the downstream system sees only the resulting blocks and delays. Both prompts already discuss tension and reveals.

Consequence: there is no shared addressable moment meaning 'the reveal starts here' that can coordinate a held breath, music entrance and effect. More descriptive prose in the prompt is not an executable control.

Proposed fix:

- Guided: extend the existing scene answer with a small list of beat boundaries referencing utterance IDs, plus only actionable fields: focus, pause/transition intent and sound/music changes. Feed the same boundaries to direction and sound.
- Single-pass: keep its single response, but support stable line IDs and the same cue/automation references in a compact optional staging field or documented script attributes. Do not require a second planning call or duplicate the whole source.
- Both: validate these references and translate semantic choices to bounded renderer parameters. Keep the existing script-only response usable.

Implement only fields consumed by timing or mixing. A large unused story-planning schema would add complexity without improving audio.

## 3. Important cue timing is guessed before the performance exists

Evidence: Guided compilation places a `during` effect at `expected_seconds * 0.5` and reactions initially at `* 0.6`. Single-pass writes literal second delays, including a 1.2-second example, before speech is synthesized. Alignment runs later, but it does not resolve these sound anchors.

Consequence: a slower or more expressive take can move the spoken action away from its sound. Regenerating a take does not make a guessed cue follow the relevant word.

New shared function: resolve an event anchor `(line_id, phrase, occurrence, start/end, offset)` against the final rendered speech timeline. Preserve phrase references until after take trimming, alignment and any duration-changing effects. Use sentence/line-boundary fallback when precise alignment is unavailable.

- Guided supplies these fields in SoundEvent instead of compiling a midpoint delay.
- Single-pass emits the same anchors through the new supported attributes/sidecar instead of estimating seconds for semantic cues.
- Existing explicit delays remain supported for intentional absolute timing.
- Align only takes requiring anchors beyond existing alignment work. This runs automatically and is independent of the subtitle UI setting.

Acceptance: two takes with different pacing keep a latch click at the end of 'handle', including when the phrase repeats. Timing changes must not regenerate unrelated speech.

## 4. Guided cannot express a score that enters and exits within a scene

Evidence: Staging stores a single music name. The compiler adds it as a loop beside the complete scene timeline. Kind defaults fix music at 40 seconds and volume 0.2. Single-pass has richer nested blocks and custom durations, but its prompt also steers it toward one full-scene music loop. Neither SoundNode nor the mixing track contract carries a musical gain envelope.

Consequence: instructions to thin the music before a reveal cannot reliably become a score entrance, mid-scene withdrawal or evolving gain. Splitting into additional loop blocks can restart the asset and introduce another bed lead-in.

New shared function: bounded bed spans and gain automation. Support start/end line anchors, short fade durations and a small gain-envelope vocabulary such as establish, under-dialogue, rise and withdraw. Keep ambience and music distinct. Allow continuation of a selected bed with its playback cursor, without restarting it at every structural block.

- Guided: extend Staging from one music name to a short list of anchored bed cues; retain the old field as a default.
- Single-pass: expose the same bed span/continuation controls and update examples to use them.
- Both: reuse the existing generated asset. This does not require another music model or full composition system.

Acceptance: music enters on discovery, drops out before a whisper and resumes without a fresh 0.8-second introduction or repeated opening note. A continuing ambience does not restart at a beat boundary.

## 5. Sound roles are erased before room effects and mixing

Evidence: Guided distinguishes ambience/music/effect/stinger, but `sound_line` serializes only loop, volume, duration, placement and description. SoundNode has no kind or diegetic field. `_placed_sound` applies scene-room processing to every non-loop sound. The mixer principally distinguishes looped beds from other sounds.

Consequence: a non-diegetic orchestral stinger can acquire the hall reverb intended for footsteps. Ambience and score cannot receive genuinely different policies based on their roles. Prompts that request echoing assets can also duplicate room processing.

New shared contract: carry `kind`, `in_world` and an acoustic treatment flag through writer output, parser, SoundNode and mix tracks.

- Guided already knows kind; preserve it. Default effects to in-world and score/stingers to outside-world, with explicit exceptions such as music playing from an in-scene radio.
- Single-pass must emit equivalent metadata; do not rely on sound names to identify music.
- Shared renderer applies room sends only to appropriate dry in-world assets and keeps externally scored music out of the scene room. A wet/dry flag is provenance supplied by the asset plan, not a claim that an arbitrary generated waveform has been measured as dry.
- Legacy scripts preserve their current behavior unless metadata is supplied.

Acceptance: a latch and an orchestral hit placed in the same hall use different room routing. Changing the hall affects the latch, not the external score.

## 6. Dramatic holds within a narrator passage are prevented

Evidence: Single-pass instructions prohibit pauses between same-speaker passage lines. Guided's compiler actively drops a requested pause there. A local compiler probe produced `u2: no pause inside one speaker's passage` for two narrator lines with an explicit long pause.

Consequence: the system protects speech continuity at the expense of a deliberate suspense beat during narration. Merely asking the writer to 'hold before the reveal' does not overcome that restriction.

New shared function: a post-synthesis pause anchored to a line or phrase boundary. Generate the continuous passage as today, then insert the intentional hold at its aligned boundary while continuing the ambience and allowing natural tails.

- Guided retains the requested pause rather than dropping it.
- Single-pass gets a separate anchored hold control, avoiding a markup split that would force a cold-start synthesis request.
- Preserve breath boundaries with short crossfades; keep room tails separate from the edited dry speech.

Acceptance: a deliberate hold appears between two narrator sentences without splitting the speech into two generation requests or creating a click.

## 7. The two-effect rule conflates identity, distance and device

Evidence: Single-pass instructs at most two tags. Guided `_ordered_effects` sorts character/place/device and truncates to two; the compiler also keeps only two. A local probe of robotic_voice + far + walkie_talkie returned only robotic_voice + far.

Consequence: a valid combination can lose the device that explains how a voice reaches the listener. Adding more effects indiscriminately would create a different problem.

Proposed shared normalization function: separate character processing, device processing and spatial parameters. Distance belongs to spatial placement and should not consume a creative-effect slot. Retain the existing one-room policy and define compatible combinations instead of silently taking the first two tags.

- Guided emits semantic categories or normalizes existing tags into them.
- Single-pass instructions and validation use the same rules.
- Renderer distinguishes distance to a local radio speaker from distance to a remote caller; it should not automatically apply both interpretations.

Acceptance: a distant radio carrying a robotic voice retains all intended categories, with one coherent local acoustic space. Unsupported conflicts are reduced deterministically rather than dropped by list position.

## 8. Nested rendering loses the clean speech signal used for ducking

Evidence: `_render` returns mixed audio and advance. An outer ParallelNode treats a child containing both speech and an effect as a speech track because it is not sound-only. `mix_parallel` then measures that already mixed track for speech level and derives its ducking envelope from it.

Consequence: a loud inner effect can influence the outer bed's speech-driven level and ducking, including where no actual voice is active. This is a data-flow issue shared by both writers, not a prompt issue. Its audible severity needs measurement.

New shared function/return field: carry a dry dialogue sidechain alongside mixed audio and advance through recursive rendering. Use only the dialogue sidechain for speech RMS and bed ducking. Reuse it through nested blocks; do not build a general DAW or change the user's workflow.

Acceptance: a loud effect during a speech gap does not masquerade as speech for outer-bed ducking. Adding a nested effect does not change the speech-level reference used to balance the ambience.

## 9. Social and expressive sounds need a content-sensitive selection policy

The sound generator can be asked for laughter, crowd reactions or abstract accents today. The gap is not a missing synthesis category. Current planning vocabulary emphasizes ambience, physical events and musical reveal stingers; the schema stores playback kind but not narrative function or stylistic appropriateness. Character laughs already exist as speech performance events, which must remain distinct from independent sound assets.

Add a small semantic layer above the existing four playback kinds:

| Narrative role | Examples | Existing playback/generation route |
| --- | --- | --- |
| Character performance | A nervous chuckle, sigh, gasp, angry shout | Named speaker's speech cues/scoped delivery; preserve that voice's identity |
| Social atmosphere | Market conversation, tavern chatter, children playing, crowd murmuring | Ambience bed, preferably indistinct voices without intelligible invented dialogue |
| Group response | A crowd gasps, cheers, laughs, falls into an uneasy murmur | Short effect; only when a group is present or the requested presentation deliberately includes an audience |
| Subjective emotion | A low pressure swell for dread, a muffled wash for disorientation | Effect or music, explicitly subjective rather than a literal event in the room |
| Editorial/comic accent | A tiny plucked note after an awkward answer, a brief descending accent after a failed boast | Short effect/stinger outside the story's acoustic space; only where the scene's tone supports it |
| Transition or punctuation | A soft tonal bridge into memory, a brief accent at a reveal | Effect/stinger or music envelope; avoid decorating every scene boundary |

### Better guidance that can help immediately

Both writer prompts should evaluate the scene's tone and purpose before selecting sounds, then consider social and expressive roles as candidates—not quotas. Include paired examples with the same situation in contrasting tones:

- **Playful embarrassment:** a boast fails; a brief awkward hold followed by one quiet plucked accent may fit. Do not add an audience laugh unless an audience belongs in the scene or presentation.
- **Painful embarrassment:** someone makes a vulnerable confession; retain the pause, subtle room sound and natural breath. No comic accent.
- **Busy market:** quiet diffuse bargaining/chatter establishes people nearby; foreground calls belong only where supported by the content. Exact meaningful vendor dialogue goes through the speech system, not a generic sound prompt.
- **Serious public announcement:** restrained murmurs or a collective gasp can be appropriate if grounded in the audience's reaction. Serious content does not imply a lifeless soundscape.
- **Horror crowd scene:** tense murmurs or startled shouts can serve the scene without celebratory cheering or comic punctuation.

Judge tone per scene/beat under the story's overall style, not with one rigid comedy/serious label for the entire book. A funny story can contain a sincere moment. Default editorial stylization to restrained when evidence is ambiguous. Do not mistake a character's laughter for permission to make the production comedic.

Each chosen expressive cue should have an internal purpose and source/beat anchor. Permit no cue. Avoid repeating a generic comedy sting after every joke or scoring every feeling. Preserve mode must not invent new spoken words or consequential crowd actions absent from the source; unobtrusive setting texture and editorial accents must not change what happened.

### Small functions needed to make this reliable

1. **Role and perspective metadata:** retain `role` and `perspective` (in-world, subjective, editorial) alongside existing kind. These drive room routing, placement and mix policy; do not add a separate synthesizer for each role. The sound-role preservation in section 5 is the prerequisite.
2. **Duration classes:** support short accent, brief reaction and sustained texture instead of applying Guided's four-second effect/six-second stinger default to everything. Code can map these to bounded generation durations and natural-tail handling. Single-pass uses the same policy while retaining explicit durations. Do not blindly truncate a generated four-second sound to imitate a crisp half-second accent.
3. **Duplicate-event prevention:** if a named speaker already performs a laugh/gasp, do not also place an unrelated generic version on that beat. Group laughter remains a distinct event when the content calls for it.
4. **Small repetition budget:** limit editorial accents by beat/scene, avoid identical accents in adjacent beats and preserve quiet stretches. This is a deterministic check over the generated plan, not another model evaluation loop.

Guided can return the small style policy in its existing content response and pass it to direction and sound; Single-pass chooses and emits the equivalent policy in the same response. Source intent takes precedence over the inferred style. No style questionnaire or manual cue selection is required.

### Validate with contrasting content

Use paired comic/serious embarrassment scenes, a market, a celebratory crowd and a tense audience. Check that both writers can choose appropriate social sound, refrain from comic accents in sincere moments, avoid duplicate character laughter and produce expressive cues with fitting duration. Asset quality remains dependent on the audio generator; prompts alone cannot guarantee clean crowd vocalizations or precise musical attacks. Compare actual generated examples before claiming improvement.

## Implementation priority

1. **Broaden creative choices and preserve their meaning:** add tone-sensitive social/expressive examples to both writers, preserve Guided character profiles, and retain sound roles/perspective through rendering. Add duration classes and duplicate-event prevention so new expressive choices are rendered appropriately.
2. **Small rendering fixes with clear evidence:** normalize effect categories and carry a clean dialogue sidechain.
3. **Shared timing capability:** stable event references, word/phrase anchor resolution and post-synthesis holds. Update both writer formats/prompts together.
4. **Actual score control:** bed start/end/continuation and gain envelopes. Feed them from compact actionable beat plans; improve both writers' examples against the same renderer contract.

Do not merely repeat existing generic scene-planning instructions: add missing selection examples and the controls needed to realize them. Do not force Single-pass through Guided. Keep one shared capability contract and validate equivalent scenes emitted by both writers.

## Validation and limitations

Three local model-free probes confirmed Guided's dropped same-speaker pause, dropped third effect and loss of the stinger's kind in compiled markup. Other findings above are source-traced capabilities/data-flow limits; no real generated audio was auditioned and no full test suite was run.

Use focused tests for preserved character evidence, cue retiming, stinger room bypass, effect categories, bed continuation and clean sidechain propagation. For audible improvement, compare a few matched scenes through both writers: quiet narration with a reveal, distant device dialogue, a music entrance/withdrawal and nested action sounds. Listening is developer validation, not a new user task.
