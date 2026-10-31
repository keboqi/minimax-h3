# Nemotron and overlapping-speaker dubbing: simple first version

Plan updated 2026-10-31. Implemented 2026-10-31; the runtime is not yet smoke-tested on an RTX PRO 6000.

Implementation differs from this plan in three small ways: line keys and review marks gained the speaker instead of new stable line IDs (enough to tell simultaneous identical phrases apart); clean references are ranked by distance alone, with the 3–15 s range used only as the trim length; and the Clone from default is set when a session opens, not stored.

## Scope

Add Nemotron as the fourth transcription choice. Keep Source → Review lines → Dub, existing translators, Drama 3 and timing controls. Use fixed automatic rules; the only new choice is one extra Voice option, with a sensible default per transcriber.

| Option | Visible hint |
| --- | --- |
| Qwen on this server | Basic transcription. Best for one speaker. |
| MOSS on this server | Speaker-labeled transcription. Best for conversations where people take turns. |
| Nemotron on this server | Complex conversations with overlapping speakers, speaker labels and word timing. |
| Gemini (cloud) | Cloud transcription and translation with speaker labels. Sends audio to Gemini; requires an API key. |

Use “speaker-labeled,” not “speaker-separated”: transcription does not produce separate voice waveforms.

## Fixed workflow

1. Extract audio and run the existing optional vocals/background separation.
2. Transcribe with the selected provider. For Nemotron, retain speaker-attributed words and timing and split into readable lines without flattening simultaneous speakers.
3. Translate using the existing Hy-MT/Gemini selection.
4. Select reference audio with the chosen Voice option below.
5. Generate each translated line with Drama 3, preserve overlaps during placement, and mix simultaneous takes.
6. Display all active subtitle lines. Reuse valid native source words; align translated words against synthesized takes when the existing word-subtitle option is enabled.

Audio and video use the same workflow.

## Reference selection

Add a third choice to the existing "Clone from" radio (`voice_mode` in `h3_ui/dubbing/view.py`) rather than a new control. Keep the existing option values unchanged, since they are stored in dub records; rewrite the radio's single `info` text with one short sentence per option:

| Clone from | Hint sentence | Default for |
| --- | --- | --- |
| Each line's own voice (simple, existing) | Clones each line from its own audio, including any overlapping voices. Safe with any transcriber. | Qwen, MOSS, Gemini |
| Nearest clean voice (new) | Clones overlapping or short lines from the same speaker's closest solo passage. Needs accurate speaker labels. | Nemotron |
| One voice per speaker (existing) | Unchanged. | — |

When a session opens or finishes analysis, set the radio from its transcriber. The user's pick then holds for that view; no new per-session storage. Nearest clean depends on speaker attribution: with MOSS or Gemini a mislabeled line would clone the wrong person, so it is opt-in there. Sessions without stored speaker activity (MOSS, Gemini, older sessions) use line times as activity.

**Each line's own voice** keeps today's behavior unchanged: the line's own clip when at least two seconds, otherwise the speaker's existing best reference line.

**Nearest clean voice** honors any saved voice in speaker settings, then:

1. Use the line's own audio when clean and sufficiently long.
2. For an overlapping or too-short line, choose the closest usable clean clip from that speaker, searching before and after across the entire recording.
3. If no usable clean clip exists anywhere for that speaker, use the target line's own source audio even with overlap. Never fall back to another speaker's line. Continue automatically.

Nearest-clean rules:

- Build candidates from original speaker activity and transcript timing. Exclude other-speaker activity, including extraction padding. Deleting or moving subtitles must not make original acoustic activity disappear.
- Use complete phrases where possible. Require at least two seconds of usable speech for clean candidates, prefer the existing 3–15 second range, and trim long candidates at word/phrase boundaries to at most 15 seconds. No new quality model.
- Rank eligible clips by interval-to-interval temporal distance, then earlier start. Do not impose a local search cutoff or rank distant clips above nearby usable clips just for being longer.
- For fallback, use the line's own source slice within existing synthesis reference limits. Do not expand into unrelated speech merely to meet the clean-candidate minimum. Existing synthesis failure handling applies if it cannot be used.
- Pair clean reference audio with its actual transcript. For mixed fallback, use the target speaker's text from the selected interval. Interfering voices can affect the cloned voice; this is an accepted fallback limitation.
- One voice per speaker stays as it is today for every transcriber.
- Cache the selected interval, speaker, transcript and fallback status, and include reference identity in existing take caching. Name trimmed clips by their trimmed interval, as `_reference_clip` already does by times.

Example: A, B and C overlap at 20–23 seconds. Each uses a nearby solo passage from their own speech, including passages after the overlap. Generate and mix all three concurrently. If C never speaks alone, C uses the mixed source clip for that line.

## Minimal shared changes

- **Sessions:** persist native words and original speaker activity beside existing segments. Add stable line IDs and preserve them through edits/sorting to distinguish simultaneous identical phrases. Assign IDs to old records when loaded, without re-analysis, and persist them on the next save. Use input fingerprints to prevent stale jobs overwriting edits; no generic revision/event framework.
- **Providers:** add a Nemotron branch, add it to `PIPELINES` and `SPEAKER_PIPELINES`, and update the "MOSS or Gemini" speaker-settings messages; no provider framework rewrite.
- **References:** add the Nearest clean voice mode beside the existing modes in `_plan`, used identically by preview and full dubbing. Original activity remains the cleanliness baseline. If an edited speaker assignment cannot be reconciled, skip uncertain candidates and use fallback.
- **Timing:** retain Smart, Match and Natural. Fix only behavior that cuts or serializes intentional cross-speaker overlap; the known case is `slot_end`, which ends a take before the next line of any speaker and should consider only the same speaker's next line. No replacement scheduler or overlap-policy system.
- **Mixing:** retain additive synthesis mixing. Merge original-audio intervals for failed, disabled or untranslated lines and restore each source sample once. Keep every other line's Dub setting unchanged. Original-voice bleed during mixed original/dub overlap is accepted without a decision screen. Shared original overlap samples use the global speech gain, since separate speaker gains cannot be applied to mixed source audio.
- **Captions:** retain the multiple-active-cue player and existing exports. Do not truncate one speaker because another starts. Label simultaneous speakers where needed and verify existing subtitle rendering before introducing another export format.
- **Word timing:** reuse valid Nemotron source words. If editing invalidates them, use plain source captions for that line rather than forcing alignment on cross-talk. Existing providers keep their current behavior.

## Nemotron runtime

Use a dedicated local vLLM/NeMo environment, separate from dubbing tools and Drama 3. Pin versions after a small deployment-GPU smoke test. Start with one request at a time and no speculative decoding. Stop the server to release GPU memory before translation/synthesis or competing jobs; no sleep/wake tuning initially.

The published NVFP4 recipe uses vLLM 0.32.0, NeMo Speech and CUDA 13. Native timestamps require the V1 runner, CTC middleware and explicit capture; DFlash2 does not provide them. The example uses Blackwell hardware. Use Linux/WSL and validate the deployment GPU and memory before enabling the option. Source: [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4).

Validate structured words, speaker mapping, timing bounds, alignment errors and truncation. Invalid output produces a normal retryable analysis error; do not invent timing or switch providers. Reject recordings above the validated duration limit, never exceeding the published one-hour limit. No automatic stitching.

## Implementation order and focused checks

1. Go/no-go gate: smoke-test the pinned runtime on the deployment GPU (not locally) with clean dialogue and overlapping speech; capture response fixtures. If the NVFP4 checkpoint cannot run there, stop and revisit the checkpoint or hardware before steps 2–5.
2. Add Nemotron, selector hints, persisted words/activity and stable line identity.
3. Add the Nearest clean voice option with its mixed-source fallback and per-transcriber default.
4. Make focused timing, mixing and caption fixes.
5. Run targeted dubbing/runtime tests and an end-to-end listen/view check; update feature documentation.

Cover two/three speakers; clean clips before, after and far away; no clean clip; short fallback; contaminated padding; simultaneous identical phrases; save/reload; concurrent editing; failed/disabled lines; preview/full-dub consistency; default Voice option per transcriber and unchanged simple-mode references; cancellation and GPU release. Do not repeatedly run the full suite.

Completion means the application preserves accepted words, speaker assignments and overlapping times; intended dubbed voices remain audible; original audio is not duplicated; references follow the chosen Clone from option; and older sessions/providers still work. Recognition and mixed-reference voice quality remain best effort, not zero-error guarantees.

## Out of scope

SAM Audio or other speaker-waveform separation, visual speaker selection, manual reference assignment, new overlap-policy controls, provider-framework refactoring, a replacement scheduler, mandatory ASS export, multi-hour stitching, a new translation backend and speculative-decoding optimization.
