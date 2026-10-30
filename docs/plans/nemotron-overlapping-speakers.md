# Nemotron transcription and overlapping-speaker dubbing

Research and proposed implementation plan, 2026-10-31. No implementation changes or model execution yet.

## Recommendation

Add **Nemotron on this server** as a fourth transcription option alongside Qwen, MOSS and Gemini. Keep the existing translator selection and Drama 3 synthesis. First upgrade the common transcript, reference, timing and mixing contracts so overlapping speech survives the entire workflow, including older providers and manually edited sessions.

Treat speaker attribution and audio source separation as separate capabilities. Nemotron supplies the former; its documented outputs do not include isolated speaker waveforms. We cannot promise selective original-voice preservation during cross-talk without an additional separator.

## Verified upstream facts

- The NVFP4 recipe uses vLLM 0.32.0, NeMo Speech main and CUDA 13; the documented example uses a 96 GB Blackwell GPU. SGLang-Omni does not support this checkpoint.
- Timestamp serving requires the V1 runner and CTC middleware. Requests opt into `capture_ctc_timestamps`, retain special tokens, and use non-streaming, single-completion mode. The top-level `ctc_timestamps` contains words, diarization, speaker mapping and an error field.
- MTP or no speculation supports timestamps; DFlash2 does not. Inputs are 16 kHz mono, up to one hour. Multispeaker support varies by language.

Source: [NVIDIA model card and serving recipe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4).

The report describes speaker-specific CTC alignment, speaker tokens 0–7, and an approximately 21 GB NVFP4 checkpoint. Disk size is not runtime VRAM. Its overlap results are promising but do not beat every comparator on every dataset. Validate this exact quantization on our material rather than extrapolating from headline results. Source: [NVIDIA technical report, sections 2.5–2.7 and multispeaker evaluation](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf).

## Current code and concrete gaps

| Area | Current behavior | Required change |
| --- | --- | --- |
| `h3_app/dubbing/segments.py`, `session.py` | Normalization preserves overlapping intervals but rebuilds only six fields; table round trips also lose extra metadata. | Versioned records with stable IDs, source-word associations and provenance; metadata-preserving editing and migration. |
| `h3_app/dubbing/pipeline.py`: `_translate`, `_take_mark` | Results/marks use start time and text as identity. | Stable IDs plus content revisions; simultaneous identical phrases must remain independent. |
| `pipeline.py`: `analyze`, `_transcript_key` | Three provider branches and provider-specific cache settings. | Nemotron adapter, capability metadata and revision-aware cache keys. |
| `pipeline.py`: `_reference_clip`, `speaker_reference`, `_plan` | References come from the shared vocal mix; longest suitable line wins, and a fallback can select another speaker. | Choose clean intervals for the correct speaker, including padding; never silently use another speaker. |
| `pipeline.py`: `smart_layout`, `slot_end` | Smart timing already allows source overlaps; natural timing looks at the next global line. | Explicit overlap constraints shared by timing modes; preserve existing non-overlap behavior. |
| `pipeline.py`: `_speech_track` | Synthesized takes already add together. Each undubbed line adds a slice of the shared vocal stem. | Union original intervals to prevent duplicate mixing; resolve mixed original/dub conflicts explicitly. |
| `pipeline.py`: `_word_timings`, `subtitle_lines` | Source words are aligned again with Qwen; subtitle extension considers global next starts. | Reuse valid native source words; independent speaker cue durations and translation alignment. |
| `h3_ui/dubbing/player.py` | Already renders all active cues and highlights words. | Preserve this behavior; add clear speaker labels and deterministic stacked layout. |
| `h3_ui/dubbing/handlers.py` | Editor identity uses index, start and source text. | Stable IDs through board, inspector, table, selection and asynchronous jobs. |

## Proposed data contract

Keep the current segment fields for compatibility and add `id`, `revision`, and `source_word_ids`. Keep immutable provider output separately from editable segments.

The transcript artifact should contain provider/model/runtime revisions, raw response location, normalized words (`id`, text, start, end, speaker), diarization activity, provider-to-session speaker mapping, and alignment status. Missing confidence remains missing; do not invent scores. Preserve raw speaker tags for diagnosis while mapping them to consistent session speaker IDs.

Store source timing and rendered dub timing separately. Derive overlap intervals from speaker activity where available, and from segment intervals otherwise, recording that distinction. Overlap is many-to-many, not a flag assigning a line to one fixed pair.

Normalization validates finite, bounded times and valid speaker references but never globally forces one word/line to start after another speaker finishes. Segment each speaker's words at pauses, punctuation and subtitle length limits before merging lines into display order. Preserve transcript order within the speaker; do not concatenate interleaved speakers into one sentence.

Edits retain IDs. Text/timing/speaker changes invalidate only affected alignment/reference/take artifacts. Translation jobs apply by ID only if the relevant input revision still matches. Persist a one-time migration for old sessions; no re-analysis should be necessary just to open them. Table editing must transport hidden IDs or use a server-side row mapping, rather than matching ambiguous text and timestamps.

## Implementation sequence

### 1. Runtime and output feasibility gate

On the deployment GPU, validate a pinned checkpoint and NeMo commit with the documented vLLM version in an isolated environment. Start with one request at a time and no speculative decoding; evaluate MTP after correctness. Record exact hardware, peak VRAM, cold start and warm processing time for short and progressively longer recordings.

Capture real responses for clean dialogue, two/three simultaneous speakers, silence, Mandarin/English code switching, and more than four speakers. Verify speaker mapping and actual word overlap behavior; the presence of eight tokens is not proof of eight-speaker quality. Exercise alignment errors, token truncation and cancellation. Keep failure responses as adapter fixtures.

Use Linux/WSL for this integration. Do not assume native Windows or another runtime exposes the same timestamp contract. Set the supported duration/context limits from measurements; reject unsupported lengths clearly for the first release. Automatic long-recording chunking is deferred because it requires speaker reconciliation and boundary deduplication.

### 2. Shared transcript foundation

Implement the versioned contract, migration, stable IDs and revision checks in `segments.py`, `session.py`, pipeline jobs and UI handlers. Add provider capabilities for diarization and native word timing, replacing hard-coded MOSS/Gemini checks where appropriate. Preserve existing Qwen, MOSS and Gemini behavior through adapters with optional metadata.

Add focused tests for simultaneous identical phrases, edits/reordering during translation or synthesis, metadata-preserving table round trips, old-session loading and cache invalidation.

### 3. Nemotron runtime and adapter

Add `h3_app/dubbing/nemotron.py` and dedicated runtime setup/configuration. Reuse existing process-management patterns from MOSS, but keep vLLM/NeMo dependencies outside the dubbing-tools and Drama 3 environments. Pin model and runtime revisions; persist compiler caches and logs.

Integrate with GPU ownership, interruption and model unload. Initially stop the server to release VRAM before translation/synthesis or competing GPU jobs; enable sleep/wake only after this runtime is verified to release memory correctly. Do not copy MOSS's memory-fraction calculation into vLLM without checking its semantics.

Send full-recording speaker-tagged ASR requests through the local chat-completions endpoint. Parse structured timing, not generated timestamp prose. Validate `error`, completion finish reason, word coverage and timestamp bounds. Keep incomplete results diagnosable but do not cache them as successful analyses. An alignment failure must be visible and retryable; do not silently fabricate precise timing or switch to a cloud service.

Wire the fourth provider through `pipeline.py`, `view.py`, `presenters.py`, handlers, setup and deployment configuration. Cache identity includes model revision, prompt/parser versions, audio preprocessing and timestamp configuration. Keep Hy-MT/Gemini translation initially; translated words must be aligned against generated target speech, not source CTC positions.

### 4. Overlap-safe references, scheduling and mixing

Select references from intervals containing only the intended speaker, subtracting all other speaker activity and accounting for reference padding. Prefer a clean same-speaker clip elsewhere when a line contains cross-talk. If none exists, require a saved voice or an explicit user-selected reference; show the reason in Review lines.

For timing, retain original overlap regions and use same-speaker succession plus cross-speaker interaction boundaries as constraints. Do not let a new independent-speaker scheduler create arbitrary overlaps in previously clean turn-taking. Smart timing may borrow genuine silence, but must not move a turn through an unrelated interruption. Separate intentional source overlap from overrun caused by a long translation.

Mix generated lines concurrently on speaker tracks with gain/headroom control before the final sum. Preserve undubbed original intervals once, using their union, rather than adding the same source samples for each line.

For a region where one overlapping speaker is dubbed and another is kept original, the shared vocal stem also contains the replaced speaker. V1 should flag this conflict and offer an explicit region policy: dub all affected voices, or keep the whole affected region original. A synthesis failure should mark the region for review and use the declared fallback, never silently claim clean speaker replacement. Apply this to the actual rendered overlap intervals as well as source activity. Independent selective preservation requires a later speaker-conditioned audio separation stage with its own quality gate.

### 5. Captions and review

Reuse native source words whenever their text, timing and speaker associations remain valid. Continue aligning translation words against isolated fitted TTS takes. Avoid re-aligning original cross-talk with a single-speaker aligner as an automatic fallback.

Keep simultaneous cues active for their real durations, labeled and stacked by speaker in the player. Extend review with overlap highlighting, reference-quality warnings and mixed original/dub conflict actions. Keep source and dub timelines distinct.

For exports, generate overlap-aware ASS for deterministic burned-in placement, and retain SRT compatibility using time-sliced, speaker-labeled multiline cues where simultaneous display is otherwise ambiguous. Verify both exported and in-app captions, including source-plus-translation mode and seeking into overlaps.

### 6. Validation and rollout

Run focused dubbing/session/runtime tests per change, not the full suite repeatedly. Add audio assertions proving both synthesized voices survive an overlap, original samples are not doubled, same-speaker identity remains stable and peaks remain controlled. Exercise strict, natural and Smart timing with nested overlaps and failed/disabled lines.

Use a small annotated evaluation set covering clean dialogue, interruptions, dense overlap, repeated phrases, music, Mandarin/English and long recordings. Compare Nemotron NVFP4 with MOSS and Qwen on speaker-attributed transcription, overlap word recall, timestamp accuracy, reference contamination and complete dubbed listening quality. Report latency and memory separately from recognition quality. Include representative Gemini comparisons only when cloud processing is deliberately selected.

Ship the new option as experimental until the deployment runtime passes and the overlap fixtures work end to end. Preserve existing defaults. Update `docs/features/dubbing.md` with actual hardware requirements and measured limitations.

## Completion criteria and deferred work

Completion means two simultaneous speakers can be transcribed, edited, translated, voiced, mixed and captioned concurrently without losing words, crossing identities or duplicating originals. Old sessions still open and existing providers retain behavior. Errors and uncertain alignments remain visible; native timestamps survive save/reload and are reused.

Deferred: automatic multi-hour stitching, Nemotron as a separate translator, native Windows serving, speculative performance tuning beyond measured MTP, and isolated speaker waveform extraction. The last item is necessary if the product must preserve one original speaker while replacing another inside cross-talk with no leakage; diarization alone cannot meet that requirement.
