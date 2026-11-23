"""Overlap prompts, model routing, speaker attribution, and reference reuse."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from h3_app.dubbing import gemini, nemotron, overlap, pipeline, segments, session, tools_worker, voices
from h3_app.dubbing.text import manual_transcript
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled


def take_result(generator):
    try:
        while True:
            next(generator)
    except StopIteration as done:
        return done.value


def words(text, offset=0.0):
    return [{"text": token, "start": offset + i * .7, "end": offset + i * .7 + .3}
            for i, token in enumerate(text.split())]


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.rows = segments.normalize([
            {"start": 0, "end": 3, "speaker": "A", "source": "Go find that key", "translation": "A translated"},
            {"start": .2, "end": 3.2, "speaker": "B", "source": "Please get my bag", "translation": "B translated"},
        ])
        self.hints = [{"start": .2, "end": 3., "speakers": ["A", "B"]}]

    def test_gemini_prompt_and_pasted_reply_share_overlap_format(self):
        for accept_video in (False, True):
            prompt = gemini.transcribe_prompt(target_language="English", translate=False, ignore_non_speech=True,
                                             accept_video=accept_video)
            self.assertIn("simultaneous speech", prompt)
            self.assertIn('"overlaps"', prompt)
        pasted = manual_transcript(json.dumps({"segments": self.rows, "overlaps": self.hints}))
        self.assertEqual(pasted["segments"], self.rows)
        self.assertEqual(segments.transcript_overlaps(pasted["overlaps"], {"A", "B"}, 4), self.hints)
        self.assertEqual(segments.transcript_overlaps(pasted["overlaps"], {"A", "B"}, 2), [])
        self.assertEqual(segments.transcript_overlaps(pasted["overlaps"], {"A"}, 4), [])
        self.assertEqual(segments.transcript_overlaps([{**self.hints[0], "start": float("nan")}], {"A", "B"}, 4), [])
        self.assertNotIn("overlaps", manual_transcript(json.dumps(self.rows)))

    def test_automatic_gemini_preserves_overlap_metadata(self):
        raw = json.dumps({"segments": self.rows, "overlaps": self.hints})
        with patch.object(gemini.gemini, "Session"), patch.object(gemini, "_generate", return_value=raw) as call:
            transcript = gemini.transcribe(Path("source.wav"), model=next(iter(gemini.GEMINI_PROMPT_MODELS)),
                                          api_key="test", target_language="English",
                                          translate=True, ignore_non_speech=True)
        self.assertIn('"overlaps"', call.call_args.args[2][0]["text"])
        self.assertEqual(transcript["segments"], self.rows)
        self.assertEqual(transcript["overlaps"], self.hints)

    def test_model_routing_uses_actual_participants_in_local_windows(self):
        models = overlap.tools_worker.OVERLAP_MODELS
        self.assertNotEqual(models[2], models[3])
        self.assertNotEqual(models[3], models[4])
        self.assertEqual(models[4], models[5])
        transcript = {"segments": self.rows, "activity": {"A": [(0, 4)], "B": [(0, 9)], "C": [(4, 9)]}}
        _, windows = overlap.windows_for(transcript, 9)
        self.assertEqual(windows, [overlap.Window(0, 4, ("A", "B")), overlap.Window(4, 9, ("B", "C"))])
        transcript["activity"][nemotron.UNKNOWN_SPEAKER] = [(0, 4)]
        _, windows = overlap.windows_for(transcript, 9)
        self.assertEqual(windows, [overlap.Window(4, 9, ("B", "C"))])
        transcript["activity"] = {speaker: [(0, 5)] for speaker in "ABCDEF"}
        _, windows = overlap.windows_for(transcript, 5)
        self.assertEqual(windows, [])

    def test_gemini_explicit_intervals_win_over_broad_turn_bounds(self):
        transcript = {"segments": self.rows, "overlaps": []}
        self.assertEqual(overlap.windows_for(transcript, 4), ([], []))
        transcript["overlaps"] = self.hints
        self.assertEqual(overlap.windows_for(transcript, 4)[0], [overlap.Window(.2, 3, ("A", "B"))])

    def test_gemini_hint_cannot_exclude_a_third_voice_inside_the_window(self):
        extra = {"start": 1, "end": 1.12, "speaker": "C", "source": "Wait"}
        transcript = {"segments": [*self.rows, extra], "overlaps": self.hints}
        overlaps, windows = overlap.windows_for(transcript, 4)
        self.assertTrue(overlaps)
        self.assertEqual(windows, [])

    def test_attribution_verifies_order_timing_and_identity(self):
        window = overlap.Window(10, 14, ("A", "B"))
        first, second = words("Go find that key"), words("Please get my bag", .2)
        stamped = [{**word, "speaker": speaker, "start": word["start"] + 10, "end": word["end"] + 10}
                   for speaker, series in (("A", first), ("B", second)) for word in series]
        expected = overlap.expected_words({"words": stamped, "segments": self.rows}, window)
        self.assertEqual(overlap.attribute(window, expected, [second, first]), {"A": 1, "B": 0})
        self.assertEqual(overlap.attribute(window, expected, [first, first]), {})
        self.assertEqual(overlap.attribute(window, expected, [second, words("Go find that key", 1.4)]), {"B": 0})
        self.assertEqual(overlap.attribute(window, expected, [second, first + second]), {"B": 0})

    def test_stem_asr_runs_without_expected_text_and_checks_word_times(self):
        with TemporaryDirectory() as tmp:
            worker = tools_worker.Worker(Path(tmp))
            good = SimpleNamespace(text="Find the key.", time_stamps=[
                SimpleNamespace(text=item["text"], start_time=item["start"], end_time=item["end"])
                for item in words("Find the key")
            ])
            bad = SimpleNamespace(text="This is unrelated.", time_stamps=good.time_stamps)
            with patch.object(worker, "_asr") as asr, patch.dict(
                    "sys.modules", {"soundfile": SimpleNamespace(info=lambda _: SimpleNamespace(duration=3.0))}), \
                    patch.object(tools_worker.traceback, "print_exc"):
                asr.return_value.transcribe.side_effect = [[good, bad, None], RuntimeError("ASR failed")]
                result = worker.transcribe_stems({"clips": ["first.wav", "second.wav", "third.wav"],
                                                  "language": "English"})
            asr.assert_called_once_with(True)
            self.assertEqual(asr.return_value.transcribe.call_args_list[0].kwargs,
                             {"audio": ["first.wav", "second.wav", "third.wav"],
                              "language": "English", "return_time_stamps": True})
            self.assertEqual(result["words"], [words("Find the key"), [], None])

    def test_preview_and_full_plan_share_references_and_fall_back_after_audio_or_edit_changes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = session.Session(root, {"pipeline": pipeline.PIPELINE_GEMINI, "duration": 4,
                                          "segments": self.rows, "transcript": {"segments": self.rows,
                                                                              "overlaps": self.hints}})
            audio = root / "audio.wav"
            saved.set_file("audio", audio)
            t = np.arange(pipeline.RATE * 4) / pipeline.RATE
            pipeline._write_wav(.05 * np.sin(2 * np.pi * 200 * t), pipeline.RATE, audio)
            attempts = iter((None, words("Go find that key"), words("Go find that key")))

            def run(op, payload, *_args, **_kwargs):
                if op == "transcribe_stems":
                    return {"words": [words("Please get my bag", .2), next(attempts)]}
                self.assertEqual(payload["speaker_count"], 2)
                paths = [str(Path(payload["windows"][0]).with_suffix(f".{i}.wav")) for i in (1, 2)]
                for path in paths:
                    shutil.copyfile(payload["windows"][0], path)
                return {"stems": [paths]}

            def plan(*indices):
                return take_result(pipeline._prepare_plan(saved, list(indices), pipeline.VOICE_OVERLAP,
                                                          runtime=None, free_gpu=lambda: None, hf_token=None))

            with patch.object(overlap.tools, "run", side_effect=run) as worker:
                failed, note = plan(0)
                self.assertTrue(failed[0].reference.name.startswith("clip_"))
                self.assertIn("stem transcription errors", note)
                preview, _ = plan(0)
                full, note = plan(0, 1)
                self.assertEqual(worker.call_count, 4)
                self.assertEqual(preview[0].reference, full[0].reference)
                self.assertEqual([line.transcript for line in full], ["Go find that key", "Please get my bag"])
                self.assertIn("2 isolated", note)
                pipeline._write_wav(.08 * np.sin(2 * np.pi * 200 * t), pipeline.RATE, audio)
                plan(0)
                self.assertEqual(worker.call_count, 6)

            saved.segments = [{**self.rows[0], "speaker": "C"}, self.rows[1]]
            with patch.object(overlap.tools, "run") as worker:
                fallback, note = plan(0)
                worker.assert_not_called()
                self.assertTrue(fallback[0].reference.name.startswith("clip_"))
                self.assertIn("1 used", note)

    def test_cancellation_propagates(self):
        with TemporaryDirectory() as tmp:
            saved = session.Session(Path(tmp), {"pipeline": pipeline.PIPELINE_GEMINI, "segments": self.rows,
                                                "transcript": {"segments": self.rows, "overlaps": self.hints}})
            with patch.object(overlap, "prepare", side_effect=JobCancelled("stopped")):
                with self.assertRaises(JobCancelled):
                    take_result(pipeline._prepare_plan(saved, [0], pipeline.VOICE_OVERLAP, runtime=None,
                                                       free_gpu=lambda: None, hf_token=None))

    def test_attribution_handles_bleed_and_rejects_collapsed_stems(self):
        window = overlap.Window(0, 6, ("A", "B"))
        full_a = words("Go find that silver key now")
        full_b = words("Please get my red bag today", .2)
        expected = {"A": full_a, "B": full_b}
        self.assertEqual(overlap.attribute(window, expected, [full_a, words("Go")]), {"A": 0})
        self.assertEqual(overlap.attribute(window, expected, [full_a, words("Go find that")]), {"A": 0})
        first = words("Go find that key")
        expected_short = {"A": first, "B": words("Please get my bag", .2)}
        self.assertEqual(overlap.attribute(window, expected_short, [first, first[:-1]]), {})
        self.assertEqual(overlap.attribute(window, expected, [full_a, full_a[:-1]]), {})

    def test_adjacent_same_speaker_overlaps_merge_into_one_window(self):
        hints = [{"start": 1.0, "end": 1.2, "speakers": ["A", "B"]},
                 {"start": 1.8, "end": 2.0, "speakers": ["A", "B"]}]
        transcript = {"segments": self.rows, "overlaps": hints}
        overlaps, windows = overlap.windows_for(transcript, 4.0)
        self.assertEqual(overlaps, [overlap.Window(1.0, 1.2, ("A", "B")), overlap.Window(1.8, 2.0, ("A", "B"))])
        self.assertEqual(windows, [overlap.Window(0.0, 3.25, ("A", "B"))])

    def test_alignment_language_and_unsupported_overlap_fallback(self):
        self.assertEqual(tools_worker.alignment_language("en"), "English")
        self.assertEqual(tools_worker.alignment_language("ZH"), "Chinese")
        self.assertEqual(tools_worker.alignment_language("Auto-detect"), "")
        self.assertIsNone(tools_worker.alignment_language("Arabic"))
        with TemporaryDirectory() as tmp:
            saved = session.Session(Path(tmp), {
                "pipeline": pipeline.PIPELINE_GEMINI,
                "source_language": "Arabic",
                "duration": 4,
                "segments": self.rows,
                "transcript": {"segments": self.rows, "overlaps": self.hints},
            })
            with patch.object(overlap.tools, "run") as worker:
                found, note = overlap.prepare(saved, {0: self.rows[0]}, runtime=None,
                                              free_gpu=lambda: None, hf_token=None, report=lambda _m: None)
            worker.assert_not_called()
            self.assertEqual(found, {})
            self.assertIn("word timing does not support Arabic", note)

    def test_worker_outage_skips_later_batches_and_solo_short_lines_skip_prepare(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            many_rows = segments.normalize([
                {"start": t, "end": t + 2.5, "speaker": "A", "source": "Go find that key", "translation": "A"}
                for t in (0.0, 10.0, 20.0, 30.0)
            ] + [
                {"start": t + .2, "end": t + 2.7, "speaker": "B", "source": "Please get my bag", "translation": "B"}
                for t in (0.0, 10.0, 20.0, 30.0)
            ])
            many_hints = [{"start": t + .2, "end": t + 2.5, "speakers": ["A", "B"]} for t in (0.0, 10.0, 20.0, 30.0)]
            saved = session.Session(root, {"pipeline": pipeline.PIPELINE_GEMINI, "duration": 35.0,
                                          "segments": many_rows, "transcript": {"segments": many_rows,
                                                                                "overlaps": many_hints}})
            audio = root / "audio.wav"
            saved.set_file("audio", audio)
            pipeline._write_wav(np.zeros(pipeline.RATE * 35, dtype=np.float32), pipeline.RATE, audio)
            lines = {i: many_rows[i] for i in range(len(many_rows))}
            with patch.object(overlap.tools, "run", side_effect=H3Error("worker down")) as worker:
                found, note = overlap.prepare(saved, lines, runtime=None,
                                              free_gpu=lambda: None, hf_token=None, report=lambda _m: None)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(found, {})
            self.assertIn("4 windows with worker errors", note)

            solo_rows = segments.normalize([
                {"start": 0.0, "end": 1.0, "speaker": "A", "source": "Hi.", "translation": "Hi translated"},
                {"start": 5.0, "end": 8.0, "speaker": "A", "source": "Go find that key.", "translation": "Long translated"},
            ])
            solo_dir = root / "solo"
            solo = session.Session(solo_dir, {"pipeline": pipeline.PIPELINE_GEMINI, "duration": 9.0,
                                              "segments": solo_rows,
                                              "transcript": {"segments": solo_rows, "overlaps": []}})
            solo_audio = solo_dir / "audio.wav"
            solo.set_file("audio", solo_audio)
            pipeline._write_wav(np.zeros(pipeline.RATE * 9, dtype=np.float32), pipeline.RATE, solo_audio)
            with patch.object(overlap, "prepare") as prepare_call:
                planned, note = take_result(pipeline._prepare_plan(
                    solo, [0], pipeline.VOICE_OVERLAP, runtime=None, free_gpu=lambda: None, hf_token=None))
            prepare_call.assert_not_called()
            self.assertEqual(len(planned), 1)
            self.assertEqual(note, "")

    def test_speaker_activity_handles_null_overlaps_and_normalizes_speaker_ids(self):
        with TemporaryDirectory() as tmp:
            saved = session.Session(Path(tmp), {
                "pipeline": pipeline.PIPELINE_GEMINI,
                "segments": self.rows,
                "transcript": {"segments": self.rows, "overlaps": None},
            })
            self.assertIn("A", voices.speaker_activity(saved))
            saved.data["transcript"]["overlaps"] = [
                None,
                {"start": 1.0, "end": 2.0, "speakers": ["Speaker 2", "spk_1"]},
                {"start": 2.0, "end": 1.0, "speakers": ["A"]},
                {"start": "bad", "end": 2.0, "speakers": ["A"]},
            ]
            activity = voices.speaker_activity(saved)
            self.assertIn((1.0, 2.0), activity["speaker2"])
            self.assertIn((1.0, 2.0), activity["speaker1"])
