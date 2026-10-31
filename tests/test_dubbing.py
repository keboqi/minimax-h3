"""Dub & Translate and Media downloads: pure helpers, storage, handlers and an ffmpeg-backed dub."""

from __future__ import annotations

import json
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import patch
import wave

import numpy as np

from h3_app import video_download
from h3_app.dubbing import moss, pipeline, segments, session as sessions, tools, tools_worker
from h3_app.errors import H3Error

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
REAL_WORD_TIMINGS = pipeline._word_timings


def runtime_for(root: Path):
    return SimpleNamespace(outputs_dir=root / "outputs", output_dir=root / "comfy-output",
                           comfy_dir=root / "ComfyUI", script_dir=root, generation_timeout=600)


class Workspace(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory(prefix="h3-dub-")
        self.root = Path(self._tmp.name)
        self.runtime = runtime_for(self.root)
        self.runtime.outputs_dir.mkdir(parents=True)
        env = patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(self.root / "workspace")})
        env.start()
        self.addCleanup(env.stop)

        # A dub times its subtitle words on the real tools worker; tests of that call REAL_WORD_TIMINGS.
        def no_word_timing(*_args, **_kwargs):
            return {}
            yield

        words = patch.object(pipeline, "_word_timings", no_word_timing)
        words.start()
        self.addCleanup(words.stop)

    def tearDown(self):
        self._tmp.cleanup()


class SegmentTests(unittest.TestCase):
    def test_times_parse_leniently_and_format_back(self):
        self.assertEqual(segments.parse_time("01:13.500"), 73.5)
        self.assertEqual(segments.parse_time("0:01:13,5"), 73.5)
        self.assertEqual(segments.parse_time(2.25), 2.25)
        self.assertEqual(segments.parse_time("4.5s"), 4.5)
        self.assertIsNone(segments.parse_time("soon"))
        self.assertEqual(segments.format_time(73.5), "01:13.500")
        self.assertEqual(segments.format_time(3725.004), "01:02:05.004")
        self.assertEqual(segments.srt_time(3725.004), "01:02:05,004")

    def test_gemini_reply_with_fence_speakers_and_bad_rows(self):
        reply = "```json\n" + json.dumps({
            "speakers": [{"id": "Speaker 2", "description": "older man, calm"}],
            "segments": [
                {"start": "00:03.000", "end": "00:04.500", "speaker": "Speaker 2",
                 "source_text": "Second", "translated_text": "Deuxième"},
                {"start": "00:01.000", "end": "00:02.000", "speaker": 1, "source_text": "First",
                 "translated_text": "Premier"},
                {"start": "later", "end": "00:05", "source_text": "dropped"},
            ]}) + "\n```"
        found, speakers = segments.parse_transcript_reply(reply)
        self.assertEqual([s["source"] for s in found], ["First", "Second"])
        self.assertEqual([s["speaker"] for s in found], ["speaker1", "speaker2"])
        self.assertEqual(speakers, {"speaker2": "older man, calm"})
        self.assertTrue(all(s["dub"] for s in found))

    def test_translation_reply_requires_every_line(self):
        self.assertEqual(segments.parse_translation_reply('Sure: ["a", {"text": "b"}]', 2), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            segments.parse_translation_reply('["a", "b"]', 3)

    def test_table_round_trip_keeps_edits_and_skips_empty_rows(self):
        lines = segments.normalize([
            {"start": 1, "end": 2, "speaker": "speaker1", "source": "Hi", "translation": "Hola"},
            {"start": 3, "end": 4, "speaker": "speaker2", "source": "Bye", "translation": ""},
        ])
        rows = segments.to_rows(lines)
        rows[1][5] = "Adiós"
        rows[1][6] = "false"
        rows.append([3, "", "", "", "", "", True])
        restored = segments.from_rows(rows)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[1]["translation"], "Adiós")
        self.assertFalse(restored[1]["dub"])
        self.assertEqual(restored[0]["start"], 1.0)

    def test_srt_numbers_only_lines_with_text(self):
        lines = segments.normalize([
            {"start": 0.5, "end": 1.25, "source": "One", "translation": ""},
            {"start": 2, "end": 3, "source": "Two", "translation": "Dos"},
        ])
        self.assertEqual(segments.to_srt(lines, "translation"), "1\n00:00:02,000 --> 00:00:03,000\nDos\n")

    def test_line_numbers(self):
        self.assertEqual(segments.parse_line_numbers("3, 5-6", 6), [2, 4, 5])
        self.assertEqual(segments.parse_line_numbers("", 6), [])
        with self.assertRaisesRegex(ValueError, "No line 7"):
            segments.parse_line_numbers("7", 6)
        with self.assertRaisesRegex(ValueError, "not a line number"):
            segments.parse_line_numbers("two", 6)


class WorkerHelperTests(unittest.TestCase):
    def test_spans_merge_across_short_pauses(self):
        self.assertEqual(tools_worker.merge_spans([(3, 4), (0, 1), (1.2, 2)], 0.35), [(0.0, 2.0), (3.0, 4.0)])

    def test_words_join_with_cjk_and_punctuation(self):
        self.assertEqual(tools_worker.join_words(["Hello", ",", "world", "!"]), "Hello, world!")
        self.assertEqual(tools_worker.join_words(["你", "好", "。", "OK"]), "你好。OK")

    def test_alignment_lines_cut_at_sentence_ends_and_offset(self):
        item = lambda text, start, end: SimpleNamespace(text=text, start_time=start, end_time=end)  # noqa: E731
        lines = tools_worker.lines_from_alignment(
            [item("Hi.", 0.0, 0.4), item("How", 0.6, 0.8), item("are", 0.8, 1.0), item("you?", 1.0, 1.3)],
            10.0, 12.0, "fallback")
        self.assertEqual(lines, [{"start": 10.0, "end": 10.4, "text": "Hi."},
                                 {"start": 10.6, "end": 11.3, "text": "How are you?"}])
        self.assertEqual(tools_worker.lines_from_alignment([], 1.0, 2.0, "Whole span"),
                         [{"start": 1.0, "end": 2.0, "text": "Whole span"}])

    def test_translation_cleanup(self):
        self.assertEqual(tools_worker.clean_translation("Translation: Bonjour<|im_end|>"), "Bonjour")

    def test_translation_commentary_check_preserves_dialogue(self):
        source = "而如果我们往地下看下一代已经在孕育之中了"
        for reply in ("根据提供的背景信息，可以翻译为：\n***而如果我们往地下看……***\n（说明：结合背景……）",
                      "Based on the provided context, the translation is: Hello.",
                      "Here is the translation: Hello.", ""):
            with self.subTest(reply=reply):
                self.assertTrue(tools_worker.invalid_translation(reply, source))
        for reply in (source, "Hello!", "He said, ‘Here is the translation.’", "说明情况后，我们走了。"):
            self.assertFalse(tools_worker.invalid_translation(reply, source))
        self.assertFalse(tools_worker.invalid_translation("根据背景，我们决定出发。", "根据背景，我们决定出发。"))

    def translation_worker(self, replies):
        class Encoded(dict):
            def to(self, _device):
                return self

        tokenizer = unittest.mock.Mock(pad_token_id=0)
        tokenizer.side_effect = lambda prompts, **_kw: Encoded(
            input_ids=np.zeros((len(prompts), 2)), attention_mask=np.ones((len(prompts), 2)))
        tokenizer.decode.side_effect = replies
        model = unittest.mock.Mock(device="cpu")
        model.generate.side_effect = lambda **kw: np.zeros((len(kw["input_ids"]), 3))
        worker = object.__new__(tools_worker.Worker)
        worker._translator = lambda: (tokenizer, model)
        worker.torch = lambda: SimpleNamespace(inference_mode=nullcontext)
        worker.progress = unittest.mock.Mock()
        return worker, tokenizer, model

    def test_hy_mt_retries_only_bad_lines_without_context(self):
        worker, tokenizer, model = self.translation_worker([
            "1. Plus tôt\n2. Bonjour", "根据提供的背景信息，可以翻译为：……", "下一代已经在孕育之中了。"])
        result = worker.translate({"texts": ["Hello", "", "The next generation is developing."],
                                   "contexts": ["earlier line", "", "previous line"],
                                   "notes": "Keep the tone.", "target_language": "Chinese"})
        self.assertEqual(result["texts"], ["Bonjour", "", "下一代已经在孕育之中了。"])
        prompts = [call.args[0][0]["content"] for call in tokenizer.apply_chat_template.call_args_list]
        self.assertIn("[Translation Notes - Reference Only]\nKeep the tone.\n[/Translation Notes]", prompts[0])
        self.assertIn("[Source Text]\n1. earlier line\n2. Hello\n[/Source Text]", prompts[0])
        self.assertIn("1. previous line", prompts[1])
        self.assertNotIn("previous line", prompts[2])
        self.assertNotIn("Keep the tone.", prompts[2])
        self.assertIn("[Source Text]\nThe next generation is developing.\n[/Source Text]", prompts[2])
        self.assertEqual([call.kwargs["do_sample"] for call in model.generate.call_args_list], [True, False])

    def test_hy_mt_keeps_repeated_commentary_or_empty_results_for_review(self):
        for replies in (["Here is the translation: Hi", "Here is the translation: Hi"], ["", ""],
                        ["Here is the translation: Hi", ""]):
            with self.subTest(replies=replies):
                worker, _tokenizer, model = self.translation_worker(replies)
                self.assertEqual(worker.translate({"texts": ["", "Hello"]}),
                                 {"texts": ["", replies[1] or replies[0]]})
                self.assertEqual(model.generate.call_count, 2)

    def test_hy_mt_accepts_translation_without_retry(self):
        worker, _tokenizer, model = self.translation_worker(["Bonjour"])
        self.assertEqual(worker.translate({"texts": ["Hello", ""]}), {"texts": ["Bonjour", ""]})
        self.assertEqual(model.generate.call_count, 1)

    def test_player_identifies_missing_word_timing_per_track(self):
        from h3_ui.dubbing.player import player_html

        cue = {"start": 0, "end": 1, "shown_start": 0, "shown_end": 1,
               "speaker": "speaker1", "source": "Hello", "translation": "Salut"}
        source = [{"text": "Hello", "start": 0, "end": 1}]
        translation = [{"text": "Salut", "start": 0, "end": 1}]
        for source_words, translation_words, missing in (([], translation, "original"),
                                                       (source, [], "translated"),
                                                       ([], [], "original and translated"),
                                                       (source, translation, "")):
            with self.subTest(missing=missing):
                player = pipeline.Player(None, Path("dub.mp3"), Path("original.mp3"), None, 1,
                                         [{**cue, "source_words": source_words,
                                           "translation_words": translation_words}], word_timing=True)
                html = player_html(player, key="test")
                if missing:
                    self.assertIn(f"Word timing unavailable: {missing}. Plain subtitles shown.", html)
                else:
                    self.assertNotIn("Word timing unavailable", html)
                player.word_timing = False
                self.assertNotIn("Word timing unavailable", player_html(player, key="test"))

    def test_worker_protocol_reports_failures_and_keeps_serving(self):
        replies = []
        stream = SimpleNamespace(write=replies.append, flush=lambda: None)
        with TemporaryDirectory() as models, patch.object(tools_worker.Worker, "translate",
                                                          return_value={"texts": ["x"]}):
            tools_worker.serve(models, requests=[json.dumps({"id": 1, "op": "bogus"}),
                                                 json.dumps({"id": 2, "op": "translate", "texts": ["a"]})],
                               replies=stream)
        first, second = (json.loads(line) for line in replies)
        self.assertFalse(first["ok"])
        self.assertIn("Unknown dubbing tools request", first["error"])
        self.assertEqual(second, {"id": 2, "loaded": [], "ok": True, "texts": ["x"]})

    def test_first_use_install_compiles_with_gcc_when_clang_is_missing(self):
        from h3_app.workers import env as worker_env

        with TemporaryDirectory() as root, \
                patch.dict(os.environ, {"H3_DUB_TOOLS_VENV": str(Path(root) / "venv")}), \
                patch.object(worker_env.shutil, "which", side_effect=lambda name: None if name == "clang" else name), \
                patch.object(worker_env, "uv_command", return_value=["uv"]), \
                patch.object(worker_env, "run_process",
                             side_effect=lambda *a, **k: (Path(root) / "venv").mkdir(exist_ok=True)) as run:
            os.environ.pop("CC", None)
            tools.prepare(runtime_for(Path(root)), Path(root) / "setup.log")
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(call.kwargs["env"] == {"CC": "gcc", "CXX": "g++"} for call in run.call_args_list))

    def test_modal_builds_the_dub_tools_with_gcc(self):
        source = (Path(__file__).resolve().parents[1] / "modal_h3.py").read_text(encoding="utf-8")
        self.assertIn('f"CC=gcc CXX=g++ {shlex.join(command)}" for command in dub_tools_install_commands', source)

    def test_moss_replies_become_timed_speaker_lines(self):
        reply = {"text": "ignored", "segments": [
            {"start": 0.48, "end": 1.66, "text": "[S01]Welcome everyone"},
            {"start": 2.1, "end": 4.25, "text": "[S02] The new pipeline [3.0] is ready"},
            {"start": 5.0, "end": 5.2, "text": "[S01]"}]}
        self.assertEqual(moss.reply_lines(reply), [
            {"start": 0.48, "end": 1.66, "speaker": "S01", "text": "Welcome everyone"},
            {"start": 2.1, "end": 4.25, "speaker": "S02", "text": "The new pipeline is ready"}])
        # Without parsed segments the raw transcript is read; a missing end runs to the next start.
        compact = ("[0.48][S01]Welcome everyone[1.66][5.00][S01]No end time here"
                   "[6.20][S02]Last words")
        self.assertEqual(moss.reply_lines({"text": compact, "segments": []}, duration=7.0), [
            {"start": 0.48, "end": 1.66, "speaker": "S01", "text": "Welcome everyone"},
            {"start": 5.0, "end": 6.2, "speaker": "S01", "text": "No end time here"},
            {"start": 6.2, "end": 7.0, "speaker": "S02", "text": "Last words"}])
        self.assertEqual(moss.parse_transcript("no tags at all"), [])

    def test_long_moss_turns_split_into_subtitle_lines(self):
        long_text = "This is one sentence. " * 8
        lines = pipeline.moss_lines([{"start": 0.0, "end": 20.0, "speaker": "S02", "text": long_text},
                                     {"start": 21.0, "end": 22.0, "speaker": "S01", "text": "Short."}])
        self.assertGreater(len(lines), 2)
        self.assertTrue(all(line["speaker"] == "speaker2" for line in lines[:-1]))
        self.assertEqual(lines[-1], {"start": 21.0, "end": 22.0, "speaker": "speaker1", "source": "Short.",
                                     "translation": "", "dub": True})

    def test_moss_serves_from_drama3s_sglang_environment(self):
        from h3_app.drama3.serving import environment

        runtime = runtime_for(Path("/tmp/h3"))
        with patch.dict(os.environ):
            os.environ.pop("H3_MOSS_SGLANG_MEMORY_GIB", None)
            settings = moss.MossSettings.from_environment(runtime)
        self.assertEqual(settings.venv, environment.default_venv(runtime.script_dir))
        # SGLang takes the fraction of the memory free at launch: 19 GiB on an empty RTX PRO 6000
        # is 0.2 of it, and beside an awake Drama 3 (about 46 GiB left) a larger share.
        self.assertEqual(settings.memory_fraction(95.6 * 1024), 0.199)
        self.assertEqual(settings.memory_fraction(46 * 1024), 0.413)
        self.assertEqual(settings.memory_fraction(15 * 1024), moss.MAX_FRACTION)
        self.assertAlmostEqual(settings.footprint_mib() / 1024, 21.1, places=1)
        with patch.object(moss, "gpu_memory", return_value=(46 * 1024.0, 95.6 * 1024, "")):
            command = moss.MossServer(settings).command()
        self.assertEqual(command[1:3], ["serve", "--model-path"])
        flag = lambda name: command[command.index(name) + 1]
        self.assertEqual(flag("--asr.engine.mem_fraction_static"), "0.413")
        self.assertEqual((flag("--asr.engine.max_running_requests"), flag("--asr.engine.cuda_graph_max_bs")),
                         ("1", "1"))

    def test_other_jobs_sleep_moss_and_unload_stops_it(self):
        with patch.object(moss, "release_all") as release, patch.object(moss, "stop_all") as stop:
            moss.release_for_job(moss.FAMILY)
            release.assert_not_called()
            for family in ("dubbing", "dubbing_lines", "drama3", "ltx"):
                moss.release_for_job(family)
            self.assertEqual(release.call_count, 4)
            moss.release_for_job("unload")
            stop.assert_called_once()

    def test_moss_sleeps_once_its_lines_are_back(self):
        server = unittest.mock.Mock()
        server.transcribe.return_value = {"segments": [{"start": 0.0, "end": 1.0, "text": "[S01]Hi"}]}
        runtime = runtime_for(Path("/tmp/h3"))
        with patch.object(moss, "supported", return_value=True), patch.object(moss, "shared_server", return_value=server), \
                patch.object(moss.environment, "prepare"), patch.object(moss, "ensure_model"), \
                patch.object(moss, "_make_room"):
            lines = moss.transcribe(runtime, Path("speech.wav"), duration=10.0, free_gpu=lambda: None)
            self.assertEqual(lines, [{"start": 0.0, "end": 1.0, "speaker": "S01", "text": "Hi"}])
            server.ensure_ready.assert_called_once()
            server.release.assert_called_once()
            server.transcribe.side_effect = H3Error("boom")
            with self.assertRaises(H3Error):
                moss.transcribe(runtime, Path("speech.wav"), duration=10.0, free_gpu=lambda: None)
            self.assertEqual(server.release.call_count, 2)
            with self.assertRaisesRegex(H3Error, "up to 90 minutes"):
                moss.transcribe(runtime, Path("speech.wav"), duration=91 * 60.0, free_gpu=lambda: None)

    def test_other_jobs_stop_the_worker(self):
        with patch.object(tools, "stop") as stop:
            tools.release_for_job("dubbing")
            stop.assert_not_called()
            tools.release_for_job("ltx")
            stop.assert_called_once()


class TimingTests(unittest.TestCase):
    def test_slot_ends_before_the_next_line_but_never_before_its_own_end(self):
        lines = segments.normalize([{"start": 0, "end": 2, "source": "a"}, {"start": 3, "end": 4, "source": "b"},
                                    {"start": 3.5, "end": 3.8, "source": "c"}])
        self.assertAlmostEqual(pipeline.slot_end(lines, 0, 10), 2.95)
        # Line c overlaps b in the source (another speaker), so it does not end b's slot.
        self.assertEqual(pipeline.slot_end(lines, 1, 10), 10)
        self.assertEqual(pipeline.slot_end(lines, 2, 10), 10)

    def test_fit_leaves_short_takes_and_caps_the_speed_up(self):
        take = np.zeros(48000, dtype=np.float32)
        same, speed, note = pipeline.fit(take, 24000, 3.0, 1.35)
        self.assertIs(same, take)
        self.assertEqual((speed, note), (1.0, ""))
        with patch.object(pipeline, "_stretch",
                          side_effect=lambda s, r, f, cache=None: s[: int(len(s) / f)]) as stretch:
            fitted, speed, note = pipeline.fit(take, 24000, 1.0, 1.35)
        self.assertAlmostEqual(stretch.call_args.args[2], 1.35)
        self.assertEqual(speed, 1.35)
        self.assertIn("longer than its slot", note)

    def test_speaker_reference_prefers_the_longest_clip_in_range(self):
        lines = segments.normalize([
            {"start": 0, "end": 1, "speaker": "a", "source": "short"},
            {"start": 2, "end": 7, "speaker": "a", "source": "five"},
            {"start": 8, "end": 16, "speaker": "a", "source": "eight"},
            {"start": 20, "end": 40, "speaker": "a", "source": "twenty"},
            {"start": 41, "end": 42, "speaker": "b", "source": "one"},
        ])
        self.assertEqual(pipeline.speaker_reference(lines, "a"), 2)
        self.assertEqual(pipeline.speaker_reference(lines, "b"), 4)
        self.assertIsNone(pipeline.speaker_reference(lines, "c"))

    def test_subtitles_stay_up_while_a_line_is_spoken_but_not_over_the_next(self):
        lines = segments.normalize([{"start": 0, "end": 1, "source": "a", "translation": "A"},
                                    {"start": 3, "end": 4, "source": "b", "translation": "B"}])
        shown = pipeline.subtitle_lines(lines, {0: (0.0, 5.0), 1: (3.0, 6.5)})
        self.assertEqual([s["end"] for s in shown], [2.95, 6.5])
        self.assertEqual([s["end"] for s in pipeline.subtitle_lines(lines, {})], [1.0, 4.0])

    def smart(self, spans, natural, tightened=None, max_speed=1.35):
        lines = segments.normalize([{"start": s, "end": e, "source": "x", "translation": "y"} for s, e in spans])
        return pipeline.smart_layout(lines, natural, tightened or (lambda index: natural[index]), max_speed)

    # Line 1 sits between a short line 0 (its take ends at 0.8) and line 2 at 2.6: its own slot ends at
    # 2.5 (before the 0.1 s gap) and it may start from 0.9 (0.1 s after line 0's take).
    SPANS = [(0, 1), (1.2, 2.0), (2.6, 3.0)]

    def test_smart_keeps_a_take_that_fits_at_its_own_start_and_pace(self):
        placed = self.smart(self.SPANS, {0: 0.8, 1: 1.2, 2: 0.3})
        self.assertEqual((placed[1].start, placed[1].tempo, placed[1].tighten), (1.2, 1.0, False))
        self.assertAlmostEqual(placed[1].seconds, 1.2)

    def test_smart_starts_a_long_take_only_as_early_as_it_needs(self):
        placed = self.smart(self.SPANS, {0: 0.8, 1: 1.4, 2: 0.3})[1]
        self.assertEqual((placed.start, placed.tempo, placed.tighten), (1.1, 1.0, False))

    def test_smart_shortens_pauses_before_speeding_up(self):
        placed = self.smart(self.SPANS, {0: 0.8, 1: 1.8, 2: 0.3}, tightened=lambda index: 1.5)[1]
        self.assertTrue(placed.tighten)
        self.assertEqual((placed.start, placed.tempo), (1.0, 1.0))
        self.assertAlmostEqual(placed.seconds, 1.5)

    def test_smart_speeds_up_gently_within_its_own_slot(self):
        placed = self.smart(self.SPANS, {0: 0.8, 1: 1.8, 2: 0.3})[1]
        self.assertEqual(placed.start, 0.9)
        self.assertAlmostEqual(placed.tempo, 1.8 / 1.6)
        self.assertFalse(placed.squeezed)

    def test_smart_pushes_the_next_line_into_time_it_does_not_need(self):
        placed = self.smart(self.SPANS, {0: 0.8, 1: 2.3, 2: 0.3})
        self.assertAlmostEqual(placed[1].tempo, 2.3 / 2.1)  # 1.6 s of its own slot and 0.5 s of line 2's
        self.assertAlmostEqual(placed[2].start, 3.1)  # line 2 starts 0.5 s late, 0.1 s after line 1
        self.assertAlmostEqual(placed[2].start - (placed[1].start + placed[1].seconds), 0.1)
        self.assertFalse(placed[1].squeezed)

    def test_smart_speeds_up_to_the_limit_marks_and_never_moves_original_voice_lines(self):
        placed = self.smart(self.SPANS, {0: 0.8, 1: 3.0})  # line 2 keeps its original voice
        self.assertNotIn(2, placed)
        self.assertTrue(placed[1].squeezed)
        self.assertEqual(placed[1].tempo, 1.35)
        self.assertIn("overlaps the next line by 0.6s", pipeline.smart_note(placed[1]))
        self.assertIn("sped up 1.35x", pipeline.smart_note(self.smart(self.SPANS, {0: 0.8, 1: 2.16})[1]))

    def test_smart_never_starts_a_line_more_than_half_a_second_late(self):
        placed = self.smart([(0, 1), (1.2, 2.0), (2.1, 3.0)], {0: 0.8, 1: 5.0, 2: 0.5})
        self.assertAlmostEqual(placed[2].start, 2.1 + pipeline.SMART_DELAY)

    def test_smart_lets_speakers_who_overlap_in_the_source_overlap(self):
        placed = self.smart([(0, 2), (1, 3)], {0: 2.0, 1: 2.0})
        self.assertEqual([placed[0].start, placed[1].start], [0.0, 1.0])
        self.assertEqual([placed[0].tempo, placed[1].tempo], [1.0, 1.0])

    def test_shorten_pauses_keeps_edges_and_shortens_long_inner_silence(self):
        from h3_app.drama3.serving.audio import shorten_pauses

        rate, sound = 24000, np.full(24000 // 2, 0.3, dtype=np.float32)
        quiet = lambda seconds: np.zeros(int(seconds * rate), dtype=np.float32)
        take = np.concatenate([quiet(0.2), sound, quiet(0.6), sound, quiet(0.2), sound, quiet(0.2)])
        short = shorten_pauses(take, rate, longer_than=0.25, keep=0.15)
        self.assertAlmostEqual(len(short) / rate, len(take) / rate - 0.45, delta=0.02)

    def test_atempo_chain_keeps_each_stage_in_range(self):
        self.assertEqual(pipeline.atempo_chain(3.0), "atempo=2.000000,atempo=1.500000")
        self.assertEqual(pipeline.atempo_chain(0.3), "atempo=0.500000,atempo=0.600000")
        self.assertEqual(pipeline.atempo_chain(1.25), "atempo=1.250000")

    def test_match_reports_large_tempo_changes(self):
        take = np.zeros(4 * 24000, dtype=np.float32)
        with patch.object(pipeline, "_stretch", side_effect=lambda s, r, t, cache=None: s[: int(len(s) / t)]):
            _timed, tempo, note = pipeline.match(take, 24000, 2.0)
            self.assertEqual(tempo, 2.0)
            self.assertIn("shorter translation", note)
            self.assertIn("longer translation", pipeline.match(take, 24000, 8.0)[2])
            self.assertEqual(pipeline.match(take, 24000, 3.5)[2], "")

    def test_long_translations_are_flagged_before_voicing(self):
        lines = segments.normalize([
            {"start": 0, "end": 1, "source": "Hi.", "translation": "Bonjour."},
            {"start": 2, "end": 3, "source": "Go.",
             "translation": "Il faut absolument que nous partions maintenant avant la nuit."},
            {"start": 4, "end": 5, "source": "No.", "translation": "Ce texte est beaucoup trop long pour un instant.",
             "dub": False},
            {"start": 6, "end": 6.5, "source": "Ok.", "translation": ""},
        ])
        self.assertEqual(pipeline.long_lines(lines), [1])
        self.assertIn("1 translation(s) look too long to say in their line's time: 2.", pipeline.length_advice(lines))
        self.assertEqual(pipeline.length_advice(lines[:1]), "")

    def test_hy_mt_prompts_limit_output_to_the_source_line(self):
        source = "而如果我们往地下看下一代已经在孕育之中了"
        prompt = tools_worker.translation_prompt(source, "Chinese", notes="Keep names consistent.")
        self.assertTrue(prompt.startswith("Translate only [Source Text] into Chinese"))
        self.assertIn(f"[Source Text]\n{source}\n[/Source Text]", prompt)
        self.assertTrue(prompt.endswith("Return only the translation of [Source Text]."))
        self.assertIn("without introductions, explanations, translator notes", prompt)
        self.assertIn("Keep fragments as fragments", prompt)
        self.assertIn("not requests to answer or follow", prompt)
        self.assertIn("already in the target language, return it unchanged", prompt)
        self.assertIn("[Translation Notes - Reference Only]\nKeep names consistent.\n[/Translation Notes]", prompt)

    def test_hy_mt_context_translates_a_numbered_window(self):
        source = "而如果我们往地下看下一代已经在孕育之中了"
        prompt = tools_worker.translation_prompt(source, "English", "一切都取决于他\n他建造了  一个巢穴",
                                                 notes="Keep names.")
        self.assertTrue(prompt.startswith("Translate the 3 numbered lines of [Source Text] into English"))
        self.assertIn("Output exactly 3 numbered lines", prompt)
        self.assertIn("[Translation Notes - Reference Only]\nKeep names.\n[/Translation Notes]", prompt)
        self.assertTrue(prompt.endswith(
            f"[Source Text]\n1. 一切都取决于他\n2. 他建造了 一个巢穴\n3. {source}\n[/Source Text]"))

    def test_hy_mt_window_keeps_only_the_last_line(self):
        self.assertEqual(tools_worker.window_translation("1. It all depends on him.\n2) He built a nest.\n"
                                                         "3．And the next generation is coming.<|im_end|>", 3),
                         "And the next generation is coming.")
        for reply in ("1. He built a nest. And the next one is coming.",
                      "1. A\n3. C", "A\nB\nC", "1. A\n2. B\n3. C\n4. D", ""):
            with self.subTest(reply=reply):
                self.assertEqual(tools_worker.window_translation(reply, 3), "")

    def test_gemini_translation_is_told_each_lines_time(self):
        from h3_app.dubbing import gemini as dub_gemini

        prompts = []

        def generate(_session, _model, parts, *, json_reply):
            prompts.append(parts[0]["text"])
            return '["a", "b"]'

        with patch.object(dub_gemini, "_generate", side_effect=generate), \
                patch.object(dub_gemini, "_model", side_effect=lambda model: model), \
                patch.object(dub_gemini.gemini, "api_key", return_value="key"), \
                patch.object(dub_gemini.gemini, "Session"):
            self.assertEqual(dub_gemini.translate(["One", "Two"], model="m", api_key="", target_language="French",
                                                  seconds=[1.24, 3.0]), ["a", "b"])
            dub_gemini.translate(["One", "Two"], model="m", api_key="", target_language="French")
        self.assertIn('"seconds": 1.2', prompts[0])
        self.assertIn("seconds it lasts on screen", prompts[0])
        self.assertIn("about as long to say as the original", prompts[1])
        self.assertNotIn("seconds", prompts[1])

    def test_filter_paths_escape_drive_colons(self):
        self.assertEqual(pipeline._filter_path(Path("C:/a b/it's.srt"))[:12], "'C\\:/a b/it\\")


class DownloadTests(Workspace):
    def test_format_fallbacks_end_with_any_format(self):
        chain = video_download.format_fallbacks("720p")
        self.assertIn("height<=720", chain[0])
        self.assertEqual(chain[-2:], ["best", None])
        self.assertEqual(len(chain), len(set(chain)))

    def test_domains(self):
        self.assertEqual(video_download.site_domain("https://www.youtube.com/watch?v=x"), "youtube.com")
        self.assertEqual(video_download.site_domain("https://youtu.be/x"), "youtube.com")
        self.assertEqual(video_download.site_domain("https://m.bilibili.com/video/1"), "bilibili.com")
        self.assertEqual(video_download.normalize_domain("https://www.X.com/home"), "x.com")
        with self.assertRaises(H3Error):
            video_download.normalize_domain("not a domain")

    def test_curl_cookies_are_saved_per_site_and_found_for_its_urls(self):
        store = video_download.CookieStore(video_download.cookies_dir(self.runtime))
        curl = ("curl 'https://www.youtube.com/youtubei/v1/next' -H 'accept: */*' "
                "-H 'cookie: SID=abc; HSID=def=1' --compressed")
        self.assertEqual(video_download.parse_curl_cookies(curl), {"SID": "abc", "HSID": "def=1"})
        self.assertEqual(video_download.import_curl(store, curl), ("youtube.com", 2))
        self.assertEqual(store.sites(), {"youtube.com": 2})
        saved = store.file_for_url("https://youtu.be/abc")
        self.assertTrue(saved.read_text().startswith("# Netscape HTTP Cookie File"))
        self.assertIn(".youtube.com\tTRUE\t/\tTRUE\t", saved.read_text())
        self.assertTrue(saved.is_relative_to(self.root / "workspace"))
        self.assertIsNone(store.file_for_url("https://vimeo.com/1"))
        self.assertTrue(store.delete("youtube.com"))
        self.assertEqual(store.sites(), {})
        with self.assertRaisesRegex(H3Error, "no Cookie header"):
            video_download.import_curl(store, "curl 'https://x.com'")

    def test_download_publishes_only_the_finished_file_and_registers_it(self):
        class FakeYDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, url, download):
                folder = Path(self.options["outtmpl"]).parent
                for hook in self.options["progress_hooks"]:
                    hook({"status": "downloading", "downloaded_bytes": 5, "total_bytes": 10, "speed": 2**20})
                (folder / "abc123.mp4").write_bytes(b"video")
                (folder / "abc123.f137.mp4.part").write_bytes(b"partial")
                return {"id": "abc123", "title": 'A: "great" clip', "duration": 61}

        messages = []
        with patch.object(video_download, "_youtube_dl", return_value=SimpleNamespace(YoutubeDL=FakeYDL)), \
                patch("h3_app.provenance.write_snapshot") as snapshot:
            path = video_download.download_video("https://youtu.be/abc123", "720p", self.runtime,
                                                 report=messages.append)
        self.assertEqual(path.name, "A great clip [abc123].mp4")
        self.assertEqual(path.parent, self.runtime.outputs_dir.resolve() / "downloads")
        self.assertEqual(path.read_bytes(), b"video")
        self.assertFalse((path.parent / ".processing").exists() and any((path.parent / ".processing").iterdir()))
        self.assertIn("Downloading 50%", messages[0])
        self.assertEqual(snapshot.call_args.args[1]["family"], "Downloaded")

    def test_download_stops_when_asked(self):
        class FakeYDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, url, download):
                try:
                    self.options["progress_hooks"][0]({"status": "downloading"})
                except Exception as exc:
                    raise RuntimeError(f"ERROR: wrapped {exc}") from exc

        with patch.object(video_download, "_youtube_dl", return_value=SimpleNamespace(YoutubeDL=FakeYDL)):
            with self.assertRaisesRegex(H3Error, "Download interrupted"):
                video_download.download_video("https://x.com/a/status/1", "Best available", self.runtime,
                                              stopped=lambda: True)
        self.assertEqual(list((self.runtime.outputs_dir / "downloads").glob("*.mp4")), [])


class SessionTests(Workspace):
    def test_sessions_copy_the_source_and_round_trip_lines(self):
        source = self.root / "talk.mp4"
        source.write_bytes(b"data")
        created = sessions.create(self.runtime, source, video=True)
        source.unlink()
        self.assertEqual(created.path("source").read_bytes(), b"data")
        self.assertTrue(created.root.is_relative_to(self.root / "workspace" / "dubbing"))
        sessions.update_segments(self.runtime, created.id, [{"start": "00:01.000", "end": 2, "source": "x"}])
        loaded = sessions.load(self.runtime, created.id)
        self.assertEqual(loaded.segments[0]["start"], 1.0)
        self.assertEqual(loaded.speech_audio, None)
        with self.assertRaisesRegex(H3Error, "Analyze a source"):
            sessions.load(self.runtime, "../escape")

    def test_sessions_of_the_same_file_find_each_other(self):
        source, other = self.root / "talk.mp4", self.root / "other.mp4"
        source.write_bytes(b"data")
        other.write_bytes(b"different")
        first = sessions.create(self.runtime, source, video=True)
        unrelated = sessions.create(self.runtime, other, video=True)
        second = sessions.create(self.runtime, source, video=True)
        self.assertEqual(first.data["source_hash"], second.data["source_hash"])
        self.assertNotEqual(first.data["source_hash"], unrelated.data["source_hash"])
        self.assertEqual([found.id for found in sessions.same_source(second)], [first.id])


class HandlerTests(Workspace):
    def setUp(self):
        super().setUp()
        from h3_ui.dubbing import handlers as module
        from h3_ui.dubbing.handlers import DubbingHandlers, DubbingServices

        self.module = module
        self.handlers = DubbingHandlers(DubbingServices(
            runtime_config=lambda: self.runtime, free_gpu=lambda: None,
            list_media=lambda mode: [self.root / f"{mode}.file"]))

    def test_library_choices_and_table_saves(self):
        update = self.handlers.library_choices()
        self.assertEqual([label for label, _path in update["choices"]], ["Video · Video.file", "Audio · Audio.file"])
        self.assertIn("Analyze a source first", self.handlers.save_lines("", [])[1])
        source = self.root / "a.wav"
        source.write_bytes(b"x")
        created = sessions.create(self.runtime, source, video=False)
        rows, status = self.handlers.save_lines(
            created.id, [[1, "00:00.000", "00:01.000", "speaker1", "Hi", "Hola", True],
                         [2, "00:02.000", "00:03.000", "speaker1", "Yo", "", True]])
        self.assertEqual(status, "2 lines · 1 speaker · <strong>1 ready</strong> to dub into the target "
                                 "language · 1 untranslated")
        self.assertEqual(rows, self.module.gr.skip())  # nothing moved: the table is left as typed

    def test_a_library_pick_survives_a_refresh_while_still_listed(self):
        listed = str(self.root / "Video.file")
        self.assertEqual(self.handlers.library_choices(listed)["value"], listed)
        self.assertIsNone(self.handlers.library_choices(str(self.root / "gone.mp4"))["value"])

    def test_an_unfinished_new_row_stays_in_the_editor(self):
        source = self.root / "a.wav"
        source.write_bytes(b"x")
        created = sessions.create(self.runtime, source, video=False)
        typed = [[1, "00:00.000", "00:01.000", "speaker1", "Hi", "Hola", True],
                 [None, "00:04.000", "", "", "", "", True]]
        rows, status = self.handlers.save_lines(created.id, typed)
        self.assertEqual(rows, self.module.gr.skip())  # redrawing would wipe the row being typed
        self.assertIn("1 unfinished row(s) not saved yet", status)
        # Once complete it is saved, numbered and sorted into place.
        typed[1] = [None, "00:00.500", "00:00.900", "speaker2", "Oh", "Ah", True]
        rows, status = self.handlers.save_lines(created.id, typed)
        self.assertEqual([(row[0], row[4]) for row in rows], [(1, "Hi"), (2, "Oh")])
        self.assertNotIn("unfinished", status)

    def test_reopening_a_session_restores_its_lines_and_language(self):
        source = self.root / "talk.wav"
        source.write_bytes(b"x")
        empty = sessions.create(self.runtime, source, video=False)
        created = sessions.create(self.runtime, source, video=False)
        created.segments = [{"start": 0, "end": 1, "source": "Hi", "translation": "Salut"}]
        created.data["target_language"] = "French"
        created.save()
        recent = self.handlers.recent_sessions()
        self.assertEqual([entry["id"] for entry in recent["sessions"]], [created.id])  # unanalyzed ones are hidden
        self.assertEqual((recent["sessions"][0]["name"], recent["sessions"][0]["detail"]),
                         ("talk", "French · 1 line · 1 speaker"))
        session_id, board, rows, status, target = self.handlers.open_recent({**recent, "chosen": 0})
        self.assertEqual((session_id, rows[0][4], target), (created.id, "Hi", "French"))
        self.assertEqual((board["focus"], board["lines"][0]["translation"]), (0, "Salut"))
        self.assertIn("Opened **talk**", status)
        self.assertIn("Error", self.handlers.open_session("f" * 32)[3])
        self.assertEqual(self.handlers.open_recent({**recent, "chosen": 7}), (self.module.gr.skip(),) * 5)
        self.assertNotEqual(empty.id, created.id)

    def board_session(self):
        source = self.root / "talk.wav"
        source.write_bytes(b"x")
        created = sessions.create(self.runtime, source, video=False)
        created.segments = [
            {"start": 0.0, "end": 1.0, "speaker": "speaker1", "source": "One <b>", "translation": "Un"},
            {"start": 2.0, "end": 3.0, "speaker": "speaker2", "source": "Two", "translation": ""},
            {"start": 4.0, "end": 5.0, "speaker": "speaker1", "source": "Three", "translation": "Trois",
             "dub": False}]
        created.save()
        return created

    def test_the_board_escapes_text_and_flags_lines(self):
        created = self.board_session()
        board = self.handlers.board_for(created.id)
        self.assertEqual(board["lines"][0]["source"], "One &lt;b&gt;")
        self.assertEqual([line["flags"] for line in board["lines"]], [[], ["empty"], ["off"]])
        self.assertEqual([(s["label"], s["count"]) for s in board["speakers"]], [("Speaker 1", 2), ("Speaker 2", 1)])
        self.assertEqual(board["focus"], 0)
        # The viewer's focus, picks and filter carry over while the session stays the same.
        kept = self.handlers.board_for(created.id, {**board, "focus": 2, "picked": [1, 2, 9], "filter": "empty"})
        self.assertEqual((kept["focus"], kept["picked"], kept["filter"]), (2, [1, 2], "empty"))
        other = self.handlers.board_for(created.id, {**board, "session": "x", "focus": 2})
        self.assertEqual(other["focus"], 0)

    def test_the_inspector_follows_the_focus_and_picks_drive_the_line_actions(self):
        created = self.board_session()
        board = {**self.handlers.board_for(created.id), "focus": 1}
        heading, start, end, speaker, original, translation, dub, fit, numbers, key = self.handlers.focus_line(
            created.id, board)
        self.assertIn("Line 2 of 3", heading)
        self.assertEqual((start, end, speaker["value"], original, translation, dub, numbers),
                         ("00:02.000", "00:03.000", "speaker2", "Two", "", True, "2"))
        self.assertEqual(key, {"index": 1, "start": 2.0, "speaker": "speaker2", "source": "Two"})
        self.assertIn("No translation yet", fit)
        self.assertEqual(self.handlers.pick_lines({**board, "picked": [0, 1, 2]}), "1-3")
        self.assertEqual(self.handlers.pick_lines({**board, "picked": []}), "2")
        moved = self.handlers.step_line(created.id, board, 1)
        self.assertEqual(moved["focus"], 2)
        self.assertEqual(self.handlers.step_line(created.id, moved, 1)["focus"], 2)  # stays on the last line

    def test_inspector_edits_save_and_follow_the_line_when_it_moves(self):
        created = self.board_session()
        board = {**self.handlers.board_for(created.id), "focus": 0}
        key = self.handlers.focus_line(created.id, board)[-1]
        value, rows, heading, fit, status, key = self.handlers.save_focused(
            created.id, key, board, "00:06.000", "00:07.000", "speaker3", "One", "Uno", True)
        self.assertEqual(status, "")
        self.assertEqual((value["focus"], rows[2][4], rows[2][3]), (2, "One", "speaker3"))
        self.assertEqual(key, {"index": 2, "start": 6.0, "speaker": "speaker3", "source": "One"})
        self.assertFalse(value["keepFocus"])  # the line moved, and the focus moves with it
        self.assertIn("Line 3 of 3", heading)
        self.assertIn("Not saved", self.handlers.save_focused(created.id, key, value, "soon", "", "", "x", "",
                                                              True)[4])
        self.assertIn("Not saved", self.handlers.save_focused(created.id, key, value, "6", "7", "", "", "",
                                                              True)[4])

    def test_a_late_save_reaches_the_inspected_line_not_the_newly_focused_one(self):
        created = self.board_session()
        board = {**self.handlers.board_for(created.id), "focus": 0}
        key = self.handlers.focus_line(created.id, board)[-1]
        # The viewer left line 1's translation by clicking line 3: the save arrives with the new focus,
        # and tells the browser to keep whichever line the viewer has focused since.
        value = self.handlers.save_focused(created.id, key, {**board, "focus": 2}, "00:00.000", "00:01.000",
                                           "speaker1", "One <b>", "Un !", True)[0]
        self.assertTrue(value["keepFocus"])
        lines = sessions.load(self.runtime, created.id).segments
        self.assertEqual([line["translation"] for line in lines], ["Un !", "", "Trois"])
        # A line edited elsewhere (here, its text in the table) is no longer the one the inspector shows.
        sessions.update_segments(self.runtime, created.id, [{**lines[0], "source": "Changed"}, *lines[1:]])
        status = self.handlers.save_focused(created.id, key, board, "00:00.000", "00:01.000", "speaker1",
                                            "One <b>", "Un", True)[4]
        self.assertIn("click it on the board again", status)

    def test_dub_switches_and_adding_or_deleting_lines(self):
        created = self.board_session()
        board = self.handlers.board_for(created.id)
        value, _rows, status = self.handlers.toggle_dub(created.id, {**board, "changed": {"line": 2, "dub": True}})
        self.assertEqual((status, value["focus"], value["lines"][2]["dub"]), ("", 2, True))
        added, _rows, status = self.handlers.add_line(created.id, {**value, "focus": 0})
        self.assertEqual((len(added["lines"]), added["focus"]), (4, 1))  # the new line, focused
        self.assertIn("Added a line", status)
        self.assertEqual(sessions.load(self.runtime, created.id).segments[1]["start"], 1.05)
        deleted, _rows, status = self.handlers.delete_line(created.id, added)
        self.assertEqual((len(deleted["lines"]), status), (3, "Line deleted."))
        only = self.handlers.board_for(created.id)
        for _ in range(2):
            only = self.handlers.delete_line(created.id, {**only, "focus": 0})[0]
        self.assertIn("at least one line", self.handlers.delete_line(created.id, only)[2])

    def test_saving_redraws_the_table_when_numbering_changes(self):
        source = self.root / "a.wav"
        source.write_bytes(b"x")
        created = sessions.create(self.runtime, source, video=False)
        # Line 2 now starts first, and an empty row was added.
        rows, _status = self.handlers.save_lines(
            created.id, [[1, "00:05.000", "00:06.000", "speaker1", "Late", "Tard", True],
                         [2, "00:01.000", "00:02.000", "speaker1", "Early", "Tôt", True],
                         [None, "", "", "", "", "", True]])
        self.assertEqual([(row[0], row[4]) for row in rows], [(1, "Early"), (2, "Late")])
        self.assertEqual([s["source"] for s in sessions.load(self.runtime, created.id).segments], ["Early", "Late"])

    def test_interrupt_signals_this_sessions_download(self):
        started, release = threading.Event(), threading.Event()
        seen = []

        def fake(url, quality, runtime, *, report, stopped):
            started.set()
            release.wait(5)
            seen.append(stopped())
            raise H3Error("Download interrupted.")

        request = SimpleNamespace(session_hash="tab-1")
        with patch.object(self.module.video_download, "download_video", side_effect=fake):
            updates = []
            worker = threading.Thread(target=lambda: updates.extend(
                self.handlers.download_video("https://x.com/v", "720p", request)))
            worker.start()
            self.assertTrue(started.wait(5))
            self.assertEqual(self.handlers.stop_downloads(SimpleNamespace(session_hash="other")),
                             "No download is running.")
            self.assertIn("Interrupting", self.handlers.stop_downloads(request))
            release.set()
            worker.join(10)
        self.assertEqual(seen, [True])
        self.assertIn("Download interrupted", updates[-1][0])
        self.assertEqual(self.module._DOWNLOAD_STOPS, {})

    def test_download_handler_streams_progress_then_the_path(self):
        target = self.root / "clip.mp4"

        def fake(url, quality, runtime, *, report, stopped):
            report("Downloading 10%")
            self.assertFalse(stopped())
            return target

        with patch.object(self.module.video_download, "download_video", side_effect=fake):
            updates = list(self.handlers.download_video("https://x.com/v", "720p"))
        self.assertEqual(updates[-1], ("Downloaded **clip.mp4** to Media.", str(target)))
        self.assertTrue(all(path == "" for _status, path in updates[:-1]))

    def test_speaker_cards_show_only_for_sessions_with_speakers(self):
        source = _tone_wav(self.root / "a.wav", 3.0)
        session = sessions.create(self.runtime, source, video=False)
        session.segments = [{"start": 0.0, "end": 1.0, "speaker": "speaker1", "source": "a", "translation": "A"},
                            {"start": 1.5, "end": 2.5, "speaker": "speaker2", "source": "b", "translation": "B"}]
        session.data["speakers"] = {"speaker2": "older man"}
        session.save()
        self.assertEqual(self.handlers.speakers(session.id)[0]["visible"], False)
        session.data["pipeline"] = pipeline.PIPELINE_MOSS
        session.save()
        with patch.object(self.handlers, "_voice_choices", return_value=[("Cloned", pipeline.ORIGINAL_VOICE)]):
            panel, cards = self.handlers.speakers(session.id)
        self.assertEqual(panel["visible"], True)
        self.assertEqual([(card["label"], card["detail"], card["volume"]) for card in cards["speakers"]],
                         [("Speaker 1", "1 line", 100.0), ("Speaker 2", "1 line · older man", 100.0)])
        cards["speakers"][1]["volume"] = 60
        self.assertEqual(self.handlers.save_speaker(session.id, {**cards, "changed": 1}), "")
        self.assertEqual(pipeline.speaker_settings(sessions.load(self.runtime, session.id)),
                         {"speaker2": {"voice": pipeline.ORIGINAL_VOICE, "volume": 60.0}})
        cards["speakers"][1]["voice"] = "Nobody"
        self.assertIn("Not saved", self.handlers.save_speaker(session.id, {**cards, "changed": 1}))
        self.assertEqual(self.handlers.save_speaker(session.id, {**cards, "changed": None}), self.module.gr.skip())

    def test_listen_original_uses_exact_source_cuts_without_translation_or_gpu(self):
        from h3_app.drama3.serving.audio import decode_wav

        source = _tone_wav(self.root / "original.wav", 2.0, level=0.2)
        session = sessions.create(self.runtime, source, video=False)
        session.set_file("audio", session.path("source"))
        vocals = _tone_wav(session.root / "vocals.wav", 2.0, level=0.8)
        session.set_file("vocals", vocals)
        session.segments = [{"start": 0.25, "end": 0.75, "source": "Hello", "translation": "", "dub": False}]
        session.save()
        with patch.object(self.handlers, "_gpu", side_effect=AssertionError("Original playback needs no GPU")):
            value, message = self.handlers.listen_original(session.id, "1")
        samples, rate = decode_wav(value["value"])
        expected = pipeline.read_slice(session.path("audio"), 0.25, 0.75)
        self.assertEqual((rate, len(samples)), (pipeline.RATE, int(0.5 * pipeline.RATE)))
        np.testing.assert_allclose(samples, expected, atol=2 / 32768)
        self.assertIn("Original line 1", message)
        self.assertIn("00:00.250", value["label"])
        # Reload the edited cuts; the old slice is never cached.
        session.segments = [{**session.segments[0], "start": 0.5, "end": 1.5}]
        session.save()
        value, _ = self.handlers.listen_original(session.id, "1")
        self.assertEqual(len(decode_wav(value["value"])[0]), pipeline.RATE)

    def test_listen_original_reports_missing_selection_or_source(self):
        source = _tone_wav(self.root / "original.wav", 1.0)
        session = sessions.create(self.runtime, source, video=False)
        session.segments = [{"start": 0, "end": 1, "source": "Hello"}]
        session.save()
        self.assertIn("Select a line", self.handlers.listen_original(session.id, "")[1])
        self.assertIn("soundtrack is missing", self.handlers.listen_original(session.id, "1")[1])

    def test_preview_player_preserves_levels_including_mute_and_boost(self):
        from h3_app.drama3.serving.audio import decode_wav

        source = _tone_wav(self.root / "a.wav", 1.0)
        session = sessions.create(self.runtime, source, video=False)
        session.segments = [{"start": 0, "end": 1, "source": "Hi", "translation": "Salut"}]
        session.save()
        player = self.module.gr.Audio(render=False)
        player.GRADIO_CACHE = str(self.root / "preview-cache")
        for handler in (self.handlers.preview_line, self.handlers.preview_new_take):
            for level in (0.0, 0.2, 0.4, 0.8):
                with self.subTest(handler=handler.__name__, level=level):
                    samples = np.tile(np.array([-level, level], dtype=np.float32), 100)
                    preview = pipeline.LinePreview(pipeline.RATE, samples, "Preview ready")
                    with patch.object(self.handlers, "_hf_token", return_value=None), patch.object(
                        pipeline, "preview_line", return_value=iter([preview])
                    ):
                        value, message = list(handler(session.id, "1", pipeline.VOICE_LINE, "Balance",
                                                       pipeline.TIMING_MATCH, 1.35, 16))[-1]
                    played, rate = decode_wav(Path(player.postprocess(value).path).read_bytes())
                    self.assertEqual((rate, message), (pipeline.RATE, "Preview ready"))
                    np.testing.assert_allclose(played, samples, atol=2 / 32768)

    def test_preview_forces_fresh_take_and_clears_previous_audio_on_failure(self):
        session = self.board_session()
        with patch.object(self.handlers, "_hf_token", return_value=None), \
                patch.object(pipeline, "preview_line", return_value=iter([
                    pipeline.LinePreview(pipeline.RATE, np.zeros(100), "Fresh preview ready")])) as preview:
            updates = list(self.handlers.preview_line(session.id, "1", pipeline.VOICE_LINE, "Balance",
                                                      pipeline.TIMING_MATCH, 1.35, 16))
        self.assertIsNone(updates[0][0])
        self.assertEqual(preview.call_args.args[0].regenerate, (0,))
        with patch.object(self.handlers, "_hf_token", return_value=None), \
                patch.object(pipeline, "preview_line", side_effect=H3Error("translated generation failed")):
            updates = list(self.handlers.preview_line(session.id, "1", pipeline.VOICE_LINE, "Balance",
                                                      pipeline.TIMING_MATCH, 1.35, 16))
        self.assertIsNone(updates[-1][0])
        self.assertIn("translated generation failed", updates[-1][1])

    def test_errors_become_status_lines(self):
        updates = list(self.handlers.analyze_source(None, None, True, "Balanced (BS-Roformer)", pipeline.PIPELINE_LOCAL,
                                                    "Auto-detect", "English", True, pipeline.TRANSLATOR_HYMT,
                                                    True, 0.35, "gemini-3.8-flash", "", ""))
        self.assertIn("Upload a video or audio file", updates[-1][3])
        updates = list(self.handlers.generate_dub("", pipeline.VOICE_LINE, "Balance", pipeline.TIMING_MATCH, 1.35,
                                                  8, 100, 100, True, ""))
        self.assertIn("Analyze a source first", updates[-1][4])

    def test_cookie_handlers(self):
        status, sites, cleared = self.handlers.import_curl_cookies(
            "curl 'https://www.bilibili.com/' -b 'SESSDATA=1'", "")
        self.assertEqual((status, cleared), ("Saved 1 cookie(s) for bilibili.com.", ""))
        self.assertIn("**bilibili.com** (1)", sites)
        self.assertIn("Deleted", self.handlers.delete_cookies("bilibili.com")[0])

    def test_subtitle_export_uses_finished_files_and_clears_with_the_session(self):
        source = {"video": "finished.mp4", "subtitles": "finished.french.srt"}
        with patch.object(pipeline, "export_subtitled_video", return_value=Path("export.mp4")) as export:
            updates = list(self.handlers.export_subtitles(source))
        export.assert_called_once_with("finished.mp4", "finished.french.srt", runtime=self.runtime)
        self.assertEqual(updates[-1], ("export.mp4", "Export saved to Media."))
        self.assertIn("Generate a video dub first", list(self.handlers.export_subtitles(None))[-1][1])
        cleared = self.handlers.clear_results()
        self.assertIsNone(cleared[9])
        self.assertFalse(cleared[10]["interactive"])
        self.assertIsNone(cleared[11])

    def test_translate_again_can_include_or_omit_context(self):
        created = self.board_session()
        board = {"session": created.id, "picked": [1]}
        for include in (True, False):
            with self.subTest(include=include), \
                    patch.object(self.handlers, "_hf_token", return_value=None), \
                    patch.object(pipeline.tools, "run", return_value={"texts": ["Deux"]}) as run:
                updates = list(self.handlers.translate_lines(
                    created.id, board, "French", pipeline.TRANSLATOR_HYMT, "", "", "Keep names.", include))
                self.assertIn("Translated 1 line", updates[-1][2])
                payload = run.call_args.args[1]
                self.assertEqual(payload["contexts"], ["One <b>" if include else ""])
                self.assertEqual(payload["notes"], "Keep names.")
                self.assertEqual(payload["texts"], ["Two"])
                self.assertEqual(sessions.load(self.runtime, created.id).segments[1]["translation"], "Deux")

    def test_export_is_bound_separately_from_generation(self):
        from h3_ui.dubbing.view import build_dubbing_view
        from h3_ui.dubbing.bindings import bind_dubbing

        gr = self.module.gr
        interrupt = unittest.mock.Mock(return_value="Cancellation requested")
        with gr.Blocks() as demo:
            task = gr.Radio(["Dub"], value="Dub")
            with gr.Tab("Create") as tab:
                view = build_dubbing_view(gr.Column(), gemini_models=["model"], default_gemini_model="model")
            bind_dubbing(view, task=task, create_tab=tab, interrupt=interrupt, actions=self.handlers)
        generation = next(fn for fn in demo.fns.values() if fn.api_name == "dub_generate")
        self.assertEqual(len(generation.inputs), 11)
        self.assertEqual(len(generation.outputs), 9)
        export = next(fn for fn in demo.fns.values() if fn.fn == self.handlers.export_subtitles)
        self.assertEqual(export.inputs, [view.export_source])
        self.assertEqual(export.outputs, [view.exported_video, view.export_status])
        self.assertFalse(view.export_subtitles.interactive)
        translation = next(fn for fn in demo.fns.values()
                           if view.translation_context in fn.inputs and fn.inputs[:2] == [view.session, view.board])
        self.assertEqual(translation.inputs[:8], [view.session, view.board, view.target_language,
                                                view.translator, view.gemini_model, view.api_key,
                                                view.notes, view.translation_context])
        self.assertFalse(view.translation_context.value)
        analysis = next(fn for fn in demo.fns.values() if fn.api_name == "dub_analyze_source")
        self.assertEqual(analysis.inputs[-1], view.translation_context)
        config = demo.get_config_file()
        self.assertTrue(any((view.align_words._id, "click") in [tuple(target) for target in dep["targets"]]
                            for dep in config["dependencies"]))
        self.assertTrue(any(fn.outputs == [view.player, view.alignment_status] for fn in demo.fns.values()))
        self.assertFalse(view.align_words.interactive)
        from h3_app.jobs import JOBS
        self.assertEqual(JOBS.callbacks[tools.FAMILY_DUB].__name__, "generate_dub")
        self.assertEqual(len(JOBS.callbacks[tools.FAMILY_DUB].job_input_names), 11)
        self.assertEqual(JOBS.callbacks[tools.FAMILY_WORDS].__name__, "align_words")
        self.assertEqual(JOBS.callbacks[tools.FAMILY_WORDS].job_input_names, ("session_id",))
        self.assertIs(view.align_words.h3_stop, view.stop_align_words)
        self.assertIsNot(view.align_words.h3_stop, view.run.h3_stop)
        for control, family in ((view.stop, tools.FAMILY_DUB), (view.stop_align_words, tools.FAMILY_WORDS)):
            event = next(dep for dep in config["dependencies"]
                         if (control._id, "click") in [tuple(target) for target in dep["targets"]]
                         and getattr(demo.fns[dep["id"]].fn, "__name__", "") == "stop")
            demo.fns[event["id"]].fn("request")
            interrupt.assert_called_with("request", family=family)

    def test_word_alignment_handler_refreshes_player_and_reports_errors(self):
        player = pipeline.Player(None, Path("dub.mp3"), Path("original.mp3"), None, 1, [], word_timing=True)
        result = pipeline.WordAlignment(player, "All word timings generated.")
        with patch.object(self.handlers, "_hf_token", return_value=None), \
                patch.object(pipeline, "align_dub_words", return_value=iter(["Aligning", result])) as align:
            updates = list(self.handlers.align_words("session-id"))
        self.assertEqual(updates[0][1], "Aligning")
        self.assertEqual(updates[-1][1], result.message)
        self.assertTrue(updates[-1][0]["visible"])
        self.assertIn("h3-dub-player", updates[-1][0]["value"])
        self.assertEqual(align.call_args.args, ("session-id",))
        with patch.object(self.handlers, "_hf_token", return_value=None), \
                patch.object(pipeline, "align_dub_words", side_effect=H3Error("Generate a dub first")):
            failed = list(self.handlers.align_words(""))
        self.assertIn("Generate a dub first", failed[-1][1])



def _tone_wav(path: Path, seconds: float, level: float = 0.5) -> Path:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(pipeline.RATE)
        out.writeframes(np.full((int(seconds * pipeline.RATE), 2), int(level * 32768), dtype="<i2").tobytes())
    return path


class ReviewFixTests(Workspace):
    def session_with_vocals(self, lines):
        source = _tone_wav(self.root / "a.wav", 3.0)
        session = sessions.create(self.runtime, source, video=False)
        session.set_file("vocals", _tone_wav(session.root / "vocals.wav", 3.0))
        session.segments = lines
        session.save()
        return session

    def test_failed_translated_speech_stops_dub_before_original_audio_fallback(self):
        session = self.session_with_vocals([
            {"start": 0, "end": 2.5, "source": "English original", "translation": "中文译文"}])
        with patch.object(pipeline, "_synthesize", return_value=iter(())), \
                patch.object(pipeline, "_speech_track") as mix:
            with self.assertRaisesRegex(H3Error, "Translated speech generation failed: line 1"):
                list(pipeline.dub(pipeline.DubRequest(session.id), runtime=self.runtime, free_gpu=lambda: None))
        mix.assert_not_called()

    def test_chinese_translation_reaches_tts_and_fresh_requests_bypass_cached_take(self):
        from unittest.mock import MagicMock
        from h3_app.drama3.serving import server, speech, audio

        text = "穿过树叶，一边走一边蹦跳着他们的小靴子"
        original = "Slicing through leaves and bouncing their booties as they go"
        session = self.session_with_vocals([{ "start": 0, "end": 2.5,
                                             "source": original, "translation": text}])
        fake = MagicMock()
        fake.payload.side_effect = server.SglangOmniServer(None).payload
        tone = (.2 * np.sin(np.arange(pipeline.RATE * 3) / 20)).astype(np.float32)
        fake.speech.return_value = audio.encode_wav(tone, pipeline.RATE)

        def synthesize(fresh=False):
            planned = pipeline._plan(session, [0], pipeline.VOICE_LINE, self.runtime)
            list(pipeline._synthesize(planned, pipeline.DubRequest(session.id, regenerate=(0,) if fresh else ()),
                                      session, runtime=self.runtime, free_gpu=lambda: None))
            return planned[0]

        with patch.dict(os.environ, {"H3_DRAMA3_SGLANG_DATA": str(self.root / "data")}), \
                patch.object(speech, "shared_server", return_value=fake), \
                patch.object(speech.LineSpeaker, "ready", return_value=iter(())), \
                patch.object(speech, "persist_caches"), patch.object(pipeline.tools, "stop"), \
                patch.object(pipeline.tools, "run", side_effect=AssertionError("No ASR checks for generation")):
            first = synthesize()
            cached = synthesize()
            self.assertEqual(fake.speech.call_count, 1)
            fresh = synthesize(fresh=True)
            self.assertEqual(fake.speech.call_count, 2)
            payload = fake.speech.call_args.args[0]
            self.assertEqual(payload["input"], text)
            self.assertEqual(payload["references"][0]["text"], original)
            self.assertEqual(first.key, cached.key)
            self.assertEqual(first.key, fresh.key)
            lines = session.segments
            lines[0]["translation"] = text + "."
            session.segments = lines
            session.save()
            punctuated = synthesize()
            self.assertNotEqual(first.key, punctuated.key)
            self.assertEqual(fake.speech.call_count, 3)
            self.assertEqual(fake.speech.call_args.args[0]["input"], text + ".")

    def test_failed_word_timings_are_reported_and_retried_including_old_null_cache(self):
        import hashlib

        session = self.session_with_vocals([
            {"start": i, "end": i + .8, "source": f"source {i}", "translation": f"translated {i}"}
            for i in range(3)])
        lines = session.segments
        clips = {i: np.full(pipeline.RATE, .2, dtype=np.float32) for i in range(3)}
        spoken = {i: (i + .1, i + 1.1) for i in range(3)}
        notes, calls, failed_paths = [], [], []

        def fake_tools(_op, payload, _runtime, **_kwargs):
            calls.append([clip["text"] for clip in payload["clips"]])
            result = []
            for clip in payload["clips"]:
                if len(calls) == 1 and clip["text"] in ("source 0", "translated 2"):
                    key = hashlib.sha256(Path(clip["audio"]).read_bytes() + clip["text"].encode()).hexdigest()
                    failed_paths.append(session.root / "word_timing" / f"{key}.json")
                    result.append(None)
                else:
                    result.append([{"text": word, "start": i * .1, "end": i * .1 + .08}
                                   for i, word in enumerate(clip["text"].split())])
            return {"words": result}

        def collect():
            result = yield from REAL_WORD_TIMINGS(session, lines, spoken, clips, runtime=self.runtime,
                                                  free_gpu=lambda: None, notes=notes)
            yield result

        with patch.object(pipeline.tools, "run", side_effect=fake_tools):
            first = list(collect())[-1]
            self.assertNotIn("source", first[0])
            self.assertEqual(set(first[1]), {"source", "translation"})
            self.assertNotIn("translation", first[2])
            self.assertIn("original lines 1; translated lines 3", " ".join(notes))
            self.assertTrue(all(not path.exists() for path in failed_paths))
            # Older app versions saved failed results as null or []. Both must retry.
            for path, value in zip(failed_paths, ("null", "[]")):
                path.write_text(value, encoding="utf-8")
            notes.clear()
            again = list(collect())[-1]
            self.assertEqual(calls[1], ["source 0", "translated 2"])
            self.assertTrue(all(set(again[i]) == {"source", "translation"} for i in range(3)))
            self.assertEqual(notes, [])
            self.assertEqual(again[0]["translation"][0]["start"], .1)
            self.assertEqual(again[2]["source"][0]["start"], 2.0)
            self.assertEqual(list(collect())[-1], again)
            self.assertEqual(len(calls), 2)

    def test_word_timings_outside_audio_are_retried_and_never_reported_as_complete(self):
        session = self.session_with_vocals([
            {"start": 0, "end": .8, "source": "Hello", "translation": "Salut"}])
        clips = {0: np.full(pipeline.RATE, .2, dtype=np.float32)}
        spoken = {0: (0, 1)}
        notes, calls = [], []
        valid = True
        existing = {0: {"source": [{"text": "Hello", "start": 100, "end": 101}],
                        "translation": [{"text": "Salut", "start": 100, "end": 101}]}}

        def fake_tools(_op, payload, _runtime, **_kwargs):
            calls.append(len(payload["clips"]))
            return {"words": [[{"text": clip["text"], "start": .1 if valid else 100,
                                "end": .4 if valid else 101}] for clip in payload["clips"]]}

        def collect(previous=None):
            result = yield from REAL_WORD_TIMINGS(session, session.segments, spoken, clips, runtime=self.runtime,
                free_gpu=lambda: None, notes=notes, existing=previous)
            yield result

        with patch.object(pipeline.tools, "run", side_effect=fake_tools):
            first = list(collect(existing))[-1]
            self.assertEqual(calls, [2])
            self.assertEqual(set(first[0]), {"source", "translation"})
            cache = session.root / "word_timing"
            for path in cache.glob("*.json"):
                words = json.loads(path.read_text(encoding="utf-8"))
                words[0].update(start=100, end=101)
                path.write_text(json.dumps(words), encoding="utf-8")
            self.assertEqual(list(collect())[-1], first)
            self.assertEqual(calls, [2, 2])
            for path in cache.glob("*.json"):
                path.unlink()
            valid = False
            notes.clear()
            self.assertEqual(list(collect())[-1], {})
            self.assertIn("Word timing falls outside the audio clip", " ".join(notes))
            self.assertEqual(list(cache.glob("*.json")), [])
            valid = True
            self.assertEqual(list(collect())[-1], first)
            self.assertEqual(calls, [2, 2, 2, 2])

        # Allow rounding at the clip edge without retaining words beyond playback.
        self.assertEqual(pipeline._validated_word_timing("Hi", [{"text": "Hi", "start": .1, "end": 1.01}], 1),
                         [{"text": "Hi", "start": .1, "end": 1}])

    def test_untranslated_lines_keep_their_original_voice(self):
        session = self.session_with_vocals([
            {"start": 0.0, "end": 1.0, "source": "a", "translation": "A"},
            {"start": 1.5, "end": 2.5, "source": "b", "translation": ""},
        ])
        line = pipeline._Line(0, session.segments[0], session.path("vocals"), "a",
                              take=np.zeros(pipeline.RATE // 2, dtype=np.float32))
        track, _notes, _ends = pipeline._speech_track(session, [line], pipeline.DubRequest(session.id), 3.0)
        self.assertAlmostEqual(float(track[int(2.0 * pipeline.RATE)]), 0.5, places=3)
        self.assertEqual(float(track[int(0.25 * pipeline.RATE)]), 0.0)

    def test_smart_timing_places_takes_shifts_subtitles_and_marks_lines_still_too_long(self):
        from h3_ui.dubbing import presenters

        session = self.session_with_vocals([
            {"start": 0.0, "end": 0.8, "source": "a", "translation": "A"},
            {"start": 1.0, "end": 1.6, "source": "b", "translation": "B"},
            {"start": 2.0, "end": 2.8, "source": "c", "translation": "C"},
        ])
        take = lambda seconds: np.full(int(seconds * pipeline.RATE), 0.3, dtype=np.float32)
        planned = [pipeline._Line(i, session.segments[i], session.path("vocals"), "", take=take(seconds))
                   for i, seconds in ((0, 0.5), (1, 3.0), (2, 0.3))]
        request = pipeline.DubRequest(session.id, timing=pipeline.TIMING_SMART)
        track, notes, spoken = pipeline._speech_track(session, planned, request, 3.0)
        # Line 1 starts 0.3 s early, plays at the 1.35x limit, still overlaps and is noted.
        self.assertAlmostEqual(spoken[1][0], 0.7, places=3)
        self.assertAlmostEqual(spoken[1][1] - spoken[1][0], 3.0 / 1.35, delta=0.03)
        self.assertEqual(spoken[0], (0.0, 0.5))
        self.assertTrue(planned[1].timing_problem.startswith("too long even at 1.35x"))
        self.assertTrue(any(note.startswith("line 2: too long even at 1.35x") for note in notes))
        self.assertAlmostEqual(spoken[2][0], 2.5, places=3)  # pushed by at most SMART_DELAY
        self.assertAlmostEqual(float(track[int(0.75 * pipeline.RATE)]), 0.3, places=3)
        shown = pipeline.subtitle_lines(session.segments, spoken)
        self.assertEqual([line["start"] for line in shown], [0.0, 0.7, 2.5])

        pipeline._record_timing_problems(session, planned)
        latest = sessions.load(self.runtime, session.id)
        self.assertEqual(list(pipeline.timing_problems(latest)), [1])
        board = presenters.board_value(latest)
        self.assertEqual(board["lines"][1]["flags"], ["rushed"])
        self.assertIn("overlaps the next line", board["lines"][1]["timing"])
        self.assertIn("1 too long in the last dub", board["summary"])
        self.assertIn("Line 2 did not fit", presenters.readiness_html(latest))
        # The next dub replaces the marks: in another timing nothing is marked.
        for line in planned:
            line.timing_problem = ""
        pipeline._record_timing_problems(session, planned)
        self.assertEqual(pipeline.timing_problems(sessions.load(self.runtime, session.id)), {})

    def test_failed_takes_are_marked_for_review_until_a_good_take_replaces_them(self):
        from h3_app.drama3.serving.audio import encode_wav
        from h3_app.drama3.serving.speech import SpokenLine
        from h3_ui.dubbing import presenters

        session = self.session_with_vocals([
            {"start": 0.0, "end": 0.8, "source": "a", "translation": "A"},
            {"start": 1.0, "end": 1.8, "source": "b", "translation": "B"},
            {"start": 2.0, "end": 2.8, "source": "c", "translation": "C"},
        ])
        tone = (0.2 * np.sin(np.arange(24000 // 2) / 8)).astype(np.float32)
        outcomes = {"A": "", "B": "no speech", "C": "generation failed: bad request"}

        class FakeSpeaker:
            def __init__(self, _runtime):
                pass

            reference = staticmethod(Path)
            budget = staticmethod(lambda _text: 100)
            ready = staticmethod(lambda *_args: iter(()))

            @staticmethod
            def speak(lines, **_kwargs):
                spoken = []
                for key, text, _reference, _transcript in lines:
                    rejected = outcomes[text].startswith("generation failed")
                    samples = np.zeros(0, dtype=np.float32) if rejected else tone
                    spoken.append(SpokenLine(key, b"" if rejected else encode_wav(tone, 24000), samples,
                                             0 if rejected else 24000, outcomes[text]))
                yield spoken

        def synthesize(regenerate=()):
            latest = sessions.load(self.runtime, session.id)
            planned = pipeline._plan(latest, [0, 1, 2], pipeline.VOICE_LINE, self.runtime)
            with patch("h3_app.drama3.serving.speech.LineSpeaker", FakeSpeaker), \
                    patch("h3_app.drama3.serving.speech.persist_caches"), patch.object(pipeline.tools, "stop"):
                list(pipeline._synthesize(planned, pipeline.DubRequest(session.id, regenerate=regenerate), latest,
                                          runtime=self.runtime, free_gpu=lambda: None))
            return planned

        planned = synthesize()
        self.assertIsNone(planned[2].take)
        self.assertIsNone(planned[1].take)  # failed checks must not become cached translated speech
        from h3_app.drama3.serving.takes import takes_root
        self.assertFalse((takes_root(self.runtime) / f"{planned[1].key}.wav").exists())
        latest = sessions.load(self.runtime, session.id)
        self.assertEqual(pipeline.take_problems(latest), {1: "no speech", 2: "generation failed: bad request"})
        board = presenters.board_value(latest)
        self.assertEqual([line["flags"] for line in board["lines"]], [[], ["failed"], ["failed"]])
        self.assertEqual(board["lines"][1]["problem"], "no speech")
        self.assertIn("2 with a failed take", board["summary"])
        self.assertIn("Lines 2-3: generation failed", presenters.readiness_html(latest))

        # Editing a line's translation makes its mark stale.
        segments = latest.segments
        segments[2]["translation"] = "C again"
        sessions.update_segments(self.runtime, session.id, segments)
        self.assertEqual(pipeline.take_problems(sessions.load(self.runtime, session.id)), {1: "no speech"})

        # A fresh take that passes clears the mark; reused takes leave the others as they were.
        outcomes.update({"B": "", "C again": "generation failed: still bad"})
        synthesize(regenerate=(1,))
        self.assertEqual(pipeline.take_problems(sessions.load(self.runtime, session.id)),
                         {2: "generation failed: still bad"})

        # A line added before it moves the mark along; a line no longer dubbed is not reported.
        segments = sessions.load(self.runtime, session.id).segments
        sessions.update_segments(self.runtime, session.id, [
            {"start": 0.0, "end": 0.4, "source": "new", "translation": "N"}, *segments])
        self.assertEqual(pipeline.take_problems(sessions.load(self.runtime, session.id)),
                         {3: "generation failed: still bad"})
        segments = sessions.load(self.runtime, session.id).segments
        segments[3]["dub"] = False
        sessions.update_segments(self.runtime, session.id, segments)
        self.assertEqual(pipeline.take_problems(sessions.load(self.runtime, session.id)), {})

    def test_translations_land_on_the_latest_saved_lines(self):
        session = self.session_with_vocals([
            {"start": 0.0, "end": 1.0, "source": "one"},
            {"start": 1.5, "end": 2.5, "source": "two"},
        ])

        def translate(op, payload, runtime, **_kwargs):
            # The editor saves while Hy-MT works: a new first line and a hand translation for "two".
            sessions.update_segments(self.runtime, session.id, [
                {"start": 0.0, "end": 0.4, "source": "zero", "translation": "zéro"},
                {"start": 0.5, "end": 1.0, "source": "one"},
                {"start": 1.5, "end": 2.5, "source": "two", "translation": "deux (edited)"}])
            return {"texts": [f"{text}-fr" for text in payload["texts"]]}

        stale = sessions.load(self.runtime, session.id)
        stale.segments = [{"start": 0.5, "end": 1.0, "source": "one"}, {"start": 1.5, "end": 2.5, "source": "two"}]
        with patch.object(pipeline.tools, "run", side_effect=translate):
            list(pipeline._translate(stale, [0], pipeline.AnalyzeRequest("", target_language="French"),
                                     runtime=self.runtime, free_gpu=lambda: None))
        saved = sessions.load(self.runtime, session.id)
        self.assertEqual([(s["source"], s["translation"]) for s in saved.segments],
                         [("zero", "zéro"), ("one", "one-fr"), ("two", "deux (edited)")])
        self.assertEqual(saved.data["target_language"], "French")

    def test_hy_mt_gets_the_lines_before_each_one(self):
        session = self.session_with_vocals([
            {"start": 0.0, "end": 0.5, "source": "one"},
            {"start": 0.6, "end": 1.0, "source": "two"},
            {"start": 1.1, "end": 1.5, "source": "three"},
            {"start": 1.6, "end": 2.0, "source": "four"},
        ])
        for include, expected in ((False, ["", ""]), (True, ["", "two\nthree"])):
            with patch.object(pipeline.tools, "run", return_value={"texts": ["x", "y"]}) as run:
                request = pipeline.AnalyzeRequest("", target_language="French", translation_context=include)
                list(pipeline._translate(session, [0, 3], request, runtime=self.runtime, free_gpu=lambda: None))
            self.assertEqual(run.call_args.args[1]["contexts"], expected)
        self.assertFalse(pipeline.AnalyzeRequest("").translation_context)

    def test_suspect_translation_is_saved_and_flagged_until_corrected(self):
        from h3_ui.dubbing import presenters, widgets

        session = self.session_with_vocals([
            {"start": 0.0, "end": 0.5, "source": "Hello"},
            {"start": 1.0, "end": 5.0, "source": "Goodbye"},
        ])
        explanation = "根据提供的背景信息，可以翻译为：你好。（说明：这是问候。）"
        request = pipeline.AnalyzeRequest("", target_language="Chinese")
        with patch.object(pipeline.tools, "run", return_value={"texts": [explanation, "再见"]}):
            updates = list(pipeline.retranslate(session.id, [0, 1], request,
                                                runtime=self.runtime, free_gpu=lambda: None))
        saved = sessions.load(self.runtime, session.id)
        self.assertEqual(saved.segments[0]["translation"], explanation)
        self.assertEqual(pipeline.translation_issues(saved.segments), [0])
        self.assertIn("may contain explanations: 1", updates[-1].message)
        board = presenters.board_value(saved)
        self.assertIn("translation", board["lines"][0]["flags"])
        self.assertIn("long", board["lines"][0]["flags"])
        self.assertNotIn("translation", board["lines"][1]["flags"])
        self.assertIn("1 translation needs review", board["summary"])
        self.assertIn("Review translation", presenters.fit_note(saved.segments[0]))
        self.assertIn("May sound rushed", presenters.fit_note(saved.segments[0]))
        self.assertIn("Review translations: 1", presenters.readiness_html(saved))
        self.assertIn("'rushed', 'translation'", widgets.BOARD_TEMPLATE)
        self.assertIn("Review translation</span>", widgets.BOARD_TEMPLATE)
        # Editing the saved text clears the notice, including when reopening a session.
        lines = saved.segments
        lines[0]["translation"] = "你好"
        saved.segments = lines
        saved.save()
        corrected = sessions.load(self.runtime, saved.id)
        self.assertEqual(pipeline.translation_issues(corrected.segments), [])
        self.assertNotIn("translation", presenters.board_value(corrected)["lines"][0]["flags"])
        self.assertNotIn("Review translation", presenters.fit_note(corrected.segments[0]))
        self.assertNotIn("Review translations", presenters.readiness_html(corrected))

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg is required")
    def test_fitted_takes_are_cached_and_stale_ones_pruned(self):
        session = self.session_with_vocals([
            {"start": 0.0, "end": 1.0, "source": "a", "translation": "A"},
            {"start": 1.5, "end": 2.5, "source": "b", "translation": "B"},
        ])
        tone = (0.2 * np.sin(np.arange(int(1.5 * pipeline.RATE)) / 20)).astype(np.float32)
        planned = [pipeline._Line(i, session.segments[i], session.path("vocals"), "", take=tone) for i in (0, 1)]
        cache = session.root / "timed"
        cache.mkdir()
        stale = cache / "old.wav"
        stale.write_bytes(b"x")
        old = time.time() - 3600
        os.utime(stale, (old, old))
        first, _notes, _ends = pipeline._speech_track(session, planned, pipeline.DubRequest(session.id), 3.0)
        self.assertFalse(stale.exists())
        self.assertEqual(len(list(cache.glob("*.wav"))), 1)  # both lines fit the same take the same way
        with patch.object(pipeline, "_ffmpeg", side_effect=AssertionError("cached")):
            again, _notes, _ends = pipeline._speech_track(session, planned, pipeline.DubRequest(session.id), 3.0)
        np.testing.assert_allclose(again, first, atol=1e-4)

    def gemini_session(self):
        session = self.session_with_vocals([
            {"start": 0.0, "end": 1.0, "speaker": "speaker1", "source": "a", "translation": "A"},
            {"start": 1.5, "end": 2.5, "speaker": "speaker2", "source": "b", "translation": "B"},
        ])
        session.data.update(pipeline=pipeline.PIPELINE_GEMINI, speakers={"speaker2": "older man"})
        session.save()
        return session

    def saved_voice(self, name="Alice"):
        from h3_app.drama3 import speakers as drama3_speakers

        return drama3_speakers.save_speaker(name, _tone_wav(self.root / f"{name}.wav", 3.0), "Hello there.",
                                            self.runtime)

    def test_speaker_settings_are_offered_only_for_sessions_with_speakers(self):
        local = self.session_with_vocals([{"start": 0.0, "end": 1.0, "source": "a", "translation": "A"}])
        with self.assertRaisesRegex(H3Error, "transcribed with MOSS, Nemotron or Gemini"):
            pipeline.set_speaker(self.runtime, local.id, "speaker1", voice="", volume=50)
        local.data["speaker_settings"] = {"speaker1": {"voice": "", "volume": 50}}
        self.assertEqual(pipeline.speaker_settings(local), {})

        session = self.gemini_session()
        voice = self.saved_voice()
        pipeline.set_speaker(self.runtime, session.id, "speaker2", voice=voice.key, volume=250)
        self.assertEqual(pipeline.speaker_settings(sessions.load(self.runtime, session.id)),
                         {"speaker2": {"voice": voice.key, "volume": 200.0}})
        with self.assertRaisesRegex(H3Error, "this session's speakers"):
            pipeline.set_speaker(self.runtime, session.id, "speaker9", voice="", volume=100)
        with self.assertRaisesRegex(H3Error, "no longer available"):
            pipeline.set_speaker(self.runtime, session.id, "speaker2", voice="Nobody", volume=100)
        # Back to the defaults clears the entry.
        pipeline.set_speaker(self.runtime, session.id, "speaker2", voice="", volume=100)
        self.assertEqual(sessions.load(self.runtime, session.id).data["speaker_settings"], {})

    def test_a_speakers_saved_voice_replaces_its_cloned_reference(self):
        session = self.gemini_session()
        voice = self.saved_voice()
        pipeline.set_speaker(self.runtime, session.id, "speaker2", voice=voice.key, volume=100)
        session = sessions.load(self.runtime, session.id)
        planned = pipeline._plan(session, [0, 1], pipeline.VOICE_LINE, self.runtime)
        self.assertEqual((planned[1].reference, planned[1].transcript), (voice.audio, "Hello there."))
        self.assertEqual(planned[0].reference.parent.name, "references")  # speaker1 is still cloned

        from h3_app.drama3 import speakers as drama3_speakers

        drama3_speakers.delete_speaker(voice.key, self.runtime)
        with self.assertRaisesRegex(H3Error, "Choose another voice for speaker2 under Speakers"):
            pipeline._plan(session, [1], pipeline.VOICE_LINE, self.runtime)

    def test_speaker_volume_scales_its_dubbed_and_original_lines(self):
        session = self.gemini_session()
        session.segments = [*session.segments,
                            {"start": 2.6, "end": 2.9, "speaker": "speaker2", "source": "c", "dub": False}]
        session.save()
        pipeline.set_speaker(self.runtime, session.id, "speaker2", voice="", volume=50)
        session = sessions.load(self.runtime, session.id)
        take = np.full(pipeline.RATE, 0.4, dtype=np.float32)
        planned = [pipeline._Line(i, session.segments[i], session.path("vocals"), "", take=take) for i in (0, 1)]
        track, _notes, _ends = pipeline._speech_track(session, planned, pipeline.DubRequest(session.id), 3.0)
        self.assertAlmostEqual(float(track[int(0.5 * pipeline.RATE)]), 0.4, places=3)
        self.assertAlmostEqual(float(track[int(2.0 * pipeline.RATE)]), 0.2, places=3)
        self.assertAlmostEqual(float(track[int(2.75 * pipeline.RATE)]), 0.25, places=3)  # kept original, halved

    def test_line_preview_matches_dub_speaker_volume_without_changing_the_take(self):
        session = self.gemini_session()
        take = np.full(pipeline.RATE, 0.4, dtype=np.float32)

        def synthesize(planned, *_args, **_kwargs):
            for line in planned:
                line.take = take
            return iter(())

        for volume in (0, 50, 100, 200):
            pipeline.set_speaker(self.runtime, session.id, "speaker2", voice="", volume=volume)
            session = sessions.load(self.runtime, session.id)
            request = pipeline.DubRequest(session.id)
            for index, gain in ((0, 1.0), (1, volume / 100.0)):
                with self.subTest(volume=volume, speaker=index), patch.object(
                    pipeline, "_synthesize", side_effect=synthesize
                ):
                    preview = list(pipeline.preview_line(request, index, runtime=self.runtime,
                                                        free_gpu=lambda: None))[-1]
                    planned = pipeline._plan(session, [index], request.voice_mode, self.runtime)
                    synthesize(planned)
                    track, _notes, _ends = pipeline._speech_track(session, planned, request, 3.0)
                    start = round(session.segments[index]["start"] * pipeline.RATE)
                    np.testing.assert_allclose(preview.samples, 0.4 * gain)
                    np.testing.assert_array_equal(preview.samples, track[start:start + len(preview.samples)])
                    np.testing.assert_array_equal(take, np.full(pipeline.RATE, 0.4, dtype=np.float32))

    def test_translating_into_a_new_language_replaces_every_translation(self):
        session = self.session_with_vocals([
            {"start": 0.0, "end": 1.0, "source": "one", "translation": "un"},
            {"start": 1.5, "end": 2.5, "source": "two", "translation": ""},
        ])
        session.data["target_language"] = "French"
        session.save()

        def translate(op, payload, runtime, **_kwargs):
            return {"texts": [f"{text}-{payload['target_language']}" for text in payload["texts"]]}

        def run(target):
            request = pipeline.AnalyzeRequest("", target_language=target)
            with patch.object(pipeline.tools, "run", side_effect=translate) as tools_run:
                updates = list(pipeline.retranslate(session.id, None, request, runtime=self.runtime,
                                                    free_gpu=lambda: None))
            return tools_run.call_count, updates[-1].message

        # Same language: only the untranslated line.
        self.assertEqual(run("French"), (1, "Translated 1 line(s) into French."))
        self.assertEqual(run("French")[0], 0)
        # A new language: every line, and the session now dubs into it.
        self.assertEqual(run("German"), (1, "Translated 2 line(s) into German."))
        saved = sessions.load(self.runtime, session.id)
        self.assertEqual([s["translation"] for s in saved.segments], ["one-German", "two-German"])
        self.assertEqual(saved.data["target_language"], "German")

    def test_job_bookkeeping_keeps_lines_edited_meanwhile(self):
        session = self.session_with_vocals([{"start": 0.0, "end": 1.0, "source": "old"}])
        sessions.update_segments(self.runtime, session.id, [{"start": 0.0, "end": 1.0, "source": "new"}])
        sessions.modify(session, lambda latest: latest.data.update(last_output="dub.mp4"))
        saved = sessions.load(self.runtime, session.id)
        self.assertEqual((saved.segments[0]["source"], saved.data["last_output"]), ("new", "dub.mp4"))
        self.assertEqual(session.segments[0]["source"], "new")

    def test_reference_clips_follow_edited_times(self):
        session = self.session_with_vocals([])
        short = pipeline._reference_clip(session, {"start": 0.5, "end": 1.0})
        longer = pipeline._reference_clip(session, {"start": 0.5, "end": 2.0})
        self.assertNotEqual(short, longer)
        with wave.open(str(short)) as a, wave.open(str(longer)) as b:
            self.assertGreater(b.getnframes(), a.getnframes())

    def test_old_sessions_are_pruned_when_a_new_one_starts(self):
        old = self.session_with_vocals([])
        stale = time.time() - sessions.KEEP_SECONDS - 60
        os.utime(old.root / sessions.MANIFEST, (stale, stale))
        fresh = self.session_with_vocals([])
        self.assertFalse(old.root.exists())
        self.assertTrue(fresh.root.exists())


class WorkerRoomTests(unittest.TestCase):
    class Worker:
        def __init__(self, loaded):
            self.loaded, self.requests = set(loaded), []

        def request(self, payload, timeout):
            self.requests.append(payload)
            self.loaded.discard(payload.get("model"))
            return {"ok": True}

    def memory(self, *free):
        values = iter(free)
        return patch("h3_app.drama3.serving.server.gpu_memory", side_effect=lambda: (next(values), 12288.0, ""))

    def test_idle_tools_models_unload_before_other_engines(self):
        worker, free_gpu = self.Worker({"asr+aligner"}), unittest.mock.Mock()
        with self.memory(3000, 9000), patch("h3_app.drama3.serving.server.release_all") as release:
            tools._make_room("translate", worker, free_gpu)
        self.assertEqual(worker.requests, [{"op": "unload", "model": "asr+aligner"}])
        free_gpu.assert_not_called()
        release.assert_not_called()

    def test_other_engines_go_only_when_still_short(self):
        worker, free_gpu = self.Worker(set()), unittest.mock.Mock()
        with self.memory(3000, 3000, 3000), patch("h3_app.drama3.serving.server.release_all") as release,                 patch.object(tools.audio_tools, "stop") as stop_voice_design:
            tools._make_room("transcribe", worker, free_gpu)
        free_gpu.assert_called_once()
        stop_voice_design.assert_called_once()
        release.assert_called_once()

    def test_a_loaded_model_needs_no_room(self):
        worker = self.Worker({"translator"})
        with patch("h3_app.drama3.serving.server.gpu_memory") as memory:
            tools._make_room("translate", worker, lambda: None)
        memory.assert_not_called()

    def test_dubbing_steps_share_the_gpu_with_the_worker_and_drama3(self):
        from h3_app.drama3.serving import server

        with patch.object(tools, "stop") as stop, patch.object(server, "release_all") as release:
            for family in tools.FAMILIES:
                tools.release_for_job(family)
                server.release_for_job(family)
        stop.assert_not_called()
        release.assert_not_called()

    def test_spans_without_word_timing_split_at_sentences(self):
        text = "Short one. Another short. " + "A very long sentence " * 6 + "ends here."
        lines = tools_worker.proportional_lines(10.0, 20.0, text)
        self.assertEqual(lines[0]["text"], "Short one. Another short.")
        self.assertEqual((lines[0]["start"], lines[-1]["end"]), (10.0, 20.0))
        self.assertEqual(len(lines), 2)
        self.assertEqual(tools_worker.lines_from_alignment([], 0.0, 4.0, "One. Two."),
                         [{"start": 0.0, "end": 4.0, "text": "One. Two."}])

def _run(command):
    subprocess.run(command, check=True, capture_output=True)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg is required")
class DubRenderTests(Workspace):
    """Real ffmpeg extraction, mixing and muxing; Drama 3 and the tools worker are mocked."""

    def make_video(self) -> Path:
        video = self.root / "Source talk.mp4"
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
              "testsrc=size=160x120:rate=10:duration=4", "-f", "lavfi", "-i", "sine=frequency=330:duration=4",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)])
        return video

    def analyze(self, video, *, separate, calls=None, **settings):
        def fake_tools(op, payload, runtime, **_kwargs):
            if calls is not None:
                calls.append(op)
            if op == "separate":
                out = Path(payload["output_dir"])
                out.mkdir(parents=True, exist_ok=True)
                for name in ("vocals", "backing"):
                    shutil.copyfile(payload["input"], out / f"x_({name}).wav")
                return {"vocals": str(out / "x_(vocals).wav"), "backing": str(out / "x_(backing).wav")}
            if op == "transcribe":
                return {"language": "English", "segments": [
                    {"start": 0.2, "end": 2.6, "text": "First line here."},
                    {"start": 2.8, "end": 3.6, "text": "Second."}]}
            if op == "translate":
                return {"texts": [f"[{text}] in French" for text in payload["texts"]]}
            raise AssertionError(op)

        def fake_moss(runtime, audio, *, duration, **_kwargs):
            if calls is not None:
                calls.append("moss")
            return [{"start": 0.2, "end": 2.6, "speaker": "S01", "text": "First line here."},
                    {"start": 2.8, "end": 3.6, "speaker": "S02", "text": "Second."}]

        request = pipeline.AnalyzeRequest(str(video), target_language="French", separate=separate, **settings)
        with patch.object(pipeline.tools, "run", side_effect=fake_tools), \
                patch.object(pipeline.moss, "transcribe", side_effect=fake_moss):
            updates = list(pipeline.analyze(request, runtime=self.runtime, free_gpu=lambda: None))
        result = updates[-1]
        self.assertIsInstance(result, pipeline.Analysis)
        return result.session

    def fake_synthesize(self, planned, request, session, **_kwargs):
        for line in planned:
            seconds = 3.0 if line.index == 0 else 0.5  # line 1 is longer than its 2.55 s slot
            line.take = (0.2 * np.sin(np.arange(int(seconds * pipeline.RATE)) / 20)).astype(np.float32)
        yield "voiced"

    def assert_constant_bitrate(self, *paths):
        # LAME tags a constant-bitrate MP3 "Info" and a variable one "Xing"; browsers seek only the
        # former exactly, so the player's tracks must stay constant or the dub drifts from its subtitles.
        for path in paths:
            head = Path(path).read_bytes()[:4096]
            self.assertIn(b"Info", head, path)
            self.assertNotIn(b"Xing", head, path)

    def streams(self, path):
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                                "-of", "json", str(path)], check=True, capture_output=True, text=True)
        return [stream["codec_type"] for stream in json.loads(probe.stdout)["streams"]]

    def test_analyze_then_dub_writes_video_audio_and_subtitles(self):
        session = self.analyze(self.make_video(), separate=True)
        self.assertEqual([s["translation"] for s in session.segments],
                         ["[First line here.] in French", "[Second.] in French"])
        self.assertTrue(session.path("vocals").is_file())
        self.assertAlmostEqual(session.data["duration"], 4.0, delta=0.1)
        with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize):
            request = pipeline.DubRequest(session.id, timing=pipeline.TIMING_MATCH)
            updates = list(pipeline.dub(request, runtime=self.runtime,
                                        free_gpu=lambda: None))
        result = updates[-1]
        self.assertIsInstance(result, pipeline.Dub)
        self.assertEqual(self.streams(result.video), ["video", "audio", "audio", "subtitle", "subtitle"])
        self.assertTrue(result.audio.is_file())
        self.assertEqual([p.name.rsplit(".", 2)[-2] for p in result.subtitles], ["french", "original"])
        self.assertIn("matched to the original timing (1 sped up, 1 slowed down)", result.message)
        self.assertAlmostEqual(self.duration(result.audio), 4.0, delta=0.1)
        # The first line's clip is long enough to be its own reference; the short one uses its speaker's best.
        references = sorted(p.name for p in (session.root / "references").iterdir())
        self.assertEqual(len(references), 1)
        snapshot = json.loads(Path(str(result.video) + ".settings.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["family"], "Dub & Translate")

    def test_dub_keeps_separate_player_tracks_out_of_media(self):
        from h3_ui.dubbing.player import player_html

        session = self.analyze(self.make_video(), separate=True)
        with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize):
            result = list(pipeline.dub(pipeline.DubRequest(session.id), runtime=self.runtime,
                                       free_gpu=lambda: None))[-1]
        player = result.player
        folder = self.runtime.outputs_dir / "dubbing" / pipeline.PLAYER_DIR / result.audio.stem
        self.assertEqual({player.dub.parent, player.original.parent, player.music.parent}, {folder})
        self.assertEqual(player.video, result.video)
        self.assertAlmostEqual(self.duration(player.music), 4.0, delta=0.15)
        self.assert_constant_bitrate(player.dub, player.original, player.music, result.audio)
        self.assertEqual([cue["translation"] for cue in player.lines],
                         ["[First line here.] in French", "[Second.] in French"])
        # The translated subtitle stays up while the long first take plays, but not into line 2.
        self.assertGreaterEqual(player.lines[0]["shown_end"], player.lines[0]["end"])
        self.assertLess(player.lines[0]["shown_end"], player.lines[1]["start"])
        html = player_html(player, key=result.audio.stem)
        self.assertIn("data-music", html)
        self.assertIn('data-shown-end="', html)
        self.assertIn("Dub (French)", html)
        # Deleting the dub from Media lets the next dub drop its player tracks.
        result.audio.unlink()
        pipeline._prune_players(self.runtime.outputs_dir / "dubbing")
        self.assertFalse(folder.exists())

    def test_dub_player_lights_aligned_words_and_reuses_their_timing(self):
        import re
        from h3_ui.dubbing.player import player_html

        calls = []

        def fake_tools(op, payload, runtime, **_kwargs):
            calls.append(op)
            return {"words": [[{"text": word, "start": .1 * number, "end": .1 * number + .08}
                               for number, word in enumerate(re.findall(r"\w+", clip["text"]))]
                              for clip in payload["clips"]]}

        session = self.analyze(self.make_video(), separate=True)
        timed = pipeline.DubRequest(session.id, word_timing=True)
        with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize), \
                patch.object(pipeline, "_word_timings", REAL_WORD_TIMINGS), \
                patch.object(pipeline.tools, "run", side_effect=fake_tools):
            plain = list(pipeline.dub(pipeline.DubRequest(session.id), runtime=self.runtime,
                                      free_gpu=lambda: None))[-1]
            self.assertEqual(calls, [])  # off by default
            self.assertEqual(plain.player.lines[0]["translation_words"], [])
            self.assertFalse(plain.player.word_timing)
            first = list(pipeline.dub(timed, runtime=self.runtime, free_gpu=lambda: None))[-1]
            again = list(pipeline.dub(timed, runtime=self.runtime, free_gpu=lambda: None))[-1]
        self.assertEqual(calls, ["align_words"])  # the second dub's clips are unchanged
        cue = first.player.lines[0]
        self.assertTrue(first.player.word_timing)
        self.assertEqual(cue["translation_words"][0]["start"], round(cue["shown_start"], 3))
        self.assertEqual(cue["source_words"][0]["start"], round(cue["start"], 3))
        self.assertEqual(again.player.lines, first.player.lines)
        self.assertIn('class="h3-word"', player_html(first.player, key="k"))

        with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize), \
                patch.object(pipeline, "_word_timings", REAL_WORD_TIMINGS), \
                patch.object(pipeline.tools, "run", side_effect=H3Error("no aligner")):
            shutil.rmtree(session.root / "word_timing", ignore_errors=True)
            failed = list(pipeline.dub(timed, runtime=self.runtime, free_gpu=lambda: None))[-1]
        self.assertIn("word-by-word subtitles unavailable", failed.message)
        self.assertEqual(failed.player.lines[0]["translation_words"], [])
        self.assertIn("Word timing unavailable: original and translated", player_html(failed.player, key="failed"))

    def test_separate_word_alignment_repairs_latest_dub_without_regenerating_media(self):
        from h3_ui.dubbing.player import player_html

        session = self.analyze(self.make_video(), separate=True)
        with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize):
            dub = list(pipeline.dub(pipeline.DubRequest(session.id), runtime=self.runtime,
                                    free_gpu=lambda: None))[-1]
        assets = [dub.audio, dub.video, dub.player.dub, dub.player.original, *dub.subtitles]
        before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in assets}
        # Later edits must not be aligned against the already recorded dialogue.
        latest = sessions.load(self.runtime, session.id)
        lines = latest.segments
        lines[0]["source"] = "Edited original"
        lines[0]["translation"] = "Edited translation"
        latest.segments = lines
        latest.save()
        calls = []

        def fake_tools(op, payload, _runtime, **_kwargs):
            self.assertEqual(op, "align_words")
            calls.append([clip["text"] for clip in payload["clips"]])
            return {"words": [None if len(calls) == 1 and i == 0 else
                              [{"text": word, "start": j * .1, "end": j * .1 + .08}
                               for j, word in enumerate(re.findall(r"\w+", clip["text"]))]
                              for i, clip in enumerate(payload["clips"])],
                    "errors": ["Cannot align unknown language" if len(calls) == 1 and i == 0 else ""
                               for i in range(len(payload["clips"]))]}

        import re
        with patch.object(pipeline, "_word_timings", REAL_WORD_TIMINGS), \
                patch.object(pipeline.tools, "run", side_effect=fake_tools), \
                patch.object(pipeline, "_synthesize", side_effect=AssertionError("must not generate voices")), \
                patch.object(pipeline, "_ffmpeg", side_effect=AssertionError("must not re-encode media")):
            first = list(pipeline.align_dub_words(session.id, runtime=self.runtime, free_gpu=lambda: None))[-1]
            self.assertIn("1 still missing", first.message)
            self.assertIn("original lines 1", first.message)
            self.assertIn("original line 1: Cannot align unknown language", first.message)
            self.assertIn("Word timing unavailable: original", player_html(first.player, key="first"))
            fixed = list(pipeline.align_dub_words(session.id, runtime=self.runtime, free_gpu=lambda: None))[-1]
            self.assertIn("All word timings generated", fixed.message)
            self.assertIn("Newly aligned this run: 1", fixed.message)
            self.assertEqual(calls[1], [dub.player.lines[0]["source"]])
            self.assertNotIn("Edited", " ".join(calls[0]))
            self.assertNotIn("Word timing unavailable", player_html(fixed.player, key="fixed"))
            cached = list(pipeline.align_dub_words(session.id, runtime=self.runtime, free_gpu=lambda: None))[-1]
            self.assertEqual(cached.player.lines, fixed.player.lines)
            self.assertIn("Newly aligned this run: 0", cached.message)
            self.assertEqual(len(calls), 2)
        self.assertEqual(before, {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in assets})
        self.assertEqual(sessions.load(self.runtime, session.id).segments[0]["translation"], "Edited translation")
        # Old dubs give an actionable message instead of guessing their audio/text mapping.
        folder = self.runtime.outputs_dir / "dubbing" / pipeline.PLAYER_DIR / dub.audio.stem
        (folder / "alignment.npz").unlink()
        with self.assertRaisesRegex(H3Error, "Generate the dub once"):
            list(pipeline.align_dub_words(session.id, runtime=self.runtime, free_gpu=lambda: None))

    def test_optional_alignment_storage_failure_keeps_completed_dub(self):
        session = self.analyze(self.make_video(), separate=True)
        real_write = Path.write_text

        def fail_manifest(path, *args, **kwargs):
            if path.name == "alignment.json":
                raise OSError("alignment manifest write failed")
            return real_write(path, *args, **kwargs)

        def fail_archive(path, **_clips):
            Path(path).write_bytes(b"partial archive")
            raise OSError("alignment archive write failed")

        for stage in ("archive", "manifest"):
            with self.subTest(stage=stage):
                storage = (patch.object(pipeline.np, "savez_compressed", side_effect=fail_archive)
                           if stage == "archive" else patch.object(Path, "write_text", fail_manifest))
                with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize), storage:
                    result = list(pipeline.dub(pipeline.DubRequest(session.id), runtime=self.runtime,
                                               free_gpu=lambda: None))[-1]
                self.assertIsInstance(result, pipeline.Dub)
                self.assertTrue(result.audio.is_file())
                self.assertTrue(result.video.is_file())
                self.assertIsNotNone(result.player)
                self.assertIn("Dub saved, but word-alignment repair audio could not be saved", result.message)
                self.assertEqual(sessions.load(self.runtime, session.id).data["last_output"], str(result.video))
                folder = self.runtime.outputs_dir / "dubbing" / pipeline.PLAYER_DIR / result.audio.stem
                self.assertFalse((folder / "alignment.npz").exists())
                self.assertFalse((folder / "alignment.json").exists())

    def test_unseparated_player_switches_between_the_mix_and_the_original(self):
        from h3_ui.dubbing.player import player_html

        session = self.analyze(self.make_video(), separate=False)
        with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize):
            result = list(pipeline.dub(pipeline.DubRequest(session.id), runtime=self.runtime,
                                       free_gpu=lambda: None))[-1]
        self.assertIsNone(result.player.music)
        self.assertEqual(result.player.dub, result.audio)
        self.assertTrue(result.player.original.is_file())
        self.assert_constant_bitrate(result.player.dub, result.player.original)
        self.assertIn("needs vocal separation", player_html(result.player, key="k"))

    def test_moss_transcription_tells_speakers_apart(self):
        calls = []
        session = self.analyze(self.make_video(), separate=False, calls=calls, pipeline=pipeline.PIPELINE_MOSS)
        self.assertEqual(calls, ["moss", "translate"])
        self.assertEqual([s["speaker"] for s in session.segments], ["speaker1", "speaker2"])
        self.assertTrue(pipeline.speaker_settings_available(session))
        self.assertIn(moss.MODEL_REPO, json.loads(session.data["transcript"]["key"])["model"])

    def test_reanalysis_reuses_stems_and_transcript_of_the_same_file(self):
        video = self.make_video()
        first = self.analyze(video, separate=True)
        calls = []
        second = self.analyze(video, separate=True, calls=calls)
        self.assertEqual(calls, ["translate"])
        self.assertNotEqual(second.id, first.id)
        self.assertEqual([s["source"] for s in second.segments], [s["source"] for s in first.segments])
        self.assertEqual(second.path("vocals").read_bytes(), first.path("vocals").read_bytes())
        self.assertEqual(second.data["source_language"], "English")
        # Another transcription setting keeps the stems; Reuse off redoes both.
        calls.clear()
        self.analyze(video, separate=True, calls=calls, merge_gap=0.6)
        self.assertEqual(calls, ["transcribe", "translate"])
        calls.clear()
        self.analyze(video, separate=True, calls=calls, reuse=False)
        self.assertEqual(calls, ["separate", "transcribe", "translate"])

    def test_unseparated_dub_then_separate_subtitle_export(self):
        session = self.analyze(self.make_video(), separate=False)
        self.assertIsNone(session.path("backing"))
        with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize):
            result = list(pipeline.dub(pipeline.DubRequest(session.id, keep_original_track=False),
                                       runtime=self.runtime, free_gpu=lambda: None))[-1]
        self.assertEqual(self.streams(result.video), ["video", "audio", "subtitle", "subtitle"])
        original_bytes = result.video.read_bytes()
        exported = pipeline.export_subtitled_video(str(result.video), str(result.subtitles[0]), runtime=self.runtime)
        self.assertNotEqual(exported, result.video)
        self.assertEqual(result.video.read_bytes(), original_bytes)
        self.assertEqual(self.streams(exported), ["video", "audio", "subtitle", "subtitle"])
        self.assertEqual(result.player.video, result.video)
        snapshot = pipeline.read_snapshot(exported)
        self.assertTrue(snapshot["settings"]["burned_subtitles"])
        self.assertEqual(snapshot["settings"]["source"], result.video.name)
        with self.assertRaises(pipeline.H3Error):
            pipeline.export_subtitled_video(str(result.video), str(result.subtitles[1]), runtime=self.runtime)

    def long_last_line(self, planned, request, session, **_kwargs):
        # Line 2 starts at 2.8 s in a 4 s source: 2.5 s at 1.35x still ends near 4.65 s.
        for line in planned:
            seconds = 1.0 if line.index == 0 else 2.5
            line.take = (0.2 * np.sin(np.arange(int(seconds * pipeline.RATE)) / 20)).astype(np.float32)
        yield "voiced"

    def duration(self, path, stream=None):
        command = ["ffprobe", "-v", "error", "-of", "csv=p=0"]
        command += ["-select_streams", stream, "-show_entries", "stream=duration"] if stream else             ["-show_entries", "format=duration"]
        return float(subprocess.run([*command, str(path)], check=True, capture_output=True,
                                    text=True).stdout.split()[0])

    def test_matched_timing_keeps_a_long_last_line_inside_the_source(self):
        session = self.analyze(self.make_video(), separate=True)
        with patch.object(pipeline, "_synthesize", side_effect=self.long_last_line):
            request = pipeline.DubRequest(session.id, timing=pipeline.TIMING_MATCH)
            result = list(pipeline.dub(request, runtime=self.runtime,
                                       free_gpu=lambda: None))[-1]
        self.assertAlmostEqual(self.duration(result.audio), 4.0, delta=0.1)
        self.assertNotIn("past the end", result.message)

    def test_a_natural_pace_last_line_longer_than_the_source_is_not_cut(self):
        session = self.analyze(self.make_video(), separate=True)
        with patch.object(pipeline, "_synthesize", side_effect=self.long_last_line):
            result = list(pipeline.dub(pipeline.DubRequest(session.id, timing=pipeline.TIMING_NATURAL),
                                       runtime=self.runtime, free_gpu=lambda: None))[-1]
        self.assertGreater(self.duration(result.audio), 4.55)
        self.assertGreater(self.duration(result.video, "a:0"), 4.55)
        self.assertIn("past the end of the source", result.message)
        last_cue = result.subtitles[0].read_text(encoding="utf-8").strip().splitlines()[-2]
        self.assertGreater(segments.parse_time(last_cue.split("-->")[1]), 4.55)

    def test_burned_video_holds_its_last_frame_under_a_long_last_line(self):
        session = self.analyze(self.make_video(), separate=False)
        with patch.object(pipeline, "_synthesize", side_effect=self.long_last_line):
            result = list(pipeline.dub(pipeline.DubRequest(session.id, timing=pipeline.TIMING_NATURAL),
                                       runtime=self.runtime, free_gpu=lambda: None))[-1]
        exported = pipeline.export_subtitled_video(str(result.video), str(result.subtitles[0]), runtime=self.runtime)
        self.assertGreater(self.duration(exported, "v:0"), 4.55)
        self.assertEqual(self.streams(exported), ["video", "audio", "audio", "subtitle", "subtitle"])

    def test_match_stretches_a_take_to_exactly_its_line(self):
        tone = (0.2 * np.sin(np.arange(2 * pipeline.RATE) / 20)).astype(np.float32)
        for target in (0.6, 1.0, 3.5):  # beyond 2x faster, faster, and slower
            timed, tempo, _note = pipeline.match(tone, pipeline.RATE, target)
            self.assertEqual(len(timed), int(round(target * pipeline.RATE)))
            self.assertAlmostEqual(tempo, 2.0 / target)
            # Stretched, not padded or cut: the tone runs to within a few ms of the end.
            self.assertGreater(float(np.abs(timed[-int(0.05 * pipeline.RATE):]).max()), 0.05)

    def test_read_slice_reads_part_of_a_wav(self):
        target = self.root / "tone.wav"
        with wave.open(str(target), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(pipeline.RATE)
            out.writeframes((np.full((pipeline.RATE * 2, 2), 16384, dtype="<i2")).tobytes())
        part = pipeline.read_slice(target, 0.5, 1.0)
        self.assertEqual(len(part), pipeline.RATE // 2)
        self.assertAlmostEqual(float(part.mean()), 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
