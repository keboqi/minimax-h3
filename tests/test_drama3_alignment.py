"""Script alignment, cache invalidation and render timing regression tests."""
import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from h3_app.dubbing.tools_worker import Worker, aligned_line_bounds, aligned_line_words
from h3_app.drama3.serving.alignment import align_take
from h3_app.drama3.serving import generate
from h3_app.drama3.serving.planner import Chunk
from h3_app.drama3.serving.tree import parse_drama3_tree, iter_speech_nodes
from h3_ui.subtitle_words import word_spans_html


class AlignmentTests(unittest.TestCase):
    def test_uneven_delivery_uses_aligned_words_and_ignores_internal_pause(self):
        items = [{"text": "Wait", "start": .1, "end": 1},
                 {"text": "please", "start": 1.3, "end": 4.3},
                 {"text": "Yes", "start": 4.6, "end": 5.6}]
        self.assertEqual(aligned_line_bounds(["Wait, please!", "Yes."], items), [(.1, 4.3), (4.6, 5.6)])

    def test_cjk_and_repeated_lines(self):
        items = [{"text": c, "start": i, "end": i + .5} for i, c in enumerate("你好你好")]
        self.assertEqual(aligned_line_bounds(["你好！", "你好。"], items), [(0, 1.5), (2, 3.5)])

    def test_invalid_alignment_is_rejected(self):
        for items in ([{"text": "wrong", "start": 0, "end": 1}],
                      [{"text": "yes", "start": float("nan"), "end": 1}]):
            with self.assertRaises(ValueError):
                aligned_line_bounds(["yes"], items)

    def test_quantized_point_words_and_one_tick_overlap_are_preserved(self):
        from h3_app.dubbing.tools_worker import aligned_clip_words

        words = aligned_clip_words("你好呀", [
            {"text": "你", "start": .08, "end": .24},
            {"text": "好", "start": .24, "end": .24},
            {"text": "呀", "start": .24, "end": .48}], .45)
        self.assertEqual(words[1], {"text": "好", "start": .24, "end": .24})
        self.assertEqual(words[-1]["end"], .45)
        overlap = aligned_clip_words("Hi there", [
            {"text": "Hi", "start": .08, "end": .4},
            {"text": "there", "start": .38, "end": .8}], 1)
        self.assertEqual(overlap[1]["start"], .4)
        for invalid in ([{"text": "Hi", "start": .2, "end": .2}],
                        [{"text": "Hi", "start": 100, "end": 101}]):
            with self.assertRaises(ValueError):
                aligned_clip_words("Hi", invalid, 1)

    def test_soundstage_reuses_cached_point_word_boundaries(self):
        take = SimpleNamespace(chunk=SimpleNamespace(lines=("你好呀",), text="你好呀"),
                               samples=np.full(1000, .2, dtype=np.float32), rate=1000)
        reply = {"bounds": [[.08, .88]], "words": [[
            {"text": "你", "start": .08, "end": .24},
            {"text": "好", "start": .24, "end": .24},
            {"text": "呀", "start": .24, "end": .88}]]}
        with tempfile.TemporaryDirectory() as root, patch("h3_app.dubbing.tools.run", return_value=reply) as run:
            first = align_take(take, root, None, free_gpu=lambda: None)
            self.assertEqual(first[1][0][1], ("好", .24, .24))
            self.assertEqual(align_take(take, root, None, free_gpu=lambda: None), first)
            self.assertEqual(run.call_count, 1)

    def test_worker_realignment_with_silence_padding_fixes_persistent_timestamp_failures(self):
        from unittest.mock import MagicMock
        from h3_app.drama3.serving.audio import encode_wav

        worker = object.__new__(Worker)
        model = MagicMock()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "clip.wav"
            original = encode_wav(np.full(16000, .2, np.float32), 16000)
            path.write_bytes(original)
            for bad_start, bad_end in ((.2, .2), (.1, 1.12)):
                with self.subTest(start=bad_start, end=bad_end):
                    bad = [[SimpleNamespace(text="Hi", start_time=bad_start, end_time=bad_end)]]
                    padded = [[SimpleNamespace(text="Hi", start_time=.34, end_time=.74)]]
                    model.forced_aligner.align.reset_mock()
                    model.forced_aligner.align.side_effect = [bad, bad, padded]
                    with patch.object(worker, "_asr", return_value=model):
                        result = worker.align_words({"clips": [{"audio": str(path), "text": "Hi!",
                                                                 "language": "English", "duration": 1}]})
                    self.assertEqual(result["errors"], [""])
                    self.assertAlmostEqual(result["words"][0][0]["start"], .1)
                    self.assertAlmostEqual(result["words"][0][0]["end"], .5)
                    self.assertEqual(model.forced_aligner.align.call_count, 3)
                    samples, rate = model.forced_aligner.align.call_args.kwargs["audio"]
                    self.assertEqual((rate, len(samples)), (16000, 16000 + 2 * 3840))
                    self.assertTrue(np.all(samples[:3840] == 0))
                    self.assertTrue(np.all(samples[-3840:] == 0))
                    np.testing.assert_allclose(samples[3840:-3840], .2, atol=1 / 32768)
                    self.assertEqual(path.read_bytes(), original)

            invalid = [[SimpleNamespace(text="Hi", start_time=100, end_time=101)]]
            model.forced_aligner.align.side_effect = [invalid, invalid, invalid]
            with patch.object(worker, "_asr", return_value=model):
                result = worker.align_words({"clips": [{"audio": str(path), "text": "Hi!",
                                                         "language": "English", "duration": 1}]})
            self.assertEqual(result["words"], [None])
            self.assertIn("padded retry", result["errors"][0])
            self.assertIn("1.000s clip", result["errors"][0])

    def test_worker_aligns_supplied_script_instead_of_recognized_paraphrase(self):
        from unittest.mock import MagicMock
        worker = Worker(Path("."))
        model = MagicMock()
        model.transcribe.return_value = [SimpleNamespace(language="English", text="incorrect")]
        model.forced_aligner.align.return_value = [[SimpleNamespace(text="Yes", start_time=.2, end_time=.8)]]
        with patch.object(worker, "_asr", return_value=model):
            self.assertEqual(worker.handle({"op": "align_lines", "audio": "take.wav", "lines": ["Yes!"]}),
                             {"bounds": [(.2, .8)], "words": [[{"text": "Yes", "start": .2, "end": .8}]]})
        self.assertEqual(model.forced_aligner.align.call_args.kwargs["text"], "Yes!")

    def test_cache_tracks_audio_and_line_boundaries(self):
        take = SimpleNamespace(chunk=SimpleNamespace(lines=("Hello.", "Yes.")),
                               samples=np.full(2000, .2, dtype=np.float32), rate=1000)
        with tempfile.TemporaryDirectory() as root, patch("h3_app.dubbing.tools.run", return_value={
                "bounds": [[.1, .8], [1.2, 1.8]],
                "words": [[{"text": "Hello", "start": .1, "end": .8}],
                          [{"text": "Yes", "start": 1.2, "end": 1.8}]]}) as run:
            self.assertEqual(align_take(take, root, None, free_gpu=lambda: None)[0], [(100, 800), (1200, 1800)])
            align_take(take, root, None, free_gpu=lambda: None)
            self.assertEqual(run.call_count, 1)
            take.samples = take.samples * .5
            align_take(take, root, None, free_gpu=lambda: None)
            self.assertEqual(run.call_count, 2)
        with tempfile.TemporaryDirectory() as root, patch("h3_app.dubbing.tools.run", return_value={
                "bounds": [[.1, .8], [2.005, 2.015]]}):
            with self.assertRaises(ValueError):
                align_take(take, root, None, free_gpu=lambda: None)

    def test_render_uses_aligned_offsets_after_resampling(self):
        tree = parse_drama3_tree("<|speaker:0|> Hello.")
        speech = list(iter_speech_nodes(tree))[0]
        chunk = Chunk(0, "Speaker 0", "Hello. Yes.", (), ("Hello.", "Yes."))
        take = generate.Take(chunk, np.full(2000, .2, dtype=np.float32), 1000, 1, None,
                             line_bounds=[(100, 800), (1200, 1800)])
        spans = {}
        generate._render_ast(tree, {id(speech): [chunk]}, {0: take}, {}, 2000, spans=spans)
        self.assertEqual(spans[0][2], [(200, 1600), (2400, 3600)])
        self.assertEqual(take.alignment_scale, 1.0)

    def test_repair_mapping_records_duration_changing_effects(self):
        tree = parse_drama3_tree("<|speaker:0|> Hello.")
        speech = list(iter_speech_nodes(tree))[0]
        chunk = Chunk(0, "Speaker 0", "Hello.", ("robot",), ("Hello.",))
        take = generate.Take(chunk, np.full(2000, .2, dtype=np.float32), 1000, 1, None,
                             line_bounds=[(100, 800)], line_words=[[("Hello", .1, .8)]])
        words, spans = {}, {}
        with patch.object(generate, "apply_speaker_effects", side_effect=lambda samples, *_a, **_kw:
                          np.repeat(samples, 2)):
            generate._render_ast(tree, {id(speech): [chunk]}, {0: take}, {}, 2000, spans=spans, words=words)
        self.assertEqual(take.alignment_scale, 2.0)
        self.assertEqual(words[0], [[("Hello", 400, 3200)]])

    def test_words_are_grouped_by_script_line(self):
        items = [{"text": "Wait", "start": .1, "end": 1}, {"text": "please", "start": 1.3, "end": 4.3},
                 {"text": "Yes", "start": 4.6, "end": 5.6}]
        self.assertEqual([[word["text"] for word in line] for line in aligned_line_words(["Wait, please!", "Yes."], items)],
                         [["Wait", "please"], ["Yes"]])

    def test_old_line_only_cache_is_aligned_again_for_words(self):
        take = SimpleNamespace(chunk=SimpleNamespace(lines=("Yes.",), text="Yes."),
                               samples=np.full(2000, .2, dtype=np.float32), rate=1000)
        reply = {"bounds": [[.1, .8]], "words": [[{"text": "Yes", "start": .1, "end": .8}]]}
        with tempfile.TemporaryDirectory() as root, patch("h3_app.dubbing.tools.run", return_value=reply) as run:
            bounds, _words = align_take(take, root, None, free_gpu=lambda: None)
            cache = next(Path(root).glob("*.alignment.json"))
            cache.write_text("[[0.1, 0.8]]", encoding="utf-8")
            self.assertEqual(align_take(take, root, None, free_gpu=lambda: None), (bounds, [[("Yes", .1, .8)]]))
            self.assertEqual(run.call_count, 2)

    def test_incomplete_and_invalid_word_cache_is_retried(self):
        take = SimpleNamespace(chunk=SimpleNamespace(lines=("Yes.",), text="Yes."),
                               samples=np.full(2000, .2, dtype=np.float32), rate=1000)
        reply = {"bounds": [[.1, .8]], "words": [[{"text": "Yes", "start": .1, "end": .8}]]}
        with tempfile.TemporaryDirectory() as root, patch("h3_app.dubbing.tools.run", return_value=reply) as run:
            align_take(take, root, None, free_gpu=lambda: None)
            cache = next(Path(root).glob("*.alignment.json"))
            for bad in (None, {"bounds": [[.1, .8]], "words": [[]]},
                        {"bounds": [[.1, .8]], "words": [[{"text": "Wrong", "start": .1, "end": .8}]]},
                        {"bounds": [[.1, .8]], "words": [[{"text": "Yes", "start": 5, "end": 6}]]}):
                with self.subTest(bad=bad):
                    cache.write_text(json.dumps(bad), encoding="utf-8")
                    before = run.call_count
                    self.assertEqual(align_take(take, root, None, free_gpu=lambda: None)[1], [[("Yes", .1, .8)]])
                    self.assertEqual(run.call_count, before + 1)
            align_take(take, root, None, free_gpu=lambda: None)
            self.assertEqual(run.call_count, before + 1)
        with tempfile.TemporaryDirectory() as root, patch("h3_app.dubbing.tools.run", return_value={
                "bounds": [[.1, .8]], "words": [[]]}):
            with self.assertRaises(ValueError):
                align_take(take, root, None, free_gpu=lambda: None)
            self.assertEqual(list(Path(root).glob("*.alignment.json")), [])

    def test_soundstage_repair_reuses_good_chunks_and_preserves_audio(self):
        from h3_app.drama3.serving import alignment, audio, takes
        from h3_app.errors import H3Error
        from h3_app.provenance import read_snapshot, write_snapshot

        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"H3_WORKSPACE_DIR": directory}):
            root = Path(directory)
            runtime = SimpleNamespace(outputs_dir=root)
            result = root / "result.mp3"
            result.write_bytes(b"unchanged mixed audio")
            original_mtime = result.stat().st_mtime_ns
            chunks = []
            for index, text in enumerate(("Hello", "Bye")):
                key = str(index) * 32
                takes.save_take(takes.takes_root(runtime), key, audio.encode_wav(np.full(2000, .2), 1000))
                samples, rate = takes.load_take(takes.takes_root(runtime), key)
                chunks.append({"text": text, "take": key, "start": 1 + index * 3, "end": 3 + index * 3,
                               "lines": [{"text": text, "start": 1 + index * 3, "end": 3 + index * 3}],
                               "alignment": {"start": 1 + index * 3, "scale": 1.5,
                                             "audio": alignment.take_audio_digest(samples, rate)}})
            write_snapshot(result, {"family": "drama3", "settings": {"chunks": chunks, "duration": 8}})
            calls = []

            def run(_op, payload, *_args, **_kwargs):
                text = payload["lines"][0]
                calls.append(text)
                if calls == ["Hello"]:
                    raise H3Error("temporary alignment failure")
                return {"bounds": [[.1, .8]], "words": [[{"text": text, "start": .1, "end": .8}]]}

            with patch("h3_app.dubbing.tools.run", side_effect=run):
                self.assertIn("1 still missing", list(alignment.repair_result(result, runtime, free_gpu=lambda: None))[-1])
                self.assertIn("All word timings generated: 2/2", list(alignment.repair_result(
                    result, runtime, free_gpu=lambda: None))[-1])
                self.assertIn("All word timings generated", list(alignment.repair_result(
                    result, runtime, free_gpu=lambda: None))[-1])
            self.assertEqual(calls, ["Hello", "Bye", "Hello"])
            words = read_snapshot(result)["settings"]["chunks"][0]["lines"][0]["words"]
            self.assertEqual(words, [{"text": "Hello", "start": 1.15, "end": 2.2}])
            self.assertEqual(result.read_bytes(), b"unchanged mixed audio")
            self.assertEqual(result.stat().st_mtime_ns, original_mtime)
            snapshot = read_snapshot(result)
            snapshot["settings"]["chunks"][0]["lines"][0]["words"] = []
            write_snapshot(result, snapshot)
            takes.save_take(takes.takes_root(runtime), "0" * 32, audio.encode_wav(np.full(2000, .1), 1000))
            message = list(alignment.repair_result(result, runtime, free_gpu=lambda: None))[-1]
            self.assertIn("saved take changed", message)
            self.assertIn("1 still missing", message)
            snapshot = read_snapshot(result)
            snapshot["settings"]["chunks"][0].pop("alignment")
            write_snapshot(result, snapshot)
            self.assertIn("Remix once", list(alignment.repair_result(result, runtime, free_gpu=lambda: None))[-1])

    def test_soundstage_repair_handler_refreshes_subtitles_only(self):
        from h3_ui.drama3.handlers import Drama3Handlers, Drama3Services

        handlers = Drama3Handlers(Drama3Services(runtime_config=lambda: None, stage_file=lambda *_: None,
            generation_services=lambda: None, free_gpu=lambda: None, resolve_result=lambda _path: "original.mp3"))
        with patch("h3_app.drama3.serving.alignment.repair_result", return_value=iter([
                "Aligning saved speech", "All word timings generated: 1/1 lines aligned."])) as repair, \
                patch.object(handlers, "drama3_subtitles", return_value={"value": "updated subtitles"}) as subtitles, \
                patch("h3_models.resolve_hf_token", return_value=None), patch("h3_app.dubbing.tools.stop"):
            updates = list(handlers.repair_drama3_words("cached/result.mp3"))
        self.assertEqual(updates[-1], ({"value": "updated subtitles"}, "All word timings generated: 1/1 lines aligned."))
        self.assertEqual(repair.call_args.args[0], "original.mp3")
        subtitles.assert_called_once_with("cached/result.mp3")

    def test_repair_preserves_original_generation_provenance_and_records_repair_separately(self):
        from h3_app.drama3.serving.alignment import repair_result
        from h3_app.jobs import CURRENT_JOB
        from h3_app.provenance import read_snapshot, write_snapshot

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.mp3"
            path.write_bytes(b"unchanged audio")
            identity = {"application_job_id": "original-job", "variant": 3, "retry_of": "earlier-job",
                        "source_asset_ids": ["original-source"], "image_canvas": {"width": 512}}
            write_snapshot(path, {"family": "Drama 3", **identity, "settings": {"duration": 1,
                "chunks": [{"text": "Hello", "lines": [{"text": "Hello", "start": 0, "end": 1,
                    "words": [{"text": "Hello", "start": .1, "end": .8}]}]}]}})
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            job = SimpleNamespace(id="repair-job", variant=0, retry_of=None, source_asset_ids=["repair-source"],
                                  canvas_request={"width": 1024}, check=lambda: None)
            token = CURRENT_JOB.set(job)
            try:
                messages = list(repair_result(path, SimpleNamespace(outputs_dir=Path(directory)), free_gpu=lambda: None))
            finally:
                CURRENT_JOB.reset(token)
            saved = read_snapshot(path)
            self.assertEqual({key: saved[key] for key in identity}, identity)
            self.assertEqual(saved["settings"]["word_alignment"]["repair_job_id"], "repair-job")
            self.assertIn("All word timings generated", messages[-1])
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_worker_times_words_per_clip_and_skips_clips_it_cannot_align(self):
        from unittest.mock import MagicMock
        worker = Worker(Path("."))
        model = MagicMock()
        model.transcribe.return_value = [SimpleNamespace(language="English"), SimpleNamespace(language="")]
        model.forced_aligner.align.return_value = [[SimpleNamespace(text="Hi", start_time=.1, end_time=.4)]]
        with patch.object(worker, "_asr", return_value=model):
            reply = worker.handle({"op": "align_words", "clips": [{"audio": "a.wav", "text": "Hi!"},
                                                                  {"audio": "b.wav", "text": "Hm"}]})
        self.assertEqual(reply["words"], [[{"text": "Hi", "start": .1, "end": .4}], None])
        self.assertEqual(reply["errors"][0], "")
        self.assertIn("unknown or mixed language", reply["errors"][1])

    def test_invalid_batch_alignment_retries_individual_clip_and_reports_failures(self):
        from unittest.mock import MagicMock

        worker = object.__new__(Worker)
        model = MagicMock()
        model.transcribe.return_value = [SimpleNamespace(language="English")]
        wrong = [[SimpleNamespace(text="Wrong", start_time=.1, end_time=.4)]]
        right = [[SimpleNamespace(text="Hi", start_time=.1, end_time=.4)]]
        request = {"clips": [{"audio": "a.wav", "text": "Hi!"}]}
        with patch.object(worker, "_asr", return_value=model):
            model.forced_aligner.align.side_effect = [wrong, right]
            fixed = worker.align_words(request)
            self.assertEqual(fixed["words"], [[{"text": "Hi", "start": .1, "end": .4}]])
            self.assertEqual(fixed["errors"], [""])
            self.assertEqual(model.forced_aligner.align.call_args.kwargs["audio"], "a.wav")
            model.forced_aligner.align.side_effect = [wrong, wrong]
            failed = worker.align_words(request)
            self.assertEqual(failed["words"], [None])
            self.assertIn("does not match the script", failed["errors"][0])

    def test_dub_alignment_uses_known_language_without_redetecting_short_clip(self):
        from unittest.mock import MagicMock

        worker = object.__new__(Worker)
        model = MagicMock()
        model.transcribe.side_effect = AssertionError("must not redetect a known language")
        model.forced_aligner.align.return_value = [[SimpleNamespace(text="Hi", start_time=.1, end_time=.4)]]
        with patch.object(worker, "_asr", return_value=model):
            reply = worker.align_words({"clips": [{"audio": "a.wav", "text": "Hi!", "language": "en"}]})
            self.assertTrue(reply["words"][0])
            self.assertEqual(model.forced_aligner.align.call_args.kwargs["language"], ["English"])
            failed = worker.align_words({"clips": [{"audio": "a.wav", "text": "Hi!", "language": "Unsupported"}]})
            self.assertEqual(failed["words"], [None])
            self.assertIn("does not support language", failed["errors"][0])

    def test_render_places_words_with_their_take(self):
        tree = parse_drama3_tree("<|speaker:0|> Hello.")
        speech = list(iter_speech_nodes(tree))[0]
        chunk = Chunk(0, "Speaker 0", "Hello.", (), ("Hello.",))
        take = generate.Take(chunk, np.full(2000, .2, dtype=np.float32), 1000, 1, None,
                             line_bounds=[(100, 800)], line_words=[[("Hello", .1, .8)]])
        words = {}
        generate._render_ast(tree, {id(speech): [chunk]}, {0: take}, {}, 2000, words=words)
        self.assertEqual(words[0], [[("Hello", 200, 1600)]])


class WordSpanTests(unittest.TestCase):
    def test_words_take_their_trailing_punctuation(self):
        html = word_spans_html("Hello, world!", [{"text": "hello", "start": 0, "end": .4},
                                                  {"text": "world", "start": .5, "end": .9}])
        self.assertEqual(html, '<span class="h3-word" data-start="0.000" data-end="0.400">Hello, </span>'
                               '<span class="h3-word" data-start="0.500" data-end="0.900">world!</span>')

    def test_cjk_characters_are_words(self):
        html = word_spans_html("你好！", [{"text": "你", "start": 0, "end": .2}, {"text": "好", "start": .2, "end": .4}])
        self.assertIn('data-start="0.200" data-end="0.400">好！</span>', html)

    def test_text_the_words_do_not_cover_stays_plain(self):
        for words in ([{"text": "hello", "start": 0, "end": .4}], [{"text": "bye", "start": 0, "end": .4}], None):
            self.assertEqual(word_spans_html("Hello <there>", words), "Hello &lt;there&gt;")
