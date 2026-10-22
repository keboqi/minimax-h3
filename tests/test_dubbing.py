"""Dub & Translate and Media downloads: pure helpers, storage, handlers and an ffmpeg-backed dub."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

from h3_app import video_download
from h3_app.dubbing import pipeline, segments, session as sessions, tools, tools_worker
from h3_app.errors import H3Error

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


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
        self.assertEqual(pipeline.slot_end(lines, 1, 10), 4.0)
        self.assertEqual(pipeline.slot_end(lines, 2, 10), 10)

    def test_fit_leaves_short_takes_and_caps_the_speed_up(self):
        take = np.zeros(48000, dtype=np.float32)
        same, speed, note = pipeline.fit(take, 24000, 3.0, 1.35)
        self.assertIs(same, take)
        self.assertEqual((speed, note), (1.0, ""))
        with patch.object(pipeline, "_stretch", side_effect=lambda s, r, f: s[: int(len(s) / f)]) as stretch:
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


class HandlerTests(Workspace):
    def setUp(self):
        super().setUp()
        from h3_ui import dubbing_handlers

        self.handlers = dubbing_handlers
        previous = dubbing_handlers._SERVICES
        dubbing_handlers.install(dubbing_handlers.DubbingServices(
            runtime_config=lambda: self.runtime, free_gpu=lambda: None,
            list_media=lambda mode: [self.root / f"{mode}.file"]))
        self.addCleanup(dubbing_handlers.install, previous)

    def test_library_choices_and_table_saves(self):
        update = self.handlers.library_choices()
        self.assertEqual([label for label, _path in update["choices"]], ["Video · Video.file", "Audio · Audio.file"])
        self.assertIn("Analyze a source first", self.handlers.save_lines("", []))
        source = self.root / "a.wav"
        source.write_bytes(b"x")
        created = sessions.create(self.runtime, source, video=False)
        status = self.handlers.save_lines(created.id, [[1, "00:00.000", "00:01.000", "speaker1", "Hi", "Hola", True],
                                                       [2, "00:02.000", "00:03.000", "speaker1", "Yo", "", True]])
        self.assertEqual(status, "Saved 2 line(s); 1 ready to dub.")

    def test_download_handler_streams_progress_then_the_path(self):
        target = self.root / "clip.mp4"

        def fake(url, quality, runtime, *, report, stopped):
            report("Downloading 10%")
            self.assertFalse(stopped())
            return target

        with patch.object(self.handlers.video_download, "download_video", side_effect=fake):
            updates = list(self.handlers.download_video("https://x.com/v", "720p"))
        self.assertEqual(updates[-1], ("Downloaded **clip.mp4** to Media.", str(target)))
        self.assertTrue(all(path == "" for _status, path in updates[:-1]))

    def test_errors_become_status_lines(self):
        updates = list(self.handlers.analyze_source(None, None, True, "Balanced (BS-Roformer)", pipeline.PIPELINE_LOCAL,
                                                    "Auto-detect", "English", True, pipeline.TRANSLATOR_HYMT,
                                                    True, 0.35, "gemini-3.8-flash", "", ""))
        self.assertIn("Upload a video or audio file", updates[-1][2])
        updates = list(self.handlers.generate_dub("", pipeline.VOICE_LINE, "Balance", 1.35, 8, 100, 100,
                                                  True, False, ""))
        self.assertIn("Analyze a source first", updates[-1][3])

    def test_cookie_handlers(self):
        status, sites, cleared = self.handlers.import_curl_cookies(
            "curl 'https://www.bilibili.com/' -b 'SESSDATA=1'", "")
        self.assertEqual((status, cleared), ("Saved 1 cookie(s) for bilibili.com.", ""))
        self.assertIn("**bilibili.com** (1)", sites)
        self.assertIn("Deleted", self.handlers.delete_cookies("bilibili.com")[0])


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

    def analyze(self, video, *, separate):
        def fake_tools(op, payload, runtime, **_kwargs):
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

        request = pipeline.AnalyzeRequest(str(video), target_language="French", separate=separate)
        with patch.object(pipeline.tools, "run", side_effect=fake_tools):
            updates = list(pipeline.analyze(request, runtime=self.runtime, free_gpu=lambda: None))
        result = updates[-1]
        self.assertIsInstance(result, pipeline.Analysis)
        return result.session

    def fake_synthesize(self, planned, request, session, **_kwargs):
        for line in planned:
            seconds = 3.0 if line.index == 0 else 0.5  # line 1 is longer than its 2.55 s slot
            line.take = (0.2 * np.sin(np.arange(int(seconds * pipeline.RATE)) / 20)).astype(np.float32)
        yield "voiced"

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
            updates = list(pipeline.dub(pipeline.DubRequest(session.id), runtime=self.runtime,
                                        free_gpu=lambda: None))
        result = updates[-1]
        self.assertIsInstance(result, pipeline.Dub)
        self.assertEqual(self.streams(result.video), ["video", "audio", "audio", "subtitle", "subtitle"])
        self.assertTrue(result.audio.is_file())
        self.assertEqual([p.name.rsplit(".", 2)[-2] for p in result.subtitles], ["french", "original"])
        self.assertIn("1 sped up to fit", result.message)
        # The first line's clip is long enough to be its own reference; the short one uses its speaker's best.
        references = sorted(p.name for p in (session.root / "references").iterdir())
        self.assertEqual(len(references), 1)
        snapshot = json.loads(Path(str(result.video) + ".settings.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["family"], "Dub & Translate")

    def test_unseparated_dub_ducks_the_original_and_burns_subtitles(self):
        session = self.analyze(self.make_video(), separate=False)
        self.assertIsNone(session.path("backing"))
        with patch.object(pipeline, "_synthesize", side_effect=self.fake_synthesize):
            result = list(pipeline.dub(pipeline.DubRequest(session.id, keep_original_track=False,
                                                           burn_subtitles=True),
                                       runtime=self.runtime, free_gpu=lambda: None))[-1]
        self.assertEqual(self.streams(result.video), ["video", "audio", "subtitle", "subtitle"])

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
