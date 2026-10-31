"""Nemotron transcription and overlapping speakers: reply parsing, clean references, timing and mixing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from h3_app.dubbing import nemotron, pipeline, segments, session as sessions
from h3_app.errors import H3Error
from h3_requirements import nemotron_install_commands
from tests.test_dubbing import Workspace, _tone_wav


def reply(words, *, error=None, diarization=None, mapping=None, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": ""}}],
            "ctc_timestamps": {"words": words, "error": error, "diarization": diarization or [],
                               "speaker_tag_to_diarization_speaker": mapping or {}}}


def word(text, start, end, speaker):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


def lines_of(*spans):
    return segments.normalize({"start": s, "end": e, "speaker": who, "source": text, "translation": text.upper()}
                              for s, e, who, text in spans)


class ReplyTests(unittest.TestCase):
    def test_words_become_one_speaker_lines_and_overlaps_stay_separate(self):
        found = nemotron.reply_words(reply([
            word("Hello", 0.0, 0.4, "<spk:0>"), word("there.", 0.5, 2.2, "<spk:0>"),
            word("Wait", 1.0, 1.4, "<spk:1>"), word("no", 1.5, 1.8, "<spk:1>"),
            word("Okay", 4.0, 4.5, None),
        ]), 10.0)
        self.assertEqual([w["speaker"] for w in found], ["speaker1", "speaker1", "speaker2", "speaker2", "speaker2"])
        lines = nemotron.word_lines(found)
        self.assertEqual([(line["speaker"], line["source"]) for line in lines],
                         [("speaker1", "Hello there."), ("speaker2", "Wait no"), ("speaker2", "Okay")])
        self.assertLess(lines[1]["start"], lines[0]["end"])  # the overlap survives

    def test_cjk_words_join_without_spaces(self):
        self.assertEqual(nemotron.join_words(["你", "好", "，", "Bob", "!"]), "你好， Bob!")

    def test_bad_replies_are_retryable_errors_not_patched(self):
        for bad, message in ((reply([], error="alignment failed"), "could not time"),
                             (reply([word("a", 0, 1, 0)], finish="length"), "cut off"),
                             ({"choices": [{}]}, "no word timings"),
                             (reply([word("a", 0, 30, 0)]), "outside")):
            with self.subTest(message), self.assertRaisesRegex(H3Error, message):
                nemotron.reply_words(bad, 10.0)

    def test_activity_merges_words_and_files_unmapped_diarization_as_unknown(self):
        words = nemotron.reply_words(reply([word("a", 0, 0.5, 0), word("b", 0.6, 1.0, 0)]), 10)
        activity = nemotron.reply_activity(reply([], diarization=[
            {"start": 2, "end": 3, "speaker": "D0"}, [5, 6, "D9"]], mapping={"<spk:0>": "D0"}), words)
        self.assertEqual(activity["speaker1"], [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(activity[nemotron.UNKNOWN_SPEAKER], [[5.0, 6.0]])

    def test_install_keeps_the_editable_checkout_beside_the_venv(self):
        commands = nemotron_install_commands(Path("/opt/nemotron-venv"))
        self.assertIn(str(Path("/opt/nemotron-venv-src")), commands[1])
        self.assertIn("--locked", commands[4])
        self.assertIn("-e", commands[5])


class NearestCleanTests(unittest.TestCase):
    def activity(self, lines):
        return {line["speaker"]: [] for line in lines} | {
            speaker: [(line["start"], line["end"]) for line in lines if line["speaker"] == speaker]
            for speaker in {line["speaker"] for line in lines}}

    def test_three_speakers_overlapping_each_take_their_closest_solo_passage(self):
        lines = lines_of((0, 4, "a", "A alone"), (5, 8, "b", "B alone"), (20, 23, "a", "a over"),
                         (20, 23, "b", "b over"), (20.5, 22, "c", "c over"), (30, 33, "b", "B later"),
                         (24, 27, "a", "A after"), (100, 104, "c", "C far"))
        clean = pipeline.clean_lines(lines, self.activity(lines))
        pick = {line["source"]: lines[pipeline.nearest_clean(lines, i, clean)]["source"]
                for i, line in enumerate(lines) if "over" in line["source"]}
        self.assertEqual(pick, {"a over": "A after", "b over": "B later", "c over": "C far"})

    def test_ties_go_earlier_and_a_speaker_never_alone_gets_none(self):
        lines = lines_of((0, 3, "a", "before"), (5, 6, "a", "short"), (8, 11, "a", "after"),
                         (12, 14, "c", "c talks"), (12.5, 14, "d", "d over"))
        clean = pipeline.clean_lines(lines, self.activity(lines))
        self.assertEqual(lines[pipeline.nearest_clean(lines, 1, clean)]["source"], "before")
        self.assertIsNone(pipeline.nearest_clean(lines, 4, clean))
        self.assertIsNone(pipeline.nearest_clean(lines, 3, clean))

    def test_speech_in_the_padding_or_heard_but_not_in_any_line_is_not_clean(self):
        lines = lines_of((0, 3, "a", "touching"), (3.05, 4, "b", "next"), (10, 13, "a", "far"))
        activity = self.activity(lines) | {nemotron.UNKNOWN_SPEAKER: [(11, 11.5)]}
        self.assertEqual(pipeline.clean_lines(lines, activity), set())


class OverlapSessionTests(Workspace):
    def session(self, lines, transcript=None, seconds=30.0):
        session = sessions.create(self.runtime, _tone_wav(self.root / "a.wav", 1.0), video=False)
        session.set_file("vocals", _tone_wav(session.root / "vocals.wav", seconds))
        session.segments = lines
        if transcript is not None:
            session.data["transcript"] = transcript
        session.data["pipeline"] = pipeline.PIPELINE_NEMOTRON
        session.save()
        return session

    def test_nearest_clean_plans_a_clean_clip_or_falls_back_to_the_line_itself(self):
        session = self.session(lines_of((0, 3, "a", "solo a"), (5, 6, "a", "a over"), (5, 6, "b", "b over")))
        planned = {line.segment["source"]: line for line in
                   pipeline._plan(session, [0, 1, 2], pipeline.VOICE_NEAREST, self.runtime)}
        self.assertEqual(planned["a over"].transcript, "solo a")
        self.assertEqual(planned["a over"].reference, planned["solo a"].reference)
        self.assertEqual(planned["b over"].transcript, "b over")  # B never speaks alone
        self.assertIn("5.000_6.000", planned["b over"].reference.name)

    def test_a_line_moved_to_another_speaker_still_counts_as_cross_talk(self):
        original = lines_of((0, 3, "b", "said by b"), (10, 13, "a", "a solo"), (20, 21, "a", "a short"))
        edited = lines_of((0, 3, "a", "said by b"), (10, 13, "a", "a solo"), (20, 21, "a", "a short"))
        session = self.session(edited, {"segments": original})
        planned = pipeline._plan(session, [2], pipeline.VOICE_NEAREST, self.runtime)
        self.assertEqual(planned[0].transcript, "a solo")

    def test_long_clean_clip_is_trimmed_at_a_sentence_with_native_words(self):
        words = [{"text": f"w{i}.", "start": float(i), "end": i + 0.8, "speaker": "speaker1"} for i in range(20)]
        line = {"start": 0.0, "end": 19.8, "speaker": "speaker1",
                "source": nemotron.join_words([w["text"] for w in words]), "translation": "x"}
        session = self.session([line], {"words": words})
        clip, text = pipeline._trimmed(session, session.segments[0])
        self.assertEqual(clip["end"], 14.8)
        self.assertTrue(text.endswith("w14."))

    def test_overlapping_kept_lines_restore_the_original_once(self):
        session = self.session(lines_of((0, 2, "a", "one"), (1, 3, "b", "two")))
        track, _notes, _spoken = pipeline._speech_track(session, [], pipeline.DubRequest(session.id), 4.0)
        self.assertAlmostEqual(float(track[int(1.5 * pipeline.RATE)]), 0.5, places=3)
        self.assertEqual(pipeline.kept_original(session.segments, set()), [(0.0, 3.0, {"a", "b"})])

    def test_take_marks_tell_simultaneous_speakers_apart(self):
        session = self.session(lines_of((1, 2, "a", "yes"), (1, 2, "b", "yes")))
        pipeline._record_take_problems(session, {0: (session.segments[0], "failed check")})
        self.assertEqual(pipeline.take_problems(sessions.load(self.runtime, session.id)), {0: "failed check"})

    def test_defaults_follow_the_transcription(self):
        self.assertEqual(pipeline.default_voice_mode(pipeline.PIPELINE_NEMOTRON), pipeline.VOICE_NEAREST)
        for other in (pipeline.PIPELINE_LOCAL, pipeline.PIPELINE_MOSS, pipeline.PIPELINE_GEMINI, None):
            self.assertEqual(pipeline.default_voice_mode(other), pipeline.VOICE_LINE)


class OverlapTimingTests(unittest.TestCase):
    def test_another_speaker_talking_over_a_line_does_not_cut_its_slot_or_subtitle(self):
        lines = lines_of((0, 2, "a", "long one"), (1, 1.5, "b", "over"), (4, 5, "a", "next"))
        self.assertAlmostEqual(pipeline.slot_end(lines, 0, 10), 3.95)
        shown = pipeline.subtitle_lines(lines, {0: (0.0, 2.5)})
        self.assertEqual(shown[0]["end"], 2.5)


class SleepTests(unittest.TestCase):
    """vLLM's own sleep mode between analyses, with its endpoints and the GPU faked."""

    def setUp(self):
        settings = nemotron.NemotronSettings(venv=Path("/venv"), data_dir=Path("/data"), model_root=Path("/models"))
        self.server = nemotron.NemotronServer(settings)
        self.calls = []
        self.awake_reply = False
        self.server.running = lambda: True
        self.server.healthy = lambda timeout=2.0: True
        self.server._engine = lambda path, **params: self.calls.append((path, params))
        self.server._is_sleeping = lambda: bool(self.calls) and self.calls[-1][0] == "/sleep" or self.awake_reply
        self.stops = []
        self.server.stop = lambda: (self.stops.append(True), setattr(self.server, "_asleep", False))
        self.patches = [patch.object(nemotron, "gpu_memory", return_value=(80 * 1024, 96 * 1024, "")),
                        patch.object(nemotron.checkpoint, "available_ram", return_value=200 * 2**30)]
        for started in self.patches:
            started.start()
        self.server.command()  # a sleep-mode launch

    def tearDown(self):
        for started in self.patches:
            started.stop()

    def test_settings_default_to_sleep_and_reject_unknown_modes(self):
        runtime = SimpleNamespace(script_dir=Path("/app"), comfy_dir=Path("/app/ComfyUI"))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(nemotron.NemotronSettings.from_environment(runtime).sleep_mode, "auto")
        with patch.dict("os.environ", {"H3_NEMOTRON_SLEEP": "OFF"}, clear=True):
            self.assertEqual(nemotron.NemotronSettings.from_environment(runtime).sleep_mode, "off")
        with self.assertRaises(ValueError):
            replace(self.server.settings, sleep_mode="deep")

    def test_launch_enables_sleep_unless_it_is_off_or_failed(self):
        self.assertIn("--enable-sleep-mode", self.server.command())
        self.server._sleep_disabled = True
        self.assertNotIn("--enable-sleep-mode", self.server.command())
        off = nemotron.NemotronServer(replace(self.server.settings, sleep_mode="off"))
        self.assertNotIn("--enable-sleep-mode", off.command())

    def test_release_sleeps_and_the_next_analysis_wakes(self):
        self.server.release()
        self.assertTrue(self.server.sleeping)
        self.assertEqual(self.calls, [("/sleep", {"level": 1})])
        self.server.release()  # already asleep
        self.assertEqual(len(self.calls), 1)
        self.server.ensure_ready(lambda: None)
        self.assertFalse(self.server.sleeping)
        self.assertEqual(self.calls[-1][0], "/wake_up")
        self.assertEqual(self.stops, [])

    def test_short_host_ram_stops_without_disabling_sleep(self):
        with patch.object(nemotron.checkpoint, "available_ram", return_value=10 * 2**30):
            self.server.release()
        self.assertEqual((self.calls, self.stops, self.server._sleep_disabled), ([], [True], False))

    def test_a_failed_sleep_stops_and_later_launches_skip_sleep(self):
        def refuse(path, **params):
            raise RuntimeError("/sleep answered HTTP 404")

        self.server._engine = refuse
        self.server.release()
        self.assertEqual(self.stops, [True])
        self.assertTrue(self.server._sleep_disabled)
        self.assertNotIn("--enable-sleep-mode", self.server.command())

    def test_waking_without_room_keeps_it_asleep(self):
        self.server.release()
        with patch.object(nemotron, "gpu_memory", return_value=(10 * 1024, 96 * 1024, "drama3")), \
                patch.object(nemotron, "reclaim_audio_tools", return_value=None):
            with self.assertRaises(H3Error):
                self.server.ensure_ready(lambda: None)
        self.assertTrue(self.server.sleeping)

    def test_other_jobs_sleep_it_and_an_unload_stops_it(self):
        with patch.object(nemotron, "_SERVER", self.server):
            nemotron.release_for_job(nemotron.FAMILY)
            self.assertFalse(self.server.sleeping)
            nemotron.release_for_job("drama3")
            self.assertTrue(self.server.sleeping)
            nemotron.release_for_job("unload")
        self.assertEqual(self.stops, [True])


if __name__ == "__main__":
    unittest.main()
