"""Same-language dubbing: lines are copied as said or rewritten, and only changed lines are voiced."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from h3_app.dubbing import gemini, pipeline, session as sessions
from h3_ui.dubbing.handlers import DubbingHandlers, DubbingServices

from tests.test_dubbing import HAS_FFMPEG, Workspace, _run

SAME = pipeline.SAME_LANGUAGE


def lines_of(session):
    return [(line["translation"], line["dub"]) for line in session.segments]


class SameLanguageLines(Workspace):
    def setUp(self):
        super().setUp()
        source = self.root / "talk.wav"
        source.write_bytes(b"data")
        self.session = sessions.create(self.runtime, source, video=False)
        self.session.data.update(target_language=SAME, source_language="English")
        self.session.segments = [
            {"start": 1, "end": 2, "speaker": "speaker1", "source": "Add three grams.", "translation": "",
             "dub": True},
            {"start": 3, "end": 4, "speaker": "speaker1", "source": "Then stir.", "translation": "", "dub": True},
        ]
        self.session.save()
        self.handlers = DubbingHandlers(DubbingServices(lambda: self.runtime, lambda: None, lambda mode: []))

    def retranslate(self, indices=None, **settings):
        request = pipeline.AnalyzeRequest("", target_language=SAME, **settings)
        return list(pipeline.retranslate(self.session.id, indices, request, runtime=self.runtime,
                                         free_gpu=None))[-1]

    def test_copy_fills_new_text_with_what_was_said_and_keeps_the_audio(self):
        result = self.retranslate()
        self.assertEqual(lines_of(result.session), [("Add three grams.", False), ("Then stir.", False)])
        self.assertIn("Copied 2 line(s)", result.message)

    def test_rewrite_voices_only_lines_the_rewriter_changed(self):
        with patch.object(pipeline.gemini, "rewrite", return_value=["Add two grams.", "Then stir."]) as rewrite:
            result = self.retranslate(rewrite=True, style="Children", translator=pipeline.TRANSLATOR_GEMINI)
        self.assertEqual(lines_of(result.session), [("Add two grams.", True), ("Then stir.", False)])
        self.assertEqual(rewrite.call_args.kwargs["language"], "English")
        self.assertIn("children", rewrite.call_args.kwargs["style"])
        self.assertEqual(result.session.data["rewrite_style"], "Children")
        self.assertEqual(pipeline.dub_label(result.session), "English · for Children")

    def test_hymt_rewrite_asks_for_the_spoken_language(self):
        calls = []

        def fake_tools(op, payload, runtime, **_kwargs):
            calls.append(payload)
            return {"texts": ["Add 3 g.", "Then stir."]}

        with patch.object(pipeline.tools, "run", side_effect=fake_tools):
            result = self.retranslate(rewrite=True, style="Simpler words")
        self.assertEqual(calls[0]["target_language"], "English")
        self.assertEqual(lines_of(result.session), [("Add 3 g.", True), ("Then stir.", False)])

    def test_an_edit_voices_a_changed_line_and_a_revert_keeps_the_original(self):
        self.retranslate()
        key = {"index": 0, "start": 1, "speaker": "speaker1", "source": "Add three grams."}
        board = {"session": self.session.id, "focus": 0}
        self.handlers.save_focused(self.session.id, key, board, "00:01.0", "00:02.0", "speaker1",
                                   "Add three grams.", "Add two grams.", False)
        self.assertEqual(lines_of(sessions.load(self.runtime, self.session.id))[0], ("Add two grams.", True))
        self.handlers.save_focused(self.session.id, key, board, "00:01.0", "00:02.0", "speaker1",
                                   "Add three grams.", "Add three grams.", True)
        self.assertEqual(lines_of(sessions.load(self.runtime, self.session.id))[0], ("Add three grams.", False))

    def test_correcting_what_was_said_keeps_an_unedited_copy_in_step(self):
        self.retranslate()
        key = {"index": 0, "start": 1, "speaker": "speaker1", "source": "Add three grams."}
        board = {"session": self.session.id, "focus": 0}
        self.handlers.save_focused(self.session.id, key, board, "00:01.0", "00:02.0", "speaker1",
                                   "Add three grams of salt.", "Add three grams.", False)
        line = sessions.load(self.runtime, self.session.id).segments[0]
        self.assertEqual((line["source"], line["translation"], line["dub"]),
                         ("Add three grams of salt.", "Add three grams of salt.", False))

    def test_the_dub_switch_set_by_hand_holds_while_the_text_is_unchanged(self):
        self.retranslate()
        board = {"session": self.session.id, "focus": 1, "changed": {"line": 1, "dub": True}}
        self.handlers.toggle_dub(self.session.id, board)
        self.assertEqual(lines_of(sessions.load(self.runtime, self.session.id))[1], ("Then stir.", True))

    def test_table_saves_follow_the_same_rule(self):
        self.retranslate()
        rows = [[1, "00:01.0", "00:02.0", "speaker1", "Add three grams.", "Add two grams.", True],
                [2, "00:03.0", "00:04.0", "speaker1", "Then stir.", "Then stir.", True]]
        self.handlers.save_lines(self.session.id, rows)
        self.assertEqual(lines_of(sessions.load(self.runtime, self.session.id)),
                         [("Add two grams.", True), ("Then stir.", True)])

    def test_pasted_rewrite_applies_with_the_rule(self):
        prompt, request = pipeline.translation_request(self.session, None, SAME, style="Teens")
        self.assertIn("Rewrite each line", prompt)
        self.assertIn("in English", prompt)
        self.assertIn("teenagers", prompt)
        result = pipeline.apply_translation_reply(self.session.id, request, '["Add two grams.", "Then stir."]',
                                                  runtime=self.runtime)
        self.assertEqual(lines_of(result.session), [("Add two grams.", True), ("Then stir.", False)])
        self.assertIn("Applied 2 rewritten line(s)", result.message)

    def test_leaving_the_same_language_translates_and_voices_every_line(self):
        self.retranslate()
        request = pipeline.AnalyzeRequest("", target_language="French", translator=pipeline.TRANSLATOR_GEMINI)
        with patch.object(pipeline.gemini, "translate", return_value=["Ajoutez trois grammes.", "Puis remuez."]):
            result = list(pipeline.retranslate(self.session.id, None, request, runtime=self.runtime,
                                               free_gpu=None))[-1]
        self.assertEqual(lines_of(result.session), [("Ajoutez trois grammes.", True), ("Puis remuez.", True)])


class SameLanguagePrompts(unittest.TestCase):
    def test_transcription_for_the_same_language_does_not_translate(self):
        prompt = gemini.transcribe_prompt(target_language=SAME, translate=True, ignore_non_speech=True,
                                          notes="Keep Mei.")
        self.assertIn("without translating it", prompt)
        self.assertNotIn(SAME, prompt)

    def test_rewrite_prompt_keeps_timing_and_terms(self):
        prompt = gemini.rewrite_prompt(["Hello there."], language="English", notes="Keep 'dashi'.",
                                       seconds=[1.2], style="simple words")
        self.assertIn("Rewrite each line of this video dialogue in English for simple words", prompt)
        self.assertIn("Do not translate", prompt)
        self.assertIn("1.2", prompt)
        self.assertIn("dashi", prompt)
        self.assertIn("exactly 1 strings", prompt)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg is required")
class SameLanguageAnalysis(Workspace):
    def make_audio(self) -> Path:
        audio = self.root / "talk.wav"
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
              "sine=frequency=330:duration=4", str(audio)])
        return audio

    def analyze(self, **settings):
        def fake_tools(op, payload, runtime, **_kwargs):
            if op == "transcribe":
                return {"language": "English", "segments": [
                    {"start": 0.2, "end": 2.6, "text": "First line here."},
                    {"start": 2.8, "end": 3.6, "text": "Second."}]}
            if op == "translate":
                return {"texts": ["First line, simply.", "Second."]}
            raise AssertionError(op)

        request = pipeline.AnalyzeRequest(str(self.make_audio()), target_language=SAME, separate=False,
                                          refine_timestamps=False, **settings)
        with patch.object(pipeline.tools, "run", side_effect=fake_tools):
            return list(pipeline.analyze(request, runtime=self.runtime, free_gpu=lambda: None))[-1]

    def test_copy_analysis_skips_translation(self):
        result = self.analyze()
        self.assertEqual(lines_of(result.session), [("First line here.", False), ("Second.", False)])
        self.assertIn("copied them as said", result.message)
        self.assertEqual(pipeline.dub_label(result.session), "English · edited")

    def test_rewrite_analysis_voices_the_changed_lines(self):
        result = self.analyze(rewrite=True, style="Children")
        self.assertEqual(lines_of(result.session), [("First line, simply.", True), ("Second.", False)])
        self.assertIn("rewrote them, 1 changed", result.message)

    def test_gemini_transcribes_only(self):
        with patch.object(pipeline.gemini, "transcribe",
                          return_value=([{"start": 0.2, "end": 2.6, "speaker": "speaker1",
                                          "source": "Hello.", "translation": ""}], {"speaker1": ""}, "")) as call:
            result = self.analyze(pipeline=pipeline.PIPELINE_GEMINI, gemini_model="gemini-x")
        self.assertFalse(call.call_args.kwargs["translate"])
        self.assertEqual(lines_of(result.session), [("Hello.", False)])

    def test_dub_voices_only_edited_lines(self):
        session = self.analyze().session
        lines = [dict(line) for line in session.segments]
        lines[0]["translation"] = "First line, fixed."
        pipeline.apply_same_language_rule(session, lines)
        session = sessions.update_segments(self.runtime, session.id, lines)
        voiced = []

        def synthesize(planned, request, working, **_kwargs):
            voiced.extend(line.index for line in planned)
            for line in planned:
                line.take = (0.2 * np.sin(np.arange(int(2.0 * pipeline.RATE)) / 20)).astype(np.float32)
            yield "voiced"

        with patch.object(pipeline, "_synthesize", side_effect=synthesize):
            result = list(pipeline.dub(pipeline.DubRequest(session.id, timing=pipeline.TIMING_MATCH),
                                       runtime=self.runtime, free_gpu=lambda: None))[-1]
        self.assertEqual(voiced, [0])
        self.assertIn("Voiced 1 edited line(s)", result.message)
        self.assertEqual(result.player.target_language, "English · edited")
        # Word alignment repair reads the dub's language, not its label.
        import json
        folder = self.runtime.outputs_dir / "dubbing" / pipeline.PLAYER_DIR / result.audio.stem
        self.assertEqual(json.loads((folder / "alignment.json").read_text(encoding="utf-8"))["language"], "English")


if __name__ == "__main__":
    unittest.main()
