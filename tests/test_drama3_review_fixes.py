"""Regression cases from the Sound Stage subsystem review."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from h3_app.drama3 import voice_looks
from h3_app.drama3.gemma_checks import fix_readings
from h3_app.drama3.guided.enhance import add_pronunciations, apply_content_annotations
from h3_app.drama3.guided.model import Utterance
from h3_app.drama3.results import script_fingerprint
from h3_app.drama3.serving import audio
from h3_app.drama3.serving.cue_timing import resolve_anchor
from h3_app.drama3.serving.planner import plan_turns_chunks
from h3_app.drama3.serving.tree import iter_speech_nodes, parse_drama3_tree, strip_drama3_sound_tags
from h3_app.drama3.visual_plan import expand_prompt
from h3_app.jobs import JobCancelled
from h3_app.workflows.drama3 import request_cast
from h3_ui.drama3.page_common import timeline_payload
from h3_ui.drama3.page_script import _source_line_numbers, preview_payload, spoken_payload
from h3_ui.drama3.project_visuals import project_cast, project_cues
from h3_ui.drama3.result_view import script_preview_settings


class ReviewFixTests(unittest.TestCase):
    def test_source_line_matching_handles_interleaved_speakers_and_requires_complete_text(self):
        estimate = [dict(speaker="A", text="Hello friend", line=1), dict(speaker="B", text="Reply", line=2)]
        actual = [dict(speaker="B", text="Reply"), dict(speaker="A", text="Hello"), dict(speaker="A", text="friend")]
        self.assertEqual(_source_line_numbers(estimate, actual), [2, 1, 1])
        self.assertIsNone(_source_line_numbers(estimate, actual[:2]))
        self.assertIsNone(_source_line_numbers(estimate, [dict(speaker="A", text="Different")]))

    def test_project_cues_follow_the_real_cast_and_basic_parser(self):
        text = 'Mia: Read the sign.\nWarning: High voltage.\n<sound>Rain</sound>\n<#1#>\n<|speaker:0|> Stay back.'
        data = dict(text=text, shown_slots=1, voice_0_save_name="Mia", script_mode="Basic")
        _, default, resolve = request_cast(SimpleNamespace(voice_names=((0, "Mia"),), voices=((0, "clip", "ref"),)))
        tree = parse_drama3_tree(strip_drama3_sound_tags(text), default_speaker=default, resolve=resolve)
        actual = [dict(speaker=t.speaker, text=t.text) for n in iter_speech_nodes(tree) for t in n.turns]
        self.assertEqual(project_cues(data), actual)
        self.assertIn("Warning: High voltage.", [cue["text"] for cue in actual])
        preview = preview_payload(text, "Basic", **project_cast(data))
        self.assertTrue(all(line["speaker"] == "Mia" for line in preview["lines"]))
        self.assertIn("Warning: High voltage.", [line["text"] for line in preview["lines"]])

    def test_phrase_fallback_ignores_delivery_tags_and_reading_notes(self):
        for text, phrase in (("Hello [warmly] there.", "Hello there"),
                             ("Take <emphasis>the</emphasis> key.", "Take the key"),
                             ("去银行{yin2 hang2}取钱。", "去银行取钱")):
            with self.subTest(text=text):
                self.assertEqual(resolve_anchor(phrase, [], [(text, 10, 20)]), 20)
                self.assertEqual(resolve_anchor(phrase, [], [(text, 10, 20)], edge="start"), 10)
        self.assertIsNone(resolve_anchor("he", [], [("the key", 0, 10)]))

    def test_existing_multi_character_notes_remain_authoritative(self):
        self.assertEqual(add_pronunciations("银行{yin2 hang2}在银行附近", {"银": "yin2", "银行": "yin2 hang2"}),
                         "银行{yin2 hang2}在银行{yin2 hang2}附近")

    def test_duplicate_content_fixes_are_idempotent(self):
        utterances = [Utterance("u1", "Mia", "teh door", sources=("s1",))]
        answer = {"fixes": [("s1", "teh", "the"), ("s1", "teh", "the")]}
        self.assertEqual(apply_content_annotations(utterances, answer)[0].text, "the door")

    def test_age_requires_a_person_age_rather_than_clothing_or_history(self):
        self.assertIsNone(voice_looks.age("A man holding an old lantern"))
        self.assertEqual(voice_looks.age("A woman in her twenties wearing an 80s jacket"), (20, 29))
        self.assertEqual(voice_looks.age("An elderly man who spent twenty years at sea"), (60, 100))
        self.assertEqual(voice_looks.age("A man in his 80s"), (80, 89))

    def test_subject_key_does_not_match_a_longer_undefined_name(self):
        text, mentioned = expand_prompt("An empty @alley beside @al.", {"al": ("characters", "Al in a hat")})
        self.assertIn("empty alley", text)
        self.assertEqual(mentioned, ("al",))
        self.assertEqual(text.count("Al in a hat"), 1)

    def test_reading_repair_can_insert_missing_pinyin_spaces(self):
        with patch("h3_app.drama3.gemma_checks.gemma.ask_json", return_value={"readings": ["yin2 hang2"]}):
            for reading in ("yin2hang2", "yinhang2"):
                self.assertEqual(fix_readings([("银行", reading)]), ["yin2 hang2"])
            self.assertEqual(fix_readings([("银行", "yin2")]), [None])

    def test_explicit_lead_in_applies_to_a_resumed_bed(self):
        voice = np.full(100, .03, dtype=np.float32)
        tracks = [dict(audio=voice), dict(audio=voice, loop=True, is_sound=True, cursor=50)]
        self.assertEqual(audio.mix_parallel(tracks, 100).speech_offset, 0)
        self.assertEqual(audio.mix_parallel(tracks, 100, lead_in=2.5).speech_offset, 250)

    def test_long_script_turn_uses_actual_timing_after_chunk_splitting(self):
        text = "Mia: " + "A long sentence with several words. " * 35
        estimate = timeline_payload(script_preview_settings(text, "Basic"))
        estimate["lines"][0]["line"] = 7
        chunks = [chunk for node in iter_speech_nodes(parse_drama3_tree(text))
                  for chunk in plan_turns_chunks(node.turns, preserve_lines=True)]
        self.assertGreater(len(chunks), 1)
        settings = dict(duration=100, script_mode="Basic", script_fingerprint=script_fingerprint(text), chunks=[
            dict(name=c.speaker, text=c.text, start=i * 10, end=(i + 1) * 10,
                 lines=[dict(text=t, start=i * 10, end=(i + 1) * 10) for t in c.lines]) for i, c in enumerate(chunks)])
        result = spoken_payload(estimate, settings, "Basic", text)
        self.assertTrue(result["actual"])
        self.assertEqual([line["line"] for line in result["lines"]], [7] * len(chunks))

    def test_speech_cancellation_is_not_reported_as_a_generation_failure(self):
        from h3_app.generation.drama3 import generate_drama3
        with patch("h3_app.generation.drama3.validate_drama3"), \
                patch("h3_app.generation.drama3.record_failure") as failed, \
                patch("h3_models.resolve_hf_token", return_value=None), \
                patch("h3_app.drama3.serving.generate.generate_drama3_served", side_effect=JobCancelled("stopped")), \
                self.assertRaises(JobCancelled):
            list(generate_drama3(None, SimpleNamespace(models=Mock()), None))
        failed.assert_not_called()

    def test_voice_transcription_and_sound_handlers_preserve_cancellation(self):
        from h3_ui.drama3.voices import VoiceHandlers
        from h3_ui.drama3.sounds import SoundHandlers
        handler = SimpleNamespace(stage_file=lambda *_: "clip", _generation_services=lambda: None,
                                  _runtime_config=lambda: None, free_comfy_vram=lambda: None)
        with patch("h3_app.generation.drama3.transcribe_drama3_audio", side_effect=JobCancelled("stop")), \
                self.assertRaises(JobCancelled):
            VoiceHandlers.transcribe_drama3_voice(handler, "clip")
        with patch("h3_app.voice_design.generate_reference", side_effect=JobCancelled("stop")), \
                self.assertRaises(JobCancelled):
            list(VoiceHandlers.generate_drama3_voice(handler, "warm", "Hello", "Auto"))
        with patch("h3_models.resolve_hf_token", return_value=None), \
                patch("h3_app.stable_audio.generate_sound_reference", side_effect=JobCancelled("stop")), \
                self.assertRaises(JobCancelled):
            list(SoundHandlers.generate_drama3_sound(handler, "rain"))
