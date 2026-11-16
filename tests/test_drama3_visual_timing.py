import json
from copy import deepcopy
import unittest

from h3_app.drama3.visual_plan import check_plan_reply, plan_request
from h3_app.drama3.visual_timing import bind_plan, preview_settings, script_cues
from h3_app.errors import H3Error


class VisualTimingTests(unittest.TestCase):
    def setUp(self):
        self.settings = preview_settings('Mia: Hello there.\nLeo: Come inside.')
        self.plan = {"shots": [dict(cue=0, offset=0, beat="Hello", prompt="Mia waves.", transition="cut"),
                               dict(cue=1, offset=0, beat="Inside", prompt="An open door.", transition="dissolve")]}

    def test_request_and_reply_need_no_audio(self):
        request = json.loads(plan_request(self.settings, 2))
        self.assertEqual(request["timing"], "untimed")
        self.assertIsNone(request["duration"])
        self.assertEqual(request["cues"][1]["speaker"], "Leo")
        plan, shots = check_plan_reply(json.dumps(self.plan), self.settings, 2)
        self.assertEqual(len(shots), 2)
        self.assertNotIn("start", plan["shots"][1])
        self.assertEqual(plan["script_cues"], self.settings["script_cues"])

    def test_binding_uses_actual_line_times_and_handles_chunk_splitting(self):
        plan, _ = check_plan_reply(json.dumps(self.plan), self.settings, 2)
        settings = dict(duration=15, chunks=[dict(lines=[dict(text="Hello", start=0, end=1),
            dict(text="there.", start=1.5, end=4)]), dict(text="Come inside.", start=9, end=12)])
        bound = bind_plan(plan, settings)
        self.assertEqual(bound["shots"][1]["start"], 9)
        settings["chunks"][1].update(start=11, end=14)
        self.assertEqual(bind_plan(plan, settings)["shots"][1]["start"], 11)
        self.assertNotIn("start", plan["shots"][1])

    def test_changed_words_are_rejected(self):
        plan, _ = check_plan_reply(json.dumps(self.plan), self.settings, 2)
        with self.assertRaisesRegex(H3Error, "differs"):
            bind_plan(plan, dict(duration=8, chunks=[dict(text="A different script.", start=0, end=8)]))

    def test_bad_anchors_are_rejected(self):
        for change in (dict(cue=-1), dict(cue=2), dict(offset=1), dict(offset=float('nan'))):
            with self.subTest(change=change), self.assertRaises(H3Error):
                plan = deepcopy(self.plan)
                plan["shots"][1].update(change)
                check_plan_reply(json.dumps(plan), self.settings, 2)

    def test_fractional_anchor_uses_word_alignment(self):
        plan, _ = check_plan_reply(json.dumps({"shots": [self.plan["shots"][0],
            dict(cue=0, offset=0.5, beat="Pause", prompt="Mia waits.", transition="cut")]}), self.settings, 2)
        settings = dict(duration=15, chunks=[dict(text="Hello there.", start=0, end=8, words=[
            dict(text="Hello", start=0, end=1), dict(text="there", start=6, end=8)]),
            dict(text="Come inside.", start=9, end=12)])
        self.assertEqual(bind_plan(plan, settings)["shots"][1]["start"], 6)

    def test_advanced_script_cues_exclude_sounds_and_staging(self):
        cues = script_cues('<parallel room="small_room">\n<sound>Rain</sound>\nMia: Hi.\n</parallel>\nLeo: Hello.')
        self.assertEqual(cues, [dict(speaker="Mia", text="Hi."), dict(speaker="Leo", text="Hello.")])
