import json
from copy import deepcopy
import unittest

from h3_app.drama3.visual_plan import check_plan_reply, plan_request
from h3_app.drama3.visual_timing import adapt_plan, anchor_plan, bind_plan, preview_settings, script_cues
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

    def test_finished_audio_uses_the_same_cue_plan_as_workshop(self):
        settings = dict(duration=15, chunks=[dict(name="Mia", text="Hello there.", start=0, end=4),
                                             dict(name="Leo", text="Come inside.", start=9, end=12)])
        request = json.loads(plan_request(settings, 2))
        self.assertEqual(request["timing"], "anchored")
        plan, shots = check_plan_reply(json.dumps(self.plan), settings, 2)
        self.assertEqual([row["cue"] for row in plan["shots"]], [0, 1])
        self.assertEqual([shot.first for shot in shots], [0, 216])
        settings["chunks"][1].update(start=6, end=8)
        self.assertEqual(bind_plan(plan, settings)["shots"][1]["start"], 6)

    def test_wording_corrections_preserve_shot_ids_and_follow_unchanged_words(self):
        settings = preview_settings("Mia: Hello there, my friend.\nLeo: Come inside.")
        raw = {"shots": [self.plan["shots"][0],
                        dict(cue=0, offset=10 / 18, beat="Friend", prompt="A friend.", transition="cut"),
                        self.plan["shots"][1]]}
        plan, _ = check_plan_reply(json.dumps(raw), settings, 2)
        corrected = script_cues("Mia: Hello there, my dear friend.\nLeo: Come inside.")
        adapted = adapt_plan(plan, corrected)
        self.assertEqual([row["id"] for row in adapted["shots"]], [row["id"] for row in plan["shots"]])
        self.assertEqual([row["prompt"] for row in adapted["shots"]], [row["prompt"] for row in plan["shots"]])
        # The anchor follows the same text boundary before 'my', not the old fraction.
        self.assertAlmostEqual(adapted["shots"][1]["offset"], 10 / 22)
        self.assertEqual(plan["script_cues"], settings["script_cues"])

    def test_structural_or_unrelated_edits_need_review(self):
        plan, _ = check_plan_reply(json.dumps(self.plan), self.settings, 2)
        for text in ("Mia: Hello there.", "Leo: Hello there.\nMia: Come inside.",
                     "Mia: The spaceship explodes.\nLeo: Come inside."):
            with self.subTest(text=text), self.assertRaisesRegex(H3Error, "wording correction"):
                adapt_plan(plan, script_cues(text))

    def test_legacy_timestamp_plan_migrates_including_a_sound_only_tail(self):
        settings = dict(duration=8, chunks=[dict(name="Mia", text="Hello there.", start=0, end=2)])
        raw = {"shots": [dict(start=start, beat="Room", prompt="An empty room.", transition="cut")
                          for start in (0, 4, 6)]}
        plan = anchor_plan(raw, settings)
        self.assertEqual([row["start"] for row in bind_plan(plan, settings)["shots"]], [0, 4, 6])
        self.assertNotIn("start", plan["shots"][1])
        self.assertEqual(plan["shots"][1]["phase"], "after")

    def test_short_cjk_typo_and_collapsed_word_anchors_keep_the_storyboard(self):
        settings = preview_settings("Mia: 你好。")
        plan, _ = check_plan_reply(json.dumps({"shots": [self.plan["shots"][0]]}), settings, 1)
        self.assertEqual(adapt_plan(plan, script_cues("Mia: 您好。"))["script_cues"][0]["text"], "您好。")
        settings = preview_settings("Mia: Hello dear old friend.")
        plan, _ = check_plan_reply(json.dumps({"shots": [dict(cue=0, offset=offset, beat="Greeting",
            prompt="A friend waves.", transition="cut") for offset in (0, 5 / 18, 7 / 18)]}), settings, 1)
        cues = script_cues("Mia: Hello old friend.")
        adapted = adapt_plan(plan, cues)
        bound = bind_plan(adapted, preview_settings("Mia: Hello old friend."))
        self.assertGreater(bound["shots"][2]["start"], bound["shots"][1]["start"])
