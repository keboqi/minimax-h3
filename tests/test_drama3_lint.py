"""Writer-script repairs and the source coverage check."""

from __future__ import annotations

import json
import unittest

from h3_app import prompt_service
from h3_app.drama3_lint import coverage_note, lint_drama3_script, missing_source_phrases


class LintTests(unittest.TestCase):
    def test_risky_cue_placements_are_repaired(self):
        script, notes = lint_drama3_script(
            "<|speaker:3|> <whisper>不知道。</whisper>\n"
            "<|speaker:2|> 好好上课！[sharp authoritative scold, sudden burst]\n"
            "<|speaker:1|> <phone_call> <soft>Hello</soft> there.\n"
            "<sound name=\"rain\">Rain</sound>"
        )
        self.assertEqual(script.split("\n"), [
            "<|speaker:3|> 不知道。",
            "<|speaker:2|> 好好上课！",
            "<|speaker:1|> <phone_call> Hello there.",
            "<sound name=\"rain\">Rain</sound>",
        ])
        self.assertEqual(len(notes), 2)

    def test_valid_placements_are_kept_and_long_cues_reported(self):
        original = ("<|speaker:0|> I did it! [laughs]\n"
                    "<|speaker:1|> 共生关系，[hoarse yet resonant teacher voice, methodical cadence] 指两种生物。\n"
                    "<|speaker:2|> [angry]")
        script, notes = lint_drama3_script(original)
        self.assertEqual(script, original)  # a trailing event stays; a cue-only line is not emptied
        self.assertEqual(notes, ["1 cue(s) run over 5 words; shorter cues are more reliable"])


class CoverageTests(unittest.TestCase):
    SOURCE = ("“陈念，”班主任站在门口，一贯威严的人难得和颜悦色，冲她招招手，“你出来一下。”\n"
              "一道阴影笼罩过来。是班主任，身后跟着两个jing察。\n"
              "陈念咬紧牙，不属于她的羞耻在她脸上炸开。")

    def test_reordered_and_normalized_lines_count_as_spoken(self):
        script = ("<|speaker:2|> 陈念，[stern] 你出来一下。\n"
                  "<|speaker:0|> 班主任站在门口，一贯威严的人难得和颜悦色，冲她招招手。\n"
                  "<|speaker:0|> 一道阴影笼罩过来。是班主任，身后跟着两个警察。\n"
                  "<|speaker:0|> 陈念咬紧牙，不属于她的羞耻在她脸上炸开。")
        self.assertEqual(missing_source_phrases(self.SOURCE, script), [])
        self.assertEqual(coverage_note(self.SOURCE, script), "")

    def test_dropped_narration_is_reported(self):
        script = ("<|speaker:2|> 陈念，你出来一下。\n"
                  "<|speaker:0|> 班主任站在门口，一贯威严的人难得和颜悦色，冲她招招手。\n"
                  "<|speaker:0|> 一道阴影笼罩过来。是班主任，身后跟着两个警察。")
        self.assertEqual(missing_source_phrases(self.SOURCE, script), ["陈念咬紧牙", "不属于她的羞耻在她脸上炸开"])
        self.assertIn("2 source phrase(s)", coverage_note(self.SOURCE, script))

    def test_a_brief_is_not_checked(self):
        self.assertIsNone(missing_source_phrases(
            "two friends arguing about pizza toppings for thirty seconds",
            "<|speaker:0|> No way, pineapple belongs on pizza.\n<|speaker:1|> Absolutely not."))

    def test_pasted_reply_status_reports_repairs_and_gaps(self):
        reply = json.dumps({
            "script": "<|speaker:0|> 班主任站在门口。\n<|speaker:1|> 你出来一下！[angry and impatient]",
            "speakers": [
                {"speaker_id": 0, "name": "Narrator", "description": "Calm.", "preview_text": "Hello there."},
                {"speaker_id": 1, "name": "Teacher", "description": "Stern.", "preview_text": "Hello there."},
            ],
        })
        script, status, _cast = prompt_service.apply_drama3_reply(
            reply, "班主任站在门口，冲她招招手：“你出来一下！”教室里顿时鸦雀无声。", (), "Auto")
        self.assertEqual(script.split("\n")[1], "<|speaker:1|> 你出来一下！")
        self.assertIn("Writer check:", status)
        self.assertIn("Removed 1 cue(s) at line ends", status)
        self.assertIn("教室里顿时鸦雀无声", status)


class PromptExampleTests(unittest.TestCase):
    def test_advanced_example_is_a_clean_parseable_script(self):
        from pathlib import Path

        from h3_app.drama3_serving.tree import iter_sound_nodes, iter_speech_nodes, parse_drama3_tree

        prompt = (Path(__file__).resolve().parents[1] / "prompt_drama3.txt").read_text(encoding="utf-8")
        example = prompt.split("Advanced script-field example", 1)[1]
        script = example.split("Script:\n", 1)[1].split("\nWhy:", 1)[0].strip()
        self.assertEqual(lint_drama3_script(script), (script, []))
        tree = parse_drama3_tree(script)
        turns = [turn for node in iter_speech_nodes(tree) for turn in node.turns]
        self.assertTrue(all(not turn.speaker_effects for turn in turns if turn.speaker == 0))  # narrator stays dry
        self.assertIn(("fairy_tiny", "reverb_small_room"), {turn.speaker_effects for turn in turns})
        sounds = iter_sound_nodes(tree)
        self.assertTrue(all(sound.duration for sound in sounds))
        self.assertEqual({sound.name for sound in sounds if sound.loop},
                         {"temple_rumble", "tense_strings", "mystery_pad"})


if __name__ == "__main__":
    unittest.main()
