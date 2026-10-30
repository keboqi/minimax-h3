"""Writer-script repairs and the source coverage check."""

from __future__ import annotations

import json
import unittest

from h3_app.prompt_writers import drama3 as drama3_writer
from h3_app.drama3.lint import (
    coverage_note,
    direction_share,
    lint_drama3_script,
    merge_direction,
    missing_source_phrases,
    under_directed,
)


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
        self.assertEqual(len(notes), 3)
        self.assertIn("only 0 of 3 lines carry delivery direction", notes)

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

    def test_one_corrected_slip_per_phrase_counts_as_spoken(self):
        self.assertEqual(missing_source_phrases(
            "We walked to teh river at dawn, and I said see you their.",
            "<|speaker:0|> We walked to the river [softly] at dawn, and I said see you there.",
            source_mode="preserve"), [])
        self.assertEqual(missing_source_phrases(
            "陈念以经走了，她说明天在见。",
            "<|speaker:0|> 陈念已经走了，[softly] 她说明天再见。",
            source_mode="preserve"), [])
        self.assertEqual(missing_source_phrases(  # a removed accidental repeat
            "She opened the the door slowly.",
            "<|speaker:0|> She opened the door slowly.",
            source_mode="preserve"), [])

    def test_a_phrase_with_two_changed_words_is_reported(self):
        self.assertEqual(missing_source_phrases(
            "She opened the heavy door slowly",
            "<|speaker:0|> She closed the wooden door slowly.",
            source_mode="preserve"), ["She opened the heavy door slowly"])

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
        script, status, _cast = drama3_writer.apply_drama3_reply(
            reply, "班主任站在门口，冲她招招手：“你出来一下！”教室里顿时鸦雀无声。", (), "Auto")
        self.assertEqual(script.split("\n")[1], "<|speaker:1|> 你出来一下！")
        self.assertIn("Writer check:", status)
        self.assertIn("Removed 1 cue(s) at line ends", status)
        self.assertIn("教室里顿时鸦雀无声", status)


class PromptExampleTests(unittest.TestCase):
    def test_advanced_example_is_a_clean_parseable_script(self):
        from pathlib import Path

        from h3_app.drama3.serving.tree import iter_sound_nodes, iter_speech_nodes, parse_drama3_tree

        prompt = (Path(__file__).resolve().parents[1] / "prompts" / "prompt_drama3.txt").read_text(encoding="utf-8")
        example = prompt.split("Advanced script-field example", 1)[1]
        script = example.split("Script:\n", 1)[1].split("\nWhy:", 1)[0].strip()
        self.assertEqual(lint_drama3_script(script), (script, []))
        tree = parse_drama3_tree(script)
        turns = [turn for node in iter_speech_nodes(tree) for turn in node.turns]
        self.assertTrue(all(not turn.speaker_effects for turn in turns if turn.speaker == 0))  # narrator stays dry
        self.assertIn(("fairy_tiny",), {turn.speaker_effects for turn in turns})
        # Only the echoing temple needs a room; the hiding place keeps its stage without one.
        scenes = [node for node in tree.children if getattr(node, "room", "")]
        self.assertEqual([scene.room for scene in scenes], ["large_hall"])
        staged = [node for node in tree.children if getattr(node, "stage", ())]
        self.assertEqual(staged[0].room, "")
        self.assertEqual(staged[0].stage, (("守护精灵", 0.4), ("林夕", -0.3)))
        self.assertFalse(any("reverb" in effect for turn in turns for effect in turn.speaker_effects))
        sounds = iter_sound_nodes(tree)
        self.assertTrue(all(sound.duration for sound in sounds))
        self.assertEqual({sound.name for sound in sounds if sound.loop},
                         {"temple_rumble", "tense_strings", "mystery_pad"})


class RepairTests(unittest.TestCase):
    def test_opening_cues_move_after_the_opening_words(self):
        script, notes = lint_drama3_script(
            "<|speaker:0|> [softly] The house was quiet, too quiet.\n"
            "<|speaker:1|> <phone_call> [urgent] Leave now!\n"
            "<|speaker:2|> [whispering] 不知道你在说什么。\n"
            "<|speaker:3|> [laughs] Yes.")
        self.assertEqual(script.split("\n"), [
            "<|speaker:0|> The house was quiet, [softly] too quiet.",
            "<|speaker:1|> <phone_call> Leave [urgent] now!",
            "<|speaker:2|> 不知 [whispering] 道你在说什么。",
            "<|speaker:3|> Yes.",  # no words left after a one-word line's opening
        ])
        self.assertIn("moved 3 line-opening cue(s) after the opening words", notes)
        self.assertIn("removed 1 line-opening cue(s) with no room after the opening words", notes)

    def test_ellipses_pauses_and_extra_cues_are_removed(self):
        script, notes = lint_drama3_script(
            "<|speaker:0|> Well... I [pause] never said that…\n"
            "<|speaker:1|> 我不知道……你在说什么。Hello…?\n"
            "<|speaker:0|> She waited [nervous] and waited [tense] and then [scared] she ran. [laughs]")
        self.assertEqual(script.split("\n"), [
            "<|speaker:0|> Well, I never said that.",
            "<|speaker:1|> 我不知道，你在说什么。Hello?",
            "<|speaker:0|> She waited [nervous] and waited and then she ran. [laughs]",
        ])
        self.assertIn("replaced 4 ellipsis(es) with commas or periods", notes)
        self.assertIn("removed 1 pause or silence cue(s); punctuation paces the line", notes)
        self.assertIn("removed 2 cue(s) beyond one per sentence and 2 per line", notes)

    def test_direction_share_and_repeated_cues_are_reported(self):
        flat = "\n".join(f"<|speaker:0|> Line number {n} goes here." for n in range(5))
        self.assertTrue(under_directed(flat))
        self.assertEqual(direction_share(flat), (0, 5))
        self.assertIn("only 0 of 5 lines carry delivery direction", lint_drama3_script(flat)[1])
        directed = ("<|speaker:0|> One, [sad] two.\n<|speaker:1|> Three, [sad] four.\n"
                    "<|speaker:0|> Five <emphasis>six</emphasis>.")
        self.assertFalse(under_directed(directed))
        self.assertEqual(lint_drama3_script(directed)[1], ["1 line(s) open with the same cue as the line before"])
        self.assertFalse(under_directed("<|speaker:0|> Hi.\n<|speaker:1|> Yo."))  # too short to judge

    def test_redundant_room_tags_and_narrator_effect_tags_are_stripped(self):
        raw = (
            '<parallel room="small_room">\n'
            '  narrator: <far> The room [hushed] was cold.\n'
            '  Mia: <reverb_small_room> Who [nervous] is there?\n'
            '  Leo: <phone_call> Stay [urgent] where you are.\n'
            '</parallel>'
        )
        cleaned, notes = lint_drama3_script(raw)
        self.assertEqual(cleaned.split("\n"), [
            '<parallel room="small_room">',
            '  narrator: The room [hushed] was cold.',
            '  Mia: Who [nervous] is there?',
            '  Leo: <phone_call> Stay [urgent] where you are.',
            '</parallel>',
        ])
        self.assertIn("removed 1 redundant room effect tag(s) already set by the scene", notes)
        self.assertIn("removed 1 vocal effect or distance tag(s) from narration", notes)



class DirectionMergeTests(unittest.TestCase):
    SCRIPT = ("<|speaker:0|> The door creaked open. Nobody was there.\n"
              "<parallel>\n"
              '  <sound name="wind" duration="3">Wind gust</sound>\n'
              "  <|speaker:1|> <phone_call> Hello? Is someone there?\n"
              "</parallel>\n"
              "<|speaker:2|> It's just the wind.")

    def test_inserted_direction_is_merged_and_changed_words_are_refused(self):
        directed = ("<|speaker:0|> The door creaked open. [hushed] Nobody was <slow>there</slow>.\n"
                    "<parallel>\n"
                    '<sound name="wind" duration="3">Wind gust</sound>\n'
                    "<|speaker:1|> <phone_call> Hello? [nervous] Is someone here?\n"  # a word changed
                    "</parallel>\n"
                    "<|speaker:2|> It's just [gently teasing] the wind.")
        merged, taken, refused = merge_direction(self.SCRIPT, directed)
        self.assertEqual((taken, refused), (2, 1))
        self.assertEqual(merged.split("\n"), [
            "<|speaker:0|> The door creaked open. [hushed] Nobody was <slow>there</slow>.",
            "<parallel>",
            '  <sound name="wind" duration="3">Wind gust</sound>',
            "  <|speaker:1|> <phone_call> Hello? Is someone there?",
            "</parallel>",
            "<|speaker:2|> It's just [gently teasing] the wind.",
        ])

    def test_word_boundaries_count_but_cue_spacing_does_not(self):
        for original, directed, taken in (
            ("We are now here.", "We are [sad] nowhere.", 0),  # two words merged into one
            ("We are nowhere.", "We are now [sad] here.", 0),
            ("We are now here.", "We are [sad] now <emphasis>here</emphasis>.", 1),
            ("Hello? Is someone there?", "Hello? [nervous] Is someone there [scared]?", 1),
            ("我不知道你在说什么。", "我不知 [whispering] 道你在<emphasis>说什么</emphasis>。", 1),
            ("“你好，”她说。", "“你好，” [warm] 她说。", 1),
        ):
            with self.subTest(directed=directed):
                _merged, merged_lines, _refused = merge_direction(
                    f"<|speaker:0|> {original}", f"<|speaker:0|> {directed}")
                self.assertEqual(merged_lines, taken)

    def test_changed_structure_merges_nothing(self):
        for directed in ("<|speaker:0|> The door creaked open. [hushed] Nobody was there.",  # lines dropped
                         self.SCRIPT.replace("Wind gust", "Strong wind"),  # a sound changed
                         self.SCRIPT.replace("<|speaker:2|>", "<|speaker:1|>"),  # a speaker changed
                         self.SCRIPT.replace("<phone_call> ", "")):  # an effect removed
            with self.subTest(directed=directed[-40:]):
                merged, taken, _refused = merge_direction(self.SCRIPT, directed)
                self.assertEqual(taken, 0)
                self.assertEqual(merged.split("\n")[3], "  <|speaker:1|> <phone_call> Hello? Is someone there?")

    def test_shallow_direction_also_gets_a_direction_pass(self):
        from h3_app.drama3.lint import needs_direction, shallow_direction

        bare = "\n".join(f"<|speaker:{n % 2}|> Line {n}, [{cue}] spoken." for n, cue in
                         enumerate(["serious", "calm", "sad", "serious", "angry"]))
        self.assertFalse(under_directed(bare))  # every line carries a cue ...
        self.assertIn("single bare word", shallow_direction(bare))  # ... but a weak one
        self.assertTrue(needs_direction(bare))
        self.assertIn("direction is shallow", " ".join(lint_drama3_script(bare)[1]))
        repeated = "\n".join(f"<|speaker:1|> Line {n}, [{cue}] spoken." for n, cue in
                             enumerate(["tense, low", "warm and kind", "tense, low", "quietly hurt", "tense, low"]))
        self.assertIn("repeats [tense, low] 3 times", shallow_direction(repeated))
        directed = ["warm and kind", "quietly hurt", "to him, pleading", "trying to sound casual", "close to tears",
                    "turning on him", "with heavy sadness", "bright and eager"]
        flat_run = "\n".join([f"<|speaker:1|> Hi, [{cue}] there." for cue in directed[:5]]
                             + [f"<|speaker:0|> Plain line {n}." for n in range(5)]
                             + [f"<|speaker:1|> Bye, [{cue}] now." for cue in directed[5:]])
        self.assertFalse(under_directed(flat_run))
        self.assertIn("5 lines in a row", shallow_direction(flat_run))
        rich = ("<|speaker:0|> The door [hushed] creaked open.\n<|speaker:1|> Hello [nervous, speaking quickly] there?\n"
                "<|speaker:0|> Nobody [tense, low] answered.\n<|speaker:1|> I [to herself, bitter] knew it.")
        self.assertEqual(shallow_direction(rich), "")

    def test_direction_pass_runs_only_for_under_directed_scripts(self):
        cast = [{"speaker_id": 0, "name": "narrator", "description": "Calm."},
                {"speaker_id": 1, "name": "Mia", "description": "Bright."}]
        flat = ("<|speaker:0|> The door creaked open.\n<|speaker:1|> Hello there?\n"
                "<|speaker:0|> Nobody answered her.")
        requests = []

        def ask(request):
            requests.append(request)
            return ("narrator: The door [hushed] creaked open.\nMia: Hello [nervous] there?\n"
                    "narrator: Nobody [tense, low] answered her.", "Enhanced Drama 3 direction prompt.")

        script, note = drama3_writer.direct_drama3_script(flat, cast, "Basic", ask)
        self.assertEqual(script, "<|speaker:0|> The door [hushed] creaked open.\n<|speaker:1|> Hello [nervous] there?\n"
                                 "<|speaker:0|> Nobody [tense, low] answered her.")
        self.assertEqual(note, "A direction pass added cues to 3 line(s), 3 of 3 now directed.")
        self.assertIn("Mia: Hello there?", requests[0])  # the director reads names, not tags
        self.assertIn("- Mia: Bright.", requests[0])
        self.assertEqual(drama3_writer.direct_drama3_script(script, cast, "Basic", ask), (script, ""))
        self.assertEqual(len(requests), 1)  # an already directed script needs no second call

        def failing(_request):
            return "", "Prompt enhancement failed: quota exceeded."

        script, note = drama3_writer.direct_drama3_script(flat, cast, "Basic", failing)
        self.assertEqual(script, flat)
        self.assertIn("quota exceeded", note)


class SourceModeTests(unittest.TestCase):
    SOURCE = "第一句话说得很长很长。第二句话也说得很长。第三句话同样很长很长。"

    def test_preserve_mode_always_reports_missing_source(self):
        shortened = "<|speaker:0|> 第一句话说得很长很长。"
        self.assertIsNone(missing_source_phrases(self.SOURCE, shortened))  # mostly gone: looks like a brief
        self.assertEqual(len(missing_source_phrases(self.SOURCE, shortened, source_mode="preserve")), 2)
        self.assertIn("2 source phrase(s)", coverage_note(self.SOURCE, shortened, source_mode="preserve"))
        self.assertIsNone(missing_source_phrases(self.SOURCE, "<|speaker:0|> 第一句话说得很长很长。第二句话也说得很长。",
                                                 source_mode="brief"))

    def test_the_writer_reply_carries_its_source_mode(self):
        reply = json.dumps({
            "source_mode": "preserve",
            "script": "narrator: 第一句话说得很长很长。",
            "speakers": [{"speaker_id": 0, "name": "narrator", "description": "Calm.", "preview_text": "你好。"}],
        }, ensure_ascii=False)
        _script, status, _cast = drama3_writer.apply_drama3_reply(reply, self.SOURCE, (), "Auto")
        self.assertIn("2 source phrase(s)", status)


class VoiceSexTests(unittest.TestCase):
    def test_descriptions_that_never_say_male_or_female_are_reported(self):
        from h3_app.drama3.cast import voices_without_sex

        cast = [{"name": "胡母", "description": "A weary, hardworking middle-aged mother, warm, rapid-fire."},
                {"name": "Mia", "description": "A female voice: bright, early twenties."},
                {"name": "Leo", "description": "Young man, dry humour."},
                {"name": "旁白", "description": "沉稳的中年男声。"},
                {"name": "Kid", "description": "A boy's voice, about ten."},
                {"name": "Spirit", "description": "Human-like, manner calm, mysterious."}]
        self.assertEqual(voices_without_sex(cast), ["胡母", "Spirit"])
        reply = json.dumps({"script": "胡母: 杀人犯！", "speakers": [
            {"speaker_id": 0, "name": "胡母", "description": cast[0]["description"], "preview_text": "你好。"}]},
            ensure_ascii=False)
        _script, status, _cast = drama3_writer.apply_drama3_reply(reply, "胡母: 杀人犯！", (), "Auto")
        self.assertIn("The voice description for 胡母 does not say male or female", status)


class PromptBasicExampleTests(unittest.TestCase):
    def test_basic_examples_are_clean_directed_and_complete(self):
        from pathlib import Path

        prompt = (Path(__file__).resolve().parents[1] / "prompts" / "prompt_drama3.txt").read_text(encoding="utf-8")
        for heading in ("Basic script-field example", "Basic prose example"):
            with self.subTest(heading=heading):
                example = prompt.split(heading, 1)[1]
                source = example.split("User text:\n", 1)[1].split("\nScript:\n", 1)[0]
                script = example.split("\nScript:\n", 1)[1].split("\n\n", 1)[0].strip()
                names = {}
                for line in script.split("\n"):
                    names.setdefault(line.split(":", 1)[0], len(names))
                tagged = "\n".join(f"<|speaker:{names[line.split(':', 1)[0]]}|> {line.split(': ', 1)[1]}"
                                   for line in script.split("\n"))
                self.assertEqual(lint_drama3_script(tagged), (tagged, []))
                self.assertEqual(missing_source_phrases(source, tagged, source_mode="preserve"), [])
                directed, total = direction_share(tagged)
                self.assertGreaterEqual(directed, total - 1)


if __name__ == "__main__":
    unittest.main()
