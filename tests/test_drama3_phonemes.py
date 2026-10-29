"""Pronunciation notes (pinyin, and Arpabet for English): written after the words in scripts, sent to Drama 3 as phoneme tags."""

import json
import unittest

from h3_app.drama3.phonemes import note_problems, strip_notes, to_phoneme_tags
from h3_app.errors import H3Error

TAG = "<|phoneme_start|>{}<|phoneme_end|>"


class PhonemeNoteTests(unittest.TestCase):
    def test_notes_become_one_phoneme_tag_per_character(self):
        self.assertEqual(to_phoneme_tags("把书还{huan2}给他。"), f"把书{TAG.format('huan2')}给他。")
        self.assertEqual(to_phoneme_tags("尉迟{yu4 chi2}恭"), f"{TAG.format('yu4')}{TAG.format('chi2')}恭")
        # ü is written v, as pypinyin's tone3 style does.
        self.assertEqual(to_phoneme_tags("绿{lü4}"), TAG.format("lv4"))
        self.assertEqual(to_phoneme_tags("No notes here."), "No notes here.")
        self.assertEqual(strip_notes("他又长{zhang3}高了，尉迟{yu4 chi2}恭"), "他又长高了，尉迟恭")

    def test_malformed_notes_are_explained(self):
        cases = {
            "还{Huan2}": "lowercase pinyin",
            "还{huan}": "tone number",
            "还 {huan2}": "exactly 1 Chinese character",
            "行{yin2 hang2}": "exactly 2 Chinese character",
            "OK{ou1}": "exactly 1 Chinese character",
            "还{}": "1-8 pinyin syllables",
            "还{huan2": "brace without its pair",
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                problems = note_problems(text)
                self.assertTrue(problems and reason in problems[0], problems)
                with self.assertRaises(H3Error):
                    to_phoneme_tags(text)

    def test_script_validation_accepts_notes_and_points_raw_tags_to_them(self):
        from h3_app.workflows.drama3 import validate_drama3_text

        self.assertEqual(validate_drama3_text("<|speaker:0|> 把书还{huan2}给他。"), "<|speaker:0|> 把书还{huan2}给他。")
        with self.assertRaisesRegex(H3Error, "tone number"):
            validate_drama3_text("<|speaker:0|> 把书还{huan}给他。")
        with self.assertRaisesRegex(H3Error, r"银行\{yin2 hang2\}"):
            validate_drama3_text("<|speaker:0|> 把书<|phoneme_start|>huan2<|phoneme_end|>给他。")

    def test_only_the_drama3_request_carries_phoneme_tags(self):
        from h3_app.drama3.serving.server import SglangOmniServer

        payload = SglangOmniServer.payload(None, text="他又长{zhang3}高了。", reference=None, transcript=None,
                                           max_new_tokens=100, sampling={"top_k": 30})
        self.assertEqual(payload["input"], f"他又{TAG.format('zhang3')}高了。")

    def test_planning_and_cues_treat_notes_as_markup(self):
        from h3_app.drama3.serving.planner import expected_seconds, insert_cue, spoken_text

        self.assertEqual(spoken_text("尉迟{yu4 chi2}恭来了"), "尉迟 恭来了")
        self.assertEqual(expected_seconds("尉迟{yu4 chi2}恭来了"), expected_seconds("尉迟恭来了"))
        # A cue after the opening words never separates a word from its note.
        self.assertEqual(insert_cue("银行{yin2 hang2}关门了", "[sad]"), "银行{yin2 hang2} [sad] 关门了")
        # The opening two characters may end inside a longer noted word, or start one.
        self.assertEqual(insert_cue("尉迟恭{yu4 chi2 gong1}来了", "[sad]"), "尉迟恭{yu4 chi2 gong1} [sad] 来了")
        self.assertEqual(insert_cue("我银行{yin2 hang2}关门了", "[sad]"), "我银行{yin2 hang2} [sad] 关门了")
        self.assertEqual(insert_cue("我去银行{yin2 hang2}了", "[sad]"), "我去 [sad] 银行{yin2 hang2}了")
        # Nothing spoken after the noted word leaves no place for the cue.
        self.assertIsNone(insert_cue("尉迟恭{yu4 chi2 gong1}", "[sad]"))

    def test_coverage_subtitles_and_direction_keep_the_characters(self):
        from h3_app.drama3.lint import merge_direction, missing_source_phrases
        from h3_ui.drama3.generation import _CUE

        self.assertEqual(missing_source_phrases("他把书还给了老师。", "<|speaker:0|> 他把书还{huan2}给了老师。",
                                                source_mode="preserve"), [])
        self.assertEqual(" ".join(_CUE.sub(" ", "把书还{huan2}[softly] 给他").split()), "把书还 给他")
        script = "<|speaker:0|> 他把书还{huan2}给了老师。"
        kept, taken, _refused = merge_direction(script, "<|speaker:0|> 他把书还{huan2}给了 [warmly] 老师。")
        self.assertEqual((kept, taken), ("<|speaker:0|> 他把书还{huan2}给了 [warmly] 老师。", 1))
        # The director may not drop a note: the line keeps the writer's version.
        kept, taken, refused = merge_direction(script, "<|speaker:0|> 他把书还给了 [warmly] 老师。")
        self.assertEqual((kept, taken, refused), (script, 0, 1))
        # Nor detach one: a cue or space between the word and its note would break the request.
        for detached in ("<|speaker:0|> 他把书还 [gently] {huan2}给了老师。", "<|speaker:0|> 他把书还 {huan2}给了老师。"):
            kept, taken, refused = merge_direction(script, detached)
            self.assertEqual((kept, taken, refused), (script, 0, 1), detached)

    def test_previews_and_writer_replies(self):
        from h3_app.drama3.cast import parse_script_cast_and_sounds

        speaker = {"speaker_id": 0, "name": "narrator", "description": "A female voice: warm.",
                   "preview_text": "雨下了一整夜。"}
        script, _cast, _sounds = parse_script_cast_and_sounds(json.dumps(
            {"script": "narrator: 他又长{zhang3}高了。", "speakers": [speaker]}, ensure_ascii=False))
        self.assertIn("长{zhang3}", script)
        with self.assertRaisesRegex(H3Error, "only spoken words"):
            parse_script_cast_and_sounds(json.dumps(
                {"script": "narrator: 他又长高了。", "speakers": [{**speaker, "preview_text": "长{zhang3}大"}]},
                ensure_ascii=False))
        with self.assertRaisesRegex(H3Error, "tone number"):
            parse_script_cast_and_sounds(json.dumps(
                {"script": "narrator: 他又长{zhang}高了。", "speakers": [speaker]}, ensure_ascii=False))

    def test_english_notes_become_one_phoneme_tag_for_the_word(self):
        self.assertEqual(to_phoneme_tags("Ask Siobhan{SH IH0 V AO1 N} first."),
                         f"Ask {TAG.format('SH IH0 V AO1 N')} first.")
        self.assertEqual(to_phoneme_tags("O'Neill{OW0 N IY1 L} and 银行{yin2 hang2}"),
                         f"{TAG.format('OW0 N IY1 L')} and {TAG.format('yin2')}{TAG.format('hang2')}")
        self.assertEqual(strip_notes("Leicester{L EH1 S T ER0} Square"), "Leicester Square")
        self.assertEqual(note_problems("Leicester{L EH1 S T ER0}"), [])
        self.assertIn("stress digit", note_problems("Leicester{L EH S T ER0}")[0])  # a vowel without stress
        self.assertIn("follow the English word", note_problems("Hello {HH AH0 L OW1}")[0])
        self.assertIn("no vowel", note_problems("Psst{P S T}")[0])
        with self.assertRaises(H3Error):
            to_phoneme_tags("Hello {HH AH0 L OW1}")

    def test_a_cue_never_splits_an_english_word_from_its_note(self):
        from h3_app.drama3.serving.planner import insert_cue

        self.assertEqual(insert_cue("Siobhan{SH IH0 V AO1 N} waited.", "[softly]"),
                         "Siobhan{SH IH0 V AO1 N} [softly] waited.")

    def test_writer_prompt_asks_for_risky_readings_only(self):
        from pathlib import Path

        prompt = (Path(__file__).resolve().parents[1] / "prompts" / "prompt_drama3.txt").read_text(encoding="utf-8")
        self.assertIn("mark a reading only where it is easy to get wrong", prompt)
        self.assertIn("Do not mark ordinary words", prompt)
        director = (Path(__file__).resolve().parents[1] / "prompts" / "prompt_drama3_direction.txt").read_text(
            encoding="utf-8")
        self.assertIn("pinyin pronunciation note", director)


if __name__ == "__main__":
    unittest.main()
