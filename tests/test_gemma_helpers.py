"""JSON repair and the free Gemma checks, which must always fall back to the fixed rules."""

import os
import unittest
from unittest import mock

from h3_app import song_brief
from h3_app.drama3 import gemma_checks
from h3_app.drama3.guided.fidelity import accepted_repairs_note, validate_source_fidelity
from h3_app.drama3.guided.schema import StageError, load_answer
from h3_app.dubbing.segments import parse_translation_reply
from h3_app.errors import H3Error
from h3_app.llm import gemma, repair_json
from h3_app.prompt_writers import song


class Response:
    def __init__(self, status, payload=None):
        self.status_code, self.ok, self._payload = status, 200 <= status < 300, payload or {}

    def json(self):
        return self._payload


def reply(text, tokens=40):
    return Response(200, {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                                  {"text": text}]}}],
                          "usageMetadata": {"totalTokenCount": tokens}})


class JsonRepairTests(unittest.TestCase):
    def test_repairs_malformed_replies_and_rejects_prose(self):
        self.assertEqual(repair_json('{"a": [1, 2,], "b": "x}"}'), {"a": [1, 2], "b": "x}"})
        self.assertIsNone(repair_json("no JSON here"))
        self.assertEqual(load_answer('Here: {"speakers": {"s1": "Mia"},}'), {"speakers": {"s1": "Mia"}})
        with self.assertRaises(StageError):
            load_answer("I could not do that.")
        self.assertEqual(parse_translation_reply('["Hola", "Adiós",]', 2), ["Hola", "Adiós"])

    def test_never_closes_a_cut_off_reply(self):
        self.assertIsNone(repair_json('{"a": [1, 2,]'))
        # Slicing to the last "}" must not hide that the outer object never closed.
        with self.assertRaises(StageError):
            load_answer('{"scenes": [{"lines": [{"speaker": "Mia", "text": "Hi"}, {"speaker": "Al", "text": "She op')
        with self.assertRaises(ValueError):
            parse_translation_reply('["Hola", "Adi', 2)


class GemmaClientTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"H3_GEMMA": "1", "GEMINI_API_KEY": "env-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        budget = mock.patch.object(gemma, "_BUDGET", gemma._Budget())
        budget.start()
        self.addCleanup(budget.stop)

    def test_off_without_key_or_when_disabled(self):
        with mock.patch("requests.post") as post:
            with mock.patch.dict(os.environ, {"H3_GEMMA": "0"}):
                self.assertIsNone(gemma.ask("hi"))
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                self.assertIsNone(gemma.ask("hi"))
        post.assert_not_called()

    def test_skips_thoughts_and_switches_model_on_429(self):
        with mock.patch("requests.post", side_effect=[Response(429), reply('{"ok": true}')]) as post:
            self.assertEqual(gemma.ask_json("check"), {"ok": True})
        urls = [call.args[0] for call in post.call_args_list]
        self.assertIn(gemma.MODELS[0], urls[0])
        self.assertIn(gemma.MODELS[1], urls[1])
        self.assertNotIn("systemInstruction", post.call_args.kwargs["json"])

    def test_a_missing_model_rests_while_the_other_answers(self):
        with mock.patch("requests.post", side_effect=[Response(404), reply("a"), reply("b")]) as post:
            self.assertEqual(gemma.ask("x"), "a")
            self.assertEqual(gemma.ask("y"), "b")
        self.assertEqual([gemma.MODELS[0], gemma.MODELS[1], gemma.MODELS[1]],
                         [next(m for m in gemma.MODELS if m in call.args[0]) for call in post.call_args_list])

    def test_budget_refuses_past_the_token_limit(self):
        with mock.patch("requests.post", return_value=reply("ok", tokens=9000)) as post:
            self.assertEqual(gemma.ask("a", max_output_tokens=9000), "ok")
            self.assertEqual(gemma.ask("b", max_output_tokens=9000), "ok")  # the second model's own minute
            self.assertIsNone(gemma.ask("c", max_output_tokens=9000))
        self.assertIn(gemma.MODELS[1], post.call_args.args[0])
        self.assertEqual(post.call_count, 2)

    def test_writer_key_is_shared_only_from_a_gemini_writer(self):
        @gemma.uses_writer_key
        def writer(prompt, temporary_api_key="", backend="Gemini"):
            return gemma._api_key()

        self.assertEqual(writer("x", "ui-key"), "ui-key")
        self.assertEqual(writer("x", temporary_api_key="ui-key", backend="OpenAI-compatible"), "env-key")
        self.assertEqual(gemma._api_key(), "env-key")


class SongStyleTests(unittest.TestCase):
    def test_gemma_profile_and_tempo_with_keyword_fallback(self):
        with mock.patch.object(gemma, "ask_json", return_value={"genre": "ballad", "bpm": 70}):
            self.assertEqual(song._song_style("一首慢慢的爵士情歌"), (song_brief.GENRES["ballad"], 70))
        with mock.patch.object(gemma, "ask_json", return_value={"genre": "rap", "bpm": 300}):
            self.assertEqual(song._song_style("trap"), (song_brief.GENRES["rap"], None))
        # A tempo written in BPM is exact and wins over Gemma's reading.
        self.assertIn("~140 BPM", song_brief.build_brief(engine="YuE2", max_seconds=60, idea="trap, 140 bpm", bpm=90))
        self.assertIn("~90 BPM", song_brief.build_brief(engine="YuE2", max_seconds=60, idea="fast trap", bpm=90))
        with mock.patch.object(gemma, "ask_json", return_value=None):
            self.assertEqual(song._song_style("rap"), (None, None))
        plan = song_brief.plan_song(60, idea="rap", genre=song_brief.GENRES["ambient"])
        self.assertEqual(plan.genre.name, "ambient")
        self.assertIn("ambient", song_brief.build_brief(engine="YuE2", max_seconds=60, idea="rap",
                                                        genre=song_brief.GENRES["ambient"]))


class ReadingTests(unittest.TestCase):
    def test_only_checked_readings_replace_broken_notes(self):
        text = "<|speaker:0|> 去银行{yin hang2}找Siobhan{SH IH V AO1 N}，还{huan2}钱。"
        with mock.patch.object(gemma, "ask_json",
                               return_value={"readings": ["yin2 hang2", "SH IH0 V AO1 N"]}) as ask:
            self.assertEqual(gemma_checks.repair_reading_notes(text),
                             "<|speaker:0|> 去银行{yin2 hang2}找Siobhan{SH IH0 V AO1 N}，还{huan2}钱。")
        self.assertEqual(ask.call_count, 1)
        with mock.patch.object(gemma, "ask_json", return_value={"readings": ["yin2", None]}):
            self.assertEqual(gemma_checks.repair_reading_notes(text), text)  # wrong syllable count
        with mock.patch.object(gemma, "ask_json") as ask:
            self.assertEqual(gemma_checks.repair_reading_notes("还{huan2}钱"), "还{huan2}钱")
        ask.assert_not_called()

    def test_word_readings_must_cover_the_whole_word(self):
        with mock.patch.object(gemma, "ask_json", return_value={"readings": ["hang2", "yin2 hang2"]}):
            self.assertEqual(gemma_checks.fix_readings([("银行", "yin hang2"), ("银行", "yin hang2")], words=True),
                             [None, "yin2 hang2"])


class SourceJudgeTests(unittest.TestCase):
    SOURCE = "Mia: Bring the keey to Siobhan."
    SCRIPT = "<|speaker:0|> Bring the key to Siobhan."

    def test_strict_check_accepts_only_small_changes_gemma_approves(self):
        with mock.patch.object(gemma, "ask_json", return_value={"ok": [True]}) as ask:
            accepted = validate_source_fidelity(self.SOURCE, self.SCRIPT)
        self.assertIn("[keey]", ask.call_args.args[0])
        self.assertEqual(accepted, [("words", "bring the[keey]", "bring the[key]")])
        self.assertIn("1 small typo repair", accepted_repairs_note(accepted))
        for verdict in ({"ok": [False]}, None, {"ok": []}):
            with mock.patch.object(gemma, "ask_json", return_value=verdict):
                with self.assertRaises(H3Error):
                    validate_source_fidelity(self.SOURCE, self.SCRIPT)

    def test_numbers_speakers_and_long_changes_never_reach_gemma(self):
        cases = (("Mia: It costs 5 dollars.", "<|speaker:0|> It costs 6 dollars."),
                 ("Mia: We walked home slowly that night.", "<|speaker:0|> They ran all the way back."),
                 ("Speaker 1: Hi.\nSpeaker 2: Bye.", "<|speaker:1|> Hi.\n<|speaker:1|> Bye."))
        with mock.patch.object(gemma, "ask_json") as ask:
            for source, script in cases:
                with self.subTest(source=source), self.assertRaises(H3Error):
                    validate_source_fidelity(source, script)
        ask.assert_not_called()
        self.assertEqual(validate_source_fidelity(self.SCRIPT, self.SCRIPT), [])


if __name__ == "__main__":
    unittest.main()
