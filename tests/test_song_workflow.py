"""Song planning, writer guards, and YuE2 score reuse; CPU only, no network."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h3_models as _models
from h3_app import prompt_service, song_brief
from h3_app.provenance import read_score, score_path, write_snapshot
from h3_app.workflows.yue2 import build_yue2_graph, planned_score, required_yue2_nodes
from h3_ui import application as app

VERSE = "[Verse]\nMidnight and the canvas glows\nDragging little wires where the current flows\n"
SONG = (
    VERSE
    + "\n[Chorus]\nLet it render on\nTake your time, take your time\n"
    + "\n" + VERSE
    + "\n[Chorus]\nLet it render on\nTake your time, take your time\n"
)


class SongPlanTests(unittest.TestCase):
    def test_forms_fill_but_never_exceed_common_lengths(self) -> None:
        for seconds in (30, 60, 90, 120, 180, 240, 300):
            with self.subTest(seconds=seconds):
                plan = song_brief.plan_song(seconds)
                self.assertLessEqual(plan.seconds, seconds)
                self.assertGreater(plan.seconds, seconds * 0.7)
                self.assertIn("Chorus", [section.tag for section in plan.sections])

    def test_short_limits_get_a_hook_and_long_ones_a_bridge(self) -> None:
        self.assertEqual(
            [s.tag for s in song_brief.plan_song(30).sections], ["Intro", "Chorus", "Outro"]
        )
        self.assertIn("Bridge", [s.tag for s in song_brief.plan_song(180).sections])
        slow = song_brief.plan_song(30, bpm=70)
        self.assertLessEqual(slow.seconds, 30)
        self.assertEqual(slow.sections[1].bars % 2, 0)

    def test_instrumental_comes_from_choice_use_or_idea(self) -> None:
        self.assertTrue(song_brief.plan_song(120, vocals="Instrumental").instrumental)
        self.assertTrue(song_brief.plan_song(120, use="Background music").instrumental)
        self.assertTrue(song_brief.plan_song(120, idea="lo-fi beat, no vocals").instrumental)
        self.assertFalse(
            song_brief.plan_song(120, vocals="Female lead", use="Background music").instrumental
        )
        plan = song_brief.plan_song(60, vocals="Instrumental")
        self.assertEqual(sum(section.lines for section in plan.sections), 0)
        tags = song_brief.instrumental_tags(plan)
        self.assertTrue(tags.startswith("[Intro]") and tags.endswith("[Outro]"))
        self.assertIn("[Instrumental]", tags)

    def test_tempo_is_read_from_the_description(self) -> None:
        self.assertEqual(song_brief.detect_bpm("dance pop, 124 BPM"), 124)
        self.assertEqual(song_brief.detect_bpm("tempo 90-100 bpm"), 95)
        self.assertEqual(song_brief.detect_bpm("bpm: 72"), 72)
        self.assertIsNone(song_brief.detect_bpm("a slow ballad"))
        # A faster tempo fits more bars, so the planner chooses a longer form.
        fast = song_brief.plan_song(150, bpm=128)
        self.assertGreater(
            sum(s.bars for s in fast.sections), sum(s.bars for s in song_brief.plan_song(150).sections)
        )

    def test_long_rap_grows_by_verse_and_hook(self) -> None:
        for seconds in (300, 360):
            with self.subTest(seconds=seconds):
                plan = song_brief.plan_song(seconds, idea="hip hop")
                self.assertLessEqual(plan.seconds, seconds)
                self.assertGreater(plan.seconds, seconds * 0.85)
                self.assertNotIn("Bridge", [s.tag for s in plan.sections])
        # Growth stops at five verses; very long limits keep a shorter song.
        longest = song_brief.plan_song(480, idea="hip hop")
        self.assertLessEqual(longest.seconds, 480)
        longest = [s.tag for s in longest.sections]
        self.assertEqual(longest.count("Verse"), 5)
        self.assertEqual(longest[-2:], ["Chorus", "Outro"])

    def test_conventions_follow_the_genre(self) -> None:
        rap = song_brief.plan_song(120, idea="a hip-hop track about my hometown")
        self.assertEqual(rap.genre.name, "rap")
        self.assertEqual(rap.bpm, 90)
        self.assertEqual(rap.sections[1].tag, "Chorus")  # hook first
        verse = next(s for s in rap.sections if s.tag == "Verse")
        self.assertEqual((verse.bars, verse.lines), (16, 16))
        dance = song_brief.plan_song(180, idea="festival EDM anthem")
        self.assertEqual(dance.bpm, 124)
        self.assertIn("drop", next(s.role for s in dance.sections if s.tag == "Chorus"))
        ballad = song_brief.plan_song(180, idea="a quiet folk song")
        self.assertIn("stripped back", [s for s in ballad.sections if s.tag == "Chorus"][-1].role)
        self.assertIn("lift through harmony", next(s.role for s in ballad.sections if s.tag == "Pre-Chorus"))
        self.assertEqual(song_brief.plan_song(120, idea="pop song").genre, song_brief.POP)
        brief = song_brief.build_brief(engine="YuE2", max_seconds=120, idea="trap banger")
        self.assertIn("Style profile: rap", brief)
        self.assertIn("a default, not a rule", brief.lower())

    def test_brief_carries_choices_length_and_lyric_policy(self) -> None:
        brief = song_brief.build_brief(
            engine="MiniMax Music 3", max_seconds=60, vocals="Male lead",
            language="Japanese", use="Auto", notes="bigger chorus",
        )
        self.assertIn("Maximum length: 60 seconds", brief)
        self.assertIn("Vocals: Male lead", brief)
        self.assertIn("Lyric language: Japanese", brief)
        self.assertIn("[Chorus] 8 bars, 4 lines", brief)
        self.assertIn("bigger chorus", brief)
        kept = song_brief.build_brief(engine="YuE2", max_seconds=60, lyrics=SONG)
        self.assertIn("keep them exactly", kept)
        self.assertIn("Existing lyric form", kept)
        self.assertNotIn("Suggested form", kept)
        polished = song_brief.build_brief(
            engine="YuE2", max_seconds=60, lyrics=SONG, lyrics_policy="Polish my lyrics",
        )
        self.assertIn("polish for singability", polished)


class SongCheckTests(unittest.TestCase):
    def check(self, **values):
        defaults = dict(engine="MiniMax Music 3", description="pop", lyrics="", max_seconds=120)
        return song_brief.check_song(**{**defaults, **values})

    def texts(self, **values):
        return " ".join(finding.text for finding in self.check(**values)[3])

    def test_parses_sections_and_estimates_length(self) -> None:
        _plan, sections, estimate, _findings = self.check(lyrics=SONG)
        self.assertEqual([s.tag for s in sections], ["Verse", "Chorus", "Verse", "Chorus"])
        self.assertEqual(len(sections[0].lines), 2)
        self.assertGreater(estimate, 30)
        self.assertLess(estimate, 60)
        self.assertEqual(song_brief.syllables("Dragging little wires"), 6)
        self.assertEqual(song_brief.syllables("想你的夜晚"), 5)

    def test_overlong_draft_warns_with_a_suggested_length(self) -> None:
        text = self.texts(lyrics=SONG * 3, max_seconds=30)
        self.assertIn("will likely be cut", text)
        self.assertIn("Set the length to about", text)

    def test_producer_checks(self) -> None:
        self.assertIn("No section tags", self.texts(lyrics="just words\nand more words"))
        self.assertIn(
            "lyrics contain words",
            self.texts(lyrics=SONG, vocals="Instrumental"),
        )
        self.assertIn("No [Chorus]", self.texts(lyrics=(VERSE * 6)))
        long_line = "[Verse]\n" + "and then we walked along the endless river in the evening light\n"
        self.assertIn("long enough to sound rushed", self.texts(lyrics=long_line))
        self.assertIn("from your lyrics", self.texts(description="", lyrics=SONG))
        self.assertIn(
            "comma-separated tags",
            self.texts(engine="YuE2", description="A song. It is sad. " * 10, lyrics=SONG),
        )
        self.assertIn("YuE2 needs lyrics", self.texts(engine="YuE2", lyrics=""))
        self.assertEqual(self.check(lyrics=SONG, max_seconds=60)[3], [])

    def test_rap_verses_allow_dense_uneven_lines(self) -> None:
        bars = "[Verse]\n" + "\n".join(
            ["Started from the corner with a notebook and a dream in my pocket",
             "Now I", "Every single word I wrote was paying off the rent", "Uh"] * 2
        ) + "\n\n[Chorus]\nHometown, hometown\nNever let me down\n"
        rap = " ".join(f.text for f in self.check(description="hip hop", lyrics=bars)[3])
        sung = " ".join(f.text for f in self.check(description="pop", lyrics=bars)[3])
        self.assertNotIn("sound rushed", rap)
        self.assertNotIn("jump around", rap)
        self.assertIn("sound rushed", sung)
        self.assertIn("jump around", sung)
        self.assertLess(
            song_brief.estimate_seconds(bars, bpm=90, genre=song_brief.detect_genre("rap")),
            song_brief.estimate_seconds(bars, bpm=90),
        )

    def test_take_note_flags_cut_off_and_early_endings(self) -> None:
        self.assertIn("may be cut", song_brief.take_note(119.4, 120))
        self.assertIn("ended on its own", song_brief.take_note(104, 120))
        self.assertIn("much shorter", song_brief.take_note(20, 120))
        self.assertEqual(song_brief.take_note(None, 120), "")

    def test_render_escapes_user_text_and_shows_a_timeline(self) -> None:
        html = song_brief.render_check(
            engine="MiniMax Music 3", description="<b>pop</b>",
            lyrics="[<Verse>]\n<script>x</script>", max_seconds=60,
        )
        self.assertNotIn("<script>", html)
        self.assertNotIn("<Verse>", html)
        self.assertIn("h3-song-timeline", html)
        empty = song_brief.render_check(
            engine="MiniMax Music 3", description="", lyrics="", max_seconds=60,
        )
        self.assertIn("Suggested shape for 1:00", empty)
        self.assertIn("h3-song-chip-hook", empty)


class SongWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        systems = {}
        for target in ("MiniMax Music 3", "YuE2"):
            systems[target] = root / f"{target}.txt"
            systems[target].write_text(f"Instructions for {target}", encoding="utf-8")
        self.runtime = SimpleNamespace(prompt_systems=systems)

    def reply(self, client: Mock, text: str) -> None:
        client.return_value.chat.completions.create.return_value = Mock(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    def request_text(self, client: Mock) -> str:
        content = client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        return content[0]["text"] if isinstance(content, list) else content

    def music3(self, caption, lyrics, **values):
        return prompt_service.enhance_music3_prompt(
            caption, "unused", "", lyrics, None, None, None, "Lightning AI", "key",
            runtime=self.runtime, **values,
        )

    def yue2(self, style, lyrics, **values):
        values.setdefault("duration", 120)
        duration = values.pop("duration")
        return prompt_service.enhance_yue2_prompt(
            style, "unused", "", lyrics, "full", duration, "Lightning AI", "key",
            runtime=self.runtime, **values,
        )

    @patch("openai.OpenAI")
    def test_music3_writer_gets_the_brief_and_keeps_user_lyrics(self, client: Mock) -> None:
        self.reply(client, "CAPTION:\nGlobal Metadata: pop\nLYRICS:\n[Verse]\nNew words")
        caption, lyrics, _ = self.music3("pop", SONG, duration=60, vocals="Duet", notes="slower")
        self.assertEqual(caption, "Global Metadata: pop")
        self.assertEqual(lyrics, SONG.strip())
        request = self.request_text(client)
        self.assertIn("Maximum length: 60 seconds", request)
        self.assertIn("Vocals: Duet", request)
        self.assertIn("slower", request)
        _, polished, _ = self.music3("pop", SONG, lyrics_policy="Polish my lyrics")
        self.assertEqual(polished, "[Verse]\nNew words")

    @patch("openai.OpenAI")
    def test_music3_writes_from_lyrics_alone_and_honours_instrumental(self, client: Mock) -> None:
        self.reply(client, "```\nCAPTION: Folk ballad\nLYRICS: N/A\n```")
        caption, lyrics, status = self.music3("", SONG)
        self.assertEqual((caption, lyrics), ("Folk ballad", SONG.strip()))
        self.assertIn("fitting music description", self.request_text(client))
        self.assertNotIn("failed", status)
        self.reply(client, "CAPTION: Ambient bed\nLYRICS:\n[Verse]\nWords that should not be sung")
        caption, lyrics, _ = self.music3("calm ambient", "", vocals="Instrumental")
        self.assertEqual((caption, lyrics), ("Ambient bed", ""))

    @patch("openai.OpenAI")
    def test_malformed_replies_keep_the_draft(self, client: Mock) -> None:
        self.reply(client, "Here is your song!\nLYRICS: [Verse] la la")
        caption, lyrics, status = self.music3("pop idea", "")
        self.assertEqual((caption, lyrics), ("pop idea", ""))
        self.assertIn("could not be split", status)
        self.reply(client, "A warm folk song with fingerpicked guitar.")
        caption, _, _ = self.music3("folk", "")
        self.assertEqual(caption, "A warm folk song with fingerpicked guitar.")
        self.reply(client, "Just a style")
        style, lyrics, status = self.yue2("folk", SONG)
        self.assertEqual((style, lyrics), ("folk", SONG.strip()))
        self.assertIn("could not be split", status)

    @patch("openai.OpenAI")
    def test_markdown_labels_and_a_preamble_still_split(self, client: Mock) -> None:
        self.reply(client, "Here is your draft.\n\n**STYLE**: indie pop, warm vocal\n\n**LYRICS:**\n[Verse]\nNew")
        style, lyrics, status = self.yue2("indie", "")
        self.assertEqual((style, lyrics), ("indie pop, warm vocal", "[Verse]\nNew"))
        self.assertNotIn("failed", status)
        self.reply(client, "## Caption: Dream pop\n## Lyrics: N/A")
        caption, _, status = self.music3("dreamy", SONG)
        self.assertEqual(caption, "Dream pop")
        self.assertNotIn("failed", status)

    @patch("openai.OpenAI")
    def test_instrumental_tags_give_way_once_a_voice_is_chosen(self, client: Mock) -> None:
        tags = "[Intro]\n\n[Instrumental]\n\n[Outro]"
        self.reply(client, "STYLE: indie pop\nLYRICS:\n[Verse]\nFresh words")
        _, lyrics, _ = self.yue2("indie", tags, vocals="Female lead")
        self.assertEqual(lyrics, "[Verse]\nFresh words")
        self.assertIn("Existing lyrics:\n(none)", self.request_text(client))
        # Still instrumental: the tag-only structure is kept.
        _, lyrics, _ = self.yue2("indie", tags, vocals="Instrumental")
        self.assertEqual(lyrics, tags)

    @patch("openai.OpenAI")
    def test_yue2_keeps_lyrics_and_turns_style_into_tags(self, client: Mock) -> None:
        self.reply(client, "STYLE:\nindie pop,\nwarm female vocal.\nclean guitar\nLYRICS:\n[Verse]\nRewritten")
        style, lyrics, _ = self.yue2("indie", SONG)
        self.assertEqual(style, "indie pop, warm female vocal, clean guitar")
        self.assertEqual(lyrics, SONG.strip())
        self.assertIn("Score mode: full", self.request_text(client))

    @patch("openai.OpenAI")
    def test_yue2_instrumental_gets_planned_section_tags(self, client: Mock) -> None:
        self.reply(client, "STYLE: lo-fi hip hop, 80 BPM\nLYRICS:\n[Verse]\nla la words")
        _, lyrics, _ = self.yue2("lo-fi", "", vocals="Instrumental", duration=60)
        self.assertFalse(any(section.lines for section in song_brief.parse_lyrics(lyrics)))
        self.assertTrue(lyrics.startswith("[Intro]"))
        self.reply(client, "STYLE: ambient\nLYRICS: N/A")
        _, lyrics, _ = self.yue2("ambient", "")
        self.assertIn("[Instrumental]", lyrics)


class YuE2ScoreTests(unittest.TestCase):
    def graph(self, mode="full", abc=""):
        return build_yue2_graph(
            model_choice=_models.DEFAULT_YUE2_MODEL, style="pop", lyrics=SONG, abc=abc,
            mode=mode, max_duration=60, seed=1, steps=32, cfg=1.0, temperature=1.0,
            top_p=0.95, top_k=100, repetition_penalty=1.2, max_abc_tokens=8192,
            abc_temperature=0.7, abc_top_p=0.9, abc_top_k=30, abc_repetition_penalty=1.005,
            abc_penalty_window=100, tiled_decode=True,
        )

    def test_only_planned_scores_are_previewed(self) -> None:
        planned = self.graph()
        preview = [k for k, n in planned.items() if n["class_type"] == "PreviewAny"]
        self.assertEqual(len(preview), 1)
        abc = next(k for k, n in planned.items() if n["class_type"] == "YuE2GenerateABC")
        self.assertEqual(planned[preview[0]]["inputs"]["source"], [abc, 0])
        self.assertIn("PreviewAny", required_yue2_nodes(tiled_decode=True, generate_abc=True))
        for graph in (self.graph(mode="off"), self.graph(abc="X:1")):
            self.assertNotIn("PreviewAny", {n["class_type"] for n in graph.values()})
        history = {"outputs": {preview[0]: {"text": ["X:1\nK:C\nCDEF|"]}}}
        self.assertEqual(planned_score(planned, history), "X:1\nK:C\nCDEF|")
        self.assertEqual(planned_score(planned, {}), "")

    def test_generation_saves_the_score_beside_the_song(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            result = Path(directory) / "yue2_00001_.mp3"
            result.write_bytes(b"fixture")
            built = {}

            def build(**values):
                built.update(graph=build_yue2_graph(**values))
                return built["graph"]

            def history(_prompt_id):
                node = next(k for k, n in built["graph"].items() if n["class_type"] == "PreviewAny")
                return {"outputs": {node: {"text": ["X:1\nK:G\nGABc|"]}}}

            for name in ("unload_prompt_rewriter", "ensure_yue2_models"):
                stack.enter_context(patch.object(app, name))
            for name in ("missing_yue2_model_names", "poll_comfy_progress"):
                stack.enter_context(patch.object(app, name, return_value=[]))
            stack.enter_context(patch.object(
                app, "object_info",
                return_value={k: {} for k in required_yue2_nodes(tiled_decode=True, generate_abc=True)},
            ))
            stack.enter_context(patch("h3_app.generation.yue2.build_yue2_graph", side_effect=build))
            stack.enter_context(patch.object(app, "submit_prompt", return_value="fixture"))
            stack.enter_context(patch.object(app, "wait_for_history", side_effect=history))
            stack.enter_context(patch.object(app, "resolve_audio_output", return_value=result))
            stack.enter_context(patch("h3_app.generation.yue2.probe_audio_seconds", return_value=59.5))
            values = {
                key: app.YUE2_DEFAULTS.get(key)
                for key in inspect.signature(app.generate_yue2).parameters
                if key != "progress"
            }
            values.update(
                model_choice=_models.DEFAULT_YUE2_MODEL, style="pop", lyrics=SONG, abc="",
                max_duration=60, seed=7, tiled_decode=True,
                progress=lambda *args, **kwargs: None,
            )
            updates = list(app.generate_yue2(**values))
            self.assertIn("score saved", updates[-1].status)
            self.assertIn("reached the 1:00 limit", updates[-1].status)
            self.assertEqual(read_score(result), "X:1\nK:G\nGABc|")
            facts = app.take_facts(str(result))
            self.assertEqual(
                (facts["score"], facts["seconds"], facts["score_mode"]),
                ("X:1\nK:G\nGABc|", 59.5, app.YUE2_DEFAULTS["mode"]),
            )
            self.assertEqual(app.take_facts(None), {})

    def test_score_sidecar_follows_the_song_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            song = Path(directory) / "take.mp3"
            song.write_bytes(b"x")
            write_snapshot(song, {"family": "YuE2"})
            self.assertEqual(score_path(song).name, "take.mp3.score.abc")
            self.assertEqual(read_score(song), "")


if __name__ == "__main__":
    unittest.main()


class SongTakeHistoryTests(unittest.TestCase):
    """The take and one-click handlers, exercised through a minimal Blocks."""

    @staticmethod
    def handler(demo, name):
        return next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == name)

    def build(self, facts, *, latest=False):
        import gradio as gr

        from h3_ui.song_takes import bind_take_history, build_take_history

        with gr.Blocks() as demo:
            style, lyrics, abc = gr.Textbox(), gr.Textbox(), gr.Textbox()
            mode = gr.Dropdown(["full", "melody", "off"], value="off")
            seed, run, output = gr.Number(), gr.Button(), gr.Audio(type="filepath")
            takes = build_take_history(with_score=True)
            extra = {"last_score": gr.Textbox(), "reuse_last_score": gr.Button()} if latest else {}
            bind_take_history(
                takes, run=run, output=output, seed=seed, take_facts=facts,
                score=abc, mode=mode, drafts={"style": style, "lyrics": lyrics, "abc": abc},
                **extra,
            )
        return demo

    def test_reused_scores_keep_their_composition_mode(self) -> None:
        import gradio as gr

        facts = {"m.mp3": {"seed": 3, "seconds": 60, "score": "X:1\nK:D", "score_mode": "melody"}}
        demo = self.build(lambda path: facts[path], latest=True)
        result = self.handler(demo, "add")("m.mp3", [], {})
        history, last_score = result[0], result[-1]
        self.assertEqual(last_score, "X:1\nK:D")
        self.assertEqual(self.handler(demo, "add")("m.mp3", history, {}), (gr.skip(),) * 8)
        self.assertEqual(self.handler(demo, "take_score")(history, "0", "full"), ("X:1\nK:D", "melody"))
        self.assertEqual(self.handler(demo, "latest_score")(history, "off"), ("X:1\nK:D", "melody"))
        self.assertEqual(self.handler(demo, "latest_score")([], "off"), (gr.skip(), gr.skip()))

    def test_takes_record_drafts_compare_and_reuse(self) -> None:
        facts = {"a.mp3": {"seed": 1, "seconds": 61, "score": "X:1"},
                 "b.mp3": {"seed": 2, "seconds": 59, "score": ""}}
        demo = self.build(lambda path: facts[path])
        add = self.handler(demo, "add")
        history, *_rest = add("a.mp3", [], {"style": "pop", "lyrics": "[Verse]\nOne", "abc": ""})
        history, take_a, take_b, audio_a, audio_b, details, panel = add(
            "b.mp3", history, {"style": "pop", "lyrics": "[Verse]\nTwo", "abc": ""},
        )
        self.assertEqual([take["number"] for take in history], [1, 2])
        self.assertEqual((take_a["value"], take_b["value"]), ("1", "0"))
        self.assertEqual((audio_a, audio_b), ("b.mp3", "a.mp3"))
        self.assertIn("lyrics, seed changed", details)
        self.assertEqual(take_a["choices"][0], ("Take 2 · 0:59 · seed 2", "1"))
        self.assertTrue(panel["open"])
        # The same file does not become a second take.
        import gradio as gr

        self.assertEqual(add("b.mp3", history, {}), (gr.skip(),) * 7)

        select = self.handler(demo, "select")
        self.assertEqual(select(history, "0", "1")[:2], ("a.mp3", "b.mp3"))
        reuse = self.handler(demo, "reuse")
        self.assertEqual(reuse(history, "0"), ("pop", "[Verse]\nOne", ""))
        self.assertEqual(self.handler(demo, "take_seed")(history, "1"), 2)
        score, mode = self.handler(demo, "take_score")(history, "0", "off")
        self.assertEqual((score, mode), ("X:1", "full"))
        capture = self.handler(demo, "<lambda>")
        self.assertEqual(capture("rock", "words", ""), {"style": "rock", "lyrics": "words", "abc": ""})

    def test_write_and_generate_applies_with_undo_or_stops_on_failure(self) -> None:
        import gradio as gr

        from h3_ui.prompt_review import build_prompt_review
        from h3_ui.song_controls import bind_write_and_generate

        replies = iter([("New caption", "New lyrics", "Done."),
                        ("idea", "", "Prompt enhancement failed: no key"),
                        ("idea", "old words", "Done.")])
        with gr.Blocks() as demo:
            caption, notes, lyrics, status = gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox()
            button = gr.Button()
            review = build_prompt_review(compact=True)
            bind_write_and_generate(
                button, lambda *values: next(replies), inputs=[caption, notes, lyrics],
                drafts=[caption, lyrics], status=status, review=review, generate_id="h3-test-generate",
            )
        write = self.handler(demo, "write")
        result = write("idea", "louder", "old words")
        self.assertEqual(result[:2], ("New caption", "New lyrics"))
        self.assertIn("Applied without review", result[2])
        self.assertEqual(result[3], ("idea", "old words"))  # Undo restores the draft
        self.assertIsNone(result[4])
        with self.assertRaises(gr.Error):
            write("idea", "", "")
        # An unchanged draft offers no Undo and does not claim it was rewritten.
        unchanged = write("idea", "", "old words")
        self.assertIn("kept your draft", unchanged[2])
        self.assertEqual((unchanged[3], unchanged[5]), (gr.skip(), gr.skip()))
        scripts = [d.get("js") or "" for d in demo.get_config_file()["dependencies"]]
        starter = next(js for js in scripts if "getElementById('h3-test-generate')" in js)
        self.assertIn("No take started", starter)
