"""Drama 3 scripts, cloned voices, speaker presets and Qwen3-ASR transcripts."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app.drama3 import speakers as drama3_speakers
from h3_app.prompt_writers import drama3 as drama3_writer
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.generation.drama3 import transcribe_drama3_audio, validate_drama3 as validate_request
from h3_app.generation.requests import Drama3Request
from h3_app.workflows.drama3 import build_drama3_asr_graph, text_speakers, validate_drama3_text

ROOT = Path(__file__).resolve().parents[1]


def completion(text):
    return Mock(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class Drama3RequestTests(unittest.TestCase):
    def test_text_validation_matches_the_node_pack(self):
        self.assertEqual(validate_drama3_text(" <|speaker:15|> [sighs] fine "), "<|speaker:15|> [sighs] fine")
        # Fish Audio documents seven scoped tags; the writer prompt offers all of them and none is rejected.
        prompt = (ROOT / "prompts" / "prompt_drama3.txt").read_text(encoding="utf-8")
        for tag in ("whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful"):
            with self.subTest(tag=tag):
                line = f"<|speaker:0|> Well, <{tag}>this matters</{tag}>."
                self.assertEqual(validate_drama3_text(line), line)
                self.assertIn(f"<{tag}>...</{tag}>", prompt)
        self.assertIn("Only these seven exist", prompt)
        for bad in ("", "<|speaker:16|> x", "<|speaker:01|> x", "<|audio_separator|> x", "<|speaker:0 x"):
            with self.subTest(bad=bad), self.assertRaises(H3Error):
                validate_drama3_text(bad)

    def test_text_speakers_treats_untagged_text_as_speaker_zero(self):
        self.assertEqual(text_speakers("hello"), {0})
        self.assertEqual(text_speakers("<|speaker:1|> a\n<|speaker:3|> b"), {1, 3})

    def test_defaults_use_the_full_frame_cap_and_uncapped_top_k(self):
        request = Drama3Request("Hello")
        self.assertEqual((request.max_frames, request.top_k, request.parallel, request.retries), (8192, 0, 16, 0))
        for frames in (0, 8193):
            with self.subTest(frames=frames), self.assertRaises(H3Error):
                validate_request(Drama3Request("Hello", voices=((0, "a.wav", "Hello"),), max_frames=frames))

    def test_request_validation_for_voice_slots(self):
        validate_request(Drama3Request("<|speaker:0|> a <|speaker:1|> b",
                                       voices=((0, "a", "t"), (1, "b", "u"))))
        for voices in (((0, "a", ""),), ((0, "a", "t"), (0, "b", "t")), ((16, "a", "t"),), ((0, "", "t"),)):
            with self.subTest(voices=voices), self.assertRaises(H3Error):
                validate_request(Drama3Request("hi", voices=voices))

    def test_uncapped_sampling_is_valid(self):
        for top_k in (0, 30, 4096):
            validate_request(Drama3Request("Hello", voices=((0, "a.wav", "Hello"),), top_k=top_k))
        for top_k in (-1, 4097):
            with self.subTest(top_k=top_k), self.assertRaises(H3Error):
                validate_request(Drama3Request("Hello", voices=((0, "a.wav", "Hello"),), top_k=top_k))

    def test_every_speaker_in_the_script_needs_a_voice(self):
        with self.assertRaisesRegex(H3Error, r"voice for Speaker 2\.$"):
            validate_request(Drama3Request("<|speaker:0|> Hi.\n<|speaker:2|> Yo.", voices=((0, "a.wav", "Hi"),)))

    def test_named_lines_need_a_cast_name_and_a_voice(self):
        voices = ((0, "a.wav", "Hi"),)
        cast = ((0, "Mia"), (2, "Leo"))
        validate_request(Drama3Request("Mia: Hi.\nmia: Again.\nSpeaker 0: Once more.", voices=voices,
                                       voice_names=cast))
        with self.assertRaisesRegex(H3Error, r"voice for Leo\.$"):
            validate_request(Drama3Request("Mia: Hi.\nLeo: Yo.", voices=voices, voice_names=cast))
        with self.assertRaisesRegex(H3Error, r'named "Bob"'):
            validate_request(Drama3Request("Mia: Hi.\nBob: Yo.", voices=voices, voice_names=cast))
        # Prose that names nobody is spoken by the first voice, headings and all.
        validate_request(Drama3Request("Chapter 1: Rain\nIt rained.", voices=voices, voice_names=cast))


class Drama3TranscriptionTests(unittest.TestCase):
    def services(self, history):
        execution = SimpleNamespace(
            object_info=lambda: {"Qwen3ASRLoader": {}, "LoadAudio": {}, "Qwen3ASRTranscribe": {}, "PreviewAny": {}},
            submit_prompt=Mock(return_value="job-1"),
            poll_comfy_progress=lambda prompt_id, graph: iter(()),
            wait_for_history=lambda prompt_id: history,
        )
        return SimpleNamespace(execution=execution, models=SimpleNamespace(ensure_model_keys=Mock()))

    def test_transcript_is_read_from_the_preview_node(self):
        graph, preview = build_drama3_asr_graph(audio="h3_gradio/drama3_asr/x.wav")
        self.assertEqual(graph[preview]["class_type"], "PreviewAny")
        services = self.services({"outputs": {preview: {"text": [" We asked over twenty people. "]}}})
        self.assertEqual(transcribe_drama3_audio("h3_gradio/drama3_asr/x.wav", services),
                         "We asked over twenty people.")
        services.models.ensure_model_keys.assert_called_once_with(("drama3_asr",))

    def test_silence_is_reported(self):
        _graph, preview = build_drama3_asr_graph(audio="x.wav")
        with self.assertRaises(H3Error):
            transcribe_drama3_audio("x.wav", self.services({"outputs": {preview: {"text": [""]}}}))

    def test_missing_nodes_are_actionable(self):
        services = self.services({})
        services.execution.object_info = lambda: {}
        with self.assertRaisesRegex(H3Error, "Qwen3ASRLoader"):
            transcribe_drama3_audio("x.wav", services)


class SpeakerPresetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        patcher = patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(root / "workspace")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runtime = RuntimeConfig(root, "http://x", root / "ComfyUI", root / "m.json", root / "outputs")
        self.clip = root / "clip.wav"
        self.clip.write_bytes(b"RIFF....WAVE")

    def test_save_reuse_replace_and_delete(self):
        saved = drama3_speakers.save_speaker("  Narrator   One ", str(self.clip), "Hello there.", self.runtime)
        self.assertEqual(saved.name, "Narrator One")
        self.assertTrue(saved.audio.is_file())
        self.assertTrue(saved.audio.is_relative_to(Path(self.temp.name) / "workspace" / "drama3_speakers"))
        drama3_speakers.save_speaker("narrator one", str(self.clip), "Replaced.", self.runtime)
        self.assertEqual(drama3_speakers.speaker_names(self.runtime), ["narrator one"])
        self.assertEqual(drama3_speakers.load_speaker("NARRATOR ONE", self.runtime).transcript, "Replaced.")
        drama3_speakers.delete_speaker("Narrator One", self.runtime)
        self.assertEqual(drama3_speakers.speaker_names(self.runtime), [])
        with self.assertRaises(H3Error):
            drama3_speakers.load_speaker("Narrator One", self.runtime)

    def test_a_failed_replace_keeps_the_previous_preset(self):
        drama3_speakers.save_speaker("Narrator", str(self.clip), "Original.", self.runtime)
        rename = Path.rename

        def failing_rename(path, target):
            if path.name.startswith(".saving-"):
                raise OSError("file in use")
            return rename(path, target)

        with patch.object(Path, "rename", autospec=True, side_effect=failing_rename), \
                self.assertRaises(OSError):
            drama3_speakers.save_speaker("Narrator", str(self.clip), "Replacement.", self.runtime)
        self.assertEqual(drama3_speakers.load_speaker("Narrator", self.runtime).transcript, "Original.")
        self.assertEqual(drama3_speakers.speaker_names(self.runtime), ["Narrator"])

    def test_presets_stay_outside_served_outputs_by_default(self):
        with patch.dict(os.environ, {"H3_WORKSPACE_DIR": ""}):
            root = drama3_speakers.speakers_root(self.runtime)
        self.assertEqual(root, (self.runtime.outputs_dir / ".h3-workspace").resolve() / "drama3_speakers")

    def test_saving_requires_a_clip_name_and_transcript(self):
        for name, audio, transcript in (("", self.clip, "t"), ("A", "", "t"), ("A", self.clip, " "),
                                        ("A", Path(self.temp.name) / "x.txt", "t"), ("x" * 65, self.clip, "t")):
            with self.subTest(name=name, audio=audio), self.assertRaises(H3Error):
                drama3_speakers.save_speaker(name, str(audio), transcript, self.runtime)


class Drama3WriterTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(prompt_systems={"Drama 3": ROOT / "prompts" / "prompt_drama3.txt"})

    @patch("openai.OpenAI")
    def test_rewrite_sends_voice_context_and_selected_openai_model(self, client):
        client.return_value.chat.completions.create.return_value = completion(
            "<|speaker:0|> [warm] Welcome back.\n<|speaker:1|> [laughs] Finally!"
        )
        script, status = drama3_writer.enhance_drama3_text(
            "ALICE: Welcome back.\nBOB: Finally!", "unused", "", [(0, "Alice")],
            "OpenAI-compatible", "key", "openai/gpt-6.1-sol", runtime=self.runtime,
        )
        self.assertTrue(script.startswith("<|speaker:0|>"))
        self.assertIn("openai/gpt-6.1-sol", status)
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-6.1-sol")  # bare ID on OpenAI's own API
        self.assertIn("Start every spoken turn line with the speaker's name", request["messages"][0]["content"])
        user = request["messages"][1]["content"][0]["text"]
        self.assertIn('- speaker_id 0 ("Alice") uses a cloned reference voice', user)
        self.assertIn("ALICE: Welcome back.", user)

    @patch("openai.OpenAI")
    def test_an_under_directed_rewrite_gets_a_direction_pass(self, client):
        import json

        writer = json.dumps({"title": "Night", "source_mode": "preserve", "script": (
            "narrator: The door creaked open.\nMia: Is someone there?\nnarrator: Nobody answered her."),
            "speakers": [{"speaker_id": 0, "name": "narrator", "description": "Calm.", "preview_text": "Hello."},
                         {"speaker_id": 1, "name": "Mia", "description": "Bright.", "preview_text": "Hello."}]})
        director = ("narrator: The door [hushed] creaked open.\nMia: Is someone [nervous] there?\n"
                    "narrator: Nobody answered [tense, low] <slow>her</slow>.")
        client.return_value.chat.completions.create.side_effect = [completion(writer), completion(director)]
        runtime = SimpleNamespace(prompt_systems={"Drama 3": ROOT / "prompts" / "prompt_drama3.txt",
                                                  "Drama 3 direction": ROOT / "prompts" / "prompt_drama3_direction.txt"})
        script, status, cast = drama3_writer.enhance_drama3_text(
            "The door creaked open. “Is someone there?” Nobody answered her.", "unused", "", (),
            "OpenAI-compatible", "key", runtime=runtime, return_cast=True)
        self.assertEqual(script, "<|speaker:0|> The door [hushed] creaked open.\n<|speaker:1|> Is someone [nervous] there?\n"
                                 "<|speaker:0|> Nobody answered [tense, low] <slow>her</slow>.")
        self.assertIn("A direction pass added cues to 3 line(s), 3 of 3 now directed.", status)
        self.assertEqual([entry["name"] for entry in cast], ["narrator", "Mia"])
        calls = client.return_value.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("performance director", calls[1].kwargs["messages"][0]["content"])
        self.assertIn("Mia: Is someone there?", calls[1].kwargs["messages"][1]["content"][0]["text"])

    @patch("openai.OpenAI")
    def test_invalid_scripts_keep_the_original_text(self, client):
        client.return_value.chat.completions.create.return_value = completion(
            "<|speaker:0|> Hi<|audio_separator|><|speaker:1|> Yo"
        )
        script, status = drama3_writer.enhance_drama3_text(
            "hi", "unused", "", (), "OpenAI-compatible", "key", runtime=self.runtime,
        )
        self.assertEqual(script, "hi")
        self.assertIn("audio_separator", status)

    def rewrite(self, client, reply, mode, voices=()):
        client.return_value.chat.completions.create.return_value = completion(reply)
        result = drama3_writer.enhance_drama3_text(
            "ALICE: Hi.\nBOB: Yo.", "unused", "", voices, "OpenAI-compatible", "key",
            speaker_mode=mode, runtime=self.runtime,
        )
        request = client.return_value.chat.completions.create.call_args
        user = request.kwargs["messages"][1]["content"][0]["text"] if request else ""
        return (*result, user)

    @patch("openai.OpenAI")
    def test_single_mode_instructs_and_folds_onto_one_speaker(self, client):
        script, _status, user = self.rewrite(
            client, "<|speaker:0|> Hi [warm] there.\n<|speaker:1|> Yo! [laughs]", "Single")
        self.assertIn("exactly ONE speaker", user)
        self.assertEqual(script, "<|speaker:0|> Hi [warm] there.\n<|speaker:0|> Yo! [laughs]")
        script, _status, user = self.rewrite(client, "<|speaker:0|> Hi.", "Single", [(2, "Bob")])
        self.assertIn("speaker_id 2", user)
        self.assertEqual(script, "<|speaker:2|> Hi.")

    @patch("openai.OpenAI")
    def test_adapt_mode_limits_speakers_to_reference_voices(self, client):
        voices = [(0, "Alice"), (2, "")]
        script, _status, user = self.rewrite(
            client, "<|speaker:0|> Hi.\n<|speaker:2|> Yo.", "Adapt to voices", voices)
        self.assertIn("exactly 2 speaker(s)", user)
        self.assertEqual(script, "<|speaker:0|> Hi.\n<|speaker:2|> Yo.")
        script, status, _user = self.rewrite(
            client, "<|speaker:0|> Hi.\n<|speaker:1|> Yo.", "Adapt to voices", voices)
        self.assertEqual(script, "ALICE: Hi.\nBOB: Yo.")
        self.assertIn("without a reference voice: Speaker 1.", status)

    @patch("openai.OpenAI")
    def test_adapt_needs_a_voice_and_auto_allows_any_speakers(self, client):
        script, status, _user = self.rewrite(client, "<|speaker:0|> Hi.", "Adapt to voices")
        self.assertEqual(script, "ALICE: Hi.\nBOB: Yo.")
        self.assertIn("needs at least one reference voice", status)
        client.return_value.chat.completions.create.assert_not_called()
        script, _status, user = self.rewrite(client, "<|speaker:0|> Hi.\n<|speaker:5|> Yo.", "Auto")
        self.assertIn("decide from the text", user)
        self.assertEqual(script, "<|speaker:0|> Hi.\n<|speaker:5|> Yo.")

    @patch("openai.OpenAI")
    def test_a_typed_model_goes_to_the_endpoint_unchanged(self, client):
        client.return_value.chat.completions.create.return_value = completion("<|speaker:0|> [warm] Hi.")
        drama3_writer.enhance_drama3_text(
            "hi", "unused", "", (), "OpenAI-compatible", "key", " my-model ",
            "https://llm.example.test/v1", runtime=self.runtime,
        )
        request = client.return_value.chat.completions.create.call_args_list[0].kwargs
        self.assertEqual(request["model"], "my-model")


class Drama3AppInputTests(unittest.TestCase):
    def test_slot_values_map_to_speakers_with_absolute_staged_paths(self):
        from h3_ui import application

        drama3_handlers = application.drama3_handlers  # importing the application installs its services

        captured = {}

        def engine(request, *args, **kwargs):
            captured["request"] = request
            yield None

        slots = ["Upload a clip", None, ""] * 16
        slots[3 * 5:3 * 5 + 3] = ["Upload a clip", "/uploads/five.wav", "Line five."]
        values = ["<|speaker:5|> Hi.", *slots, 1.0, 0.9, 30, 8192]
        with patch.object(drama3_handlers, "stage_file", return_value="h3_gradio/drama3_voices/five.wav"), \
                patch.object(drama3_handlers, "_runtime_config", return_value=SimpleNamespace(input_dir=Path("/in"))), \
                patch.object(drama3_handlers, "_generation_services"), \
                patch("h3_app.generation.drama3.generate_drama3", engine), \
                patch("h3_app.generation.drama3.validate_drama3"):
            list(drama3_handlers.generate_drama3(*values))
        request = captured["request"]
        self.assertEqual(request.voices, ((5, str(Path("/in") / "h3_gradio/drama3_voices/five.wav"), "Line five."),))
        self.assertEqual((request.top_k, request.parallel, request.retries), (30, 16, 0))
        self.assertEqual((request.regenerate, request.remix_only), ((), False))

    def captured_request(self, handler, *leading):
        from h3_ui import application

        drama3_handlers = application.drama3_handlers
        captured = {}

        def engine(request, *args, **kwargs):
            captured["request"] = request
            yield None

        slots = ["Upload a clip", None, ""] * 16
        values = ["Hi.", *slots, 1.0, 0.9, 30, 8192]
        with patch.object(drama3_handlers, "_runtime_config", return_value=SimpleNamespace(input_dir=Path("/in"))), \
                patch.object(drama3_handlers, "_generation_services"), \
                patch("h3_app.generation.drama3.generate_drama3", engine), \
                patch("h3_app.generation.drama3.validate_drama3"):
            list(getattr(drama3_handlers, handler)(*leading, *values))
        return captured["request"]

    def test_regenerate_and_remix_reuse_the_generation_inputs(self):
        import gradio as gr

        self.assertEqual(self.captured_request("regenerate_drama3_line", "abc").regenerate, ("abc",))
        remix = self.captured_request("remix_drama3")
        self.assertEqual((remix.text, remix.regenerate, remix.remix_only), ("Hi.", (), True))
        with self.assertRaisesRegex(gr.Error, "Choose a line"):
            self.captured_request("regenerate_drama3_line", None)

    def test_takes_table_comes_from_the_result_snapshot(self):
        from h3_ui.application import drama3_handlers
        from h3_app.provenance import write_snapshot

        with tempfile.TemporaryDirectory() as temp:
            result = Path(temp) / "speech.mp3"
            result.write_bytes(b"mp3")
            write_snapshot(result, {"family": "Drama 3", "settings": {"chunks": [
                {"speaker": 0, "name": "Alice", "text": "Hello  there.", "take": "k1", "reused": True,
                 "check": "reused", "seconds": 1.2},
                {"speaker": 3, "name": "", "text": "x" * 60, "take": "k2", "reused": False,
                 "check": "skipped", "seconds": 4.0},
            ]}})
            table, line, actions = drama3_handlers.drama3_takes(str(result))
        self.assertEqual(table["value"], [[1, "Alice", "Hello there.", 1.2, "reused"],
                                          [2, "Speaker 3", "x" * 60, 4.0, "new"]])
        self.assertEqual(line["choices"][0], ("1 · Alice · Hello there.", "k1"))
        self.assertTrue(line["choices"][1][0].endswith("…"))
        self.assertEqual(line["value"], "k1")
        self.assertTrue(actions["visible"])
        self.assertFalse(drama3_handlers.drama3_takes(None)[2]["visible"])

    def test_clicking_a_takes_row_picks_its_line_and_plays_its_take(self):
        import gradio as gr
        import numpy as np
        from h3_ui import application  # noqa: F401 - installs the handlers' services
        from h3_ui.application import drama3_handlers
        from h3_app.drama3.serving import audio, takes
        from h3_app.provenance import write_snapshot

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"H3_WORKSPACE_DIR": temp}):
            result = Path(temp) / "speech.mp3"
            result.write_bytes(b"mp3")
            write_snapshot(result, {"family": "Drama 3", "settings": {"chunks": [
                {"speaker": 0, "text": "Untaken."},
                {"speaker": 0, "text": "One.", "take": "a" * 32},
                {"speaker": 1, "text": "Two.", "take": "b" * 32},
            ]}})
            # Rows count only lines with a take, as the table shows them.
            self.assertEqual(drama3_handlers.drama3_take_selected(str(result), SimpleNamespace(index=[1, 2])),
                             "b" * 32)
            self.assertIs(type(drama3_handlers.drama3_take_selected(str(result), SimpleNamespace(index=[5, 0]))),
                          type(gr.skip()))
            root = takes.takes_root(None)
            takes.save_take(root, "b" * 32, audio.encode_wav(np.zeros(2400, dtype=np.float32), 24000))
            rate, samples = drama3_handlers.drama3_take_audio("b" * 32)
            self.assertEqual((rate, samples.size), (24000, 2400))
            self.assertIsNone(drama3_handlers.drama3_take_audio("a" * 32))
            self.assertIsNone(drama3_handlers.drama3_take_audio("../b"))
            self.assertIsNone(drama3_handlers.drama3_take_audio(None))

    def test_subtitles_are_timed_lines_without_cues(self):
        from h3_ui import application  # noqa: F401 - installs the handlers' services
        from h3_ui.application import drama3_handlers
        from h3_app.provenance import write_snapshot

        with tempfile.TemporaryDirectory() as temp:
            result = Path(temp) / f"Story_20261020-060911_{'f' * 32}.mp3"
            result.write_bytes(b"mp3")
            write_snapshot(result, {"family": "Drama 3", "settings": {"duration": 9.5, "chunks": [
                {"speaker": 0, "name": "Mia & Co", "text": "[whispers] Hello <there>!", "start": 0.0, "end": 1.25},
                {"speaker": 2, "text": "<phone_call> Who is it?", "start": 1.0, "end": 3.0},
                {"speaker": 0, "text": "[laughs]", "start": 3.2, "end": 4.0},
                {"speaker": 1, "name": "Leo", "text": "One. Two.", "start": 4.3, "end": 6.0, "lines": [
                    {"text": "One.", "start": 4.3, "end": 5.0}, {"text": "Two.", "start": 5.3, "end": 6.0}]},
            ], "sounds": [{"name": "rain", "loop": True, "start": 0.0, "end": 9.5},
                          {"name": "thunder", "loop": False, "start": 1.0, "end": 2.0},
                          {"name": "door <slam>", "loop": False, "start": 2.0, "end": 2.5}]}})
            update = drama3_handlers.drama3_subtitles(str(result))
            write_snapshot(result, {"family": "Drama 3", "settings": {"chunks": [{"speaker": 0, "text": "Old."}]}})
            old = drama3_handlers.drama3_subtitles(str(result))
        self.assertTrue(update["visible"])
        self.assertIn(f'data-media="{"f" * 32}"', update["value"])
        self.assertIn('data-duration="9.500"', update["value"])
        self.assertIn('<li data-start="0.000" data-end="1.250">', update["value"])
        self.assertIn("Mia &amp; Co", update["value"])
        self.assertIn('<span class="h3-subtitle-text">Hello !</span>', update["value"])
        self.assertIn("Speaker 2</span><span class=\"h3-subtitle-text\">Who is it?", update["value"])
        # Characters take the script box's name colors in order of first line.
        self.assertIn('<li data-start="5.300" data-end="6.000"><span class="h3-subtitle-speaker h3-drama3-tone-2">Leo</span>'
                      '<span class="h3-subtitle-text">Two.</span>', update["value"])
        self.assertEqual(update["value"].count("<li"), 4)  # a cue-only line has nothing to show
        # Each line is a timeline segment placed on the 9.5 s mix, and seeks from its start.
        self.assertEqual(update["value"].count('class="h3-drama3-segment h3-drama3-tone-'), 4)
        self.assertIn('data-line="3" data-start="5.300" data-end="6.000" style="left: 55.79%; width: 7.37%; --row: 0"',
                      update["value"])
        # Speaker 2 talks over Mia, so the voices lane takes a second row for them.
        self.assertIn('<div class="h3-drama3-lane" style="--rows: 2">', update["value"])
        self.assertIn('data-line="1" data-start="1.000" data-end="3.000" style="left: 10.53%; width: 21.05%; --row: 1"',
                      update["value"])
        # Sounds get their own lane: the rain bed under everything, and the thunder then the door
        # sharing the second row because one ends as the other starts.
        self.assertIn('<span style="--rows: 2">Voices</span><span style="--rows: 2">Sounds</span>', update["value"])
        self.assertIn('h3-drama3-sound h3-drama3-tone-bed" data-start="0.000" data-end="9.500" '
                      'style="left: 0.00%; width: 100.00%; --row: 0"', update["value"])
        self.assertIn('h3-drama3-tone-cue" data-start="2.000" data-end="2.500" style="left: 21.05%; width: 5.26%; --row: 1" '
                      'title="door &lt;slam&gt; (one-shot)"', update["value"])
        self.assertIn('title="Mia &amp; Co: Hello !"', update["value"])
        # Three characters spread across the stage as the mix places them.
        self.assertIn('h3-drama3-tone-0" data-row="0" style="left: 35.0%">Mia &amp; Co</span>', update["value"])
        self.assertIn('h3-drama3-tone-1" data-row="1" style="left: 65.0%">Speaker 2</span>', update["value"])
        self.assertIn('h3-drama3-tone-2" data-row="0" style="left: 42.5%">Leo</span>', update["value"])
        self.assertFalse(old["visible"])  # results from before line timings were recorded
        self.assertFalse(drama3_handlers.drama3_subtitles(None)["visible"])

    def test_takes_and_settings_are_found_from_gradios_cached_copy(self):
        import shutil
        from h3_ui import application
        from h3_app.provenance import write_snapshot

        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "output"
            result = outputs / "drama3" / "Drama3_sglang_20261020-030937_0123456789abcdef0123456789abcdef.mp3"
            result.parent.mkdir(parents=True)
            result.write_bytes(b"mp3")
            write_snapshot(result, {"family": "Drama 3", "settings": {"chunks": [
                {"speaker": 0, "name": "Alice", "text": "Hello.", "take": "k1", "reused": False,
                 "check": "skipped", "seconds": 1.0}]}})
            # Gradio passes listeners a copy in its cache, without the snapshot beside it.
            cached = Path(temp) / "gradio" / "f00d" / result.name
            cached.parent.mkdir(parents=True)
            shutil.copyfile(result, cached)
            for other_root in (Path(temp) / "outputs", outputs, outputs / ".." / "output", Path(temp)):
                with self.subTest(other_root=other_root), \
                        patch.object(application, "OUTPUT_DIR", outputs), \
                        patch.object(application, "OUTPUTS_DIR", other_root):
                    self.assertEqual(application.H3.resolve_result_path(str(cached)), result)
                    table, _line, actions = application.drama3_handlers.drama3_takes(str(cached))
                    self.assertEqual(table["value"][0][1], "Alice")
                    self.assertTrue(actions["visible"])
                    self.assertIn("Alice", application.H3.result_settings_for_media(str(cached)))

            # Two distinct outputs with the same name must still be considered ambiguous.
            other = Path(temp) / "outputs" / result.name
            other.parent.mkdir()
            shutil.copyfile(result, other)
            write_snapshot(other, {"family": "Drama 3", "settings": {}})
            with patch.object(application, "OUTPUT_DIR", outputs), \
                    patch.object(application, "OUTPUTS_DIR", other.parent):
                self.assertEqual(application.H3.resolve_result_path(str(cached)), str(cached))


class Drama3VramHandoffTests(unittest.TestCase):
    def run_release(self, allocations, timeout=60):
        from h3_ui import application

        clock = [0.0]
        response = Mock()
        response.json.side_effect = [{"devices": [{"type": "cuda", "torch_vram_total": allocation,
                                                   "vram_free": 50 * 2**30}]} for allocation in allocations]
        with patch.object(application, "unload_prompt_rewriter"), \
                patch.object(application, "unload_comfy_models") as unload, \
                patch.object(application, "api_get", return_value=response) as get, \
                patch.object(application.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(application.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            application.free_comfy_vram(timeout=timeout)
        unload.assert_called_once()
        return get.call_count

    def test_unchanged_free_vram_does_not_acknowledge_a_pending_unload(self):
        self.assertEqual(self.run_release([40 * 2**30] * 5 + [0, 0]), 7)

    def test_incomplete_unload_times_out_instead_of_starting_sglang(self):
        with self.assertRaisesRegex(H3Error, "40.00 GiB still reserved"):
            self.run_release([40 * 2**30] * 3, timeout=3)

    def test_handlers_release_comfyui_through_the_application(self):
        from h3_ui import application

        drama3_handlers = application.drama3_handlers

        with patch.object(application, "free_comfy_vram") as free:
            drama3_handlers.free_comfy_vram()
        free.assert_called_once_with()


class SoundPresetTests(unittest.TestCase):
    def setUp(self):
        from h3_app.drama3 import sounds as drama3_sounds

        self.drama3_sounds = drama3_sounds
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        patcher = patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(root / "workspace")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runtime = RuntimeConfig(root, "http://x", root / "ComfyUI", root / "m.json", root / "outputs")
        self.clip = root / "thunder.wav"
        self.clip.write_bytes(b"RIFF....WAVE")

    def test_save_reuse_replace_and_delete_sound_preset(self):
        saved = self.drama3_sounds.save_sound(
            "  Heavy   Thunder ", str(self.clip), "Loud thunderclap in a storm, 4 seconds", 4, self.runtime,
        )
        self.assertEqual(saved.name, "Heavy Thunder")
        self.assertEqual(saved.duration, 4)
        self.assertTrue(saved.audio.is_file())
        self.assertTrue(saved.audio.is_relative_to(Path(self.temp.name) / "workspace" / "drama3_sounds"))
        self.drama3_sounds.save_sound(
            "heavy thunder", str(self.clip), "Updated storm rumble, 6 seconds", 6, self.runtime,
        )
        self.assertEqual(self.drama3_sounds.sound_names(self.runtime), ["heavy thunder"])
        loaded = self.drama3_sounds.load_sound("HEAVY THUNDER", self.runtime)
        self.assertEqual(loaded.description, "Updated storm rumble, 6 seconds")
        self.assertEqual(loaded.duration, 6)
        self.drama3_sounds.delete_sound("Heavy Thunder", self.runtime)
        self.assertEqual(self.drama3_sounds.sound_names(self.runtime), [])
        with self.assertRaises(H3Error):
            self.drama3_sounds.load_sound("Heavy Thunder", self.runtime)

    def test_failed_replacement_keeps_the_previous_preset(self):
        self.drama3_sounds.save_sound("Rain", str(self.clip), "Light rain", 5, self.runtime)
        real_rename = Path.rename

        def rename(path, target):
            if path.name.startswith(".saving-"):
                raise PermissionError("clip is open")
            return real_rename(path, target)

        with patch.object(Path, "rename", rename), self.assertRaises(PermissionError):
            self.drama3_sounds.save_sound("Rain", str(self.clip), "Heavy rain", 8, self.runtime)
        kept = self.drama3_sounds.load_sound("Rain", self.runtime)
        self.assertEqual((kept.description, kept.duration), ("Light rain", 5))
        self.assertEqual(self.drama3_sounds.sound_names(self.runtime), ["Rain"])


class Drama3CastSoundTests(unittest.TestCase):
    def parse(self, sounds, script='<sound loop="true" name="rain">Rain</sound>\n<|speaker:0|> Hello there.'):
        import json
        from h3_app.drama3.cast import parse_script_cast_and_sounds

        return parse_script_cast_and_sounds(json.dumps({
            "script": script,
            "speakers": [{"speaker_id": 0, "name": "Ann", "description": "Warm alto.",
                          "preview_text": "Hello there."}],
            "sounds": sounds,
        }))

    def test_unparsable_numbers_fall_back_instead_of_crashing(self):
        _script, _cast, sounds = self.parse([
            {"name": "rain", "duration": "10s"},
            {"name": "thunder", "description": "Thunderclap", "duration": None, "volume": "loud"},
            {"name": "music", "description": "Cello drone", "loop": True, "volume": 9, "duration": 900},
        ])
        by_name = {item["name"]: item for item in sounds}
        self.assertEqual(by_name["rain"]["duration"], 10)
        self.assertEqual((by_name["thunder"]["duration"], by_name["thunder"]["volume"]), (5, 1.0))
        self.assertEqual((by_name["music"]["duration"], by_name["music"]["volume"]), (47, 2.0))

    def test_writer_briefs_are_written_into_the_script_tags(self):
        from h3_app.drama3.serving.tree import extract_script_sounds

        script, _cast, _sounds = self.parse([
            {"name": "rain", "description": "Steady rain on a <tin> roof", "duration": 20},
        ])
        self.assertIn('<sound name="rain" loop="true" duration="20">Steady rain on a tin roof</sound>', script)
        (rain,) = extract_script_sounds(script)
        self.assertEqual((rain["description"], rain["duration"], rain["loop"]),
                         ("Steady rain on a tin roof", 20, True))

    def test_writer_briefs_keep_a_sounds_placement(self):
        script, _cast, _sounds = self.parse(
            [{"name": "steps", "description": "Footsteps approaching", "duration": 4}],
            script='<sound name="steps" pan="-0.8,0" distance="far">Steps</sound>\n<|speaker:0|> Hello there.')
        self.assertIn('<sound name="steps" duration="4" pan="-0.8,0" distance="far">Footsteps approaching</sound>',
                      script)

    def test_script_is_unchanged_without_matching_briefs(self):
        script, _cast, _sounds = self.parse([{"name": "other", "description": "Wind"}])
        self.assertIn('<sound loop="true" name="rain">Rain</sound>', script)

    def test_unnamed_sound_slot_matches_an_unnamed_tag(self):
        from h3_app.drama3.serving.tree import parse_drama3_tree, iter_sound_nodes
        from h3_ui.application import drama3_handlers
        from h3_ui.drama3.controls import UPLOAD_SOUND

        (node,) = iter_sound_nodes(parse_drama3_tree("<sound>Distant church bells</sound>"))
        (sound,) = drama3_handlers._drama3_sounds((UPLOAD_SOUND, None, "", "Distant church bells", 90))
        self.assertEqual(sound, (node.name, "", "Distant church bells", 47))


class Drama3SoundAndEffectValidationTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(prompt_systems={"Drama 3": ROOT / "prompts" / "prompt_drama3.txt"})

    def test_sound_and_effect_tags_validate_in_both_script_modes(self):
        voices = ((0, "a.wav", "Ref 0."),)
        # Basic mode accepts tags left in the text; generation ignores them
        for text in (
            '<sound name="rain">Steady rain, 4 seconds</sound>\n<|speaker:0|> Hello.',
            "<parallel>\n<|speaker:0|> Hello.\n</parallel>",
            "<|speaker:0|> <#1.5#> Hello.",
            "<|speaker:0|> <phone_call> Hello.",
        ):
            with self.subTest(text=text):
                validate_request(Drama3Request(text, voices=voices))
        validate_request(Drama3Request(
            '<parallel>\n<sound loop="true" volume="0.2" name="bgm">Dark synth drone, 15 seconds</sound>\n'
            '<sequential>\n<|speaker:0|> <phone_call> [whispering] Hello? <#1.0#> Are you there?\n</sequential>\n</parallel>',
            voices=voices,
            script_mode=drama3_writer.DRAMA3_SCRIPT_ADVANCED,
        ))

    def test_basic_mode_strips_sound_and_speaker_effect_tags_while_advanced_keeps_them(self):
        import json

        raw_script = (
            "<parallel>\n"
            '<sound loop="true" volume="0.2" name="rain_ambience">Steady rain on window, 12 seconds</sound>\n'
            "<sequential>\n"
            "<|speaker:0|> <phone_call> [anxious] Hello? <#1.0#> Can you hear me?\n"
            '<sound name="thunder">Loud thunderclap, 3 seconds</sound>\n'
            "<|speaker:1|> <robotic_voice> Affirmative.\n"
            "</sequential>\n"
            "</parallel>"
        )
        reply = json.dumps({
            "speakers": [
                {"speaker_id": 0, "name": "Alice", "description": "Warm voice.", "preview_text": "One two three four five six."},
                {"speaker_id": 1, "name": "Unit 7", "description": "Cold voice.", "preview_text": "One two three four five six."},
            ],
            "script": raw_script,
        })
        basic_script, basic_status, _cast = drama3_writer.apply_drama3_reply(
            reply, "Alice calls Unit 7.", (), "Auto", script_mode=drama3_writer.DRAMA3_SCRIPT_BASIC,
        )
        self.assertFalse(basic_status.startswith("Prompt enhancement failed:"), basic_status)
        self.assertNotIn("<parallel>", basic_script)
        self.assertNotIn("<sequential>", basic_script)
        self.assertNotIn("<sound", basic_script)
        self.assertNotIn("<#1.0#>", basic_script)
        self.assertNotIn("<phone_call>", basic_script)
        self.assertNotIn("<robotic_voice>", basic_script)
        self.assertIn("<|speaker:0|> Hello? [anxious] Can you hear me?", basic_script)  # the opening cue moves in
        self.assertIn("<|speaker:1|> Affirmative.", basic_script)

        adv_script, adv_status, _cast = drama3_writer.apply_drama3_reply(
            reply, "Alice calls Unit 7.", (), "Auto", script_mode=drama3_writer.DRAMA3_SCRIPT_ADVANCED,
        )
        self.assertFalse(adv_status.startswith("Prompt enhancement failed:"), adv_status)
        self.assertIn("<parallel>", adv_script)
        self.assertIn('<sound loop="true" volume="0.2" name="rain_ambience">', adv_script)
        self.assertIn("<phone_call>", adv_script)
        self.assertIn("<robotic_voice>", adv_script)
        self.assertIn("<#1.0#>", adv_script)

    def test_switch_script_mode_and_sync_script_sounds(self):
        from h3_ui.application import drama3_handlers

        text = (
            "<parallel>\n"
            '<sound loop="true" volume="0.2" name="cafe_bgm">Warm jazz piano, 15 seconds</sound>\n'
            "<sequential>\n"
            "<|speaker:0|> <phone_call> Hi! <#1.0#> Meet me inside.\n"
            '<sound name="door_bell">Shop doorbell chime, 2 seconds</sound>\n'
            "</sequential>\n"
            "</parallel>"
        )
        # Switching to Advanced reveals the sounds panel
        panel_upd, status = drama3_handlers.switch_drama3_script_mode(drama3_writer.DRAMA3_SCRIPT_ADVANCED, text)
        self.assertTrue(panel_upd["visible"])
        self.assertIn("Advanced", status)

        # Syncing script sounds populates slot metadata
        meta = ["", "", 5] * 16
        synced = drama3_handlers.sync_drama3_script_sounds(
            text, drama3_writer.DRAMA3_SCRIPT_ADVANCED, 4, *meta
        )
        self.assertEqual(list(synced[0:3]), ["cafe_bgm", "Warm jazz piano, 15 seconds", 15])
        self.assertEqual(list(synced[3:6]), ["door_bell", "Shop doorbell chime, 2 seconds", 2])

        # Switching back to Basic keeps the authored tags (they are ignored when generating)
        basic_panel_upd, basic_status = drama3_handlers.switch_drama3_script_mode(
            drama3_writer.DRAMA3_SCRIPT_BASIC, text
        )
        self.assertFalse(basic_panel_upd["visible"])
        self.assertIn("kept but ignored", basic_status)

    def test_batch_sounds_report_clips_that_were_not_returned(self):
        from h3_ui.application import drama3_handlers
        from h3_ui.drama3.controls import UPLOAD_SOUND

        with tempfile.TemporaryDirectory() as temp:
            clip = Path(temp) / "rain.wav"
            clip.write_bytes(b"RIFF....WAVE")
            slots = [UPLOAD_SOUND, False, "", "", 5] * 16
            slots[0:5] = [UPLOAD_SOUND, False, "rain", "Steady rain", 4]
            slots[5:10] = [UPLOAD_SOUND, False, "bell", "Church bell", 3]
            with patch("h3_app.stable_audio.generate_sound_references", return_value=[clip, None]), \
                    patch.object(drama3_handlers, "_runtime_config"), patch("h3_models.resolve_hf_token"), \
                    patch.object(drama3_handlers, "free_comfy_vram"):
                frames = list(drama3_handlers.generate_drama3_sounds(2, True, *slots))
        final = frames[-1]
        self.assertEqual(final[0], str(clip))
        self.assertIn("ready to save", final[1])
        self.assertIn("failed", final[3])
        self.assertEqual(final[-1], "Generated 1 of 2 sound clips. Failed: sound 1; see each sound's panel.")

    def test_job_admission_signatures_match_bound_inputs(self):
        from h3_ui.application import drama3_handlers
        from h3_ui.job_bindings import owned_generation

        gen_drama = owned_generation(drama3_handlers.generate_drama3, "drama3")
        # 1 text + 48 speaker inputs + 6 settings + 1 script_mode + 80 sound slot inputs + 1 title
        # + 16 speaker names + 1 word_timing = 154
        self.assertEqual(len(gen_drama.job_input_names), 154)
        gen_sounds = owned_generation(drama3_handlers.generate_drama3_sounds, "drama3_sound")
        # 1 shown_sound_slots + 1 skip_existing_sounds + 80 sound slot inputs = 82
        self.assertEqual(len(gen_sounds.job_input_names), 82)
        self.assertEqual(len(owned_generation(drama3_handlers.remix_drama3, "drama3").job_input_names), 154)
        self.assertEqual(len(owned_generation(drama3_handlers.regenerate_drama3_line, "drama3").job_input_names), 155)
        gen_sound = owned_generation(drama3_handlers.generate_drama3_sound, "drama3_sound")
        self.assertEqual(len(gen_sound.job_input_names), 2)


class Drama3AutoTests(unittest.TestCase):
    def test_composer_and_auto_bindings_can_be_mounted(self):
        import gradio as gr
        from h3_ui.application import _drama3_actions
        from h3_ui.drama3.view import build_drama3_view
        from h3_ui.drama3.bindings import bind_drama3

        with gr.Blocks() as demo:
            view = build_drama3_view(gr.Group())
            bind_drama3(view, _drama3_actions())
        demo.queue()
        # In particular, Auto must not register cancellation of unqueued callbacks.
        demo.validate_queue_settings()
        self.assertEqual(len(view.voices), 16)
        self.assertEqual(len(view.sounds), 16)
        # Auto defaults to sound-enhanced, expressive speech and leaves speaker saving opt-in.
        self.assertEqual(view.script_mode.value, "Advanced (sound enhanced)")
        self.assertFalse(hasattr(view, "backend"))
        self.assertEqual((view.temperature.value, view.top_p.value, view.top_k.value), (2.0, 1.0, 30))
        self.assertTrue(view.sounds_panel.visible)
        self.assertFalse(view.auto_save_speakers.value)
        # Auto only disables and re-enables its button; the label never changes, so it cannot stick.
        import inspect

        label_writers = []
        for fn in demo.fns.values():
            outputs = [block._id for block in fn.outputs]
            if view.auto._id not in outputs or fn.inputs or not callable(fn.fn) or any(
                    parameter.default is parameter.empty
                    for parameter in inspect.signature(fn.fn).parameters.values()):
                continue
            update = fn.fn()[outputs.index(view.auto._id)]
            label_writers += ["value" in update]
        self.assertTrue(label_writers)
        self.assertFalse(any(label_writers))
        from h3_app.jobs import JOBS
        self.assertEqual(JOBS.callbacks["drama3_words"].__name__, "repair_drama3_words")
        self.assertEqual(JOBS.callbacks["drama3_words"].job_input_names, ("output",))
        self.assertIs(view.repair_words.h3_stop, view.stop_repair_words)
        config = demo.get_config_file()
        self.assertTrue(any((view.repair_words._id, "click") in [tuple(t) for t in dep["targets"]]
                            for dep in config["dependencies"]))
        self.assertTrue(any(fn.outputs == [view.subtitles, view.alignment_status] for fn in demo.fns.values()))

    def test_soundstage_is_four_steps_and_auto_on_script_presses_auto_on_speak(self):
        import gradio as gr
        from h3_ui.application import _drama3_actions
        from h3_ui.drama3.view import AUTO_ID, build_drama3_view
        from h3_ui.drama3.bindings import bind_drama3

        with gr.Blocks() as demo:
            view = build_drama3_view(gr.Group())
            bind_drama3(view, _drama3_actions())
        self.assertEqual([tab.label for tab in view.steps.tabs.children],
                         ["Script", "Cast & sounds", "Speak", "Review lines"])
        self.assertEqual(view.auto.elem_id, AUTO_ID)
        # Step buttons only run browser code, so moving between steps sends no request.
        auto_shortcuts = [fn for fn in demo.fns.values() if fn.js and AUTO_ID in fn.js]
        self.assertEqual(len(auto_shortcuts), 1)
        self.assertIsNone(auto_shortcuts[0].fn)
        self.assertFalse(view.take_actions.visible)

    def test_failed_rewrite_stops_auto_with_the_writer_reason(self):
        import gradio as gr
        from h3_ui.drama3.bindings import drama3_auto_rewritten

        with self.assertRaisesRegex(gr.Error, "no key"):
            drama3_auto_rewritten("Prompt enhancement failed: no key")
        self.assertIn("voices", drama3_auto_rewritten("Rewrote the script."))

    def test_preload_runs_on_the_gpu_queue_beside_the_rewrite(self):
        import gradio as gr
        from h3_ui.application import _drama3_actions
        from h3_ui.drama3.view import build_drama3_view
        from h3_ui.drama3.bindings import bind_drama3

        with gr.Blocks() as demo:
            view = build_drama3_view(gr.Group())
            bind_drama3(view, _drama3_actions())
        preloads = [fn for fn in demo.fns.values()
                    if [block._id for block in fn.outputs] == [view.auto_preload_status._id]]
        self.assertEqual(len(preloads), 1)
        self.assertEqual(preloads[0].concurrency_id, "h3-gpu")
        self.assertEqual([block._id for block in preloads[0].inputs], [view.voice_engine._id, view.script_mode._id])

    def test_preload_loads_the_voice_engine_then_sounds_then_sglang_and_survives_failures(self):
        from h3_app import audio_tools
        from h3_app.drama3.serving import generate
        from h3_app.drama3.voice_design import ENGINE_DRAMA3, ENGINE_QWEN
        from h3_app.generation.results import GenerationUpdate
        from h3_app.jobs import JobCancelled
        from h3_ui.application import drama3_handlers

        order = []

        def preload(model, *_args, **_kwargs):
            order.append(model)
            return model != "stable_audio"

        def served(**_kwargs):
            order.append("sglang")
            yield GenerationUpdate(None, "Starting SGLang-Omni (3s)")
            raise H3Error("SGLang-Omni needs about 50 GiB\nlast log lines")

        def run(engine, mode):
            order.clear()
            with patch.object(audio_tools, "preload", side_effect=preload), \
                    patch.object(generate, "preload_served", side_effect=served), \
                    patch.object(drama3_handlers, "_runtime_config"), patch("h3_models.resolve_hf_token"):
                return list(drama3_handlers.preload_drama3_models(engine, mode))

        frames = run(ENGINE_QWEN, drama3_writer.DRAMA3_SCRIPT_ADVANCED)
        self.assertEqual(order, ["voice_design", "stable_audio", "sglang"])
        self.assertTrue(any("Preloading SGLang-Omni: Starting SGLang-Omni (3s)" in frame for frame in frames))
        self.assertEqual(frames[-1], "Preload finished. Ready: Qwen VoiceDesign. Not preloaded, its step loads it: "
                                     "Stable Audio 3 Medium (no free GPU memory beside the loaded models); "
                                     "SGLang-Omni (SGLang-Omni needs about 50 GiB).")
        # Drama 3 designs voices on SGLang-Omni, so it loads first; Basic has no sounds.
        run(ENGINE_DRAMA3, drama3_writer.DRAMA3_SCRIPT_ADVANCED)
        self.assertEqual(order, ["sglang", "stable_audio"])
        run(ENGINE_QWEN, drama3_writer.DRAMA3_SCRIPT_BASIC)
        self.assertEqual(order, ["voice_design", "sglang"])
        with patch.object(audio_tools, "preload", side_effect=JobCancelled("cancelled")), \
                patch.object(drama3_handlers, "_runtime_config"), patch("h3_models.resolve_hf_token"), \
                self.assertRaises(JobCancelled):
            list(drama3_handlers.preload_drama3_models(ENGINE_QWEN))

    def test_only_advanced_generates_sounds(self):
        from h3_ui.drama3.bindings import drama3_auto_sound_slots

        self.assertEqual(drama3_auto_sound_slots(drama3_writer.DRAMA3_SCRIPT_BASIC, 4)[0], 0)
        self.assertEqual(drama3_auto_sound_slots(drama3_writer.DRAMA3_SCRIPT_ADVANCED, 4)[0], 4)


if __name__ == "__main__":
    unittest.main()

