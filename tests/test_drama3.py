"""Drama 3 scripts, cloned voices, speaker presets and Qwen3-ASR transcripts."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app import drama3_speakers, prompt_service
from h3_app.catalog import LIGHTNING_PROMPT_MODELS
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.generation.optional_engines import transcribe_drama3_audio, validate_request
from h3_app.generation.requests import Drama3Request
from h3_app.workflows.drama3 import (
    build_drama3_asr_graph,
    build_drama3_graph,
    speaker_transcript,
    text_speakers,
    validate_drama3_text,
)

ROOT = Path(__file__).resolve().parents[1]


def completion(text):
    return Mock(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class Drama3GraphTests(unittest.TestCase):
    def test_each_voice_is_encoded_with_its_speaker_tag(self):
        graph = build_drama3_graph(text="<|speaker:0|> Hi\n<|speaker:2|> Yo",
                                   voices=((0, "a.wav", "Hello there."), (2, "b.wav", "<|speaker:7|> Hey.")))
        encoders = [n for n in graph.values() if n["class_type"] == "Drama3EncodeVoice"]
        self.assertEqual([n["inputs"]["transcript"] for n in encoders],
                         ["<|speaker:0|> Hello there.", "<|speaker:2|> Hey."])
        generate = next(n for n in graph.values() if n["class_type"] == "Drama3Generate")
        self.assertEqual(sorted(k for k in generate["inputs"] if k.startswith("voices.")),
                         ["voices.voice0", "voices.voice1"])

    def test_without_voices_the_model_casts_from_the_seed(self):
        graph = build_drama3_graph(text="Hello")
        self.assertFalse(any(n["class_type"] == "Drama3EncodeVoice" for n in graph.values()))

    def test_text_validation_matches_the_node_pack(self):
        self.assertEqual(validate_drama3_text(" <|speaker:15|> [sighs] fine "), "<|speaker:15|> [sighs] fine")
        for bad in ("", "<|speaker:16|> x", "<|speaker:01|> x", "<|audio_separator|> x", "<|speaker:0 x"):
            with self.subTest(bad=bad), self.assertRaises(H3Error):
                validate_drama3_text(bad)

    def test_text_speakers_treats_untagged_text_as_speaker_zero(self):
        self.assertEqual(text_speakers("hello"), {0})
        self.assertEqual(text_speakers("<|speaker:1|> a\n<|speaker:3|> b"), {1, 3})
        self.assertEqual(speaker_transcript(3, "  words "), "<|speaker:3|> words")

    def test_request_validation_for_voice_slots(self):
        validate_request(Drama3Request("<|speaker:0|> a <|speaker:1|> b",
                                       voices=((0, "a", "t"), (1, "b", "u"))))
        for voices in (((0, "a", ""),), ((0, "a", "t"), (0, "b", "t")), ((16, "a", "t"),), ((0, "", "t"),)):
            with self.subTest(voices=voices), self.assertRaises(H3Error):
                validate_request(Drama3Request("hi", voices=voices))


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
        self.runtime = SimpleNamespace(prompt_systems={"Drama 3": ROOT / "prompt_drama3.txt"})

    @patch("openai.OpenAI")
    def test_rewrite_sends_voice_context_and_selected_lightning_model(self, client):
        client.return_value.chat.completions.create.return_value = completion(
            "<|speaker:0|> [warm] Welcome back.\n<|speaker:1|> [laughs] Finally!"
        )
        script, status = prompt_service.enhance_drama3_text(
            "ALICE: Welcome back.\nBOB: Finally!", "unused", "", [(0, "Alice")],
            "Lightning AI", "key", LIGHTNING_PROMPT_MODELS[1], runtime=self.runtime,
        )
        self.assertTrue(script.startswith("<|speaker:0|>"))
        self.assertIn(LIGHTNING_PROMPT_MODELS[1], status)
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "openai/gpt-6.1-sol")
        self.assertIn("<|speaker:N|>", request["messages"][0]["content"])
        user = request["messages"][1]["content"][0]["text"]
        self.assertIn('<|speaker:0|> uses a cloned reference voice saved as "Alice"', user)
        self.assertIn("ALICE: Welcome back.", user)

    @patch("openai.OpenAI")
    def test_invalid_scripts_keep_the_original_text(self, client):
        client.return_value.chat.completions.create.return_value = completion(
            "<|speaker:0|> Hi<|audio_separator|><|speaker:1|> Yo"
        )
        script, status = prompt_service.enhance_drama3_text(
            "hi", "unused", "", (), "Lightning AI", "key", runtime=self.runtime,
        )
        self.assertEqual(script, "hi")
        self.assertIn("audio_separator", status)

    def test_unknown_lightning_model_is_rejected(self):
        script, status = prompt_service.enhance_drama3_text(
            "hi", "unused", "", (), "Lightning AI", "key", "openai/unknown", runtime=self.runtime,
        )
        self.assertEqual(script, "hi")
        self.assertIn("Unsupported Lightning AI prompt model", status)


if __name__ == "__main__":
    unittest.main()
