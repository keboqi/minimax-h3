"""Voice conditioning, input isolation, and backwards-compatible API routing."""
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio_app as app
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments
from h3_app.settings import GenerationRequest, resolve_settings
from h3_ui.persistence import PERSISTED_NAMES

T8 = "MiniMaxH3AudioConditioningT8"


class Fl2vaVoiceTests(unittest.TestCase):
    def graph(self, voices=(), upscale=False, available=None):
        args = {
            name: app.UI_DEFAULTS.get(name, 0)
            for name, p in inspect.signature(app.build_fl2va_graph).parameters.items()
            if p.default is inspect.Parameter.empty
        }
        args.update(
            prompt='The woman uses the voice of <Audio 1> and says, "Hello."',
            first_image="start.png", last_image="end.png", width=864, height=480,
            duration=5, steps=4, seed=7,
            models=SimpleNamespace(text_encoder="encoder.safetensors"),
            available_nodes={T8, "LoadAudio"} if available is None else available,
            voice_reference_audios=list(voices),
            latent_upscale_model_name="upscaler.pth" if upscale else None,
            semantic_bridge=bool(voices),
        )
        with (
            patch.object(app, "stage_file", side_effect=lambda path, *a, **kw: path),
            patch.object(app, "add_model_stack", return_value=(["model", 0], ["clip", 0], ["vae", 0], ["audio", 0])),
            patch.object(app, "finish_sampling") as finish,
        ):
            graph = app.build_fl2va_graph(**args)
        return graph, finish.call_args.kwargs

    def test_voice_graph_retains_keyframes_and_generates_new_audio_in_both_stages(self):
        graph, finish = self.graph(("alice.wav", "bob.wav"), upscale=True)
        hybrids = [n for n in graph.values() if n["class_type"] == T8]
        self.assertEqual(len(hybrids), 2)
        for node in hybrids:
            inputs = node["inputs"]
            self.assertEqual(inputs["task_type"], "Hybrid")
            self.assertEqual(inputs["audio_mode"], "native")
            self.assertFalse(inputs["add_source_as_reference"])
            self.assertNotIn("drive_audio", inputs)
            self.assertNotIn("final_audio", inputs)
            self.assertEqual(inputs["video_vae"], ["vae", 0])
            self.assertEqual(inputs["audio_vae"], ["audio", 0])
            for field in ("first_frame", "last_frame", "ref_audios.ref_audio_1", "ref_audios.ref_audio_2"):
                self.assertIn(field, inputs)
        self.assertFalse(any(n["class_type"] == app.H3_SEMANTIC_BRIDGE_NODE for n in graph.values()))
        self.assertEqual(finish["model_ref"], ["model", 0])
        for field in ("conditioning_ref", "initial_conditioning_ref", "latent_ref", "initial_latent_ref"):
            self.assertEqual(graph[finish[field][0]]["class_type"], T8)
        self.assertEqual(finish["latent_ref"][1], 1)
        self.assertEqual(finish["audio_vae_ref"], ["audio", 0])

    def test_empty_voice_branch_keeps_native_graph_without_t8(self):
        graph, _ = self.graph(available=set())
        self.assertTrue(any(n["class_type"] == "MiniMaxH3ImageToVideo" for n in graph.values()))
        self.assertFalse(any(n["class_type"] in (T8, "LoadAudio") for n in graph.values()))

    def test_audio_change_and_removal_invalidate_cache(self):
        def key(voices):
            graph, _ = self.graph(voices)
            return next(n["inputs"]["cache_key"] for n in graph.values() if n["class_type"] == app.H3_CONDITIONING_CACHE_NODE)
        self.assertEqual(key(("alice.wav",)), key(("alice.wav",)))
        self.assertEqual(len({key(()), key(("alice.wav",)), key(("bob.wav",))}), 3)

    def test_missing_dependency_is_actionable(self):
        with self.assertRaisesRegex(app.H3Error, "Update provisioning"):
            self.graph(("alice.wav",), available=set())

    def test_hidden_voice_slots_are_ignored_and_speaker_ordinals_do_not_shift(self):
        for mode in ("Reference media", "Text to video"):
            self.assertEqual(app.active_fl2va_voice_references(mode, None, "hidden.wav", None), [])
        with self.assertRaisesRegex(app.H3Error, "Fill FL2VA voice slots in order"):
            app.active_fl2va_voice_references("First / last frame", None, "bob.wav", None)
        with patch.object(app, "collect_reference_slots", return_value=["alice.wav"]):
            self.assertEqual(app.active_fl2va_voice_references("First / last frame", "alice.wav", None, None), ["alice.wav"])

    def test_bridge_preference_survives_mode_switches(self):
        values = dict(semantic_bridge=True, fl2va_audio_1="alice.wav")
        plan = resolve_settings(GenerationRequest(mode="First / last frame", **values))
        self.assertFalse(plan.effective.semantic_bridge)
        self.assertTrue(plan.requested.semantic_bridge)
        self.assertTrue(resolve_settings(GenerationRequest(mode="Text to video", **values)).effective.semantic_bridge)
        gap = resolve_settings(GenerationRequest(mode="First / last frame", fl2va_audio_2="bob.wav"))
        self.assertTrue(any("slots in order" in issue for issue in gap.issues))

    def test_old_api_contracts_default_to_no_voice_inputs(self):
        for omitted in (3, 5):
            values = GenerationArguments.from_positional([None] * (len(GENERATION_FIELDS) - omitted)).values
            for i in range(1, 4):
                self.assertIsNone(values[f"fl2va_audio_{i}"])
                self.assertNotIn(f"h3.fl2va_audio_{i}", PERSISTED_NAMES)
        values = {name: None for name in GENERATION_FIELDS}
        values.update(ref_audio_1="ref2va.wav", fl2va_audio_1="fl2va.wav")
        arguments = GenerationArguments.from_positional(tuple(values.values()))
        self.assertEqual(arguments.references["audio"][0], "ref2va.wav")
        self.assertEqual(arguments.with_seed(42).values["fl2va_audio_1"], "fl2va.wav")

    def test_voice_prompt_is_preserved_without_external_enhancement(self):
        args = {name: None for name, p in inspect.signature(app.enhance_h3_prompt).parameters.items() if p.default is inspect.Parameter.empty}
        args.update(prompt="My dialogue <Audio 1>", backend="Gemini", mode="First / last frame", fl2va_audio_1="voice.wav")
        with patch.object(app, "_enhance_h3_prompt_with_gemini") as remote:
            prompt, status = app.enhance_h3_prompt(**args)
        remote.assert_not_called()
        self.assertEqual(prompt, "My dialogue <Audio 1>")
        self.assertIn("kept unchanged", status)


if __name__ == "__main__":
    unittest.main()
