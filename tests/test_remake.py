"""Remake: run rules, graph shape, source trimming, frame alignment and the tab's handlers."""

import subprocess
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate
from h3_app.jobs import CURRENT_JOB
from h3_app.media_tools import probe_video_metadata
from h3_app.remake import (
    EDIT, KEYFRAME_UNTIL, MAX_FRAMES, MIN_FRAMES, REMAKE_NODES, SECTIONS, SIGMAS, SWAP, RemakeGraph, canvas_size,
    check_pins, frames_for_seconds, missing_sections, output_size, request_overrides,
)
from h3_app.remake.model import REQUEST_STEPS
from h3_app.remake.align import align_image, comparison, load_image, measure_drift
from h3_app.remake.media import extract_frame, trim_source
from h3_app.workflows import h3
from h3_models import MODEL_SPECS
from h3_ui.engines.feature import EngineDeps
from h3_ui.engines.remake import READY, RemakeHandlers, next_run_card, readiness_message, selection_text
from h3_ui.h3.handlers import default_request
from tests.workflow_cases import H3_DEFAULTS

SIX_SECTIONS = "\n".join(f"{name}:\ntext" for name in SECTIONS)
OPTIONS = {field.name: H3_DEFAULTS[field.name] for field in fields(h3.H3GraphOptions) if field.name in H3_DEFAULTS}


def nodes(graph, kind):
    return [(node_id, node["inputs"]) for node_id, node in graph.items() if node["class_type"] == kind]


def remake_graph(**options):
    return h3.build_remake_graph(
        prompt=SIX_SECTIONS, source_video="h3_gradio/remake_sources/a.mp4",
        pictures=["h3_gradio/remake_pictures/1.png", "h3_gradio/remake_pictures/2.png"], pins=(12, 184), frames=226,
        lora_name="viggle_remake_swap_comfyui.safetensors", sigmas=SIGMAS[SWAP], keyframe_until=KEYFRAME_UNTIL,
        **{**OPTIONS, "width": 832, "height": 480, "steps": 6, "turbo_lora_name": None, **options},
    )


class RulesTests(unittest.TestCase):
    def test_lengths_are_the_longest_17n_plus_5_that_fit(self):
        self.assertEqual(frames_for_seconds(226 / 24), 226)
        self.assertEqual(frames_for_seconds(8), 192)
        self.assertEqual(frames_for_seconds(8.7), 192)
        self.assertEqual(frames_for_seconds(8.71), 209)
        self.assertEqual(frames_for_seconds(90 / 24), MIN_FRAMES)
        self.assertEqual(frames_for_seconds(60), MAX_FRAMES)
        for seconds in (4, 7.3, 11, 14):
            self.assertEqual((frames_for_seconds(seconds) - 5) % 17, 0)
        with self.assertRaises(H3Error):
            frames_for_seconds(3.7)

    def test_canvas_matches_viggles_sizes_in_either_orientation(self):
        self.assertEqual(canvas_size(480, 1920, 1080), (832, 480))
        self.assertEqual(canvas_size(736, 1920, 1080), (1280, 736))
        self.assertEqual(canvas_size(480, 1080, 1920), (480, 832))
        with self.assertRaises(H3Error):
            canvas_size(500, 1920, 1080)
        self.assertEqual(output_size(832, 480, True), (1664, 960))

    def test_pins_are_all_or_none_and_inside_the_clip(self):
        check_pins((), 3, 226)
        check_pins((0, 225), 2, 226)
        with self.assertRaises(H3Error):
            check_pins((12,), 2, 226)
        with self.assertRaises(H3Error):
            check_pins((226,), 1, 226)

    def test_sections_must_appear_in_order(self):
        self.assertEqual(missing_sections(SIX_SECTIONS), [])
        self.assertEqual(missing_sections("summary:\nx"), [name for name in SECTIONS if name != "summary"])
        self.assertEqual(missing_sections(""), list(SECTIONS))

    def test_overrides_survive_the_h3_settings_resolver(self):
        from h3_app.settings import ResolutionContext, resolve_settings
        from h3_app.settings import GenerationRequest

        for preset in ("Singularity", "Quality", "Balanced", "Fast"):
            for upscale in (False, True):
                with self.subTest(preset=preset, upscale=upscale):
                    values = default_request("")
                    values.update(preset=preset, latent_upscale=upscale, generation_mode="Turbo")
                    values.update(request_overrides(prompt=SIX_SECTIONS, first_picture="a.png", width=832, height=480,
                                                    frames=226, latent_upscale=upscale))
                    plan = resolve_settings(GenerationRequest.from_values(values),
                                            ResolutionContext(None, "unknown"))
                    self.assertEqual(plan.issues, ())
                    effective = plan.effective
                    self.assertEqual((effective.output.width, effective.output.height),
                                     (1664, 960) if upscale else (832, 480))
                    self.assertEqual(effective.sampling.steps, REQUEST_STEPS)
                    self.assertFalse(effective.semantic_bridge)

    def test_loras_are_pinned_lazily(self):
        from h3_models import LAZY_OPTIONAL_MODEL_KEYS

        for mode in (SWAP, EDIT):
            spec = MODEL_SPECS[RemakeGraph(mode, "", (), (), 226).lora_key]
            self.assertEqual(spec.revision, "21665fa7a1e038e0a5312fb15f37b18d0f06f117")
            self.assertEqual(len(spec.expected_sha256), 64)
        self.assertIn("remake_swap_lora", LAZY_OPTIONAL_MODEL_KEYS)
        self.assertIn("remake_edit_lora", LAZY_OPTIONAL_MODEL_KEYS)


class GraphTests(unittest.TestCase):
    def test_one_pass_matches_viggles_workflow(self):
        graph = remake_graph()
        (_, lora), = nodes(graph, "LoraLoaderBypassModelOnly")
        self.assertEqual((lora["lora_name"], lora["strength_model"]), ("viggle_remake_swap_comfyui.safetensors", 1.0))
        (_, sigmas), = nodes(graph, "ManualSigmas")
        self.assertEqual(sigmas["sigmas"], SIGMAS[SWAP])
        (_, sampler), = nodes(graph, "KSamplerSelect")
        self.assertEqual(sampler["sampler_name"], "euler")
        (_, keyframes), = nodes(graph, "RemakeKeyframeUntil")
        self.assertEqual(keyframes["keyframe_until"], 0.9)
        (condition_id, condition), = nodes(graph, "RemakeConditioning")
        self.assertEqual((condition["pins"], condition["frames"], condition["short_edge"]), ("12, 184", 226, 480))
        self.assertIn("ref_2", condition)
        self.assertNotIn("ref_3", condition)
        (noise_id, noise), = nodes(graph, "RemakeNoise")
        self.assertEqual(noise["positive"], [condition_id, 0])
        (_, run), = nodes(graph, "SamplerCustomAdvanced")
        self.assertEqual(run["noise"], [noise_id, 0])
        self.assertEqual(run["latent_image"], [condition_id, 1])
        self.assertEqual(nodes(graph, "RandomNoise"), [])
        self.assertEqual(nodes(graph, "RemakeHoldAudio"), [])
        video_filename = next(inputs for _, inputs in nodes(graph, "H3SaveVideoNVENC"))["filename_prefix"]
        self.assertTrue(video_filename.startswith("h3/remake_"))

    def test_no_turbo_adapter_can_join_the_stack(self):
        classes = {node["class_type"] for node in remake_graph().values()}
        self.assertNotIn("H3LightX2VBypassLoRA", classes)
        self.assertNotIn("MiniMaxH3SigmaShift", classes)

    def test_graph_only_uses_declared_or_regular_h3_nodes(self):
        classes = {node["class_type"] for node in remake_graph(
            width=1664, height=960, latent_upscale_model_name="upscaler.safetensors"
        ).values()}
        self.assertLessEqual({name for name in classes if name.startswith("Remake")}, REMAKE_NODES)

    def test_latent_upscale_rebuilds_conditioning_and_holds_the_audio(self):
        graph = remake_graph(width=1664, height=960, latent_upscale_model_name="upscaler.safetensors",
                             latent_upscale_refine_steps=2)
        conditions = nodes(graph, "RemakeConditioning")
        self.assertEqual(sorted(inputs["short_edge"] for _, inputs in conditions), [480, 960])
        by_edge = {inputs["short_edge"]: (node_id, inputs) for node_id, inputs in conditions}
        (scale_id, scale), = nodes(graph, "ImageScale")
        self.assertEqual((scale["width"], scale["height"]), (1664, 960))
        self.assertEqual(by_edge[960][1]["source"], [scale_id, 0])
        self.assertEqual(by_edge[480][1]["pins"], by_edge[960][1]["pins"])
        (hold_id, _), = nodes(graph, "RemakeHoldAudio")
        (first, second) = [inputs for _, inputs in nodes(graph, "SamplerCustomAdvanced")]
        self.assertEqual(first["latent_image"], [by_edge[480][0], 1])
        self.assertEqual(second["latent_image"], [hold_id, 0])
        self.assertEqual(first["noise"], second["noise"])
        (_, split), = nodes(graph, "SplitSigmas")
        self.assertEqual(split["step"], 4)

    def test_refinement_stays_a_tail_of_the_schedule(self):
        graph = remake_graph(width=1664, height=960, latent_upscale_model_name="upscaler.safetensors",
                             latent_upscale_refine_steps=6)
        (_, split), = nodes(graph, "SplitSigmas")
        self.assertEqual(split["step"], 1)

    def test_changed_inputs_change_the_conditioning_key(self):
        def key(**changes):
            graph = h3.build_remake_graph(**{
                "prompt": "p", "source_video": "a.mp4", "pictures": ["1.png"], "pins": (), "frames": 226,
                "lora_name": "x", "sigmas": SIGMAS[EDIT], "keyframe_until": 0.9,
                **OPTIONS, "width": 832, "height": 480, "turbo_lora_name": None, **changes,
            })
            return nodes(graph, "H3ConditioningCache")[0][1]["cache_key"]

        self.assertNotEqual(key(), key(pictures=["2.png"]))
        self.assertNotEqual(key(), key(frames=243))


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.video = cls.root / "source.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30:duration=6",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest", str(cls.video)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_trim_writes_exact_frames_on_the_canvas_with_audio(self):
        output = trim_source(self.video, self.root / "clip.mp4", start=0.5, frames=107, width=832, height=480)
        metadata = probe_video_metadata(output)
        self.assertEqual((metadata.width, metadata.height, metadata.frame_count), (832, 480, 107))
        self.assertTrue(metadata.has_audio)
        self.assertAlmostEqual(metadata.fps, 24, places=2)

    def test_trim_past_the_end_says_how_many_frames_exist(self):
        with self.assertRaisesRegex(H3Error, "only has"):
            trim_source(self.video, self.root / "short.mp4", start=4, frames=107, width=832, height=480)

    def test_extract_writes_the_requested_frame(self):
        output = extract_frame(self.video, 1.0, self.root / "frame.png")
        with Image.open(output) as image:
            self.assertEqual(image.size, (640, 480))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(3)
        noise = np.kron(random.integers(0, 255, (45, 80, 3), dtype=np.uint8), np.ones((8, 8, 1), dtype=np.uint8))
        self.source = Image.fromarray(noise)

    def test_finds_and_undoes_a_global_shift(self):
        edited = Image.fromarray(np.roll(np.asarray(self.source), (6, -14), axis=(0, 1)))
        drift = measure_drift(self.source, edited)
        self.assertLessEqual(abs(drift.dx - 14), 3)
        self.assertLessEqual(abs(drift.dy + 6), 3)
        self.assertEqual(drift.zoom, 1.0)
        self.assertTrue(drift.visible)
        aligned = np.asarray(align_image(self.source, edited, drift), dtype=np.float32)
        before = np.abs(np.asarray(edited, dtype=np.float32) - np.asarray(self.source, dtype=np.float32)).mean()
        self.assertLess(np.abs(aligned - np.asarray(self.source, dtype=np.float32)).mean(), before / 2)

    def test_an_unrelated_image_does_not_report_a_shift(self):
        other = Image.fromarray(np.random.default_rng(9).integers(0, 255, (360, 640, 3), dtype=np.uint8))
        self.assertFalse(measure_drift(self.source, other).plausible(*self.source.size))

    def test_an_unchanged_frame_reads_as_lined_up(self):
        self.assertFalse(measure_drift(self.source, self.source.copy()).visible)

    def test_finds_a_small_zoom(self):
        width, height = self.source.size
        enlarged = self.source.resize((round(width * 1.04), round(height * 1.04)))
        left, top = (enlarged.width - width) // 2, (enlarged.height - height) // 2
        drift = measure_drift(self.source, enlarged.crop((left, top, left + width, top + height)))
        self.assertLessEqual(abs(drift.zoom - 0.96), 0.011)

    def test_comparison_matches_the_source_size(self):
        small = self.source.resize((320, 180))
        self.assertEqual(comparison(self.source, small).size, self.source.size)
        self.assertEqual(comparison(self.source, small, difference=True).size, self.source.size)


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "source.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=5",
             str(self.video)], check=True,
        )
        self.picture = self.root / "edited.png"
        Image.new("RGB", (64, 64), "red").save(self.picture)
        self.stored = {}
        self.takes = []

        def stage(path, category, **_options):
            target = f"{category}/{Path(path).name}"
            self.stored[target] = Path(path).read_bytes()
            return target

        def take(values, remake, progress):
            self.takes.append((values, remake))
            yield GenerationUpdate("result.mp4", "done")

        self.handlers = RemakeHandlers(EngineDeps(
            runtime_config=lambda: None, generation_services=lambda: None, stage_file=stage, interrupt=lambda: None,
            h3_take=take, h3_defaults=lambda: {**default_request(""), "model_profile": "Default"},
        ))

    def generate(self, *, pin_frames=True, pin_1=12, picture_2=None, pin_2=None, end=4.6):
        return list(self.handlers.generate_remake(
            SWAP, str(self.video), 0, end, 480, SIX_SECTIONS, pin_frames,
            str(self.picture), picture_2, None, None, pin_1, pin_2, None, None,
        ))

    def test_generate_uses_the_captured_h3_settings_with_remakes_overrides(self):
        self.handlers.settings["me"] = {"model_profile": "Speed", "attention_mode": "Sage 2", "seed": 77,
                                        "latent_upscale": False,
                                        "steps": 30, "scheduler": "beta"}
        token = CURRENT_JOB.set(SimpleNamespace(owner="me", check=lambda: None))
        self.addCleanup(CURRENT_JOB.reset, token)
        updates = self.generate()
        self.assertEqual(updates[-1], ("result.mp4", "done"))
        (values, remake), = self.takes
        self.assertEqual((values["model_profile"], values["attention_mode"], values["seed"]), ("Speed", "Sage 2", 77))
        self.assertEqual((values["steps"], values["scheduler"], values["width"], values["height"]),
                         (REQUEST_STEPS, "simple", 832, 480))
        self.assertEqual(values["duration"], 107 / 24)
        self.assertEqual((remake.mode, remake.frames, remake.pins), (SWAP, 107, (12,)))
        self.assertEqual(remake.pictures, (f"remake_pictures/{self.picture.name}",))
        self.assertIn(remake.source_video, self.stored)

    def test_no_job_falls_back_to_the_defaults(self):
        self.generate()
        self.assertEqual(self.takes[0][0]["model_profile"], "Default")

    def test_latent_upscale_doubles_the_requested_size(self):
        self.handlers.settings["me"] = {"latent_upscale": True}
        token = CURRENT_JOB.set(SimpleNamespace(owner="me", check=lambda: None))
        self.addCleanup(CURRENT_JOB.reset, token)
        self.generate()
        self.assertEqual((self.takes[0][0]["width"], self.takes[0][0]["height"]), (1664, 960))

    def test_unpinned_runs_send_no_pins(self):
        self.generate(pin_frames=False, pin_1=None)
        self.assertEqual(self.takes[0][1].pins, ())

    def test_partly_pinned_pictures_are_refused(self):
        with self.assertRaisesRegex(H3Error, "frame number"):
            self.generate(picture_2=str(self.picture), pin_2=None)

    def test_extract_fills_the_first_empty_slot_and_records_the_raw_frame(self):
        result = self.handlers.extract(str(self.video), 0, 24, {}, str(self.picture), None, None, None)
        pictures, pins, originals = result[:4], result[4:8], result[8]
        self.assertTrue(pictures[1]["value"].endswith(".png"))
        self.assertEqual(pins[1]["value"], 24)
        self.assertNotIn("value", pictures[0])
        self.assertEqual(list(originals), ["2"])
        # A slot still holding that exact frame needs an edit before generating.
        message = readiness_message(str(self.video), 0, 4.6, SIX_SECTIONS, True, originals,
                                    [str(self.picture), pictures[1]["value"], None, None], [12, 24, None, None])
        self.assertEqual(message, "Edit Picture 2, or remove it.")

    def test_readiness_walks_through_the_steps(self):
        pictures, pins = [str(self.picture), None, None, None], [12, None, None, None]

        def message(video=str(self.video), end=4.6, prompt=SIX_SECTIONS, pin_frames=True, shown=pictures, frames=pins):
            return readiness_message(video, 0, end, prompt, pin_frames, {}, shown, frames)

        self.assertEqual(message(video=None), "Add a source video to start.")
        self.assertIn("at least 3.75", message(end=3))
        self.assertIn("Extract a frame", message(shown=[None] * 4))
        self.assertIn("frame number", message(frames=[None] * 4))
        self.assertIn("Pins must be", message(frames=[900, None, None, None]))
        self.assertIn("Rewrite your request", message(prompt=" "))
        self.assertTrue(message(prompt="free text").startswith(READY))
        self.assertIn("six-section", message(prompt="free text"))
        self.assertEqual(message(), READY)

    def test_selection_text_reports_the_snapped_length(self):
        self.assertIn("107 frames", selection_text(0, 4.6))
        self.assertIn("recommends", selection_text(0, 14))
        self.assertIn("at least", selection_text(0, 1))

    def test_next_run_card_warns_on_undocumented_settings(self):
        card = next_run_card({"model_profile": "Singularity", "latent_upscale": True}, EDIT, str(self.video), 0, 4.6,
                             480, [str(self.picture)], True)
        self.assertIn("Remake · Edit", card)
        self.assertIn("1664 × 960", card)
        self.assertIn("Untested pairing: Singularity", card)
        documented = next_run_card({"model_profile": "Original"}, SWAP, str(self.video), 0, 4.6, 480, [], False)
        self.assertNotIn("Untested", documented)

    def test_moving_start_keeps_pins_on_their_source_frames(self):
        self.assertEqual(self.handlers.start_moved(3, 2, 48, None, 10, 24), (3, 24, None, None, 0))
        self.assertEqual(self.handlers.start_moved(1, 2, 48, None, None, None), (1, 72, None, None, None))
        self.assertEqual(self.handlers.start_moved(None, 0, 48, None, None, None), (None, 48, None, None, None))

    def test_a_cleared_start_counts_as_zero(self):
        self.assertEqual(list(self.handlers.generate_remake(
            SWAP, str(self.video), None, 4.6, 480, SIX_SECTIONS, True, str(self.picture),
            None, None, None, 12, None, None, None,
        ))[-1], ("result.mp4", "done"))
        self.assertTrue(self.handlers.preview_frame(str(self.video), None, 0).endswith(".png"))

    def test_alignment_measures_the_slot_chosen_when_applied(self):
        extract_frame(self.video, 0.5, self.root / "frame.png")
        with Image.open(self.root / "frame.png") as frame:
            shifted = Image.fromarray(np.roll(np.asarray(frame.convert("RGB")), (6, -14), axis=(0, 1)))
        shifted.save(self.root / "shifted.png")
        *updates, status = self.handlers.apply_alignment(
            str(self.video), 0, "Picture 2", str(self.picture), str(self.root / "shifted.png"), None, None,
            0, 12, None, None,
        )
        self.assertIn("Aligned", status)
        self.assertNotIn("value", updates[0])
        with Image.open(updates[1]["value"]) as aligned:
            remaining = measure_drift(load_image(self.root / "frame.png"), aligned)
        self.assertFalse(remaining.visible)

    def test_a_failed_rewrite_suggests_nothing(self):
        self.handlers.deps = EngineDeps(
            runtime_config=lambda: SimpleNamespace(prompt_systems={"Viggle Remake": Path("x")}),
            generation_services=lambda: None, stage_file=lambda *a, **k: "", interrupt=lambda: None,
        )
        with patch("h3_app.prompt_writers.common.write_prompt",
                   return_value=("robot", "Prompt enhancement failed: no key")):
            self.assertEqual(self.handlers.rewrite_prompt(
                "robot", SWAP, str(self.video), 0, 4.6, 480, True, str(self.picture), None, None, None,
                12, None, None, None, "gemini-model", "", "Gemini", "", "model", None,
            ), ("", "Prompt enhancement failed: no key"))

    def test_rewrite_gives_gemini_the_clip_and_keeps_tags_in_order(self):
        calls = {}

        def write(**options):
            calls.update(options)
            calls["clip_existed"] = Path(options["media_values"][0][1]).is_file()
            return SIX_SECTIONS, "Enhanced."

        self.handlers.deps = EngineDeps(
            runtime_config=lambda: SimpleNamespace(prompt_systems={"Viggle Remake": Path("x")}),
            generation_services=lambda: None, stage_file=lambda *a, **k: "", interrupt=lambda: None,
        )
        with patch("h3_app.prompt_writers.common.write_prompt", write):
            prompt, status = self.handlers.rewrite_prompt(
                "robot", SWAP, str(self.video), 0, 4.6, 480, True, str(self.picture), None, None, None,
                12, None, None, None, "gemini-model", "", "Gemini", "", "model", None,
            )
        self.assertEqual((prompt, status), (SIX_SECTIONS, "Enhanced."))
        self.assertTrue(calls["clip_existed"])
        self.assertEqual([label for label, _ in calls["media_values"]],
                         ["<Video 1>, the source video", "<Picture 1>, a painted frame"])
        self.assertIn("4.458 seconds (107 frames", calls["context"])
        self.assertIn("<Picture 1> is pinned at 0.500 seconds (frame 12).", calls["context"])

    def test_openai_writer_gets_frames_only_and_a_missing_section_is_flagged(self):
        from h3_app.prompt_writers.remake import enhance_remake_prompt

        with patch("h3_app.prompt_writers.common.write_prompt",
                   return_value=("summary:\nx", "Enhanced.")) as write:
            _, status = enhance_remake_prompt(
                "robot", SWAP, "clip.mp4", ["p.png"], (), 107, 832, 480, "m", "", backend="OpenAI-compatible",
                runtime=SimpleNamespace(prompt_systems={"Viggle Remake": Path("x")}),
            )
        self.assertEqual([label for label, _ in write.call_args.kwargs["media_values"]],
                         ["<Picture 1>, a painted frame"])
        self.assertIn("can't read video", status)
        self.assertIn("missing subject_definitions", status)


class NodeFileTests(unittest.TestCase):
    def test_vendored_nodes_are_upstream_with_only_the_documented_change(self):
        source = (Path(__file__).resolve().parents[1] / "custom_nodes" / "H3Remake" / "nodes.py").read_text("utf-8")
        for name in ("RemakeConditioning", "RemakeKeyframeUntil", "RemakeNoise"):
            self.assertIn(f'"{name}": {name}', source)
        self.assertIn('"short_edge": ("INT", {"default": 480, "min": 480, "max": 1472, "step": 32,', source)


if __name__ == "__main__":
    unittest.main()
