"""Remake: run rules, graph shape, source trimming, frame alignment and the tab's handlers."""

import shutil
import subprocess
import tempfile
import unittest
import uuid
from dataclasses import fields, replace
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
from h3_app.remake.media import detect_cuts, extract_frame, join_windows, trim_source
from h3_app.remake.runner import Selection
from h3_app.remake.windows import LENGTHS, carried_frame, needs_windows, plan_windows
from h3_app.staging import file_content_sha256
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


def remake_graph(*, pins=(12, 184), **options):
    return h3.build_remake_graph(
        prompt=SIX_SECTIONS, source_video="h3_gradio/remake_sources/a.mp4",
        pictures=["h3_gradio/remake_pictures/1.png", "h3_gradio/remake_pictures/2.png"], pins=pins, frames=226,
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

    def test_latent_upscale_refines_with_the_canvas_sized_reference(self):
        graph = remake_graph(width=1664, height=960, latent_upscale_model_name="upscaler.safetensors",
                             latent_upscale_refine_steps=2)
        (condition_id, condition), = nodes(graph, "RemakeConditioning")
        self.assertEqual(condition["short_edge"], 480)
        self.assertEqual(nodes(graph, "ImageScale"), [])
        guiders = [inputs["conditioning"] for _, inputs in nodes(graph, "BasicGuider")]
        self.assertEqual(guiders, [[condition_id, 0], [condition_id, 0]])
        (hold_id, _), = nodes(graph, "RemakeHoldAudio")
        (first, second) = [inputs for _, inputs in nodes(graph, "SamplerCustomAdvanced")]
        self.assertEqual(first["latent_image"], [condition_id, 1])
        self.assertEqual(second["latent_image"], [hold_id, 0])
        self.assertEqual(first["noise"], second["noise"])
        (_, split), = nodes(graph, "SplitSigmas")
        self.assertEqual(split["step"], 4)

    def test_refinement_stays_a_tail_of_the_schedule(self):
        graph = remake_graph(width=1664, height=960, latent_upscale_model_name="upscaler.safetensors",
                             latent_upscale_refine_steps=6)
        (_, split), = nodes(graph, "SplitSigmas")
        # Refinement stays below the keyframe hold: the last two of six steps.
        self.assertEqual(split["step"], 4)

    def test_a_mixed_pin_list_leaves_unpinned_pictures_marked(self):
        graph = remake_graph(pins=(None, 11))
        self.assertEqual(nodes(graph, "RemakeConditioning")[0][1]["pins"], "-, 11")
        self.assertEqual(nodes(remake_graph(pins=(None, None)), "RemakeConditioning")[0][1]["pins"], "")

    def test_an_output_prefix_names_the_take(self):
        graph = remake_graph(output_prefix="h3/.remake/a/w01")
        self.assertEqual(next(inputs for _, inputs in nodes(graph, "H3SaveVideoNVENC"))["filename_prefix"],
                         "h3/.remake/a/w01")

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
        self.assertEqual(self.handlers.start_moved(3, 2, 48, None, 10, 24), (3, 3.0, 24, None, None, 0))
        self.assertEqual(self.handlers.start_moved(1, 2, 48, None, None, None), (1, 1.0, 72, None, None, None))
        self.assertEqual(self.handlers.start_moved(None, 0, 48, None, None, None), (None, 0.0, 48, None, None, None))

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
                "local", Selection(None, 0, 4.6, 480, 12, 226), 0, "robot", SWAP, str(self.video), True, str(self.picture),
                None, None, None,
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
                "local", Selection(None, 0, 4.6, 480, 12, 226), 0, "robot", SWAP, str(self.video), True, str(self.picture),
                None, None, None,
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


class PlanTests(unittest.TestCase):
    def check_tiled(self, windows, total, mix):
        self.assertEqual(windows[0]["own"][0], 0)
        self.assertEqual(windows[-1]["own"][1], total)
        for window in windows:
            self.assertIn(window["render"][1] - window["render"][0], LENGTHS)
            self.assertLessEqual(window["render"][0], window["own"][0])
            self.assertGreaterEqual(window["render"][1], window["own"][1])
        for before, after in zip(windows, windows[1:]):
            self.assertEqual(after["own"][0], before["own"][1] - (mix if after["continues"] else 0))

    def test_a_long_take_chains_with_an_overlap(self):
        windows = plan_windows(600, [], max_frames=226, mix=12)
        self.check_tiled(windows, 600, 12)
        self.assertEqual([window["continues"] for window in windows], [False, True, True])

    def test_windows_end_on_cuts_and_start_new_shots_without_a_mix(self):
        windows = plan_windows(600, [200, 410], max_frames=226, mix=12)
        self.check_tiled(windows, 600, 12)
        self.assertEqual([window["own"][1] for window in windows], [200, 410, 600])
        self.assertEqual([window["continues"] for window in windows], [False, False, False])

    def test_any_length_tiles(self):
        for total in range(245, 1400, 37):
            for cuts in ([], [300, 311, 700], list(range(40, total, 45))):
                for max_frames in (124, 226, 345):
                    if total >= max_frames + 17:
                        self.check_tiled(plan_windows(total, cuts, max_frames=max_frames, mix=12), total, 12)

    def test_a_carried_frame_with_no_mix_pins_the_first_frame(self):
        first, second = plan_windows(600, [], max_frames=226, mix=0)[:2]
        self.assertEqual(carried_frame(first, second)[1], 0)

    def test_windows_only_appear_past_one_window(self):
        self.assertFalse(needs_windows(0, 226 / 24, 226))
        self.assertFalse(needs_windows(0, 240 / 24, 226))
        self.assertTrue(needs_windows(0, 243 / 24, 226))

    def test_the_carried_frame_is_the_last_kept_one_and_pins_only_when_continuing(self):
        continuing, following = plan_windows(600, [], max_frames=226, mix=12)[:2]
        self.assertEqual(carried_frame(continuing, following), (225, 11))
        first, after_cut = plan_windows(600, [200, 410], max_frames=226, mix=12)[:2]
        self.assertEqual(carried_frame(first, after_cut), (199, None))


class JoinTests(unittest.TestCase):
    def test_a_crossfade_and_a_cut_join_to_the_exact_length_under_the_source_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, color in (("a", "red"), ("b", "blue"), ("c", "green")):
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                                f"color=c={color}:s=320x180:r=24:d=4", "-pix_fmt", "yuv420p", str(root / f"{name}.mp4")],
                               check=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                            str(root / "audio.m4a")], check=True)
            output = root / "joined.mp4"
            join_windows([
                {"path": root / "a.mp4", "first": 0, "frames": 90, "continues": False},
                {"path": root / "b.mp4", "first": 4, "frames": 90, "continues": True},
                {"path": root / "c.mp4", "first": 0, "frames": 60, "continues": False},
            ], root / "audio.m4a", start=1, frames=228, mix=12, output=output, has_audio=True)
            metadata = probe_video_metadata(output)
            self.assertEqual(metadata.frame_count, 228)
            self.assertTrue(metadata.has_audio)
            frame = extract_frame(output, 84 / 24, root / "middle.png")
            with Image.open(frame) as image:
                red, _green, blue = image.convert("RGB").getpixel((160, 90))
            self.assertGreater(red, 40)
            self.assertGreater(blue, 40)


class WindowRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "source.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=22",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=22", "-shortest", str(self.video)], check=True,
        )
        self.picture = self.root / "edited.png"
        Image.new("RGB", (832, 480), "red").save(self.picture)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        self.takes = []

        def stage(path, category, **_options):
            target = self.input_dir / category / f"{uuid.uuid4().hex}{Path(path).suffix}"
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
            return str(target.relative_to(self.input_dir))

        def take(values, remake, progress):
            self.takes.append((values, remake))
            target = self.output_dir / f"{remake.output_prefix or 'single'}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.input_dir / remake.source_video, target)
            yield GenerationUpdate(str(target), "done")

        self.runtime = SimpleNamespace(output_dir=self.output_dir,
                                       prompt_systems={"Viggle Remake": Path("x")})
        self.handlers = RemakeHandlers(EngineDeps(
            runtime_config=lambda: self.runtime, generation_services=lambda: None, stage_file=stage,
            interrupt=lambda: None, h3_take=take, h3_defaults=lambda: default_request(""),
        ))
        self.writer = patch("h3_app.remake.runner.enhance_remake_prompt", self.write)
        self.writer.start()
        self.addCleanup(self.writer.stop)
        self.written = []

    def write(self, request, mode, clip, pictures, pins, frames, width, height, *args, previous_prompt="", **_options):
        self.written.append((list(pictures), list(pins), previous_prompt))
        return SIX_SECTIONS + f"\n# {len(self.written)}", "Enhanced."

    def run_windows(self, *, index=0, prompt="", picture=None, pin=12, smart=True, end=20):
        return list(self.handlers.generate_remake(
            SWAP, str(self.video), 0, end, 480, prompt, True, picture, None, None, None, pin, None, None, None,
            226, "make a robot", smart, 12, index, True,
        ))

    def project(self):
        return self.handlers.runner.saved("local", self.selection())

    def selection(self):
        return Selection(file_content_sha256(self.video), 0, 20, 480, 12, 226)

    def test_a_long_selection_renders_every_window_and_joins_them(self):
        updates = self.run_windows(picture=str(self.picture))
        final, status = updates[-1]
        metadata = probe_video_metadata(Path(final))
        self.assertEqual((metadata.frame_count, metadata.has_audio), (480, True))
        self.assertIn("Combined", status)
        self.assertEqual(Path(final).parent, self.output_dir / "h3")
        project = self.project()
        self.assertGreater(len(project["windows"]), 1)
        self.assertTrue(all(window["state"] == "done" for window in project["windows"]))
        self.assertTrue(all(values["postprocess"] == "None" for values, _ in self.takes))
        first, second = self.takes[0][1], self.takes[1][1]
        self.assertEqual(first.pins, (12,))
        # Window 2: the frame carried from window 1, pinned, then window 1's picture for identity.
        self.assertEqual(len(second.pictures), 2)
        self.assertEqual(second.pins, (project["windows"][0]["own"][1] - 1 - project["windows"][1]["render"][0], None))
        self.assertEqual(self.written[1][2], self.written[0][2] or SIX_SECTIONS + "\n# 1")

    def test_regenerating_only_renders_what_changed_and_flags_what_follows(self):
        self.run_windows(picture=str(self.picture))
        count = len(self.takes)
        project = self.project()
        kept = project["windows"][2]["take"] if len(project["windows"]) > 2 else None
        self.handlers.runner.projects.edit(project, lambda current: current["windows"][1].update(state="changed"))
        self.run_windows(index=1, prompt=project["windows"][1]["prompt"], pin=None)
        self.assertEqual(len(self.takes), count + 1)
        windows = self.project()["windows"]
        self.assertEqual(windows[1]["state"], "done")
        if kept:
            self.assertEqual(windows[2]["state"], "outdated")
            self.assertEqual(windows[2]["take"], kept)

    def test_without_smart_a_window_with_no_frames_is_left_for_you(self):
        updates = self.run_windows(picture=str(self.picture), smart=False)
        self.assertIn("need a frame", updates[-1][1])
        self.assertEqual(len(self.takes), 1)

    def test_a_short_selection_stays_a_single_run(self):
        updates = self.run_windows(picture=str(self.picture), end=9)
        self.assertEqual(len(self.takes), 1)
        self.assertEqual(self.takes[0][1].output_prefix, None)
        self.assertEqual(updates[-1][1], "done")

    def test_a_failed_window_stops_the_run_and_keeps_finished_ones(self):
        calls = []

        def fail_second(values, remake, progress):
            calls.append(remake)
            if len(calls) == 2:
                yield GenerationUpdate(None, "Error: out of memory\n\ntimings")
                return
            target = self.output_dir / f"{remake.output_prefix}.mp4"
            shutil.copy2(self.input_dir / remake.source_video, target)
            yield GenerationUpdate(str(target), "done")

        self.handlers.deps = replace(self.handlers.deps, h3_take=fail_second)
        with self.assertRaisesRegex(H3Error, "Window 2 of .*out of memory"):
            self.run_windows(picture=str(self.picture))
        states = [window["state"] for window in self.project()["windows"]]
        self.assertEqual(states[:2], ["done", "failed"])

    def test_opening_a_window_saves_the_one_you_left(self):
        self.run_windows(picture=str(self.picture))
        project = self.project()
        with patch.object(self.handlers, "_preview", return_value="frame.png"):
            result = self.handlers.window_selected(
                "local", self.selection(), 1, 0, "new prompt", True, str(self.picture), None, None, None,
                12, None, None, None,
            )
        self.assertEqual(result[0], 1)
        window = self.project()["windows"][0]
        self.assertEqual((window["prompt"], window["state"]), ("new prompt", "changed"))
        self.assertEqual(project["windows"][1]["render"][0] / 24, result[13])

    def test_resuming_a_selection_fills_the_editor_instead_of_erasing_the_window(self):
        self.run_windows(picture=str(self.picture))
        count = len(self.takes)
        with patch.object(self.handlers, "_preview", return_value="frame.png"):
            result = self.handlers.selection_changed("local", self.selection(), 0)
        opened = result[4:]
        saved = self.project()["windows"][0]
        self.assertEqual(opened[1], saved["pictures"][0])
        self.assertEqual(opened[5], 12)
        self.assertEqual(opened[9], saved["prompt"])
        self.run_windows(prompt=opened[9], picture=opened[1], pin=opened[5])
        self.assertEqual(len(self.takes), count)

    def test_a_changed_selection_only_clears_a_later_windows_editor(self):
        selection = Selection("unsaved", 0, 20, 480, 12, 226)
        self.assertEqual(self.handlers.selection_changed("local", selection, 0, 0)[5], {"__type__": "update"})
        self.assertIsNone(self.handlers.selection_changed("local", selection, 0, 2)[5])

    def test_generating_with_an_empty_editor_keeps_an_auto_prompt(self):
        self.run_windows(picture=str(self.picture))
        count = len(self.takes)
        self.run_windows(picture=str(self.picture))
        self.assertEqual(len(self.takes), count)
        self.assertEqual(self.project()["windows"][0]["state"], "done")

    def test_an_auto_prompt_is_rewritten_when_its_pictures_change(self):
        self.run_windows(picture=str(self.picture))
        written = len(self.written)
        project = self.project()
        self.handlers.runner.projects.edit(project, lambda current: current["windows"][1].update(
            state="changed", use_carried=False,
        ))
        self.run_windows(index=1, pin=None)
        self.assertEqual(len(self.written), written)
        self.handlers.runner.projects.edit(self.project(), lambda current: current["windows"][1].update(
            state="changed", prompt_pictures=99,
        ))
        self.run_windows(index=1, pin=None)
        self.assertEqual(len(self.written), written + 1)

    def test_marking_a_window_uses_changed_not_failed(self):
        self.run_windows(picture=str(self.picture))
        self.handlers.mark_window("local", self.selection(), 1, "", True, None, None, None, None,
                                  None, None, None, None)
        self.assertEqual(self.project()["windows"][1]["state"], "changed")

    def test_rewrite_uses_the_open_windows_span(self):
        self.run_windows(picture=str(self.picture))
        project = self.project()
        first, last = project["windows"][1]["render"]
        with patch("h3_ui.engines.remake.enhance_remake_prompt", return_value=("text", "Enhanced.")) as write, \
                patch("h3_ui.engines.remake.trim_source") as trim:
            self.handlers.rewrite_prompt(
                "local", self.selection(), 1, "robot", SWAP, str(self.video), True, str(self.picture), None, None, None,
                12, None, None, None, "gemini-model", "", "Gemini", "", "model", None,
            )
        self.assertEqual(trim.call_args.kwargs["start"], first / 24)
        self.assertEqual(trim.call_args.kwargs["frames"], last - first)
        self.assertEqual(write.call_args.args[5], last - first)


class CutTests(unittest.TestCase):
    def test_cuts_are_found_and_counted_from_the_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, color in (("a", "red"), ("b", "blue")):
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                                f"testsrc=size=320x180:rate=24:d=2", "-vf", f"drawbox=c={color}:t=fill",
                                "-pix_fmt", "yuv420p", str(root / f"{name}.mp4")], check=True)
            (root / "list.txt").write_text("file a.mp4\nfile b.mp4\n", "utf-8")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-i", str(root / "list.txt"),
                            "-c", "copy", str(root / "joined.mp4")], check=True)
            self.assertEqual(detect_cuts(root / "joined.mp4", 1, 3), [24])


class NodeFileTests(unittest.TestCase):
    def test_vendored_nodes_are_upstream_with_only_the_documented_change(self):
        source = (Path(__file__).resolve().parents[1] / "custom_nodes" / "H3Remake" / "nodes.py").read_text("utf-8")
        for name in ("RemakeConditioning", "RemakeKeyframeUntil", "RemakeNoise"):
            self.assertIn(f'"{name}": {name}', source)
        self.assertIn('"short_edge": ("INT", {"default": 480, "min": 480, "max": 736, "step": 32,', source)
        self.assertIn('None if p == "-" else int(p)', source)


if __name__ == "__main__":
    unittest.main()
