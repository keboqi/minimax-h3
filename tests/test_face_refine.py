"""Opt-in face pass: off by default, wired to the pack's contract, safe without a face."""

import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app import catalog
from h3_app.contracts import GENERATION_COMPONENTS, GENERATION_FIELDS, GenerationArguments
from h3_app.generation.finish_video import NO_FACE_ERROR, finish_video, make_finishing_retry
from h3_app.jobs import CURRENT_JOB, Job
from h3_app.model_types import ModelConfig, ModelProfile
from h3_app.settings import GenerationRequest, resolve_settings
from h3_app.storyboard import model as storyboard
from h3_app.workflows.face_refine import REQUIRED_FACE_REFINE_NODES, build_h3_face_refine_graph
from h3_app.refinement import FACE_REFINE_STEPS, H3_REFINEMENT_FIELDS, resolve_face_refinement
from h3_models import MODEL_SPECS


def models():
    return ModelConfig(
        {"speed": ModelProfile("Test", "base.safetensors", "ref.safetensors")},
        "speed", "text.safetensors", "video.safetensors", "audio.safetensors",
        turbo_lora="turbo.safetensors",
    )


def by_class(graph, name):
    return next(node["inputs"] for node in graph.values() if node["class_type"] == name)


def link_id(graph, name):
    return next(key for key, node in graph.items() if node["class_type"] == name)


class FaceRefineSettingsTests(unittest.TestCase):
    def test_off_by_default_and_old_api_callers_stay_off(self):
        self.assertIs(catalog.UI_DEFAULTS["face_refine"], False)
        legacy = [None] * GENERATION_FIELDS.index("face_refine")
        values = GenerationArguments.from_positional(legacy).values
        self.assertIs(values["face_refine"], False)
        self.assertEqual(values["face_refine_strength"], 0.4)
        self.assertEqual(len(GENERATION_FIELDS), len(GENERATION_COMPONENTS))

    def test_only_video_results_can_use_it(self):
        for result, expected in (("Video", True), ("Image", False), ("Audio", False)):
            plan = resolve_settings(GenerationRequest.from_values(
                {**catalog.UI_DEFAULTS, "face_refine": True, "result_format": result}))
            self.assertIs(plan.effective.finishing.face_refine, expected, result)

    def test_strength_outside_the_range_is_refused_before_generating(self):
        for strength in (None, "0.4", True, 0, 0.1, 0.7, float("nan")):
            plan = resolve_settings(GenerationRequest.from_values(
                {**catalog.UI_DEFAULTS, "face_refine": True, "face_refine_strength": strength}))
            self.assertTrue(any("Face refine strength" in issue for issue in plan.issues), strength)
            self.assertEqual(plan.effective.finishing.face_refine_strength, 0.4)
        plan = resolve_settings(GenerationRequest.from_values(
            {**catalog.UI_DEFAULTS, "face_refine": True, "face_refine_strength": 0.6}))
        self.assertFalse(plan.issues)

    def test_existing_video_refinement_and_storyboard_takes_ignore_it(self):
        self.assertNotIn("face_refine", H3_REFINEMENT_FIELDS)
        self.assertIs(storyboard.PROFILE_OVERRIDES["face_refine"], False)


class FaceRefineGraphTests(unittest.TestCase):
    def build(self, **overrides):
        values = {**catalog.UI_DEFAULTS, "mode": "First / last frame", "prompt": "A distant singer",
                  "generation_mode": "Normal", "use_trt_vae": True, **overrides}
        config = models()
        config.turbo_lora_for = lambda mode, variant: "turbo8.safetensors"
        return build_h3_face_refine_graph(
            source_video="source.mp4", seed=7, models=config, prompt="A distant singer",
            fps=24.0, strength=0.4, refinement=values, output_nonce="abc",
        )

    def test_recipe_tracks_largest_face_without_insightface(self):
        graph = self.build()
        track = by_class(graph, "H3FaceTrackCrop")
        self.assertEqual((track["select"], track["identity_track"]), ("largest_face", False))
        self.assertEqual(track["canvas_mode"], "auto_capped_768")
        self.assertEqual(MODEL_SPECS["face_yolov8m"].local_name, track["detector"])
        schedule = by_class(graph, "BasicScheduler")
        self.assertEqual((schedule["steps"], schedule["denoise"]), (8, 0.4))
        self.assertFalse({"H3RegeneratePrepareLatent", "MiniMaxH3TRTVAELoader"}
                         & {node["class_type"] for node in graph.values()})

    def test_the_stack_plans_for_every_sampled_step(self):
        config = models()
        config.turbo_lora_for = lambda mode, variant: "turbo8.safetensors"
        resolved = resolve_face_refinement({**catalog.UI_DEFAULTS, "generation_mode": "Normal"}, config)
        self.assertEqual(resolved["steps"], FACE_REFINE_STEPS)

    def test_canvas_and_length_follow_the_tracker(self):
        graph = self.build()
        track = link_id(graph, "H3FaceTrackCrop")
        conditioning = by_class(graph, "MiniMaxH3ImageToVideo")
        self.assertEqual((conditioning["width"], conditioning["height"], conditioning["length"]),
                         ([track, 4], [track, 5], [track, 6]))

    def test_source_audio_and_per_frame_model_reach_the_sampler(self):
        graph = self.build()
        components = link_id(graph, "GetVideoComponents")
        per_frame = link_id(graph, "H3PerFrameDenoise")
        self.assertEqual(by_class(graph, catalog.H3_SOURCE_AUDIO_LATENT_NODE)["audio"], [components, 1])
        self.assertEqual(by_class(graph, "CreateVideo")["audio"], [components, 1])
        self.assertEqual(by_class(graph, "BasicGuider")["model"], [per_frame, 2])
        self.assertEqual(by_class(graph, "SamplerCustomAdvanced")["latent_image"], [per_frame, 0])

    def test_stitch_composites_over_the_source_frames(self):
        graph = self.build()
        stitch = by_class(graph, "H3FaceStitch")
        self.assertEqual(stitch["base_images"], [link_id(graph, "GetVideoComponents"), 0])
        self.assertEqual(stitch["refined_crops"], [link_id(graph, "VAEDecode"), 0])

    def test_attention_follows_the_server_backend(self):
        sol = catalog.SOL_ATTENTION_NODE
        forced = {"attention_mode": "Sol-Attn"}
        classes = lambda **values: {node["class_type"] for node in self.build(**values).values()}
        self.assertIn(sol, classes(**forced))
        self.assertNotIn(sol, classes(**forced, _attention_backend="comfy-kitchen"))

    def test_reference_mode_uses_the_reference_node(self):
        graph = self.build(mode="Reference media")
        self.assertIn("audio_vae", by_class(graph, "MiniMaxH3ReferenceToVideo"))

    def test_required_nodes_are_in_the_graph(self):
        classes = {node["class_type"] for node in self.build().values()}
        self.assertLessEqual(REQUIRED_FACE_REFINE_NODES, classes)


class FaceRefineFinishingTests(unittest.TestCase):
    def run_finish(self, *, face_refine=True, history=None, postprocess="None", job=None, face_refined=False,
                   output_prefix=None):
        source, refined = Path("source.mp4"), Path("refined.mp4")
        finishing = SimpleNamespace(
            face_refine=face_refine, face_refine_strength=0.45, postprocess=postprocess)
        request = SimpleNamespace(
            finishing=finishing, media=SimpleNamespace(prompt="Scene", mode="First / last frame"),
            values=lambda: {"prompt": "Scene"})
        build = Mock(return_value={"1": {"class_type": "LoadVideo", "inputs": {}}})
        media = SimpleNamespace(
            stage_file=Mock(return_value="staged.mp4"),
            probe_video_metadata=Mock(return_value=SimpleNamespace(fps=24.0, duration=12.5)),
            resolve_output=Mock(return_value=refined),
            postprocess_video=Mock(side_effect=lambda path, option: path))
        services = SimpleNamespace(
            workflows=SimpleNamespace(build_face_refine_graph=build), media=media,
            models=SimpleNamespace(ensure_model_keys=Mock(return_value=False),
                                   ensure_turbo_lora=Mock(return_value=False),
                                   ensure_base_video_vae=Mock(return_value=False)),
            execution=SimpleNamespace(
                submit_prompt=Mock(return_value="job"), poll_comfy_progress=Mock(return_value=[]),
                wait_for_history=history or Mock(return_value={})))
        timings = SimpleNamespace(transition=Mock())
        generator = finish_video(
            request, SimpleNamespace(actual_seed=7, models=models()), services, source,
            "client", 0.0, timings, lambda *a, **k: None, face_refined=face_refined, output_prefix=output_prefix)
        statuses = []
        token = CURRENT_JOB.set(job)
        try:
            while True:
                try:
                    statuses.append(next(generator))
                except StopIteration as stop:
                    return stop.value, services, build, statuses
        finally:
            CURRENT_JOB.reset(token)

    def test_disabled_runs_nothing(self):
        result, services, build, _ = self.run_finish(face_refine=False)
        self.assertEqual(result, Path("source.mp4"))
        build.assert_not_called()
        services.models.ensure_model_keys.assert_not_called()

    def test_enabled_hands_the_refined_clip_to_later_finishing(self):
        result, services, build, _ = self.run_finish()
        self.assertEqual(result, Path("refined.mp4"))
        self.assertEqual(build.call_args.kwargs["strength"], 0.45)
        self.assertEqual(build.call_args.kwargs["refinement"]["_source_duration"], 12.5)
        services.models.ensure_model_keys.assert_called_once_with(("face_yolov8m",))
        services.media.postprocess_video.assert_called_once_with(Path("refined.mp4"), "None")

    def test_a_take_with_its_own_prefix_keeps_the_face_pass_beside_it(self):
        _, _, build, _ = self.run_finish(output_prefix="h3/.remake/local/key/w01_abc")
        self.assertEqual(build.call_args.kwargs["output_prefix"], "h3/.remake/local/key/w01_abc_face")
        _, _, build, _ = self.run_finish()
        self.assertIsNone(build.call_args.kwargs["output_prefix"])

    def test_the_graph_saves_to_the_given_prefix(self):
        config = models()
        config.turbo_lora_for = lambda mode, variant: "turbo8.safetensors"
        values = {**catalog.UI_DEFAULTS, "mode": "First / last frame", "generation_mode": "Normal"}

        def prefix(**options):
            graph = build_h3_face_refine_graph(source_video="source.mp4", seed=7, models=config, prompt="x",
                                               fps=24.0, strength=0.4, refinement=values, output_nonce="abc",
                                               **options)
            return next(node["inputs"]["filename_prefix"] for node in graph.values()
                        if "filename_prefix" in node["inputs"])

        self.assertEqual(prefix(), "h3/face_refine_abc")
        self.assertEqual(prefix(output_prefix="h3/.remake/w01_face"), "h3/.remake/w01_face")

    def test_no_face_keeps_the_clip(self):
        history = Mock(side_effect=RuntimeError(f"{NO_FACE_ERROR}. Lower `confidence`"))
        result, _, _, _ = self.run_finish(history=history)
        self.assertEqual(result, Path("source.mp4"))

    def test_a_clip_already_refined_is_not_refined_again(self):
        result, _, build, _ = self.run_finish(face_refined=True)
        self.assertEqual(result, Path("source.mp4"))
        build.assert_not_called()

    def test_the_refined_clip_becomes_the_retained_source(self):
        job = Mock(variant=0, recoverable_sources={0: "source.mp4"})
        self.run_finish(job=job)
        self.assertEqual(job.recoverable_sources[0], "refined.mp4")

    def test_no_face_settles_the_stage_and_keeps_the_retained_source(self):
        job = Mock(variant=0, recoverable_sources={0: "source.mp4"})
        self.run_finish(job=job, history=Mock(side_effect=RuntimeError(NO_FACE_ERROR)))
        job.settle_prompt.assert_called_once_with("job")
        self.assertEqual(job.recoverable_sources[0], "source.mp4")

    def test_settling_a_prompt_closes_its_ledger_entry(self):
        job = SimpleNamespace(prompt_id="job", persist=Mock(), ledger=[
            {"prompt_id": "job", "state": "submitted"}, {"prompt_id": "other", "state": "submitted"}])
        Job.settle_prompt(job, "job")
        self.assertIsNone(job.prompt_id)
        self.assertEqual([entry["state"] for entry in job.ledger], ["completed", "submitted"])

    def test_retry_skips_the_pass_only_for_a_refined_clip(self):
        module = importlib.import_module("h3_app.generation.finish_video")
        seen = []

        def fake_finish(*args, **kwargs):
            seen.append(kwargs["face_refined"])
            return Path("out.mp4")
            yield

        with patch.object(module, "finish_video", fake_finish), patch.object(module, "write_snapshot"):
            retry = make_finishing_retry(Mock(), SimpleNamespace(actual_seed=1), Mock(), {}, raw_source="raw.mp4")
            for retained in ("raw.mp4", "refined.mp4"):
                list(retry(retained, None))
        self.assertEqual(seen, [False, True])

    def test_other_failures_still_surface(self):
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.run_finish(history=Mock(side_effect=RuntimeError("out of memory")))
