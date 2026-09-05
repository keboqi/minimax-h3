"""Policy and ownership regressions; no Gradio, SDK, network or GPU required."""

import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from h3_app.settings import *
from h3_app.jobs import JobCoordinator, CURRENT_JOB, Job, scoped_graph
from h3_app.media import recent_output_candidates, history_output_candidates
from h3_app.provenance import write_snapshot, read_snapshot
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments


class SettingsTests(unittest.TestCase):
    def test_presets_preserve_trained_counts(self):
        self.assertEqual(
            [
                preset_settings(n, "Turbo").steps
                for n in ("Fast", "Balanced", "Quality")
            ],
            [4, 6, 8],
        )
        self.assertEqual(
            [
                preset_settings(n, "Normal").steps
                for n in ("Fast", "Balanced", "Quality")
            ],
            [15, 18, 20],
        )

    def test_manual_and_required_changes_are_distinct(self):
        r = GenerationRequest(
            sampling=replace(
                preset_settings("Fast", "Turbo"), steps=8, text_encoder="BF16"
            )
        )
        p = resolve_settings(r)
        self.assertEqual(set(p.differences()), {"steps", "text_encoder"})
        self.assertTrue(p.effective.sampling.stage_model_offload)
        self.assertEqual(p.effective.cache_mode, "Off")
        self.assertFalse(p.requested.sampling.stage_model_offload)
        self.assertEqual(p.requested.cache_mode, "Spectrum")
        self.assertEqual(
            {a.field for a in p.adjustments},
            {"dimensions", "cache_mode", "stage_model_offload"},
        )

    def test_inactive_overrides_are_not_modifications(self):
        r = GenerationRequest(
            sampling=replace(SamplingSettings(), sol_tau=0.7, sol_exact_mode="exact_kv")
        )
        self.assertEqual(resolve_settings(r).differences(), {})
        r = replace(r, sampling=replace(r.sampling, attention_mode="Sol-Attn"))
        self.assertIn("sol_tau", resolve_settings(r).differences())

    def test_format_preserves_requested_finishing(self):
        r = GenerationRequest(
            output=OutputSettings(result_format="Audio"),
            finishing=FinishingSettings(True, "Upscale"),
        )
        p = resolve_settings(r)
        self.assertFalse(p.effective.finishing.latent_upscale)
        self.assertEqual(p.effective.finishing.postprocess, "None")
        self.assertEqual(
            (p.effective.output.width, p.effective.output.height), (32, 32)
        )
        self.assertEqual(p.requested.finishing, r.finishing)

    def test_start_image_and_single_frame_resolution(self):
        r = GenerationRequest(
            mode="First / last frame",
            output=OutputSettings(
                result_format="Image", image_vae="500K", image_frames=12
            ),
        )
        p = resolve_settings(r, ResolutionContext((1024, 768)))
        self.assertEqual(
            (p.effective.output.width, p.effective.output.height), (1024, 768)
        )
        self.assertEqual(p.effective.output.image_frames, 1)
        self.assertEqual(r.output.image_frames, 12)

    def test_mode_memory_and_reset_scope(self):
        v = {
            **asdict(preset_settings("Fast", "Turbo")),
            "preset": "Fast",
            "generation_mode": "Turbo",
            "cache_mode": "Off",
            "width": 1024,
        }
        mem, v = transition_modes(None, v, "edit")
        v["steps"] = 9
        mem, v = transition_modes(mem, v, "steps")
        v["generation_mode"] = "Normal"
        mem, v = transition_modes(mem, v, "generation_mode")
        self.assertEqual(v["steps"], 15)
        v["steps"] = 22
        mem, v = transition_modes(mem, v, "steps")
        v["generation_mode"] = "Turbo"
        mem, v = transition_modes(mem, v, "generation_mode")
        self.assertEqual(v["steps"], 9)
        mem, v = transition_modes(mem, v, "restore")
        self.assertEqual(v["steps"], 4)
        self.assertEqual(v["width"], 1024)
        self.assertEqual(v["cache_mode"], "Off")

    def test_invalid_saved_types(self):
        for value in ("false", 1, None):
            self.assertFalse(valid_preference(value, False))
        self.assertEqual(valid_preference(float("nan"), 4), 4)
        self.assertEqual(valid_preference(200, 4, minimum=4, maximum=30), 4)
        self.assertEqual(
            valid_preference("removed", "Fast", choices=["Fast", "Quality"]), "Fast"
        )

    def test_named_api_contract(self):
        values = list(range(len(GENERATION_FIELDS)))
        a = GenerationArguments.from_positional(values)
        self.assertEqual(len(a.references["image"]), 9)
        self.assertEqual(a.with_seed(42).values["seed"], 42)
        self.assertNotEqual(a.values["seed"], 42)
        with self.assertRaises(ValueError):
            GenerationArguments.from_positional([1, 2])


class JobTests(unittest.TestCase):
    def test_cancel_does_not_interrupt_other_session_or_tab(self):
        coordinator = JobCoordinator()
        get = Mock(
            return_value=Mock(
                json=lambda: {"queue_running": [[0, "other"]], "queue_pending": []}
            )
        )
        post = Mock()
        with coordinator.run("alice", "h3") as job:
            job.prompt_id = "mine"
            coordinator.cancel("bob", "h3", get, post)
            self.assertFalse(job.cancelled.is_set())
            coordinator.cancel("alice", "ltx", get, post)
            self.assertFalse(job.cancelled.is_set())
            coordinator.cancel("alice", "h3", get, post)
            self.assertTrue(job.cancelled.is_set())
        post.assert_not_called()
        self.assertEqual(coordinator.active, {})

    def test_only_owned_pending_job_is_removed(self):
        coordinator = JobCoordinator()
        get = Mock(
            return_value=Mock(
                json=lambda: {
                    "queue_running": [[0, "other"]],
                    "queue_pending": [[1, "mine"], [2, "theirs"]],
                }
            )
        )
        post = Mock()
        with coordinator.run("alice", "h3") as job:
            job.prompt_id = "mine"
            coordinator.cancel("alice", "h3", get, post)
        post.assert_called_once_with("/queue", json={"delete": ["mine"]})

    def test_submission_prefix_preserves_family_folder(self):
        graph = {"1": {"inputs": {"filename_prefix": "h3/image_staging/output"}}}
        scoped_graph(graph, "token")
        self.assertEqual(
            graph["1"]["inputs"]["filename_prefix"], "h3/image_staging/output_token"
        )

    def test_fallback_never_selects_another_job(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            ours = root / "h3_ours_001.mp4"
            ours.touch()
            (root / "h3_theirs_001.mp4").touch()
            self.assertEqual(recent_output_candidates(root, frozenset({".mp4"}), 0), [])
            token = CURRENT_JOB.set(Job("a", "h3", output_token="ours"))
            try:
                self.assertEqual(
                    recent_output_candidates(root, frozenset({".mp4"}), 0),
                    [ours.resolve()],
                )
            finally:
                CURRENT_JOB.reset(token)

    def test_history_paths_are_contained(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            history = {"outputs": {"1": {"filename": "outside.mp4", "subfolder": ".."}}}
            self.assertEqual(
                history_output_candidates(root, history, frozenset({".mp4"})), []
            )

    def test_result_snapshot_is_independent_of_next_request(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "clip.mp4"
            settings = {"seed": 42, "steps": 8}
            write_snapshot(path, {"settings": settings})
            settings["seed"] = 7
            self.assertEqual(read_snapshot(path)["settings"]["seed"], 42)
            self.assertIsNone(read_snapshot(Path(d) / "old.mp4"))


if __name__ == "__main__":
    unittest.main()
