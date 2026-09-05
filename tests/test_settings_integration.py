"""Gradio boundary regressions: migration, explicit actions and thread contexts."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import gradio as gr
from gradio.helpers import special_args
from h3_app.jobs import CURRENT_JOB, JOBS, JobCoordinator
from h3_app.provenance import RUN_CONTEXT
from h3_ui.job_bindings import owned_generation
from h3_ui.persistence import restore_preferences, PERSISTED_NAMES


class MigrationTests(unittest.TestCase):
    def components(self):
        return {
            "h3.preset": SimpleNamespace(
                value="Fast", choices=[("Fast", "Fast"), ("Quality", "Quality")]
            ),
            "h3.generation_mode": SimpleNamespace(
                value="Turbo", choices=[("Turbo", "Turbo"), ("Normal", "Normal")]
            ),
            "h3.steps": SimpleNamespace(value=4, minimum=4, maximum=30),
            "h3.stage_model_offload": SimpleNamespace(value=False),
            "h3.prompt": SimpleNamespace(value=""),
        }

    def test_v3_values_are_preserved_without_reapplying_preset(self):
        values, memory = restore_preferences(
            {"h3.preset": "Quality", "h3.steps": 12}, self.components()
        )
        self.assertEqual(values["h3.preset"], "Quality")
        self.assertEqual(values["h3.steps"], 12)
        self.assertEqual(memory["active"], "Turbo")
        self.assertNotIn("h3.prompt", values)

    def test_invalid_values_fall_back_individually(self):
        values, _ = restore_preferences(
            {
                "values": {
                    "h3.preset": "removed",
                    "h3.steps": float("nan"),
                    "h3.stage_model_offload": "false",
                }
            },
            self.components(),
        )
        self.assertEqual(values["h3.preset"], "Fast")
        self.assertEqual(values["h3.steps"], 4)
        self.assertFalse(values["h3.stage_model_offload"])

    def test_mode_preferences_validate_on_restore(self):
        _, memory = restore_preferences(
            {
                "schema_version": 4,
                "values": {"h3.steps": 9},
                "mode_memory": {
                    "modes": {"Normal": {"steps": 22}, "Turbo": {"steps": 999}},
                    "offload_preference": False,
                },
            },
            self.components(),
        )
        self.assertEqual(memory["modes"]["Normal"]["steps"], 22)
        self.assertEqual(memory["modes"]["Turbo"]["steps"], 4)
        self.assertFalse(memory["offload_preference"])

    def test_persistence_allowlist_excludes_secrets_and_outputs(self):
        for field in (
            "h3.prompt",
            "h3.gemini_api_key",
            "h3.lightning_api_key",
            "h3.first",
            "gallery.confirm_delete",
            "h3.image_selection",
        ):
            self.assertNotIn(field, PERSISTED_NAMES)


class JobBoundaryTests(unittest.TestCase):
    def test_gradio_injects_request_without_an_extra_input(self):
        def generate(prompt):
            yield prompt

        fn = owned_generation(generate, "ltx")
        request = gr.Request(session_hash="session")
        args, *_ = special_args(fn, ["test"], request=request)
        self.assertIs(args[-1], request)
        self.assertEqual(list(fn(*args)), ["test"])

    def test_context_is_scoped_to_each_generator_advance(self):
        seen = []

        def generate(prompt):
            for _ in range(2):
                seen.append(CURRENT_JOB.get().id)
                yield prompt

        fn = owned_generation(generate, "ltx")
        iterator = fn("test", gr.Request(session_hash="session"))
        with (
            ThreadPoolExecutor(max_workers=1) as first,
            ThreadPoolExecutor(max_workers=1) as second,
        ):
            self.assertEqual(first.submit(next, iterator).result(), "test")
            self.assertEqual(second.submit(next, iterator).result(), "test")
            second.submit(iterator.close).result()
        self.assertEqual(seen[0], seen[1])
        self.assertIsNone(CURRENT_JOB.get())
        self.assertEqual(JOBS.active, {})

    def test_h3_preset_metadata_stays_out_of_generation_arguments(self):
        seen = []

        def generate(batch_count, value):
            seen.append(RUN_CONTEXT.get())
            yield value

        fn = owned_generation(generate, "h3", ("batch_count", "value", "preset"))
        self.assertEqual(
            list(fn(2, "result", "Quality", gr.Request(session_hash="session"))),
            ["result"],
        )
        self.assertEqual(seen, [{"preset": "Quality", "batch_count": 2}])

    def test_metadata_uses_original_paths_and_retains_streaming_results(self):
        from unittest.mock import patch

        def generate(batch_count):
            yield ("original.mp4", *({} for _ in range(11)))
            yield tuple({} for _ in range(12))

        fn = owned_generation(
            generate, "h3", ("batch_count", "preset"), metadata_output=True
        )
        with patch(
            "h3_ui.job_bindings.render_snapshot", return_value="metadata"
        ) as render:
            updates = list(fn(1, "Fast", gr.Request(session_hash="session")))
        self.assertEqual([u[-1] for u in updates], ["metadata", "metadata"])
        self.assertEqual(render.call_args_list[0].args, (["original.mp4"],))
        self.assertEqual(render.call_args_list[1].args, (["original.mp4"],))

    def test_cancel_preparation_does_not_require_backend(self):
        jobs = JobCoordinator()
        get, post = Mock(), Mock()
        with jobs.run("session", "h3"):
            jobs.cancel("session", "h3", get, post)
        get.assert_not_called()
        post.assert_not_called()
