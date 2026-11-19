"""Remake editor and render state regressions, without GPU or media processing."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from h3_app.generation.results import GenerationUpdate
from h3_app.remake.project import OPTION_DEFAULTS
from h3_app.remake.runner import RunOptions, WindowRunner


class RemakeStateTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"source")
        self.picture = self.root / "picture.png"
        self.picture.write_bytes(b"first reference")
        self.used_prompts = []
        self.during_take = lambda: None

        def take(values, graph, progress):
            self.used_prompts.append(values["prompt"])
            self.during_take()
            output = self.root / "take.mp4"
            output.write_bytes(b"take")
            yield GenerationUpdate(str(output), "done")

        self.runner = WindowRunner(lambda: SimpleNamespace(output_dir=self.root),
                                   lambda *args, **kwargs: "staged", take, lambda: {})
        self.options = RunOptions("Swap", True, True, "make a robot", "", "", "")
        self.project = self.runner.projects.create(
            "local", "project", str(self.source),
            {"start": 0, "frames": 90, "width": 832, "height": 480,
             **{key: getattr(self.options, key) for key in OPTION_DEFAULTS}},
            [{"render": [0, 90], "own": [0, 90], "continues": False, "hold": 0}],
        )
        self.project = self.save()
        self.project = self.runner.projects.edit(self.project, lambda current: current["windows"][0].update(
            state="done", prompt_auto=True, prompt_pictures=1,
        ))

    def save(self, **options):
        return self.runner.store_editor(self.project, 0, "robot prompt", True,
                                        [str(self.picture)], [12], options=options)

    def render(self, options=None):
        with patch("h3_app.remake.runner.trim_source"), patch("h3_app.remake.runner._drop_extras"), \
                patch.object(WindowRunner, "_write_prompt", return_value="fresh prompt") as writer:
            list(self.runner._render(self.project, 0, options or self.options, None))
        return writer

    def test_replacing_reference_at_same_count_rewrites_automatic_prompt(self):
        self.picture.write_bytes(b"replacement reference")
        self.project = self.save()
        self.assertEqual(self.project["windows"][0]["state"], "changed")
        writer = self.render()
        writer.assert_called_once()
        self.assertEqual(self.used_prompts, ["fresh prompt"])

    def test_edit_request_change_queues_and_rewrites_automatic_prompt(self):
        self.project = self.save(request="make a dog")
        self.assertEqual(self.project["windows"][0]["state"], "changed")
        writer = self.render(replace(self.options, request="make a dog"))
        self.assertEqual(writer.call_args.args[-1].request, "make a dog")
        self.assertEqual(self.used_prompts, ["fresh prompt"])

    def test_typed_prompt_survives_reference_and_request_changes(self):
        self.project = self.runner.projects.edit(self.project, lambda current: current["windows"][0].update(
            prompt_auto=False,
        ))
        self.picture.write_bytes(b"replacement reference")
        self.project = self.save(request="make a dog")
        self.render(replace(self.options, request="make a dog")).assert_not_called()
        self.assertEqual(self.used_prompts, ["robot prompt"])

    def test_conditioning_changes_queue_all_rendered_windows(self):
        def add_windows(current):
            current["windows"].extend(
                {**current["windows"][0], "index": index, "state": state, "prompt_auto": False}
                for index, state in enumerate(("outdated", "pending", "failed"), 1)
            )
        self.project = self.runner.projects.edit(self.project, add_windows)
        for key, value in (("mode", "Edit"), ("pinned", False), ("smart", False)):
            with self.subTest(option=key):
                self.runner.projects.edit(self.project, lambda current: (
                    current.update(OPTION_DEFAULTS),
                    current["windows"][0].update(state="done"),
                    current["windows"][1].update(state="outdated"),
                ))
                saved = self.save(**{key: value})
                self.assertEqual([window["state"] for window in saved["windows"]],
                                 ["changed", "changed", "pending", "failed"])

    def test_unchanged_options_do_not_queue_finished_windows(self):
        saved = self.save(**{key: getattr(self.options, key) for key in OPTION_DEFAULTS})
        self.assertEqual(saved["windows"][0]["state"], "done")

    def test_late_autosave_cannot_overwrite_newer_unload_save(self):
        def save(index, revision, prompt, request):
            return self.runner.store_editor(
                self.project, index, prompt, True, [str(self.picture)], [12],
                options={"request": request}, save_order=["editor", revision],
            )

        self.project = self.runner.projects.edit(self.project, lambda current: current["windows"].append(
            {**current["windows"][0], "index": 1},
        ))
        save(0, 3, "robot prompt", "new request")
        save(0, 2, "discarded prompt", "old request")
        saved = save(1, 1, "second window's prompt", "old request")
        self.assertEqual(saved["request"], "new request")
        self.assertEqual([window["prompt"] for window in saved["windows"]],
                         ["robot prompt", "second window's prompt"])

    def test_prompt_settings_do_not_queue_typed_windows(self):
        self.project = self.runner.projects.edit(self.project, lambda current: current["windows"].append(
            {**current["windows"][0], "index": 1, "prompt_auto": False},
        ))
        for options in ({"request": "make a dog"}, {"auto_prompts": False}):
            with self.subTest(options=options):
                saved = self.save(**options)
                self.assertEqual([window["state"] for window in saved["windows"]], ["changed", "done"])

    def test_option_change_during_render_keeps_take_marked_changed(self):
        self.during_take = lambda: self.save(mode="Edit")
        self.render()
        saved = self.runner.projects.load("local", "project")
        self.assertEqual(saved["mode"], "Edit")
        self.assertEqual(saved["windows"][0]["state"], "changed")
        self.assertIsNotNone(saved["windows"][0]["take"])


if __name__ == "__main__":
    unittest.main()
