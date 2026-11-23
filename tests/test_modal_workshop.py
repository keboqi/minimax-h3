"""Verify the minimal CPU Workshop Modal deploy script without allocating a container."""
import ast
import os
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "modal_workshop.py"


class ModalWorkshopDeployTests(unittest.TestCase):
    def _exec_module(self, *, local: bool = True, env: dict[str, str] | None = None, requirements=None):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        image = Mock()
        for method in ("entrypoint", "apt_install", "env", "uv_pip_install", "run_commands",
                       "add_local_file", "add_local_dir", "run_function"):
            getattr(image, method).return_value = image

        from_registry = Mock(side_effect=AssertionError("minimal workshop must not use from_registry"))
        debian_slim = Mock(return_value=image)
        volume = Mock(name="volume")
        secret_from_name = Mock(side_effect=lambda name: SimpleNamespace(name=name))
        app = Mock(name="app")
        app.function.return_value = lambda fn: fn

        fake_modal = SimpleNamespace(
            is_local=lambda: local,
            Image=SimpleNamespace(debian_slim=debian_slim, from_registry=from_registry),
            Volume=SimpleNamespace(from_name=Mock(return_value=volume)),
            Secret=SimpleNamespace(from_name=secret_from_name),
            App=Mock(return_value=app),
            concurrent=lambda **kwargs: (lambda fn: fn),
            web_server=lambda *args, **kwargs: (lambda fn: fn),
        )

        body = [
            node for node in tree.body
            if not (isinstance(node, ast.Import) and any(alias.name == "modal" for alias in node.names))
        ]
        namespace = {"__file__": str(SCRIPT), "modal": fake_modal}
        patch_modules = {}
        if requirements is not None:
            patch_modules["h3_workshop_requirements"] = SimpleNamespace(
                WORKSHOP_MINIMAL_REQUIREMENTS=requirements
            )
        elif not local:
            patch_modules["h3_workshop_requirements"] = None

        with (
            patch.dict(os.environ, env or {}, clear=False),
            patch.dict(sys.modules, patch_modules),
            patch.object(sys, "path", list(sys.path)),
        ):
            exec(compile(ast.Module(body=body, type_ignores=[]), str(SCRIPT), "exec"), namespace)
        return namespace, fake_modal, image, debian_slim

    def test_minimal_image_skips_apt_playwright_and_heavy_build_steps(self):
        ns, _modal, image, debian_slim = self._exec_module(
            local=True,
            requirements=("gradio==6.31.0", "requests>=2.32"),
        )
        debian_slim.assert_called_once_with(python_version="3.12")
        image.uv_pip_install.assert_called_once_with("gradio==6.31.0", "requests>=2.32")
        image.apt_install.assert_not_called()
        image.run_commands.assert_not_called()
        image.run_function.assert_not_called()
        self.assertEqual(ns["APP"], "minimax-h3-workshop")
        self.assertEqual(ns["VOL"], "minimax-h3-data")
        self.assertEqual(ns["LABEL"], "workshop-cpu")
        self.assertEqual(ns["secrets"], [])

    def test_minimal_requirements_exclude_playwright_and_video_downloaders(self):
        from h3_workshop_requirements import WORKSHOP_MINIMAL_REQUIREMENTS

        joined = " ".join(WORKSHOP_MINIMAL_REQUIREMENTS).lower()
        for excluded in ("playwright", "yt-dlp", "deno", "openai", "json-repair"):
            self.assertNotIn(excluded, joined)
        for required in ("gradio==", "huggingface-hub", "requests", "websocket-client", "aiohttp", "wsproto=="):
            self.assertIn(required, joined)

    def test_mounts_are_runtime_only_and_deduplicated(self):
        _ns, _modal, image, _debian_slim = self._exec_module(local=True)
        file_mounts = [call for call in image.mock_calls if call[0] == "add_local_file"]
        dir_mounts = [call for call in image.mock_calls if call[0] == "add_local_dir"]
        remote_files = [call.kwargs["remote_path"] for call in file_mounts]
        remote_dirs = [call.kwargs["remote_path"] for call in dir_mounts]
        self.assertEqual(len(remote_files), len(set(remote_files)))
        self.assertEqual(len(remote_dirs), len(set(remote_dirs)))
        self.assertTrue(all(call.kwargs["copy"] is False for call in (*file_mounts, *dir_mounts)))
        self.assertIn("/opt/h3/h3_workshop.py", remote_files)
        self.assertIn("/opt/h3/h3_models.py", remote_files)
        self.assertEqual(set(remote_dirs), {"/opt/h3/h3_app", "/opt/h3/h3_ui", "/opt/h3/prompts"})

    def test_remote_import_does_not_require_workshop_requirements_module(self):
        _ns, _modal, image, _debian_slim = self._exec_module(local=False)
        image.uv_pip_install.assert_not_called()
        image.add_local_file.assert_not_called()
        image.add_local_dir.assert_not_called()

    def test_optional_secret_is_attached_only_when_configured(self):
        ns, fake_modal, _image, _debian_slim = self._exec_module(
            local=True,
            env={"H3_MODAL_HF_SECRET": "custom-secret"},
        )
        fake_modal.Secret.from_name.assert_called_once_with("custom-secret")
        self.assertEqual(len(ns["secrets"]), 1)

    def test_workshop_decorator_is_cpu_only_without_snapshots(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        workshop_fn = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "workshop"
        )
        fn_kwargs = {
            kw.arg: kw.value
            for dec in workshop_fn.decorator_list
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "function"
            for kw in dec.keywords
        }
        self.assertEqual(ast.literal_eval(fn_kwargs["max_containers"]), 1)
        for forbidden in ("gpu", "cpu", "memory", "min_containers", "enable_memory_snapshot", "experimental_options"):
            self.assertNotIn(forbidden, fn_kwargs)

    def test_workshop_entrypoint_prepares_models_and_serves_cpu_ui(self):
        ns, _modal, _image, _debian_slim = self._exec_module(local=True)
        serve_mock = Mock()
        prepare_mock = Mock()
        ns["prepare_runtime_models"] = prepare_mock
        with (
            patch.dict(sys.modules, {"h3_workshop": SimpleNamespace(serve=serve_mock)}),
            patch.dict(os.environ, {}, clear=False),
        ):
            ns["workshop"]()
            self.assertEqual(os.environ["H3_CPU_ONLY"], "1")
            self.assertEqual(os.environ["GRADIO_SERVER_NAME"], "0.0.0.0")
            self.assertEqual(os.environ["GRADIO_SERVER_PORT"], "7860")
        prepare_mock.assert_called_once_with()
        serve_mock.assert_called_once_with(
            Path(PurePosixPath("/data")),
            ns["volume"],
            port=7860,
            script_dir=Path(PurePosixPath("/opt/h3")),
        )


if __name__ == "__main__":
    unittest.main()
