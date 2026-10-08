"""Exercise GPU server lifecycle without allocating a Modal container."""
import ast
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class ModalLifecycleTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "modal_h3.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        service = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "H3Service")
        # Execute the real lifecycle bodies without SDK decoration or image builds.
        service.decorator_list = []
        for method in service.body:
            if isinstance(method, ast.FunctionDef):
                method.decorator_list = []
        self.calls = Mock()
        self.comfy = Mock(name="comfy_process")
        self.gradio = Mock(name="gradio_process")
        self.calls.launch.side_effect = [self.comfy, self.gradio]
        self.calls.sync.return_value = ["workflow.json"]
        self.env = {
            "Path": Path,
            "COMFY": PurePosixPath("/opt/h3/ComfyUI"),
            "UI": PurePosixPath("/opt/h3/gradio_app.py"),
            "COMFY_PORT": 8188,
            "UI_PORT": 7860,
            "os": SimpleNamespace(getenv=Mock(return_value=None)),
            "print": Mock(),
            "subprocess": SimpleNamespace(Popen=self.calls.launch),
            "prepare_runtime_models": self.calls.prepare,
            "sync_ltx25_workflows": self.calls.sync,
            "service_env": Mock(return_value={}),
            "wait_for_service": self.calls.ready,
            "wait_for_comfy_frontend": self.calls.frontend,
        }
        exec(compile(ast.Module(body=[service], type_ignores=[]), str(path), "exec"), self.env)
        self.service = self.env["H3Service"]()

    def test_snapshot_start_waits_for_both_servers_and_proxy(self):
        self.service.start()
        self.assertEqual([call[0] for call in self.calls.mock_calls], [
            "prepare", "sync", "launch", "ready", "launch", "ready", "frontend",
        ])
        checks = self.calls.ready.call_args_list
        self.assertEqual(checks[0].args, ("http://127.0.0.1:8188/system_stats", self.comfy))
        self.assertEqual(checks[1].args, ("http://127.0.0.1:7860/", self.gradio))
        self.assertIs(self.service.comfy_process, self.comfy)
        self.assertIs(self.service.gradio_process, self.gradio)

    def test_restore_and_endpoint_reuse_captured_servers(self):
        self.service.start()
        self.calls.reset_mock()
        self.service.check_services()
        self.assertIsNone(self.service.serve())
        self.assertEqual([call[0] for call in self.calls.mock_calls], ["ready", "ready"])
        checks = self.calls.ready.call_args_list
        self.assertIs(checks[0].args[1], self.comfy)
        self.assertIs(checks[1].args[1], self.gradio)

    def test_comfy_start_failure_does_not_launch_gradio(self):
        self.calls.ready.side_effect = RuntimeError("ComfyUI exited")
        with self.assertRaisesRegex(RuntimeError, "ComfyUI exited"):
            self.service.start()
        self.calls.launch.assert_called_once()
        self.calls.frontend.assert_not_called()

    def test_restore_health_failure_prevents_successful_enter(self):
        self.service.start()
        self.calls.reset_mock()
        self.calls.ready.side_effect = [None, RuntimeError("Gradio exited")]
        with self.assertRaisesRegex(RuntimeError, "Gradio exited"):
            self.service.check_services()
        self.calls.launch.assert_not_called()
        self.calls.prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
