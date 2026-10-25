"""Exercise GPU server lifecycle without allocating a Modal container."""
import ast
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from h3_app.workspace_store import WorkspaceStore


class ModalLifecycleTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "modal_h3.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = {node.name: node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        service = nodes["H3Service"]
        self.hooks = {
            method.name: ast.literal_eval(method.decorator_list[0].keywords[0].value)
            for method in service.body if isinstance(method, ast.FunctionDef)
            and method.name in {"start", "check_services"}
        }
        self.preloaded_hooks = {
            method.name: method.decorator_list[0].keywords[0].value.value
            for method in nodes["PreloadedService"].body if isinstance(method, ast.FunctionDef)
            and method.name in {"start", "resume"}
        }
        # Execute the real lifecycle bodies without SDK decoration or image builds.
        body = [nodes[name] for name in (
            "start_comfy", "gradio_env", "publish_gradio", "wait_for_preload",
            "H3Service", "PreloadedService",
        )]
        for node in body:
            node.decorator_list = []
            for method in getattr(node, "body", ()):
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
            "subprocess": SimpleNamespace(Popen=self.calls.launch, PIPE="pipe"),
            "prepare_runtime_models": self.calls.prepare,
            "sync_ltx25_workflows": self.calls.sync,
            "service_env": Mock(return_value={}),
            "wait_for_service": self.calls.ready,
            "wait_for_comfy_frontend": self.calls.frontend,
            "volume": SimpleNamespace(commit=self.calls.commit),
            "time": time,
            "SNAPSHOT_READY": PurePosixPath("/tmp/h3-snapshot-preload-ready"),
            "PRELOAD_TIMEOUT": 60,
        }
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), self.env)
        self.service = self.env["H3Service"]()

    def preloaded(self, target="video"):
        self.env["wait_for_preload"] = self.calls.preload
        service = type("Preloaded", (self.env["PreloadedService"],), {"preload": target})()
        return service

    def test_snapshot_captures_comfy_without_workspace_state(self):
        self.assertEqual(self.hooks, {"start": True, "check_services": False})
        self.service.start()
        self.assertEqual([call[0] for call in self.calls.mock_calls], [
            "prepare", "sync", "launch", "ready",
        ])
        checks = self.calls.ready.call_args_list
        self.assertEqual(checks[0].args, ("http://127.0.0.1:8188/system_stats", self.comfy))
        self.assertIs(self.service.comfy_process, self.comfy)
        self.assertFalse(hasattr(self.service, "gradio_process"))

    def test_restore_launches_gradio_and_commits_before_endpoint(self):
        self.service.start()
        self.calls.reset_mock()
        self.service.check_services()
        self.assertIsNone(self.service.serve())
        self.assertEqual([call[0] for call in self.calls.mock_calls], [
            "ready", "launch", "ready", "commit", "frontend",
        ])
        checks = self.calls.ready.call_args_list
        self.assertIs(checks[0].args[1], self.comfy)
        self.assertIs(checks[1].args[1], self.gradio)
        self.assertEqual(self.calls.launch.call_args.args, (["python", "-u", "/opt/h3/gradio_app.py"],))
        self.assertEqual(self.calls.launch.call_args.kwargs["env"]["GRADIO_SERVER_PORT"], "7860")
        self.assertIs(self.service.gradio_process, self.gradio)
        self.calls.prepare.assert_not_called()

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
        self.calls.launch.assert_called_once()
        self.calls.commit.assert_not_called()
        self.calls.frontend.assert_not_called()
        self.calls.prepare.assert_not_called()

    def test_restored_comfy_failure_does_not_launch_gradio(self):
        self.service.start()
        self.calls.reset_mock()
        self.calls.ready.side_effect = RuntimeError("ComfyUI exited")
        with self.assertRaisesRegex(RuntimeError, "ComfyUI exited"):
            self.service.check_services()
        self.calls.launch.assert_not_called()
        self.calls.commit.assert_not_called()

    def test_restore_initializes_missing_workspace_before_media_queries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "h3-workspace"
            self.service.start()
            self.assertFalse(root.exists())
            stores = []

            def launch_gradio(*args, **kwargs):
                stores.append(WorkspaceStore(root))
                return self.gradio

            self.calls.launch.side_effect = launch_gradio
            self.service.check_services()
            store, = stores
            self.assertTrue(store.database.is_file())
            self.assertEqual(store.catalog_page(kind="Image"), ([], 0))
            path = Path(directory) / "historical.png"
            path.write_bytes(b"historical media")
            store.index_assets([(path, "Image", {"_media": {"registered_ns": 1}})])
            rows, total = store.catalog_page(kind="Image")
            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["path"], str(path))
            self.calls.commit.assert_called_once()

    def test_restore_reads_workspace_created_after_snapshot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "h3-workspace"
            self.service.start()
            # Model a workspace committed by a serving container after the
            # ComfyUI snapshot was captured, then mounted on a new container.
            existing = WorkspaceStore(root)
            token = existing.issue_owner()
            path = Path(directory) / "existing.png"
            path.write_bytes(b"existing media")
            existing.index_assets([(path, "Image", {"_media": {"registered_ns": 1}})])
            asset = existing.catalog_page(kind="Image")[0][0]
            existing.annotate(asset["id"], tags=["saved"], favorite=True)
            stores = []

            def launch_gradio(*args, **kwargs):
                stores.append(WorkspaceStore(root))
                return self.gradio

            self.calls.launch.side_effect = launch_gradio
            self.service.check_services()
            store, = stores
            self.assertEqual(store.verify_owner(token), existing.verify_owner(token))
            restored = store.catalog_page(kind="Image", favorite=True)[0][0]
            self.assertEqual(restored["tags"], ["saved"])
            self.assertTrue(restored["favorite"])

    def test_preloaded_snapshot_loads_models_before_capture(self):
        self.assertEqual(self.preloaded_hooks, {"start": True, "resume": False})
        service = self.preloaded("soundstage")
        service.start()
        self.assertEqual([call[0] for call in self.calls.mock_calls], [
            "prepare", "sync", "launch", "ready", "launch", "preload", "commit",
        ])
        launch = self.calls.launch.call_args
        self.assertEqual(launch.args, (["python", "-u", "/opt/h3/gradio_app.py"],))
        self.assertEqual(launch.kwargs["stdin"], "pipe")
        self.assertEqual(launch.kwargs["env"]["H3_SNAPSHOT_PRELOAD"], "soundstage")
        self.assertEqual(launch.kwargs["env"]["H3_SNAPSHOT_READY_FILE"], "/tmp/h3-snapshot-preload-ready")
        self.assertIs(self.calls.preload.call_args.args[0], self.gradio)
        self.gradio.stdin.write.assert_not_called()

    def test_preloaded_restore_starts_ui_without_relaunching(self):
        service = self.preloaded()
        service.start()
        self.calls.reset_mock()
        service.resume()
        self.assertEqual([call[0] for call in self.calls.mock_calls], [
            "ready", "ready", "commit", "frontend",
        ])
        self.gradio.stdin.write.assert_called_once_with(b"serve\n")
        self.assertIs(self.calls.ready.call_args_list[1].args[1], self.gradio)
        self.calls.launch.assert_not_called()
        self.calls.prepare.assert_not_called()

    def test_preloaded_restore_waits_for_comfy_before_ui(self):
        service = self.preloaded()
        service.start()
        self.calls.reset_mock()
        self.calls.ready.side_effect = RuntimeError("ComfyUI exited")
        with self.assertRaisesRegex(RuntimeError, "ComfyUI exited"):
            service.resume()
        self.gradio.stdin.write.assert_not_called()
        self.calls.commit.assert_not_called()

    def test_preload_exit_fails_the_snapshot(self):
        process = Mock(poll=Mock(return_value=3), returncode=3)
        with self.assertRaisesRegex(RuntimeError, "exited before its models loaded: 3"):
            self.env["wait_for_preload"](process, timeout=5, label="video preload")


if __name__ == "__main__":
    unittest.main()
