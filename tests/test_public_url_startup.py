"""Exercise launcher tunnel selection and cleanup without serving publicly."""

import ast
import argparse
import contextlib
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app.public_url import public_url_settings


ROOT = Path(__file__).resolve().parents[1]


class PublicUrlStartupTests(unittest.TestCase):
    def run_launcher(self, environment, *, started=True, launch_error=None, stop_error=None):
        tree = ast.parse((ROOT / "h3_ui/application.py").read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            demo = Mock()
            demo.queue.return_value = demo
            server = Mock(started=started)
            thread = Mock()
            # The server is alive during readiness/selection, then exits.
            thread.is_alive.side_effect = [True, True, False] if started else [False, False, False]
            launch = Mock(side_effect=launch_error)
            stop = Mock(side_effect=stop_error)
            gradio = Mock()
            namespace = dict(
                argparse=argparse, os=os, Path=Path,
                INPUT_DIR=root / "input", OUTPUT_DIR=root / "output",
                OUTPUTS_DIR=root / "results", SCRIPT_DIR=root,
                COMFY_URL="http://localhost:8188", COMFY_DIR=root,
                MODELS_CONFIG=root / "models.json",
                SERVER_ATTENTION_BACKEND="sol", SERVER_DENSE_ATTENTION_BACKEND="comfy-kitchen",
                build_ui=Mock(return_value=demo), build_server=Mock(),
                public_url_settings=public_url_settings,
                threading=SimpleNamespace(Event=Mock(return_value=Mock(is_set=Mock(return_value=False))), Thread=Mock(return_value=thread)),
                signal=Mock(),
                uvicorn=SimpleNamespace(Server=Mock(return_value=server), Config=Mock()),
                UVICORN_WEBSOCKET_OPTIONS={},
                launch_cloudflare=launch, stop_cloudflare=stop, gradio_networking=gradio,
            )
            exec(compile(ast.Module(body=[main], type_ignores=[]), "<launcher>", "exec"), namespace)
            with patch.dict(os.environ, environment, clear=True), patch("sys.argv", ["gradio_app.py"]), contextlib.redirect_stdout(io.StringIO()):
                if stop_error:
                    with self.assertRaises(type(stop_error)):
                        namespace["main"]()
                else:
                    namespace["main"]()
            self.assertTrue(server.should_exit)
            self.assertTrue(server.force_exit)
            demo.close.assert_called_once()
            return launch, stop, gradio

    def test_default_starts_cloudflare_on_selected_port_and_stops_it(self):
        launch, stop, gradio = self.run_launcher({"GRADIO_SERVER_PORT": "9876"})
        self.assertEqual(launch.call_args.args, (9876,))
        stop.assert_called_once_with(launch.return_value)
        gradio.setup_tunnel.assert_not_called()

    def test_gradio_share_requires_opt_in(self):
        launch, stop, gradio = self.run_launcher({"GRADIO_SHARE": "true"})
        launch.assert_not_called()
        gradio.setup_tunnel.assert_called_once()
        stop.assert_called_once_with(None)

    def test_local_only_starts_no_tunnel(self):
        launch, _, gradio = self.run_launcher({"ENABLE_CLOUDFLARE_TUNNEL": "false"})
        launch.assert_not_called()
        gradio.setup_tunnel.assert_not_called()

    def test_failed_server_starts_no_tunnel(self):
        launch, _, gradio = self.run_launcher({}, started=False)
        launch.assert_not_called()
        gradio.setup_tunnel.assert_not_called()

    def test_interrupted_tunnel_start_still_closes_the_server(self):
        self.run_launcher({}, launch_error=KeyboardInterrupt())

    def test_tunnel_cleanup_failure_still_closes_the_server(self):
        self.run_launcher({}, stop_error=ProcessLookupError())

    def test_modal_disables_both_public_tunnels(self):
        tree = ast.parse((ROOT / "modal_h3.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "service_env")
        namespace = dict(os=os, COMFY=ROOT, CONFIG=ROOT, OUTPUT=ROOT, DATA=ROOT, COMFY_PORT=8188)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<modal env>", "exec"), namespace)
        environment = namespace["service_env"]()
        self.assertEqual(environment["ENABLE_CLOUDFLARE_TUNNEL"], "false")
        self.assertEqual(environment["GRADIO_SHARE"], "false")
