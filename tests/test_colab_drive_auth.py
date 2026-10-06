"""Reusable Drive authorization without live OAuth credentials or DriveFS."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.parse

import h3_colab_drive as drive


AUTH = dict(client_id="fixture-client", client_secret="fixture-secret", refresh_token="fixture-refresh")


class DriveAuthTests(unittest.TestCase):
    def setUp(self):
        self.google = ModuleType("google")
        self.colab = ModuleType("google.colab")
        class Missing(Exception):
            pass
        class Denied(Exception):
            pass
        self.userdata = SimpleNamespace(get=Mock(return_value=json.dumps(AUTH)),
                                        SecretNotFoundError=Missing, NotebookAccessError=Denied)
        self.colab.userdata = self.userdata
        self.google.colab = self.colab
        self.modules = {"google": self.google, "google.colab": self.colab}

    def test_valid_secret_mounts_without_browser_authorization(self):
        with patch.dict(sys.modules, self.modules), patch.object(drive, "drive_mounted", return_value=False), \
             patch.object(drive, "mount_with_credentials") as mount:
            drive.mount_from_secret("/content/drive")
        self.userdata.get.assert_called_once_with("H3_DRIVE_AUTH")
        mount.assert_called_once_with("/content/drive", AUTH)

    def test_existing_mount_does_not_read_secret(self):
        with patch.object(drive, "drive_mounted", return_value=True):
            drive.mount_from_secret("/content/drive")
        self.userdata.get.assert_not_called()

    def test_setup_skips_all_browser_ui_when_secret_works(self):
        self.colab.output = SimpleNamespace(eval_js=Mock())
        with patch.dict(sys.modules, self.modules), patch.object(drive.TokenProvider, "get"), \
             contextlib.redirect_stdout(io.StringIO()):
            drive.one_time_setup()
        self.colab.output.eval_js.assert_not_called()

    def test_setup_network_failure_does_not_restart_authorization(self):
        self.colab.output = SimpleNamespace(eval_js=Mock())
        with patch.dict(sys.modules, self.modules), \
             patch.object(drive.TokenProvider, "get", side_effect=drive.DriveAuthError("Retry later")):
            with self.assertRaises(drive.DriveAuthError):
                drive.one_time_setup()
        self.colab.output.eval_js.assert_not_called()

    def test_setup_uses_transient_ui_and_does_not_print_credentials(self):
        self.userdata.get.side_effect = self.userdata.SecretNotFoundError()
        self.colab.output = SimpleNamespace(eval_js=Mock(side_effect=[json.dumps({"installed": AUTH}), "callback", True]))
        with patch.dict(sys.modules, self.modules), patch.object(drive, "exchange_callback", return_value=AUTH), \
             patch.object(drive.TokenProvider, "get"), contextlib.redirect_stdout(io.StringIO()) as output:
            drive.one_time_setup()
        self.assertEqual(self.colab.output.eval_js.call_count, 3)
        for value in AUTH.values():
            self.assertNotIn(value, output.getvalue())
        self.assertIn("input.type = 'password'", self.colab.output.eval_js.call_args.args[0])

    def test_missing_denied_and_malformed_secret_do_not_mount(self):
        for value in (self.userdata.SecretNotFoundError(), self.userdata.NotebookAccessError(), "invalid-json", "{}"):
            with self.subTest(value=type(value).__name__), patch.dict(sys.modules, self.modules), \
                 patch.object(drive, "drive_mounted", return_value=False), patch.object(drive, "mount_with_credentials") as mount:
                self.userdata.get.side_effect = value if isinstance(value, Exception) else None
                self.userdata.get.return_value = value
                with self.assertRaises(drive.DriveAuthError):
                    drive.mount_from_secret("/content/drive")
                mount.assert_not_called()

    def test_access_token_cached_then_refreshed_without_consent(self):
        with patch.object(drive, "token_request", side_effect=[dict(access_token="first", expires_in=3600),
                                                              dict(access_token="second", expires_in=3600)]) as request:
            provider = drive.TokenProvider(AUTH)
            self.assertEqual(provider.get()["access_token"], "first")
            self.assertEqual(provider.get()["access_token"], "first")
            self.assertEqual(request.call_count, 1)
            provider.expiry = 0
            self.assertEqual(provider.get()["access_token"], "second")
            self.assertEqual(request.call_count, 2)
            self.assertEqual(request.call_args.args[0]["grant_type"], "refresh_token")

    def test_rejected_token_error_does_not_expose_credentials(self):
        error = urllib.error.HTTPError("https://oauth2.googleapis.com/token", 400, "fixture-refresh", {}, None)
        with patch.object(drive.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(drive.DriveAuthError) as raised:
                drive.token_request(dict(AUTH, grant_type="refresh_token"))
        self.assertNotIn("fixture-refresh", str(raised.exception))
        self.assertIn("replace H3_DRIVE_AUTH", str(raised.exception))

    def test_setup_uses_pkce_and_rejects_wrong_state_before_exchange(self):
        client = {k: AUTH[k] for k in ("client_id", "client_secret")}
        url, state, verifier = drive.authorization_request(client)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(query["access_type"], ["offline"])
        self.assertEqual(query["code_challenge_method"], ["S256"])
        with patch.object(drive, "token_request") as request:
            with self.assertRaises(drive.DriveAuthError):
                drive.exchange_callback(client, drive.REDIRECT_URI + "?state=wrong&code=test", state, verifier)
            request.assert_not_called()
        callback = drive.REDIRECT_URI + "?" + urllib.parse.urlencode(dict(state=state, code="test"))
        with patch.object(drive, "token_request", return_value=dict(access_token="fixture", refresh_token=AUTH["refresh_token"])) as request:
            self.assertEqual(drive.exchange_callback(client, callback, state, verifier), AUTH)
            self.assertEqual(request.call_args.args[0]["code_verifier"], verifier)

    def test_failed_mount_stops_token_server_and_reaps_process(self):
        with TemporaryDirectory() as directory:
            process = Mock()
            process.poll.side_effect = [None, 7, None]
            server = Mock(server_port=1234)
            with patch.object(drive, "drive_mounted", return_value=False), patch.object(Path, "is_file", return_value=True), \
                 patch.object(Path, "exists", return_value=True), patch.object(drive.TokenProvider, "get"), \
                 patch.object(drive.TokenProvider, "email", return_value="fixture@example.invalid"), \
                 patch.object(drive.http.server, "ThreadingHTTPServer", return_value=server), \
                 patch.object(drive.threading, "Thread"), patch.object(drive.subprocess, "Popen", return_value=process):
                with self.assertRaises(drive.DriveAuthError):
                    drive.mount_with_credentials(Path(directory), AUTH)
            server.shutdown.assert_called_once()
            server.server_close.assert_called_once()
            process.terminate.assert_called_once()
            process.wait.assert_called_once()

    def test_notebook_reusable_mount_failure_never_falls_back_to_consent(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "minimax_h3_colab.ipynb").read_text(encoding="utf-8"))
        source = next("".join(c["source"]) for c in notebook["cells"] if "def setup_google_drive_outputs(" in "".join(c["source"]))
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
        with TemporaryDirectory() as directory:
            namespace = dict(Path=Path, os=os, NOTEBOOK_DIR=Path(directory), DRIVE_AUTH_MODE="Reusable secret")
            exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)
            mount = Mock()
            self.colab.drive = SimpleNamespace(mount=mount)
            with patch.dict(sys.modules, self.modules), patch.object(os.path, "ismount", return_value=False), \
                 patch.object(drive, "mount_from_secret", side_effect=drive.DriveAuthError("H3_DRIVE_AUTH is missing")), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                namespace["setup_google_drive_outputs"](True, workspace_dir=directory)
            mount.assert_not_called()
            self.assertIn("H3_DRIVE_AUTH is missing", output.getvalue())
            self.assertFalse((Path(directory) / "h3").exists())


if __name__ == "__main__":
    unittest.main()
