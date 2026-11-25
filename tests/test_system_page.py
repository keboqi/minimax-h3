"""Tests for the standalone System page (/system/) and global session Gemini API key."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from h3_app.capabilities import ENGINES
from h3_app.config import RuntimeConfig
from h3_app.drama3.serving import config as d3_config
from h3_app.hosted_image import NANO_BANANA, hosted_image_endpoint
from h3_app.jobs import JobCoordinator, JOBS
from h3_app.llm import decisions, gemini, gemma
from h3_app.project_transfer import normalize_server_url
from h3_app.workspace_activity import workspace_task
from h3_app.workspace_store import COOKIE, WorkspaceStore, install_owner_cookie
from h3_models import MODEL_SPECS
from h3_ui.page_fields import writer_defaults, writer_rows
from h3_ui.page_runs import Run, Runs
from h3_ui.system.page import (
    SystemPageServices,
    SystemPageState,
    install_system_page,
    parse_backend_status,
)

API = "/system/api"


class FakeIndex:
    def __init__(self):
        self.calls = 0

    def rebuild(self):
        self.calls += 1
        yield "Scanning historical video files…"
        yield "Historical scan complete: 3 files indexed."


class SystemPageTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig.from_environment(
            self.root / "app",
            {
                "COMFY_DIR": str(self.root / "ComfyUI"),
                "GRADIO_OUTPUT_DIR": str(self.root / "gradio_outputs"),
            },
        )
        self.models_root = self.runtime.comfy_dir / "models"
        self.models_root.mkdir(parents=True, exist_ok=True)
        for folder in (self.runtime.input_dir, self.runtime.output_dir, self.runtime.outputs_dir):
            folder.mkdir(parents=True, exist_ok=True)

        self.backend_text = (
            "Connected · NVIDIA H100 80GB HBM3 · 54.2/79.2 GiB VRAM free · "
            "sparse: Sage 2 · dense: Flash 3 · memory: High VRAM · "
            "FirstBlockCache: available · EasyCache: available  \n"
            "**Spectrum accelerator**: available  \n"
            "**Singularity** · FL2VA `a.safetensors` · Ref2VA `b.safetensors`"
        )
        self.index = FakeIndex()
        self.changed = Mock()
        self.unloaded = Mock(side_effect=self._do_unload)
        self.specs = {
            "h3_image_dit": MODEL_SPECS["h3_image_dit"],
            "drama3_comfy_w4a8": MODEL_SPECS["drama3_comfy_w4a8"],
            "ltx25_distilled_int8": MODEL_SPECS["ltx25_distilled_int8"],
        }
        present_spec = self.specs["h3_image_dit"]
        present_file = self.models_root / present_spec.folder / present_spec.local_name
        present_file.parent.mkdir(parents=True, exist_ok=True)
        present_file.write_bytes(b"weights")

        self.services = SystemPageServices(
            backend_status=lambda: self.backend_text,
            unload_all_models=self.unloaded,
            store=self.store,
            runtime_config=lambda: self.runtime,
            index=lambda: self.index,
            model_specs=self.specs,
            models_root=lambda: self.models_root,
            on_changed=self.changed,
        )
        self.pages = SystemPageState(self.services)
        self.app = FastAPI()
        install_owner_cookie(self.app, self.store)
        install_system_page(self.app, pages=self.pages)
        self.client = TestClient(self.app)

        boot_resp = self.client.get(f"{API}/boot")
        self.assertEqual(boot_resp.status_code, 200, boot_resp.text)
        self.boot = boot_resp.json()
        self.owner = self.store.verify_owner(self.client.cookies[COOKIE])
        self.addCleanup(gemini.clear_session_key, self.owner)

    def _do_unload(self):
        self.backend_text = (
            "Connected · NVIDIA H100 80GB HBM3 · 76.8/79.2 GiB VRAM free · "
            "sparse: Sage 2 · dense: Flash 3 · memory: High VRAM · "
            "FirstBlockCache: available · EasyCache: available  \n"
            "**Spectrum accelerator**: available"
        )
        return ("All models unloaded and cached VRAM released.", self.backend_text)

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def refused(self, response, snippet):
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(snippet, response.json()["error"])

    def wait_run(self, kind: str):
        for _ in range(100):
            run = self.ok(self.client.get(f"{API}/run"))["runs"][kind]
            if run and not run["active"]:
                return run
            time.sleep(0.03)
        self.fail(f"System run {kind!r} did not finish.")

    # 1. Static routes ----------------------------------------------------------
    def test_static_routes_and_assets_are_served(self):
        redir = self.client.get("/system", follow_redirects=False)
        self.assertIn(redir.status_code, (302, 307))
        self.assertTrue(redir.headers["location"].endswith("/system/"))

        html = self.client.get("/system/")
        self.assertEqual(html.status_code, 200)
        self.assertIn("app.js", html.text)
        self.assertIn("remote-kind-filters", html.text)
        self.assertIn("remote-projects-empty", html.text)
        self.assertIn("remote-storyboard-takes-wrap", html.text)
        self.assertIn("remote-storyboard-takes", html.text)

        js = self.client.get("/system/app.js")
        self.assertEqual(js.status_code, 200)
        self.assertIn("javascript", js.headers["content-type"])
        self.assertIn("storyboard_takes", js.text)

        css = self.client.get("/system/app.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn("css", css.headers["content-type"])

        common = self.client.get("/system/common.js")
        self.assertEqual(common.status_code, 200)
        self.assertIn("H3PageCommon", common.text)

    # 2. Backend status parsing -------------------------------------------------
    def test_parse_backend_status_handles_studio_workshop_and_attention(self):
        studio = parse_backend_status(self.backend_text, cpu_only=False)
        self.assertEqual(studio["mode"], "studio")
        self.assertEqual(studio["gpu"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(studio["vram"]["free_gib"], 54.2)
        self.assertEqual(studio["vram"]["total_gib"], 79.2)
        self.assertEqual(studio["vram"]["used_gib"], 25.0)
        self.assertTrue(studio["can_open_comfyui"])
        self.assertTrue(studio["can_unload_vram"])
        caps = {c["label"]: c["value"] for c in studio["capabilities"]}
        self.assertEqual(caps["Sparse attention"], "Sage 2")
        self.assertEqual(caps["Spectrum accelerator"], "Available")
        self.assertEqual(studio["profiles"][0]["name"], "Singularity")

        workshop = parse_backend_status("CPU Workshop · No GPU backend attached", cpu_only=True)
        self.assertEqual(workshop["mode"], "workshop")
        self.assertFalse(workshop["can_open_comfyui"])
        self.assertFalse(workshop["can_unload_vram"])

        attention = parse_backend_status("Backend unavailable: Connection refused", cpu_only=False)
        self.assertEqual(attention["mode"], "attention")
        self.assertEqual(attention["status_label"], "Backend needs attention")
        self.assertFalse(attention["can_unload_vram"])

    # 3. Boot & Backend telemetry ----------------------------------------------
    def test_boot_and_backend_endpoints_return_complete_state(self):
        self.assertEqual(self.boot["backend"]["mode"], "studio")
        self.assertEqual(len(self.boot["engines"]), len(ENGINES))
        self.assertEqual(self.boot["models"]["total_count"], 3)
        self.assertEqual(self.boot["models"]["present_count"], 1)
        self.assertIn("gemini_key", self.boot)
        self.assertEqual(self.ok(self.client.get(f"{API}/backend"))["backend"]["gpu"], "NVIDIA H100 80GB HBM3")

    # 4. VRAM unload ------------------------------------------------------------
    def test_unload_runs_in_studio_and_is_rejected_in_cpu_workshop(self):
        self.ok(self.client.post(f"{API}/unload"))
        run = self.wait_run("unload")
        self.assertEqual(run["message"], "All models unloaded and cached VRAM released.")
        self.assertEqual(run["backend"]["vram"]["free_gib"], 76.8)
        self.unloaded.assert_called_once()

        with patch.dict(os.environ, {"H3_CPU_ONLY": "1"}):
            self.refused(self.client.post(f"{API}/unload"), "CPU Workshop")

    # 5. Drama 3 serving --------------------------------------------------------
    def test_drama3_updates_persist_and_enforce_a100_fp8_guard(self):
        with patch.object(d3_config, "JOBS", SimpleNamespace(store=self.store)), \
             patch("h3_app.drama3.serving.server.stop_all") as stop_all:
            res = self.ok(self.client.post(f"{API}/drama3", json={
                "backend": "comfyui",
                "model": "drama3_comfy_w4a8",
                "comfy_experts": "in RAM",
            }))
            self.assertIsNone(res["error"])
            self.assertEqual(res["drama3"]["backend"], "comfyui")
            self.assertEqual(res["drama3"]["model"], "drama3_comfy_w4a8")
            self.assertEqual(res["drama3"]["comfy_experts"], "in RAM")
            self.assertTrue(res["drama3"]["show_comfy_experts"])
            stop_all.assert_called_once()
            self.changed.assert_called_once()

            # A100 refuses SGLang FP8 while preserving the prior ComfyUI settings.
            with patch.dict(os.environ, {"H3_GPU_PROFILE": "a100"}):
                err = self.ok(self.client.post(f"{API}/drama3", json={
                    "backend": "sglang",
                    "model": "fp8",
                    "comfy_experts": "in VRAM",
                }))
                self.assertIn("SM89+", err["error"])
                self.assertEqual(err["drama3"]["backend"], "comfyui")

    # 6. Global Session Gemini API key -----------------------------------------
    def test_session_gemini_key_is_memory_only_owner_isolated_and_propagated(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "H3_DECISION": "1"}, clear=False):
            initial = self.ok(self.client.get(f"{API}/gemini-key"))["gemini_key"]
            self.assertFalse(initial["session_set"])
            self.assertEqual(initial["active_source"], "none")
            self.assertEqual(initial["masked"], "")

            secret_key = "AIzaSyTestSessionKey1234"
            saved = self.ok(self.client.post(f"{API}/gemini-key", json={"api_key": secret_key}))
            self.assertTrue(saved["gemini_key"]["session_set"])
            self.assertEqual(saved["gemini_key"]["active_source"], "session")
            self.assertEqual(saved["gemini_key"]["masked"], "AIza…1234")
            self.assertNotIn(secret_key, str(saved))

            # Never stored in SQLite settings or environment.
            self.assertIsNone(self.store.get_setting("gemini_api_key"))
            self.assertEqual(os.environ.get("GEMINI_API_KEY", ""), "")
            sqlite_bytes = (self.root / "state" / "workspace.sqlite3").read_bytes()
            self.assertNotIn(secret_key.encode(), sqlite_bytes)

            # Owner isolation: another browser session must not see Owner A's key.
            other_client = TestClient(self.app)
            other_boot = self.ok(other_client.get(f"{API}/boot"))
            other_owner = self.store.verify_owner(other_client.cookies[COOKIE])
            self.addCleanup(gemini.clear_session_key, other_owner)
            self.assertFalse(other_boot["gemini_key"]["session_set"])

            # Context propagation inside HTTP middleware -> all 4 consumers.
            @self.app.get("/test/probe-gemini")
            def probe_endpoint():
                return {
                    "api_key": gemini.api_key(""),
                    "explicit": gemini.api_key("explicit-override"),
                    "decisions": decisions.key_or_none(""),
                    "gemma": gemma.current_key(),
                    "nano": hosted_image_endpoint(NANO_BANANA).api_key,
                }

            probed = self.ok(self.client.get("/test/probe-gemini"))
            self.assertEqual(probed["api_key"], secret_key)
            self.assertEqual(probed["explicit"], "explicit-override")
            self.assertEqual(probed["decisions"], secret_key)
            self.assertEqual(probed["gemma"], secret_key)
            self.assertEqual(probed["nano"], secret_key)

            # Other browser gets no key.
            with self.assertRaises(Exception):
                other_client.get("/test/probe-gemini")

            # Propagation across workspace_task, Runs.start, and JobCoordinator.run.
            results = {}

            def bg_check():
                results["workspace_task"] = gemini.api_key("")

            with gemini.owner_context(self.owner):
                t = threading.Thread(target=workspace_task(bg_check))
                t.start()
                t.join(timeout=5)
            self.assertEqual(results["workspace_task"], secret_key)

            runs = Runs()
            run_done = threading.Event()
            test_run = Run(self.owner, "sb", "test")
            runs.start(test_run, lambda _r: (results.setdefault("run", gemini.api_key("")), run_done.set()))
            self.assertTrue(run_done.wait(5))
            self.assertEqual(results["run"], secret_key)

            coords = JobCoordinator()
            with coords.run(self.owner, "image", values=["nano_banana"]) as _job:
                self.assertEqual(hosted_image_endpoint(NANO_BANANA).api_key, secret_key)

            # Clear key -> falls back to env or error.
            cleared = self.ok(self.client.delete(f"{API}/gemini-key"))
            self.assertFalse(cleared["gemini_key"]["session_set"])
            self.assertEqual(gemini.session_key(self.owner), "")

        # Priority order: explicit > session > GEMINI_API_KEY env.
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key", "H3_DECISION": "1"}):
            with gemini.owner_context(self.owner):
                self.assertEqual(gemini.api_key(""), "env-key")
                gemini.set_session_key(self.owner, "sess-key")
                self.assertEqual(gemini.api_key(""), "sess-key")
                self.assertEqual(decisions.key_or_none(""), "sess-key")
                self.assertEqual(gemma.current_key(), "sess-key")
                self.assertEqual(hosted_image_endpoint(NANO_BANANA).api_key, "sess-key")
                self.assertEqual(gemini.api_key("req-key"), "req-key")
                self.assertEqual(decisions.key_or_none("req-key"), "req-key")

    def test_separated_gemini_key_inputs_removed_from_standalone_pages(self):
        defaults = writer_defaults()
        gemini_rows = writer_rows(defaults)
        self.assertNotIn("gemini_api_key", {r["name"] for r in gemini_rows})
        openai_rows = writer_rows({**defaults, "backend": "OpenAI-compatible", "review": True})
        self.assertNotIn("gemini_api_key", {r["name"] for r in openai_rows})

        for rel in (
            "h3_ui/h3/page/app.js",
            "h3_ui/drama3/page/script.js",
            "h3_ui/dubbing/page/source.js",
            "h3_ui/storyboard/page/index.html",
        ):
            text = (Path(__file__).resolve().parent.parent / rel).read_text(encoding="utf-8")
            self.assertNotIn("Temporary Gemini API key", text, rel)
            self.assertNotIn("Gemini API key (optional)", text, rel)
            self.assertNotIn("Gemini API key for this session", text, rel)

    # 7. Local project manage & remote project import -------------------------
    def test_local_project_rename_delete_and_remote_import(self):
        self.store.save_soundstage_project("Alpha Stage", {"text": "[Host] Hello", "title": "Alpha"})
        listed = self.ok(self.client.get(f"{API}/projects"))
        self.assertEqual(listed["counts"]["soundstage"], 1)
        pid = listed["projects"][0]["id"]

        renamed = self.ok(self.client.post(f"{API}/projects/rename", json={
            "kind": "soundstage", "id": pid, "name": "Alpha Stage Two",
        }))
        self.assertEqual(renamed["projects"][0]["name"], "Alpha Stage Two")

        with patch("h3_ui.system.page.fetch_remote_projects", return_value={
            "app": "Prism Workshop",
            "projects": [{"kind": "soundstage", "id": "remote-1", "name": "Remote Play", "summary": "Ready · 3 lines"}],
        }):
            remote = self.ok(self.client.post(f"{API}/projects/remote", json={
                "url": "https://workshop.example.com/system",
            }))
            self.assertEqual(remote["app"], "Prism Workshop")
            self.assertEqual(remote["projects"][0]["kind_label"], "Soundstage")

        with patch("h3_ui.system.page.import_remote_projects", return_value=[
            {"kind": "soundstage", "id": "remote-1", "status": "overwritten", "message": 'Overwritten Soundstage "Remote Play" (synced 1 changed file, reused 4 unchanged files).'},
            {"kind": "storyboard", "id": "remote-2", "status": "up_to_date", "message": 'Storyboard "Board" is already up to date (6 files unchanged).'},
        ]) as mock_import:
            self.ok(self.client.post(f"{API}/projects/import", json={
                "url": "https://workshop.example.com",
                "items": [
                    {"kind": "soundstage", "id": "remote-1"},
                    {"kind": "storyboard", "id": "remote-2"},
                ],
                "conflict": "overwrite",
            }))
            imp_run = self.wait_run("import")
            self.assertIn("1 overwritten, 1 up to date", imp_run["message"])
            self.assertEqual(len(imp_run["results"]), 2)
            mock_import.assert_called_once()

        deleted = self.ok(self.client.post(f"{API}/projects/delete", json={
            "kind": "soundstage", "id": renamed["projects"][0]["id"],
        }))
        self.assertEqual(deleted["counts"]["soundstage"], 0)

    def test_project_import_forwards_and_validates_storyboard_takes(self):
        with patch("h3_ui.system.page.import_remote_projects", return_value=[
            {"kind": "storyboard", "id": "sb-1", "status": "imported", "message": 'Imported Storyboard "Board" (without takes).'},
        ]) as mock_import:
            self.ok(self.client.post(f"{API}/projects/import", json={
                "url": "https://workshop.example.com",
                "items": [{"kind": "storyboard", "id": "sb-1"}],
                "conflict": "rename",
                "storyboard_takes": "none",
            }))
            self.wait_run("import")
            self.assertEqual(mock_import.call_args.kwargs["storyboard_takes"], "none")

        self.refused(
            self.client.post(f"{API}/projects/import", json={
                "url": "https://workshop.example.com",
                "items": [{"kind": "storyboard", "id": "sb-1"}],
                "conflict": "rename",
                "storyboard_takes": "bogus",
            }),
            "Unsupported storyboard takes mode",
        )

    def test_remote_projects_orders_by_kind_preserving_relative_order_and_counts(self):
        mixed = [
            {"kind": "remake", "id": "rem-1", "name": "Remake One"},
            {"kind": "dubbing", "id": "dub-1", "name": "Dub One"},
            {"kind": "soundstage", "id": "ss-1", "name": "Stage One"},
            {"kind": "dubbing", "id": "dub-2", "name": "Dub Two"},
            {"kind": "storyboard", "id": "sb-1", "name": "Board One"},
            {"kind": "soundstage", "id": "ss-2", "name": "Stage Two"},
            {"kind": "custom", "id": "cus-1", "name": "Custom One"},
        ]
        with patch("h3_ui.system.page.fetch_remote_projects", return_value={
            "app": "Prism Workshop",
            "projects": mixed,
        }):
            remote = self.ok(self.client.post(f"{API}/projects/remote", json={
                "url": "https://workshop.example.com",
            }))

        self.assertEqual(
            [(item["kind"], item["id"]) for item in remote["projects"]],
            [
                ("soundstage", "ss-1"),
                ("soundstage", "ss-2"),
                ("storyboard", "sb-1"),
                ("dubbing", "dub-1"),
                ("dubbing", "dub-2"),
                ("remake", "rem-1"),
                ("custom", "cus-1"),
            ],
        )
        self.assertEqual(
            remote["counts"],
            {
                "all": 7,
                "soundstage": 2,
                "storyboard": 1,
                "dubbing": 2,
                "remake": 1,
                "custom": 1,
            },
        )
        self.assertEqual(remote["projects"][-1]["kind_label"], "Custom")

    # 8. Model readiness caching & refresh ------------------------------------
    def test_models_inventory_caches_and_refreshes_on_request(self):
        first = self.ok(self.client.get(f"{API}/models"))
        self.assertEqual(first["present_count"], 1)
        self.assertEqual(first["missing_count"], 2)

        spec2 = self.specs["drama3_comfy_w4a8"]
        added = self.models_root / spec2.folder / spec2.local_name
        added.parent.mkdir(parents=True, exist_ok=True)
        added.write_bytes(b"weights")

        cached = self.ok(self.client.get(f"{API}/models"))
        self.assertEqual(cached["present_count"], 1)

        refreshed = self.ok(self.client.get(f"{API}/models?refresh=1"))
        self.assertEqual(refreshed["present_count"], 2)
        self.assertEqual(refreshed["missing_count"], 1)

    # 9. Historical media scan run --------------------------------------------
    def test_scan_runs_index_rebuild(self):
        self.ok(self.client.post(f"{API}/scan"))
        run = self.wait_run("scan")
        self.assertEqual(self.index.calls, 1)
        self.assertIn("3 files indexed", run["message"])

    # 10. URL normalization ----------------------------------------------------
    def test_normalize_server_url_strips_system_suffix(self):
        self.assertEqual(
            normalize_server_url("https://studio.example.modal.run/system/"),
            "https://studio.example.modal.run",
        )

    # 11. Full composed server mounts /system/ and shares session Gemini key ----
    def test_composed_app_mounts_system_frame_and_shares_session_key_with_image(self):
        from h3_app import media_catalog
        from h3_ui import application as studio_app
        from h3_ui.layout import SYSTEM_FRAME

        self.addCleanup(setattr, media_catalog, "_CATALOG", media_catalog.get_media_catalog())
        self.addCleanup(studio_app.set_runtime_override, None)

        with patch.dict(os.environ, {"H3_CPU_ONLY": "1", "GEMINI_API_KEY": ""}):
            demo = studio_app.build_ui(runtime=self.runtime, store=self.store)
            self.addCleanup(demo.close)
            html_values = [
                block.value
                for block in demo.blocks.values()
                if type(block).__name__ == "HTML" and isinstance(getattr(block, "value", None), str)
            ]
            self.assertIn(SYSTEM_FRAME, html_values)

            app = studio_app.build_server(demo, [])
            client = TestClient(app)

            self.assertEqual(client.get("/system/").status_code, 200)
            boot = self.ok(client.get("/system/api/boot"))
            self.assertEqual(boot["backend"]["mode"], "workshop")

            image_before = self.ok(client.get("/image/api/boot"))
            self.assertFalse(image_before["state"]["api_availability"]["nano_banana"])

            secret = "AIzaSyCrossPageKey9876"
            self.ok(client.post("/system/api/gemini-key", json={"api_key": secret}))
            owner = self.store.verify_owner(client.cookies[COOKIE])
            self.addCleanup(gemini.clear_session_key, owner)

            image_after = self.ok(client.get("/image/api/boot"))
            self.assertTrue(image_after["state"]["api_availability"]["nano_banana"])



if __name__ == "__main__":
    unittest.main()
