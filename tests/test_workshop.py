"""Workshop behavior without downloading models or allocating a GPU."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import h3_workshop as manager_module
from h3_workshop import (
    AUDIO_TOOLS_FAMILY, SNAPSHOT_SPECS, ModelManager, SnapshotSpec, catalog_groups, model_presets, preset_selection,
)
from h3_models import MODEL_SPECS, model_manifest_key, read_json, write_json_atomic


class ModelManagerTests(unittest.TestCase):
    def test_media_links_never_replace_existing_directories(self):
        from types import SimpleNamespace

        with TemporaryDirectory() as folder:
            root = Path(folder)
            alias = root / "app" / "ComfyUI" / "input"
            alias.mkdir(parents=True)
            marker = alias / "keep.wav"
            marker.write_bytes(b"existing")
            runtime = SimpleNamespace(comfy_dir=alias.parent, input_dir=root / "input", output_dir=root / "output")
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                manager_module.link_studio_media(runtime)
            self.assertEqual(marker.read_bytes(), b"existing")

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.volume = Mock()
        self.manager = ModelManager(Path(self.temp.name), self.volume)

    def test_presets_cover_defaults_and_optional_loras(self):
        presets = model_presets()
        for keys in presets.values():
            self.assertFalse(set(keys) - MODEL_SPECS.keys() - SNAPSHOT_SPECS.keys())
        h3 = presets["MiniMax H3 defaults"]
        self.assertTrue({"singularity_fl2va", "text_encoder", "turbo_lora",
                         "turbo_ref_lora", "video_vae_lynnreal_int8"}.issubset(h3))
        self.assertEqual(presets["Qwen Image 2.1 defaults"], (
            "qwen_image21_dit_bf16", "qwen_image21_text_bf16", "qwen_image21_vae",
        ))
        self.assertIn("ltx25_distilled_int8", presets["LTX 2.5 defaults"])
        self.assertIn("ltx25_iclora_ingredients", presets["LTX 2.5 + IC-LoRAs"])
        combined = preset_selection(["LTX 2.5 defaults", "LTX 2.5 + IC-LoRAs"])
        self.assertEqual(len(combined), len(set(combined)))

    def test_batch_continues_after_failure_and_commits_success(self):
        keys = ["qwen_image21_vae", "ltx25_audio_vae", "text_encoder"]
        attempted = []

        def download(**kwargs):
            key, = kwargs["model_keys"]
            attempted.append(key)
            if key == keys[1]:
                raise RuntimeError("access denied")
            path = self.manager._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"downloaded")

        with patch.object(manager_module, "sync_models", side_effect=download):
            message, rows = self.manager.download(keys)
        self.assertEqual(attempted, keys)
        self.assertIn("2/3", message)
        self.assertIn("ltx25_audio_vae: access denied", message)
        self.assertTrue(self.manager._path(keys[2]).is_file())
        self.assertTrue(self.manager.config.is_file())
        self.volume.reload.assert_not_called()
        self.assertEqual(self.volume.commit.call_count, len(keys) + 2)
        self.assertEqual(len(rows), len(MODEL_SPECS) + len(SNAPSHOT_SPECS))

    def test_snapshots_use_runtime_layouts(self):
        from h3_app import audio_tools_worker
        from h3_app.drama3.serving.server import ServerSettings

        root = self.manager.root
        self.assertEqual(self.manager._path("qwen_voice_design"),
                         audio_tools_worker.voice_design_cache(root))
        self.assertEqual(self.manager._path("stable_audio3_medium"),
                         audio_tools_worker.stable_audio_dir(root))
        runtime = ServerSettings(venv=Path(), data_dir=Path(self.temp.name) / "drama3-serving",
                                 model_root=root / "drama3_serving")
        self.assertEqual(self.manager._path("drama3_sglang_checkpoint"), runtime.model_dir)
        int4 = ServerSettings(venv=Path(), data_dir=Path(self.temp.name) / "drama3-serving",
                              model_root=root / "drama3_serving", checkpoint="int4")
        self.assertEqual(self.manager._path("drama3_sglang_int4"), int4.model_dir)
        groups = catalog_groups()
        self.assertIn("drama3_sglang_checkpoint", groups["Drama 3"])
        self.assertEqual(groups[AUDIO_TOOLS_FAMILY], ("qwen_voice_design", "stable_audio3_medium"))
        self.assertFalse(SNAPSHOT_SPECS.keys() & MODEL_SPECS.keys())

    def test_snapshot_inventory_reports_runtime_readiness(self):
        statuses = lambda: {row[0]: row[3] for row in self.manager._inventory()}
        self.assertEqual({statuses()[key] for key in SNAPSHOT_SPECS}, {"Missing"})
        stable = self.manager._path("stable_audio3_medium")
        (stable / "model.safetensors").parent.mkdir(parents=True)
        (stable / "model.safetensors").write_bytes(b"x")
        self.assertEqual(statuses()["stable_audio3_medium"], "Unverified / incomplete")
        drama = self.manager._path("drama3_sglang_checkpoint")
        drama.mkdir(parents=True)
        (drama / ".h3-complete").write_text("rev", encoding="utf-8")
        row = next(row for row in self.manager._inventory() if row[0] == "drama3_sglang_checkpoint")
        self.assertEqual((row[1], row[2], row[3]), ("drama3_serving", drama.name + "/", "Installed"))

    def test_snapshot_download_calls_runtime_downloaders(self):
        from h3_app import audio_tools_worker
        from h3_app.drama3.serving import server

        with patch.object(audio_tools_worker, "download") as audio, \
                patch.object(server, "ensure_model") as drama, \
                patch.object(manager_module, "sync_models") as files, \
                patch.dict("os.environ", {"HF_TOKEN": "token"}):
            message, _rows = self.manager.download(list(SNAPSHOT_SPECS))
        self.assertIn("4/4", message)
        files.assert_not_called()
        self.assertEqual([call.args for call in audio.call_args_list], [
            ("voice_design", str(self.manager.root)), ("stable_audio", str(self.manager.root)),
        ])
        self.assertEqual([call.args[0].checkpoint for call in drama.call_args_list], ["fp8", "int4"])
        settings, = drama.call_args.args
        self.assertEqual(settings.model_root, self.manager.root / "drama3_serving")
        self.assertEqual(drama.call_args.kwargs["token"], "token")
        self.assertEqual(self.volume.commit.call_count, len(SNAPSHOT_SPECS) + 2)

    def test_remove_snapshot_deletes_only_its_directory(self):
        voice = self.manager._path("qwen_voice_design")
        (voice / "models--Qwen").mkdir(parents=True)
        (voice / "models--Qwen" / "blob").write_bytes(b"model")
        stable = self.manager._path("stable_audio3_medium")
        stable.mkdir(parents=True)
        self.manager.remove(["qwen_voice_design"], True)
        self.assertFalse(voice.exists())
        self.assertTrue(stable.is_dir())
        self.volume.commit.assert_called_once()

    def test_snapshot_paths_must_stay_below_models(self):
        escaping = SnapshotSpec(
            "Drama 3", "repo", "test", lambda root: root, lambda root: False, lambda root: None,
        )
        with patch.dict(SNAPSHOT_SPECS, {"qwen_voice_design": escaping}), \
                self.assertRaises(ValueError):
            self.manager._path("qwen_voice_design")

    def test_remove_only_selected_and_clear_manifest(self):
        selected, retained = "turbo_lora", "text_encoder"
        manifest = {"files": {}}
        for key in (selected, retained):
            path = self.manager._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
            manifest["files"][model_manifest_key(MODEL_SPECS[key])] = {"size": 5}
        write_json_atomic(self.manager.manifest, manifest)
        self.manager.remove([selected], False)
        self.assertTrue(self.manager._path(selected).exists())
        self.volume.commit.assert_not_called()
        self.manager.remove([selected], True)
        self.assertFalse(self.manager._path(selected).exists())
        self.assertTrue(self.manager._path(retained).exists())
        remaining = read_json(self.manager.manifest, {})["files"]
        self.assertNotIn(model_manifest_key(MODEL_SPECS[selected]), remaining)
        self.assertIn(model_manifest_key(MODEL_SPECS[retained]), remaining)
        self.volume.commit.assert_called_once()

    def test_remove_failure_keeps_unremoved_entries_and_commits_progress(self):
        first, second = "turbo_lora", "text_encoder"
        manifest = {"files": {}}
        for key in (first, second):
            path = self.manager._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
            manifest["files"][model_manifest_key(MODEL_SPECS[key])] = {"size": 5}
        write_json_atomic(self.manager.manifest, manifest)
        real_unlink = Path.unlink

        def flaky(path, *args, **kwargs):
            if path.name == MODEL_SPECS[second].local_name:
                raise OSError("busy")
            return real_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", flaky), self.assertRaises(OSError):
            self.manager.remove([first, second], True)
        remaining = read_json(self.manager.manifest, {})["files"]
        self.assertNotIn(model_manifest_key(MODEL_SPECS[first]), remaining)
        self.assertIn(model_manifest_key(MODEL_SPECS[second]), remaining)
        self.assertTrue(self.manager._path(second).exists())
        self.volume.commit.assert_called_once()

    def test_unknown_keys_rejected_before_mutation(self):
        with self.assertRaises(ValueError):
            self.manager.download(["../anything"])
        with self.assertRaises(ValueError):
            self.manager.remove(["../anything"], True)
        self.volume.assert_not_called()
        self.assertFalse(self.manager.config.exists())

    def test_aliases_share_one_download_and_removal_target(self):
        self.assertEqual(self.manager._keys([
            "singularity_fl2va", "singularity_ref2va",
        ]), ("singularity_fl2va",))

    def test_symlink_outside_models_rejected(self):
        root = self.manager.root
        root.mkdir(parents=True)
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        spec = MODEL_SPECS["turbo_lora"]
        try:
            (root / spec.folder).symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks unavailable")
        with self.assertRaises(ValueError):
            self.manager._path("turbo_lora")

    def test_modal_workshop_serves_full_studio_ui_with_models_tab(self):
        import gradio as gr

        from h3_app import media_catalog
        from h3_app.jobs import JOBS
        from h3_app.workspace_store import WorkspaceStore
        from h3_ui import application as studio_app
        from h3_ui.media_console import media_runtime

        previous = media_catalog.get_media_catalog()
        self.addCleanup(setattr, media_catalog, "_CATALOG", previous)
        # build_ui configures the shared job registry and runtime override with this test's temporary paths.
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.addCleanup(studio_app.set_runtime_override, None)
        data = Path(self.temp.name)
        runtime = media_runtime(data, data / "app")
        self.assertEqual(runtime.output_dir, (data / "output").resolve())
        self.assertEqual(runtime.outputs_dir, (data / "output").resolve())
        ui = manager_module.build_ui(self.manager, (runtime, WorkspaceStore(data / "h3-workspace")))
        blocks = ui.blocks.values()
        labels = {getattr(block, "label", None) for block in blocks}
        self.assertTrue({"Create", "Media", "Models", "Jobs", "System"} <= labels)
        tabs = next(block for block in blocks if isinstance(block, gr.Tabs))
        self.assertEqual([child.label for child in tabs.children],
                         ["Create", "Media", "Models", "Jobs", "System"])
        self.assertTrue(all(isinstance(child, gr.Tab) for child in tabs.children))
        self.assertGreaterEqual(len(getattr(ui, "h3_routes", ())), 9)

    def test_colab_workshop_has_no_models_tab_or_volume_reload(self):
        import gradio as gr

        from h3_app import media_catalog
        from h3_app.config import RuntimeConfig
        from h3_app.jobs import JOBS
        from h3_app.workspace_store import default_store
        from h3_ui import application as studio_app

        self.addCleanup(setattr, media_catalog, "_CATALOG", media_catalog.get_media_catalog())
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.addCleanup(studio_app.set_runtime_override, None)
        data = Path(self.temp.name)
        # The Studio's own layout: run_h3.sh exports these for both apps.
        runtime = RuntimeConfig.from_environment(data, {
            "COMFY_DIR": str(data / "h3" / "ComfyUI"),
            "GRADIO_OUTPUT_DIR": str(data / "h3" / "gradio_outputs"),
        })
        ui = manager_module.build_ui(None, (runtime, default_store(runtime.outputs_dir)))
        self.assertTrue(ui.get_config_file()["fill_width"])
        tabs = next(block for block in ui.blocks.values() if isinstance(block, gr.Tabs))
        self.assertEqual([child.label for child in tabs.children],
                         ["Create", "Media", "Jobs", "System"])
        buttons = {block.value for block in ui.blocks.values() if type(block).__name__ == "Button"}
        self.assertNotIn("Reload shared volume", buttons)

    def test_dubbing_reload_picks_up_studios_analyzed_session_and_dub(self):
        from fastapi.testclient import TestClient

        from h3_app import media_catalog
        from h3_app.dubbing import projects, session as sessions
        from h3_app.jobs import JOBS
        from h3_app.workspace_store import WorkspaceStore
        from h3_ui import application as studio_app
        from h3_ui.media_console import media_runtime

        self.addCleanup(setattr, media_catalog, "_CATALOG", media_catalog.get_media_catalog())
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.addCleanup(studio_app.set_runtime_override, None)
        JOBS.store = None
        data = Path(self.temp.name)
        runtime = media_runtime(data, data / "app")
        runtime.outputs_dir.mkdir()
        store = WorkspaceStore(data / "h3-workspace")
        env = patch.dict("os.environ", {"H3_CPU_ONLY": "1", "H3_WORKSPACE_DIR": str(store.root)})
        env.start()
        self.addCleanup(env.stop)
        ui = manager_module.build_ui(self.manager, (runtime, store))
        server = manager_module._server(ui, runtime, host="127.0.0.1", port=7860, volume=self.volume)
        source = data / "interview.wav"
        source.write_bytes(b"source")
        draft = projects.save(runtime, "Interview", None, source, {})
        output = runtime.outputs_dir / "interview-dub.mp3"

        def load_studio_commit():
            analyzed = sessions.create(runtime, source, video=False)
            analyzed.segments = [{"start": 0, "end": 1, "source": "Hi", "translation": "Bonjour"}]
            output.write_bytes(b"dub")
            analyzed.data.update(target_language="French", last_output=str(output))
            analyzed.save()
            self.assertTrue(projects.attach(runtime, "Interview", analyzed.id, expected_session=draft.id))

        with TestClient(server.config.app) as client:
            api = "/dubbing/api"
            boot = client.get(f"{api}/boot").json()
            self.assertTrue(boot["reload_volume"])
            self.assertFalse(boot["gpu"])
            self.assertEqual(client.get(f"{api}/recent").json()["sessions"], [])
            project = client.post(f"{api}/project/load", json={"name": "Interview"}).json()
            self.assertEqual(project["session"]["id"], draft.id)
            self.assertEqual(project["session"]["board"]["lines"], [])
            self.volume.reload.assert_not_called()

            with patch.object(ui._queue, "active_jobs", [[object()]]):
                self.assertEqual(client.post(f"{api}/reload-volume").status_code, 409)
            self.volume.reload.assert_not_called()

            self.volume.reload.side_effect = OSError("volume busy")
            with self.assertLogs("h3_workshop", level="WARNING"):
                self.assertEqual(client.post(f"{api}/reload-volume").status_code, 503)
            self.volume.reload.side_effect = load_studio_commit
            browser = {"origin": "http://testserver"}
            with (
                client.websocket_connect("/gradio_api/heartbeat/workshop-browser", headers=browser) as heartbeat,
                client.websocket_connect("/gradio_api/queue/data?session_hash=workshop-browser", headers=browser),
            ):
                heartbeat.send_json({})
                self.assertEqual(heartbeat.receive_json()["status"], 200)
                self.assertIn(b"ALIVE", heartbeat.receive_bytes())
                self.assertEqual(client.post(f"{api}/reload-volume").json(), {"reloaded": True})

                recent, = client.get(f"{api}/recent").json()["sessions"]
                project = client.post(f"{api}/project/load", json={"name": "Interview"}).json()
                session = project["session"]
                self.assertNotEqual(session["id"], draft.id)
                self.assertEqual(session["id"], recent["id"])
                self.assertEqual(session["board"]["lines"][0]["translation"], "Bonjour")
                self.assertEqual(session["readiness"]["voiced"], 1)
                result, = session["result"]["files"]
                self.assertEqual(client.get(result["url"]).content, b"dub")

    def test_launcher_starts_workshop_before_studio_setup(self):
        source = (Path(__file__).resolve().parents[1] / "run_h3.sh").read_text(encoding="utf-8")
        self.assertIn('APP="${H3_APP:-studio}"', source)
        self.assertIn('exec "$PYTHON_BIN" -u -m h3_workshop', source)
        start = source.index('[[ "$APP" == "workshop" ]] && run_workshop')
        self.assertLess(start, source.index("setup_h3.py\" --install-dir"))
        self.assertLess(start, source.index("start_prepare \""))

    def test_shutdown_stops_fallback_tunnel_started_during_server_exit(self):
        from contextlib import ExitStack
        from threading import Event

        from h3_app import public_url

        starting = Event()
        server = Mock(started=True)
        server.run.side_effect = lambda: self.assertTrue(starting.wait(5))
        process = Mock()

        def launch(*args, **kwargs):
            starting.set()
            stop = share.call_args.kwargs["stop"]
            return process if stop.wait(5) else None

        with ExitStack() as stack:
            stack.enter_context(patch("h3_app.config.RuntimeConfig.from_environment"))
            stack.enter_context(patch("h3_app.workspace_store.default_store"))
            stack.enter_context(patch.object(manager_module, "build_ui"))
            stack.enter_context(patch.object(manager_module, "_server", return_value=server))
            stack.enter_context(patch.object(public_url, "public_url_settings", return_value=(False, True)))
            share = stack.enter_context(patch.object(public_url, "start_gradio_share", return_value=None))
            stack.enter_context(patch.object(public_url, "launch_cloudflare", side_effect=launch))
            cleanup = stack.enter_context(patch.object(public_url, "stop_cloudflare"))
            manager_module.main()
            cleanup.assert_called_once_with(process)

    def test_launcher_reads_workshop_requirements(self):
        import subprocess
        import sys

        from h3_workshop_requirements import WORKSHOP_REQUIREMENTS

        root = Path(__file__).resolve().parents[1]
        source = (root / "run_h3.sh").read_text(encoding="utf-8")
        function = source[source.index("run_workshop() {"):]
        script = function[function.index("<<'PY'\n") + 7:function.index("\nPY\n")]
        result = subprocess.run([sys.executable, "-", str(root)], input=script,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), list(WORKSHOP_REQUIREMENTS))

    def test_workshop_storyboard_prepares_everything_but_cannot_generate(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import h3_gpu
        from h3_app import media_catalog
        from h3_app import storyboard as sb
        from h3_app.jobs import JOBS
        from h3_app.workspace_store import COOKIE, WorkspaceStore, install_owner_cookie
        from h3_ui import application as studio_app
        from h3_ui.media_console import media_runtime

        self.addCleanup(setattr, media_catalog, "_CATALOG", media_catalog.get_media_catalog())
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.addCleanup(studio_app.set_runtime_override, None)
        data = Path(self.temp.name)
        store = WorkspaceStore(data / "h3-workspace")
        with patch.dict("os.environ", {"H3_CPU_ONLY": "1"}):
            ui = manager_module.build_ui(self.manager, (media_runtime(data, data / "app"), store))
            self.assertIn("Storyboard", {getattr(block, "label", None) for block in ui.blocks.values()})
            JOBS.store = store
            app = FastAPI()
            install_owner_cookie(app, store)
            for install in ui.h3_routes:
                install(app)
            client = TestClient(app)
            self.assertFalse(client.get("/storyboard/api/boot").json()["gpu"])
            created = client.post("/storyboard/api/boards", json={"name": "Workshop story", "story": "One.\n\n---\n\nTwo."})
            self.assertEqual(created.status_code, 200)
            board = created.json()
            refused = client.post(f"/storyboard/api/boards/{board['id']}/generate", json={"action": "remaining"})
            self.assertEqual(refused.status_code, 400)
            self.assertEqual(refused.json()["error"], h3_gpu.CPU_ONLY_MESSAGE)
            # Created from the defaults, and shared: the GPU endpoint's browser sees it too.
            storyboard = store.storyboard("gpu-browser", board["id"])
            expected_defaults = studio_app.storyboard_request_defaults()
            self.assertEqual(storyboard["profile"], sb.make_profile(expected_defaults, expected_defaults))
            self.assertEqual([row["name"] for row in store.storyboards("gpu-browser")], ["Workshop story"])
            self.assertIn(COOKIE, client.cookies)

    def test_cpu_endpoint_has_default_resources_and_independent_image(self):
        source = Path(__file__).resolve().parents[1] / "modal_h3.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertNotIn("provision_models", functions)
        self.assertNotIn("H3StudioUIService", classes)
        manager = functions["workshop"]
        decorator = next(d for d in manager.decorator_list
                         if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                         and d.func.attr == "function")
        arguments = {kw.arg: kw.value for kw in decorator.keywords}
        self.assertEqual(arguments["image"].id, "console_image")
        for resource in ("gpu", "cpu", "memory"):
            self.assertNotIn(resource, arguments)
        self.assertNotIn("sync_models", ast.unparse(functions["prepare_runtime_models"]))

    def test_workshop_serve_sets_cpu_only_commits_volume_and_starts_server(self):
        from h3_app import media_catalog
        from h3_app.jobs import JOBS
        from h3_ui import application as studio_app

        self.addCleanup(setattr, media_catalog, "_CATALOG", media_catalog.get_media_catalog())
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.addCleanup(studio_app.set_runtime_override, None)
        data = Path(self.temp.name)
        server = Mock()
        events = []

        def fake_build_ui(manager, media):
            events.append(("build_ui", manager_module.os.environ.get("H3_CPU_ONLY"), manager is not None))
            return Mock()

        self.volume.commit.side_effect = lambda: events.append(("commit",))

        with patch.dict("os.environ", {}, clear=False), \
                patch.object(manager_module, "link_studio_media") as link_media, \
                patch.object(manager_module, "build_ui", side_effect=fake_build_ui), \
                patch.object(manager_module, "_server", return_value=server), \
                patch("threading.Thread") as thread_cls:
            manager_module.os.environ.pop("H3_CPU_ONLY", None)
            manager_module.serve(data, self.volume, port=7860, script_dir=data / "app")
            self.assertEqual(manager_module.os.environ.get("H3_CPU_ONLY"), "1")
        link_media.assert_called_once()
        self.assertEqual(events, [("build_ui", "1", True), ("commit",)])
        thread_cls.assert_called_once_with(target=server.run, name="workshop", daemon=True)
        thread_cls.return_value.start.assert_called_once_with()

    def test_workshop_serves_the_soundstage_page_for_preparing_without_a_gpu(self):
        import gradio as gr
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from h3_app.jobs import JOBS
        from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
        from h3_ui.drama3.workshop import build_soundstage_workshop
        from h3_ui.media_console import media_runtime

        data = Path(self.temp.name)
        runtime = media_runtime(data, data / "app")
        store = WorkspaceStore(data / "workspace")
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        JOBS.store = store
        with gr.Blocks() as cpu:
            build_soundstage_workshop(gr.Tab("Sound Stage"), cpu, runtime, store)
        app = FastAPI()
        install_owner_cookie(app, store)
        for install in cpu.h3_routes:
            install(app)
        client = TestClient(app)
        api = "/soundstage/api"
        self.assertEqual(client.get("/soundstage/").status_code, 200)
        boot = client.get(f"{api}/boot").json()
        self.assertFalse(boot["gpu"])
        self.assertIsNotNone(boot["video"])  # Plans and art direction can be prepared without a GPU.
        self.assertEqual(boot["video"]["engines"], [])
        # Writing and casting work here; the project is the same one Studio opens.
        self.assertEqual(client.post(f"{api}/projects", json={"name": "Chapter one"}).status_code, 200)
        name = {"name": "Chapter one"}
        client.patch(f"{api}/project/script", params=name, json={"text": "Mia: Hello."})
        project = client.patch(f"{api}/project/slot", params={**name, "kind": "voice", "slot": 0},
                               json={"save_name": "Mia", "description": "A bright vendor."}).json()
        self.assertEqual(project["speakers"][0]["name"], "Mia")
        self.assertEqual(store.soundstage_project("Chapter one")[0]["text"], "Mia: Hello.")
        # Everything that needs the GPU says where to do it.
        for path, body in (("/speak", {"action": "generate"}), ("/auto", {}), ("/words", None),
                           ("/project/make", {"target": "prepare"})):
            reply = client.post(f"{api}{path}", params=name, json=body if body is not None else {}).json()
            self.assertIn("Studio", reply.get("error", ""), path)

    def test_cpu_image_includes_hosted_writer_dependencies_and_guidelines(self):
        source = Path(__file__).resolve().parents[1] / "modal_h3.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        install = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "uv_pip_install"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "console_image"
        )
        from h3_workshop_requirements import WORKSHOP_REQUIREMENTS

        self.assertEqual(ast.unparse(install.args[0]), "*WORKSHOP_REQUIREMENTS")
        self.assertIn("openai>=3.16.2,<4", WORKSHOP_REQUIREMENTS)
        # Only inspect the CPU image's local file mounts, not the GPU image's mounts.
        files = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                     and isinstance(node.iter, ast.Tuple)
                     and any(isinstance(item, ast.Constant) and item.value == "h3_workshop.py"
                             for item in node.iter.elts))
        self.assertIn("prompts/prompt.txt", [item.value for item in files.iter.elts])
        self.assertIn("h3_workshop_requirements.py", [item.value for item in files.iter.elts])
        self.assertTrue((source.parent / "prompts" / "prompt.txt").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
