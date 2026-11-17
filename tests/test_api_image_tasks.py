"""API images run in the CPU workspace and alongside GPU work."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from types import SimpleNamespace
import json
import os
import time
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from h3_app.jobs import CURRENT_JOB, JOBS, Job, JobCoordinator
from h3_app.generation.results import GenerationUpdate
from h3_app.drama3.visuals import Shot
from h3_app.errors import H3Error
from h3_ui.drama3.visuals import illustrate_shots


class ApiImageSchedulingTests(unittest.TestCase):
    def test_failed_shot_retains_later_success_for_retry(self):
        with TemporaryDirectory() as folder:
            paths = [str(Path(folder) / f"shot-{i}.png") for i in range(2)]
            for path in paths:
                Image.new("RGB", (4, 4)).save(path)
            calls, published = [], []
            failing = True

            def generate(engine, mode, prompt, references, options):
                calls.append(prompt)
                if failing and prompt == "first":
                    raise H3Error("API request failed")
                yield [paths[0 if prompt == "first" else 1]], "Complete"

            def publish(index, path, details):
                published.append(index)
                return str(index)

            controller = SimpleNamespace(generate=generate)
            shots = [Shot(0, 24, False, "first"), Shot(24, 24, False, "second")]
            finished = {"portraits": {}, "shots": []}
            with self.assertRaisesRegex(H3Error, "API request failed"):
                list(illustrate_shots(controller, "gpt_image", shots, {}, False,
                                     finished, on_shot=publish))
            self.assertEqual(published, [1])
            self.assertEqual(finished["shots"], [None, paths[1]])
            failing = False
            list(illustrate_shots(controller, "gpt_image", shots, {}, False,
                                 finished, on_shot=publish))
            self.assertEqual(calls.count("first"), 2)
            self.assertEqual(calls.count("second"), 1)
            self.assertEqual(finished["shots"], paths)

    def test_api_jobs_do_not_wait_for_gpu_or_run_gpu_hooks(self):
        jobs = JobCoordinator()
        self.addCleanup(jobs.close)
        hook = Mock()
        jobs.add_gpu_claim_hook(hook)
        requests = [("image", ["gpt_image"]),
                    ("remake_edit", ["frame.png", "edit", "gpt_image"]),
                    ("drama3_visuals", ["mix.wav", "gpt_image"]),
                    ("drama3_visuals", ["gpt_image", "[]"])]
        jobs.gpu.acquire()
        def run(family, values):
            accepted = Job("owner", family, snapshot_json=json.dumps(values))
            with jobs.run("owner", family, accepted=accepted) as job:
                self.assertFalse(job.has_gpu)
            return job.state
        pool = ThreadPoolExecutor(max_workers=4)
        try:
            futures = [pool.submit(run, family, values) for family, values in requests]
            self.assertEqual([future.result(timeout=3) for future in futures], ["completed"] * 4)
        finally:
            jobs.gpu.release()
            pool.shutdown(wait=True)
        hook.assert_not_called()
        self.assertTrue(Job("owner", "image", snapshot_json='["qwen"]').requires_gpu)
        with jobs.run("owner", "image", values=["gpt_image"]) as direct:
            self.assertFalse(direct.has_gpu)
            self.assertIsNone(direct.snapshot_json)

    def test_shots_overlap_keep_context_and_publish_in_completion_order(self):
        with TemporaryDirectory() as folder:
            paths = [str(Path(folder) / f"shot-{i}.png") for i in range(2)]
            for path in paths:
                Image.new("RGB", (4, 4)).save(path)
            barrier, published_second = Barrier(2), Event()
            job = Job("owner", "drama3_visuals")
            seen, published = [], []

            def generate(engine, mode, prompt, refs, options):
                self.assertIs(CURRENT_JOB.get(), job)
                self.assertEqual(options["gpt_image__batch_count"], 1)
                seen.append(options["gpt_image__models"])
                yield None, "Starting"
                barrier.wait(timeout=3)
                if prompt == "first":
                    self.assertTrue(published_second.wait(3))
                yield [paths[0 if prompt == "first" else 1]], "Complete"

            def publish(index, path, details):
                published.append(index)
                if index == 1:
                    published_second.set()
                return str(index)

            finished = {"portraits": {}, "shots": []}
            shots = [Shot(0, 24, False, "first"), Shot(24, 24, False, "second")]
            token = CURRENT_JOB.set(job)
            try:
                iterator = illustrate_shots(SimpleNamespace(generate=generate), "gpt_image", shots,
                    {"gpt_image__models": ["gpt-image-2", "gpt-image-2.5-flare"]}, False,
                    finished, on_shot=publish)
                while True:
                    try:
                        next(iterator)
                    except StopIteration as done:
                        result = done.value
                        break
            finally:
                CURRENT_JOB.reset(token)
            self.assertEqual(published, [1, 0])
            self.assertEqual(result[0], paths)
            self.assertEqual(finished["shots"], paths)
            self.assertEqual(set(seen), {("gpt-image-2",), ("gpt-image-2.5-flare",)})


class WorkshopApiImageTests(unittest.TestCase):
    def test_workshop_generates_edits_and_rejects_gpu_engines(self):
        import gradio as gr
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
        from h3_ui.api_images import build_api_images

        with TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = SimpleNamespace(outputs_dir=root / "outputs", output_dir=root / "comfy-output")
            runtime.output_dir.mkdir()
            with patch.object(JOBS, "store", WorkspaceStore(root / "state")), patch.dict(JOBS.callbacks), \
                 patch.dict(os.environ, {"GPT_IMAGE_API_URL": "https://example.invalid/v1",
                                         "GPT_IMAGE_API_KEY": "test-key"}):
                with gr.Blocks() as ui:
                    controller, deps, pages = build_api_images(ui, runtime)
                app = FastAPI()
                install_owner_cookie(app, JOBS.store)
                for install in ui.h3_routes:
                    install(app)
                client = TestClient(app)
                boot = client.get("/image/api/boot").json()
                self.assertFalse(boot["gpu"])
                self.assertEqual([item["id"] for item in boot["engines"]], ["gpt_image"])
                self.assertEqual(boot["state"]["engine"], "gpt_image")
                response = client.post("/image/api/settings", json={"action": "engine", "changes": {"engine": "qwen"}})
                self.assertEqual(response.status_code, 400)
                client.post("/image/api/prompt", json={"prompt": "a blue bird"})
                client.post("/image/api/writer", json={"changes": {"review": False}})
                target = runtime.output_dir / "result.png"
                Image.new("RGB", (4, 4)).save(target)
                requests = []
                def generate(request, runtime):
                    self.assertFalse(CURRENT_JOB.get().has_gpu)
                    requests.append(request)
                    yield GenerationUpdate([str(target)], "Complete")
                with patch("h3_app.generation.gpt_image.generate_gpt_image", generate):
                    for editing in (False, True):
                        if editing:
                            with target.open("rb") as image:
                                response = client.post("/image/api/references/upload", files={"files": ("ref.png", image)})
                            self.assertEqual(response.status_code, 200, response.text)
                        response = client.post("/image/api/generate")
                        self.assertEqual(response.status_code, 200, response.text)
                        for _ in range(100):
                            run = client.get("/image/api/run").json()["runs"]["generate"]
                            if run and not run["active"]:
                                break
                            time.sleep(0.02)
                        self.assertFalse(run["active"])
                        self.assertFalse(run.get("error"), run)
                self.assertFalse(requests[0].reference_images)
                self.assertEqual(len(requests[1].reference_images), 1)
                self.assertEqual(client.post("/image/api/interrupt").status_code, 200)


if __name__ == "__main__":
    unittest.main()
