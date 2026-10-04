"""Persistence, crash recovery and transformation checks without inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from contextlib import closing
import json
import sqlite3
import unittest
from unittest.mock import patch

from PIL import Image

from h3_app.jobs import JobCoordinator
from h3_app.workspace_store import WorkspaceStore, request_owner, COOKIE
from h3_app.image_canvas import transform_canvas
from h3_app.input_leases import create_lease, cleanup_orphans, release_lease


class WorkspaceStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = WorkspaceStore(self.root / "state")
        self.jobs = JobCoordinator()
        self.jobs.configure(self.store)

    def tearDown(self):
        self.jobs.close()
        self.directory.cleanup()

    def test_signed_owner_survives_restart_and_rejects_forgery(self):
        token = self.store.issue_owner()
        owner = self.store.verify_owner(token)
        restarted = WorkspaceStore(self.store.root)
        self.assertEqual(restarted.verify_owner(token), owner)
        self.assertIsNone(restarted.verify_owner(token + "f"))
        self.assertEqual(
            request_owner(
                SimpleNamespace(cookies={COOKIE: token}, session_hash="new-session"),
                restarted,
            ),
            owner,
        )
        self.assertEqual(
            request_owner(SimpleNamespace(cookies={}, session_hash=owner), restarted),
            "api:" + owner,
        )
        recovery = self.store.recovery_key(owner)
        self.assertEqual(restarted.recovery_owner(recovery), owner)
        self.assertIsNone(restarted.recovery_owner(recovery + "f"))
        with self.assertRaises(ValueError):
            request_owner(SimpleNamespace(cookies={}), restarted)

    def test_technical_history_excludes_unsaved_prompt_and_graph(self):
        job = self.jobs.accept(
            "a", "h3", ["Private prompt", "provider-secret"], key="click"
        )
        job.stage = "Private prompt in callback progress"
        job.observe_submission(
            "p",
            "t",
            {
                "1": {
                    "class_type": "Encode",
                    "inputs": {"text": "Private prompt", "seed": 7},
                }
            },
        )
        job.fail(RuntimeError("Private prompt echoed by backend"))
        job.finished_at = 1
        job.persist()
        serialized = json.dumps(self.store.job("a", job.id))
        self.assertNotIn("Private prompt", serialized)
        self.assertNotIn("provider-secret", serialized)
        self.assertNotIn("graph_json", serialized)
        with self.assertRaises(ValueError):
            self.store.job("outsider", job.id)

    def test_idempotency_survives_restart_and_rejects_engine_change(self):
        job = self.jobs.accept("a", "h3", ["prompt"], key="click")
        self.jobs.close()
        self.jobs = JobCoordinator()
        self.jobs.configure(self.store)
        self.assertEqual(
            self.jobs.accept("a", "h3", ["prompt"], key="click").id, job.id
        )
        with self.assertRaises(ValueError):
            self.jobs.accept("a", "ltx", ["prompt"], key="click")
        with self.assertRaises(ValueError):
            self.jobs.accept("a", "h3", ["changed"], key="click")
        # The database lookup also survives the bounded in-memory history.
        self.jobs.records.clear()
        self.jobs.idempotency.clear()
        self.assertEqual(
            self.jobs.accept("a", "h3", ["prompt"], key="click").id, job.id
        )

    def test_project_retains_inputs_and_exact_replay_across_restart(self):
        source = self.root / "source.png"
        source.write_bytes(b"source bytes")

        def callback(*args):
            return None

        job = self.jobs.accept(
            "a",
            "h3",
            ["Private prompt", str(source)],
            callback=callback,
            media_indices=(1,),
        )
        job.variant_seeds[0] = 123
        job.fail(RuntimeError("failed"))
        job.finished_at = 1
        project_id = self.store.save_project("a", job, "Example")
        job.persist()
        self.jobs.close()
        source.unlink()
        self.jobs = JobCoordinator()
        self.jobs.configure(self.store)
        self.jobs.register_callback("h3", callback)
        restored = self.jobs.owned("a", job.id)
        self.assertEqual(restored.values()[0], "Private prompt")
        self.assertEqual(Path(restored.values()[1]).read_bytes(), b"source bytes")
        retry = self.jobs.retry("a", job.id)
        self.assertEqual(retry.replay_seeds, (123,))
        with self.assertRaises(ValueError):
            self.store.delete_project("outsider", project_id)
        self.store.delete_project("a", project_id)
        self.assertEqual(self.store.projects("a"), [])

    def test_saved_finishing_restores_original_model_policy_and_seed(self):
        from h3_app.generation.finish_video import retain_finishing, restore_finishing
        from h3_app.generation.requests import H3Request
        from h3_app.contracts import GENERATION_FIELDS
        from h3_app.catalog import UI_DEFAULTS
        from h3_app.model_types import ModelConfig, ModelProfile

        request = H3Request.from_values(
            {name: UI_DEFAULTS.get(name) for name in GENERATION_FIELDS}
        )
        prepared = SimpleNamespace(
            actual_seed=123,
            resolved_width=512,
            resolved_height=512,
            models=ModelConfig(
                {"original": ModelProfile("Original", "fl2va", "ref2va")},
                "original",
                "encoder",
                "video-vae",
                "audio-vae",
            ),
        )
        job = self.jobs.accept("a", "h3", ["prompt"])
        retain_finishing(job, request, prepared, {"seed": 123})
        self.store.save_project("a", job, "Finishing")
        self.jobs.close()
        self.jobs = JobCoordinator()
        self.jobs.finishing_factory = lambda saved: restore_finishing(saved, None)
        with patch("h3_app.generation.finish_video.make_finishing_retry") as factory:
            self.jobs.configure(self.store)
        restored = self.jobs.owned("a", job.id)
        actual_request, actual_prepared, _, snapshot = factory.call_args.args
        self.assertEqual(actual_request.values(), request.values())
        self.assertEqual(actual_prepared.actual_seed, 123)
        self.assertEqual(actual_prepared.models, prepared.models)
        self.assertEqual(snapshot, {"seed": 123})
        self.assertIn(0, restored.finishing_callbacks)

    def test_deleting_text_only_project_clears_restored_private_content(self):
        job = self.jobs.accept("a", "h3", ["Private prompt"])
        job.observe_submission(
            "backend", "token", {"1": {"inputs": {"text": "Private prompt"}}}
        )
        project_id = self.store.save_project("a", job, "Text-only")
        self.jobs.close()
        self.jobs = JobCoordinator()
        self.jobs.configure(self.store)
        restored = self.jobs.owned("a", job.id)
        self.assertEqual(restored.values(), ["Private prompt"])
        self.store.delete_project("a", project_id)
        self.jobs.forget_project("a", project_id)
        self.assertIsNone(restored.snapshot_json)
        self.assertIsNone(restored.finishing_request)
        self.assertNotIn("Private prompt", json.dumps(restored.ledger))
        with self.assertRaises(ValueError):
            restored.values()

    def test_job_canvas_transforms_copies_and_preserves_snapshot(self):
        from h3_app.image_canvas import apply_job_canvas

        first, last = self.root / "first.png", self.root / "last.png"
        Image.new("RGB", (640, 320), "red").save(first)
        Image.new("RGB", (320, 640), "blue").save(last)
        callback = SimpleNamespace(
            job_input_names=("result_format", "mode", "first_image", "last_image")
        )
        job = self.jobs.accept(
            "a",
            "h3",
            ["Image", "First / last frame", str(first), str(last)],
            callback=callback,
            media_indices=(2, 3),
        )
        originals = job.values()
        job.canvas_request = {"mode": "Fit / pad", "width": 512, "height": 512}
        derived = apply_job_canvas(job, originals)
        self.assertEqual(job.values(), originals)
        self.assertNotEqual(derived[2:], originals[2:])
        for path in derived[2:]:
            with Image.open(path) as image:
                self.assertEqual(image.size, (512, 512))
        self.assertEqual(
            apply_job_canvas(job, ["Video", *originals[1:]])[2:], originals[2:]
        )

    def test_restart_reconciliation_never_replays_unknown_work(self):
        job = self.jobs.accept("a", "h3", ["prompt"])
        job.observe_submission(None, "t", {})
        self.jobs.close()
        self.jobs = JobCoordinator()
        self.jobs.configure(self.store)

        def get(path):
            return SimpleNamespace(
                json=lambda: {"queue_pending": [], "queue_running": []}
            )

        self.jobs.reconcile("a", get)
        recovered = self.jobs.owned("a", job.id)
        self.assertEqual(recovered.state, "needs_review")
        self.assertEqual(recovered.ledger[0]["state"], "submission_unknown")
        with self.assertRaises(ValueError):
            self.jobs.retry("a", job.id)

    def test_known_backend_work_is_observed_then_marked_interrupted(self):
        job = self.jobs.accept("a", "h3", ["prompt"])
        job.observe_submission("backend", "t", {})
        self.jobs.close()
        self.jobs = JobCoordinator()
        self.jobs.configure(self.store)

        def get(path):
            return SimpleNamespace(
                json=lambda: (
                    {"queue_running": [[0, "backend"]]} if path == "/queue" else {}
                )
            )

        self.jobs.reconcile("a", get)
        self.assertEqual(self.jobs.owned("a", job.id).state, "recovering")

        def complete(path):
            return SimpleNamespace(
                json=lambda: (
                    {}
                    if path == "/queue"
                    else {
                        "backend": {
                            "status": {"completed": True, "status_str": "success"}
                        }
                    }
                )
            )

        self.jobs.reconcile("a", complete)
        self.assertEqual(self.jobs.owned("a", job.id).state, "interrupted")

    def test_completed_history_recovers_outputs_without_dispatch(self):
        job = self.jobs.accept("a", "h3", ["prompt"])
        job.observe_submission("backend", "t", {})
        self.jobs.close()
        self.jobs = JobCoordinator()
        self.jobs.configure(self.store)
        self.jobs.history_outputs = lambda history, entry: ["recovered.mp4"]
        self.jobs.reconcile(
            "a",
            lambda path: SimpleNamespace(
                json=lambda: (
                    {}
                    if path == "/queue"
                    else {
                        "backend": {
                            "status": {"completed": True, "status_str": "success"}
                        }
                    }
                )
            ),
        )
        restored = self.jobs.owned("a", job.id)
        self.assertEqual(restored.outputs, ["recovered.mp4"])
        self.assertIsNone(restored.callback)
        self.assertEqual(restored.state, "interrupted")

    def test_future_schema_and_unsafe_backup_are_rejected(self):
        from h3_app.workspace_admin import restore_workspace
        import zipfile

        with self.store.connect() as db:
            db.execute("PRAGMA user_version=999")
        with self.assertRaises(RuntimeError):
            WorkspaceStore(self.store.root)
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as backup:
            backup.writestr(
                "manifest.json",
                json.dumps({"schema_version": 1, "root": str(self.store.root)}),
            )
            backup.writestr("../outside.txt", "invalid")
        with self.assertRaises(ValueError):
            restore_workspace(archive, self.root / "restored")

    def test_gallery_retained_source_is_scoped_to_running_input_lease(self):
        from h3_app.jobs import CURRENT_JOB
        from h3_ui.media_controller import MediaController

        source = self.root / "source.mp4"
        source.write_bytes(b"source")
        job = self.jobs.accept("a", "gallery", [str(source)], media_indices=(0,))
        retained = job.values()[0]
        token = CURRENT_JOB.set(job)
        try:
            self.assertIsNone(MediaController._retained_input(retained))
            job.has_gpu = True
            self.assertEqual(MediaController._retained_input(retained), Path(retained))
            self.assertIsNone(MediaController._retained_input(source))
        finally:
            CURRENT_JOB.reset(token)

    def test_index_rebuild_and_backup_preserve_authoritative_annotations(self):
        image = self.root / "result.png"
        image.write_bytes(b"output")
        asset_id = self.store.index_asset(image, "Image", {"seed": 42})
        self.store.annotate(asset_id, tags=["test", "test", "favorite"], favorite=True)
        self.store.rebuild_index([(image, "Image", {"seed": 42})])
        results = self.store.search_assets(query="test", favorite=True)
        self.assertEqual(results[0]["id"], asset_id)
        self.assertEqual(results[0]["tags"], ["favorite", "test"])
        backup = self.store.backup(self.root / "backup.sqlite3")
        with closing(sqlite3.connect(backup)) as db:
            self.assertEqual(
                db.execute("SELECT favorite FROM annotations").fetchone()[0], 1
            )
        self.store.rebuild_index([])
        self.assertEqual(self.store.search_assets(), [])
        self.store.rebuild_index([(image, "Image", {})])
        self.assertTrue(self.store.search_assets()[0]["favorite"])

    def test_batch_index_preserves_annotations_and_skips_vanished_files(self):
        image, other = self.root / "result.png", self.root / "other.png"
        image.write_bytes(b"output")
        other.write_bytes(b"other")
        asset_id = self.store.index_asset(image, "Image", {"seed": 42})
        self.store.annotate(asset_id, tags=["keep"], favorite=True)
        with patch.object(self.store, "connect", wraps=self.store.connect) as connect:
            indexed = self.store.index_assets(
                [
                    (image, "Image", {"seed": 7}),
                    (other, "Image", {}),
                    (self.root / "missing.png", "Image", {}),
                ]
            )
            self.assertEqual(indexed, {image, other})
            self.assertEqual(connect.call_count, 1)
        row = self.store.search_assets(favorite=True)[0]
        self.assertEqual(row["id"], asset_id)
        self.assertEqual(row["tags"], ["keep"])
        self.assertEqual(row["metadata"], {"seed": 7})

    def test_large_input_copy_does_not_block_monitoring(self):
        source = self.root / "source.png"
        source.write_bytes(b"source")
        entered, resume = Event(), Event()
        import shutil

        copy = shutil.copy2

        def slow_copy(*args):
            entered.set()
            self.assertTrue(resume.wait(5))
            return copy(*args)

        with (
            patch("h3_app.jobs.shutil.copy2", side_effect=slow_copy),
            ThreadPoolExecutor() as pool,
        ):
            accepted = pool.submit(
                self.jobs.accept, "a", "h3", [str(source)], media_indices=(0,)
            )
            self.assertTrue(entered.wait(2))
            read = pool.submit(self.jobs.list_owned, "a")
            try:
                self.assertEqual(read.result(timeout=1), ())
            finally:
                resume.set()
            self.assertTrue(accepted.result(timeout=3).snapshot_json)

    def test_backup_restore_relocates_saved_inputs_and_keeps_owner_identity(self):
        from h3_app.workspace_admin import backup_workspace, restore_workspace

        owner = self.store.verify_owner(self.store.issue_owner())
        source = self.root / "source.png"
        source.write_bytes(b"saved input")
        job = self.jobs.accept(
            owner, "h3", ["Private prompt", str(source)], media_indices=(1,)
        )
        project_id = self.store.save_project(owner, job, "Backup fixture")
        backup = backup_workspace(self.store, self.root / "backup.zip")
        restored = restore_workspace(backup, self.root / "restored")
        project = restored.project(owner, project_id)
        self.assertEqual(Path(project["values"][1]).read_bytes(), b"saved input")
        self.assertTrue(Path(project["values"][1]).is_relative_to(restored.root))
        self.assertEqual(restored.recovery_owner(self.store.recovery_key(owner)), owner)
        with self.assertRaises(ValueError):
            restore_workspace(backup, restored.root)

    def test_orphan_cleanup_preserves_live_process_lease(self):
        path, lock = create_lease()
        try:
            cleanup_orphans()
            self.assertTrue(Path(path).is_dir())
            lock.release()
            cleanup_orphans()
            self.assertFalse(Path(path).exists())
        finally:
            release_lease(path, lock)


class ImageCanvasTests(unittest.TestCase):
    def test_pad_and_crop_preserve_original_and_apply_different_geometry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            Image.new("RGB", (640, 320), (255, 0, 0)).save(original)
            before = original.read_bytes()
            pad, crop = root / "pad.png", root / "crop.png"
            transform_canvas(original, pad, "Fit / pad", 512, 512)
            transform_canvas(original, crop, "Fill / crop", 512, 512)
            with Image.open(pad) as image:
                self.assertEqual(image.size, (512, 512))
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            with Image.open(crop) as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(
                transform_canvas(original, pad, "Input-derived", 1, 1), str(original)
            )
            with self.assertRaises(ValueError):
                transform_canvas(original, crop, "Fill / crop", 513, 512)
            with self.assertRaises(ValueError):
                transform_canvas(original, original, "Fill / crop", 512, 512)
