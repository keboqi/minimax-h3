"""Correctness gates for workspace requests and session-owned execution."""

from concurrent.futures import ThreadPoolExecutor
import json
from h3_ui import application as app
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from h3_app.decoder_intent import VideoDecoder
from h3_app.gallery_store import AssetPage
from h3_app.jobs import CURRENT_JOB, JobCancelled, JobCoordinator, variant_seed
from h3_app.reference_bindings import ReferenceMap
from h3_app.errors import H3Error
from h3_app.provenance import write_snapshot, read_snapshot
from h3_app.execution import replay_stage
from h3_app.contracts import GENERATION_FIELDS
from h3_app.generation.finish_video import make_finishing_retry
from h3_ui.persistence import restore_preferences
from types import SimpleNamespace
from h3_ui.prompt_preview import fingerprint
from h3_ui.job_admission import execute_accepted
from h3_ui.media_actions import file_fingerprints
class ReferenceTests(unittest.TestCase):
    def test_sparse_slots_share_provider_and_generation_ordinals(self):
        values = {
            "prompt": "<Picture 9> beside <Picture 2>; <Audio 3> and <Video 2>",
            "ref_image_2": "a.png",
            "ref_image_9": "b.png",
            "ref_video_2": "v.mp4",
            "ref_audio_3": "s.wav",
        }
        mapping = ReferenceMap().update(values)
        dense = mapping.dense_values(values)
        self.assertEqual(
            dense["prompt"], "<Picture 2> beside <Picture 1>; <Audio 1> and <Video 1>"
        )
        self.assertEqual(dense["ref_image_1"], "a.png")
        self.assertEqual(dense["ref_image_2"], "b.png")
        self.assertIsNone(dense["ref_image_9"])
        self.assertEqual(mapping.restore_prompt(dense["prompt"]), values["prompt"])

    def test_replace_remove_and_readd_require_explicit_repair(self):
        initial = ReferenceMap().update({"ref_image_1": "a.png"})
        removed = initial.update({})
        with self.assertRaises(H3Error):
            removed.compile_prompt("<Picture 1>")
        replaced = removed.update({"ref_image_1": "b.png"})
        with self.assertRaises(H3Error):
            replaced.compile_prompt("<Picture 1>")
        repaired = replaced.update({"ref_image_1": "b.png"}, repair=True)
        self.assertEqual(repaired.compile_prompt("<Picture 1>"), "<Picture 1>")
        self.assertNotEqual(initial.bindings[0].asset_id, repaired.bindings[0].asset_id)
        with self.assertRaises(H3Error):
            repaired.restore_prompt("<Picture 2>")

    def test_prompt_preview_guard_tracks_media_and_draft_but_not_keys(self):
        names = ("prompt", "mode", "first", "gemini_api_key")
        original = ("draft", "First / last frame", "a.png", "secret")
        self.assertNotEqual(
            fingerprint(names, original), fingerprint(names, ("edited", *original[1:]))
        )
        self.assertNotEqual(
            fingerprint(names, original),
            fingerprint(names, (*original[:2], "b.png", "secret")),
        )
        self.assertEqual(
            fingerprint(names, original),
            fingerprint(names, (*original[:3], "different-secret")),
        )


class JobRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = JobCoordinator()

    def tearDown(self):
        self.registry.retention_seconds = -1
        for job in self.registry.records.values():
            job.finished_at = 0
        self.registry.list_owned("owner")
        self.registry.close()

    def test_concurrent_idempotency_captures_one_immutable_request(self):
        values = ["original", {"width": 768}]
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = list(
                pool.map(
                    lambda _: self.registry.accept("owner", "h3", values, key="same"),
                    range(4),
                )
            )
        self.assertEqual(len({job.id for job in jobs}), 1)
        values[1]["width"] = 2048
        self.assertEqual(jobs[0].values()[1]["width"], 768)
        with self.assertRaises(ValueError):
            self.registry.accept("owner", "h3", ["different"], key="same")

    def test_owner_isolation_and_queued_cancellation_submit_nothing(self):
        job = self.registry.accept("owner", "h3", ["draft"])
        with self.assertRaises(ValueError):
            self.registry.owned("other", job.id)
        self.assertEqual(self.registry.list_owned("other"), ())
        get, post = Mock(), Mock()
        self.registry.cancel("owner", "h3", get, post, job_id=job.id)
        with self.assertRaises(JobCancelled):
            with self.registry.run("owner", "h3", accepted=job):
                self.fail("Cancelled queued work entered GPU preparation")
        get.assert_not_called()
        post.assert_not_called()
        self.assertEqual(job.state, "cancelled")

    def test_pin_survives_upload_removal_and_expires_after_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "upload.png"
            source.write_bytes(b"original")
            job = self.registry.accept(
                "owner", "h3", ["prompt", str(source)], media_indices=(1,)
            )
            pinned = Path(job.values()[1])
            source.unlink()
            self.assertEqual(pinned.read_bytes(), b"original")
            with self.registry.run("owner", "h3", accepted=job):
                pass
            self.registry.retention_seconds = -1
            self.assertEqual(self.registry.list_owned("owner"), ())
            self.assertFalse(pinned.exists())

    def test_retry_only_failed_variants_and_resolved_seeds(self):
        job = self.registry.accept(
            "owner", "h3", ["original"], callback=lambda *_: None
        )
        with self.registry.run("owner", "h3", accepted=job):
            job.variant_seeds.update({0: 123, 1: 456, 2: 789})
            job.variant = 1
            job.fail(RuntimeError("variant 2 failed"))
        retry = self.registry.retry("owner", job.id)
        self.assertEqual(retry.values(), ["original"])
        self.assertEqual(retry.replay_seeds, (456,))
        self.assertEqual(retry.replay_indices, (1,))
        self.assertEqual(retry.retry_of, job.id)
        with self.assertRaises(ValueError):
            self.registry.retry("owner", job.id, variants=[0])
        token = CURRENT_JOB.set(retry)
        try:
            self.assertEqual(
                variant_seed(1, lambda: self.fail("random seed recomputed")), 456
            )
        finally:
            CURRENT_JOB.reset(token)

    def test_h3_batch_retries_only_recorded_failed_seeds(self):
        values = [None] * len(GENERATION_FIELDS)
        values[GENERATION_FIELDS.index("result_format")] = "Video"
        values[GENERATION_FIELDS.index("seed")] = -1
        values[GENERATION_FIELDS.index("prompt")] = "Original prompt"
        source = self.registry.accept(
            "owner", "h3", [3, *values], callback=app.generate_for_ui
        )
        captured = []

        def generate(**values):
            job = CURRENT_JOB.get()
            captured.append((values["seed"], values["prompt"]))
            if job.variant == 1:
                job.fail(RuntimeError("variant failed"))
            yield None, "mock completed"

        with patch.object(app, "generate", generate), patch.object(
            app.random, "sample", return_value=[101, 202, 303]
        ):
            token = CURRENT_JOB.set(source)
            try:
                with self.registry.run("owner", "h3", accepted=source):
                    list(app.generate_for_ui(*source.values()))
            finally:
                CURRENT_JOB.reset(token)
        retry = self.registry.retry("owner", source.id)
        with patch.object(app, "generate", generate), patch.object(
            app.random,
            "sample",
            side_effect=AssertionError("Random seeds were recomputed"),
        ):
            token = CURRENT_JOB.set(retry)
            try:
                with self.registry.run("owner", "h3", accepted=retry):
                    list(app.generate_for_ui(*retry.values()))
            finally:
                CURRENT_JOB.reset(token)
        self.assertEqual(
            captured,
            [
                (101, "Original prompt"),
                (202, "Original prompt"),
                (303, "Original prompt"),
                (202, "Original prompt"),
            ],
        )

    def test_stage_replay_preserves_seeds_and_rejects_model_drift(self):
        graph = {
            "1": {
                "class_type": "Sampler",
                "inputs": {"seed": 42, "model": "original.safetensors"},
            },
            "2": {"class_type": "LoadImage", "inputs": {"image": "old-input.png"}},
        }
        source = self.registry.accept("owner", "h3", [], callback=lambda *_: None)
        with self.registry.run("owner", "h3", accepted=source):
            source.variant_seeds[0] = 42
            source.observe_submission("backend-id", "token", graph)
            source.fail(RuntimeError("failed"))
        retry = self.registry.retry("owner", source.id)
        changed = json.loads(json.dumps(graph))
        changed["1"]["inputs"]["seed"] = 999
        changed["2"]["inputs"]["image"] = "repinned-input.png"
        replay_stage(changed, retry)
        self.assertEqual(changed["1"]["inputs"]["seed"], 42)
        retry.replay_counts.clear()
        changed["1"]["inputs"]["model"] = "different.safetensors"
        with self.assertRaises(H3Error):
            replay_stage(changed, retry)

    def test_unknown_backend_outcome_blocks_retry(self):
        source = self.registry.accept("owner", "h3", [], callback=lambda *_: None)
        with self.registry.run("owner", "h3", accepted=source):
            source.variant_seeds[0] = 42
            entry = source.observe_submission(None, "token", {})
            entry["state"] = "submission_unknown"
            source.fail(RuntimeError("Timeout after posting"))
        with self.assertRaisesRegex(ValueError, "unknown outcome"):
            self.registry.retry("owner", source.id)

    def test_failed_variant_is_not_overwritten_by_later_success(self):
        job = self.registry.accept("owner", "h3", [])
        with self.registry.run("owner", "h3", accepted=job):
            job.fail(RuntimeError("failed first"))
            job.state = "running"
        self.assertEqual(job.state, "failed")

    def test_capacity_rejection_leaves_no_phantom_record(self):
        for _ in range(8):
            self.registry.accept("owner", "h3", [])
        with self.assertRaises(ValueError):
            self.registry.accept("owner", "h3", [])
        self.assertEqual(len(self.registry.records), 8)

    def test_worker_claims_once_and_keeps_context_out_of_yields(self):
        def callback(value, request):
            self.assertIsNotNone(CURRENT_JOB.get())
            yield (value,)

        job = self.registry.accept("owner", "h3", ["original"], callback=callback)
        with patch("h3_ui.job_admission.JOBS", self.registry):
            updates = execute_accepted(job.id, Mock(session_hash="owner"))
            self.assertEqual(next(updates), ("original",))
            self.assertIsNone(CURRENT_JOB.get())
            self.assertEqual(list(updates), [])
            with self.assertRaises(ValueError):
                list(execute_accepted(job.id, Mock(session_hash="owner")))

    def test_sidecar_keeps_schema_one_and_no_private_replay_payload(self):
        job = self.registry.accept("owner", "h3", ["private prompt"])
        with tempfile.TemporaryDirectory() as directory:
            token = CURRENT_JOB.set(job)
            try:
                path = Path(directory) / "output.mp4"
                write_snapshot(path, {"seed": 42})
                snapshot = read_snapshot(path)
                self.assertEqual(snapshot["schema_version"], 1)
                self.assertEqual(snapshot["application_job_id"], job.id)
                self.assertNotIn("private prompt", json.dumps(snapshot))
            finally:
                CURRENT_JOB.reset(token)

    def test_finishing_retry_starts_at_retained_source_with_original_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            output = Path(directory) / "finished.mp4"
            source.write_bytes(b"complete H3 source")
            output.write_bytes(b"finished")
            saved = SimpleNamespace(
                finishing=SimpleNamespace(ltx25_model="Original model")
            )
            request = Mock()
            request.copy.return_value = saved
            prepared = SimpleNamespace(actual_seed=42)
            services = Mock()

            def finish(request, prepared, services, path, *_):
                self.assertEqual(path.read_bytes(), b"complete H3 source")
                self.assertEqual(prepared.actual_seed, 42)
                self.assertEqual(request.finishing.ltx25_model, "Original model")
                yield None, "Finishing only"
                return output

            callback = make_finishing_retry(
                request,
                prepared,
                services,
                {"job_id": "original-backend-id", "settings": {"seed": 42}},
            )
            job = self.registry.accept(
                "owner",
                "h3-finishing",
                [str(source)],
                callback=callback,
                media_indices=(0,),
                output_indices=(0,),
            )
            with patch("h3_ui.job_admission.JOBS", self.registry), patch(
                "h3_app.generation.finish_video.finish_video", finish
            ), patch.object(
                app, "generate", side_effect=AssertionError("H3 regenerated")
            ):
                updates = list(execute_accepted(job.id, Mock(session_hash="owner")))
            self.assertEqual(updates[-1][0], str(output))
            self.assertEqual(job.state, "completed")
            self.assertEqual(job.variant_seeds[0], 42)
            self.assertEqual(read_snapshot(output)["job_id"], "original-backend-id")
            services.execution.submit_prompt.assert_not_called()


class GalleryTests(unittest.TestCase):
    def test_deletion_detects_replacement_at_the_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.mp4"
            path.write_bytes(b"original")
            original = file_fingerprints((path,))
            replacement = Path(directory) / "replacement.mp4"
            replacement.write_bytes(b"new file")
            replacement.replace(path)
            self.assertNotEqual(file_fingerprints((path,)), original)

    def test_missing_thumbnails_advance_cursor_and_keep_selection_aligned(self):
        page = AssetPage.from_scan(
            [("thumb", "caption")], ["path"], 100, 48, "videos", 47
        )
        self.assertEqual(page.next_cursor, 48)
        self.assertIn("47 thumbnails unavailable", page.status)
        self.assertEqual(
            AssetPage.from_scan([], [], 100, 144, "videos", 100).next_cursor, None
        )
        with self.assertRaises(ValueError):
            AssetPage((), ("orphan",), 1, None, "video")

    def test_decoder_intent_round_trips_and_rejects_ambiguity(self):
        for decoder in VideoDecoder:
            self.assertEqual(
                VideoDecoder.from_flags(*decoder.workflow_flags().values()), decoder
            )
        with self.assertRaises(ValueError):
            VideoDecoder.from_flags(True, True, False)


class PreferenceTests(unittest.TestCase):
    def test_cross_task_engine_and_qwen_preferences_survive_migration(self):
        components = {
            "workspace.task": SimpleNamespace(
                value="Video",
                choices=[
                    ("Video", "Video"),
                    ("Image", "Image"),
                    ("Audio / Music", "Audio / Music"),
                ],
            ),
            "workspace.engine": SimpleNamespace(
                value="h3", choices=[("MiniMax H3", "h3"), ("LTX", "ltx")]
            ),
            "qwen_image21.seed": SimpleNamespace(value=-1, precision=0),
            "qwen_image21.prompt": SimpleNamespace(value=""),
            "qwen_image21.api_key": SimpleNamespace(value=""),
        }
        saved = {
            "schema_version": 5,
            "values": {
                "workspace.task": "Audio / Music",
                "workspace.engine": "music",
                "qwen_image21.seed": 42,
                "qwen_image21.prompt": "private",
                "qwen_image21.api_key": "secret",
            },
        }
        restored, _ = restore_preferences(saved, components)
        self.assertEqual(restored["workspace.engine"], "music")
        self.assertEqual(restored["qwen_image21.seed"], 42)
        self.assertNotIn("qwen_image21.prompt", restored)
        self.assertNotIn("qwen_image21.api_key", restored)
