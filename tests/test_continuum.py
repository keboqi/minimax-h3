"""Sequence (H3 Continuum) contracts without a GPU or a ComfyUI backend."""

from __future__ import annotations

import json
import threading
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from h3_app.continuum_policy import (
    MAX_CHUNKS,
    compile_prompts,
    parse_run_storage,
    sequence_issues,
    serialize_prompts,
)
from h3_app.continuum_store import (
    ContinuumStore,
    RunBusyError,
    StaleRunError,
    parse_project,
    run_lock,
)
from h3_app.errors import H3Error
from h3_app.generation import continuum as orchestrator
from h3_app.generation.continuum_requests import (
    FINISH,
    NEXT,
    RETRY,
    USE_TAKE,
    ContinuumAction,
    ContinuumRequest,
)
from h3_app.workflows.continuum import (
    SAMPLER_ID,
    SAVE_ID,
    build_continuum_graph,
    history_text,
    required_continuum_nodes,
)
from h3_app.workspace_store import WorkspaceStore


def make_request(**overrides) -> ContinuumRequest:
    values = dict(
        mode="Text to video",
        shared_prompt="A lighthouse at dusk.",
        actions=("The beam turns on.",),
        chunk_seconds=5.0,
        continuity="Balanced",
        width=512,
        height=512,
        seed=7,
        model_profile="Pruned",
        text_encoder="NVFP4",
        steps=20,
        scheduler="simple",
    )
    values.update(overrides)
    return ContinuumRequest(**values)


def build(request, action=ContinuumAction(NEXT), staged=None):
    return build_continuum_graph(
        request,
        action,
        project_id="1fefd563-68c4-45f7-a33f-8a2911850a9c",
        model_name="h3.safetensors",
        text_encoder_name="te.safetensors",
        video_vae_name="video.safetensors",
        audio_vae_name="audio.safetensors",
        staged_inputs=staged or {},
        available_nodes=required_continuum_nodes(request) | {"LoadImage"},
    )


class PromptAndPolicyTests(unittest.TestCase):
    def test_compiles_shared_and_action_prompts_with_line_normalization(self):
        prompts = compile_prompts("Scene\r\nstyle", ["Walks in", "", "  "], 3)
        self.assertEqual(prompts, ("Scene\nstyle\n\nWalks in", "Scene\nstyle", "Scene\nstyle"))

    def test_rejects_empty_prompt_and_count_mismatch(self):
        with self.assertRaisesRegex(H3Error, "Chunk 2 has no prompt"):
            compile_prompts("", ["one", ""], 2)
        with self.assertRaises(H3Error):
            compile_prompts("Scene", ["one"], 2)
        with self.assertRaises(H3Error):
            compile_prompts("Scene", ["x"] * (MAX_CHUNKS + 1), MAX_CHUNKS + 1)

    def test_delimiter_like_text_survives_json_serialization(self):
        prompts = compile_prompts('He says "---"\n[0-5s] hi', ["---"], 1)
        self.assertEqual(json.loads(serialize_prompts(prompts)), list(prompts))

    def test_sequence_issues_explain_unsupported_clip_settings(self):
        issues = sequence_issues(
            mode="First / last frame",
            chunk_count=4,
            chunk_seconds=9,
            width=500,
            height=512,
            has_first_image=False,
            has_last_image=True,
            reference_images=0,
            reference_videos=0,
            reference_audios=0,
        )
        text = " ".join(issues)
        for fragment in ("30 seconds", "multiples of 32", "first image", "last frame"):
            self.assertIn(fragment, text)
        self.assertEqual(
            sequence_issues(
                mode="Reference media", chunk_count=2, chunk_seconds=5, width=512, height=512,
                has_first_image=False, has_last_image=False, reference_images=1,
                reference_videos=0, reference_audios=0,
            ),
            [],
        )

    def test_parses_upstream_run_storage_status(self):
        status = (
            "V3.8 diagnostics...\nRun Storage: run_abc / revision 1a2b; 2 reused, 1 generated, "
            "3 total; resume=complete; nonce=0 (inherit_branch). auto-resume disabled: "
            "unknown wrapper.\nother"
        )
        report = parse_run_storage(status)
        self.assertEqual((report.reused, report.generated, report.total), (2, 1, 3))
        self.assertEqual(report.resume, "complete")
        self.assertEqual(report.reuse_disabled, "unknown wrapper")
        self.assertIsNone(parse_run_storage("no storage line"))


class RequestAndActionTests(unittest.TestCase):
    def test_request_round_trips_and_digest_is_stable(self):
        request = make_request(
            mode="Reference media",
            actions=("a", "b"),
            reference_images=(None, None, None, "r4.png", None, None, None, None, None),
            reference_chunks=((), (), (), (2,), (), (), (), (), ()),
        )
        restored = ContinuumRequest.from_payload(json.loads(json.dumps(request.payload())))
        self.assertEqual(restored, request)
        self.assertEqual(restored.digest(), request.digest())
        self.assertEqual(request.reference_selectors()[3], "2")
        self.assertEqual(request.reference_use, "Per chunk")

    def test_seed_must_be_resolved(self):
        with self.assertRaises(H3Error):
            make_request(seed=-1)

    def test_actions_compile_every_one_shot_field(self):
        self.assertEqual(ContinuumAction(RETRY).sampler_controls()["review_action"], "Regenerate Current")
        regen = ContinuumAction("regenerate_from", regenerate_from=2).sampler_controls()
        self.assertEqual((regen["review_action"], regen["reroll_from_chunk"]), ("Continue / Next", "Chunk 2"))
        take = ContinuumAction(USE_TAKE, take_group=2, take_revision_id="ab12").sampler_controls()
        self.assertEqual((take["take_action"], take["take_group"], take["take_revision_id"]), ("Use This Take", 2, "ab12"))
        plain = ContinuumAction(NEXT).sampler_controls()
        self.assertEqual((plain["take_action"], plain["take_group"], plain["take_revision_id"], plain["reroll_nonce"]), ("Automatic", 0, "", 0))
        with self.assertRaises(H3Error):
            ContinuumAction(USE_TAKE, take_group=1, take_revision_id="../x")
        self.assertFalse(ContinuumAction(USE_TAKE, take_group=1, take_revision_id="ab").samples)


class WorkflowTests(unittest.TestCase):
    def test_text_graph_uses_review_storage_list_prompts_and_manual_size(self):
        request = make_request(actions=("one", "two"))
        graph = build(request)
        sampler = graph[SAMPLER_ID]["inputs"]
        self.assertEqual(graph[SAMPLER_ID]["class_type"], "H3ContinuumSamplerV39")
        self.assertEqual(sampler["generation_mode"], "Review Each Chunk")
        self.assertEqual(sampler["run_storage"], "Save + Auto Resume")
        self.assertEqual(sampler["prompt_mode"], "List")
        self.assertEqual(sampler["chunks"], 2)
        self.assertEqual((sampler["size_source"], sampler["width"], sampler["height"]), ("Manual", 512, 512))
        self.assertEqual(sampler["project_id"], "1fefd563-68c4-45f7-a33f-8a2911850a9c")
        self.assertEqual(sampler["run_name"], "")
        prompt = graph[sampler["sequence_prompt"][0]]["inputs"]["value"]
        self.assertEqual(json.loads(prompt), list(request.prompts))
        self.assertNotIn("first_frame", sampler)
        self.assertNotIn("reference_images", sampler)

    def test_upstream_topology_is_identical_across_actions(self):
        request = make_request(actions=("one", "two"))
        first, retry = build(request), build(request, ContinuumAction(RETRY))
        upstream = lambda graph: {k: v for k, v in graph.items() if k != SAMPLER_ID and not k.startswith("continuum_save")}
        self.assertEqual(upstream(first), upstream(retry))
        self.assertIn(SAVE_ID, first)

    def test_reference_slots_keep_fixed_numbers(self):
        request = make_request(
            mode="Reference media",
            reference_images=(None, None, None, "/r/four.png", None, None, None, None, "/r/nine.png"),
        )
        graph = build(request, staged={"/r/four.png": "h3/four.png", "/r/nine.png": "h3/nine.png"})
        references = graph["continuum_references"]["inputs"]
        self.assertEqual(references["reference_use"], "All chunks")
        self.assertEqual(references["reference_image_4"], ["continuum_reference_4", 0])
        self.assertEqual(references["reference_image_9"], ["continuum_reference_9", 0])
        self.assertNotIn("reference_image_1", references)
        self.assertEqual(graph["continuum_reference_4"]["inputs"]["image"], "h3/four.png")

    def test_missing_nodes_are_reported(self):
        request = make_request()
        with self.assertRaisesRegex(H3Error, "H3ContinuumSamplerV39"):
            build_continuum_graph(
                request, ContinuumAction(NEXT), project_id=str(uuid.uuid4()),
                model_name="m", text_encoder_name="t", video_vae_name="v",
                audio_vae_name="a", staged_inputs={}, available_nodes=set(),
            )

    def test_history_text_reads_preview_any_outputs(self):
        self.assertEqual(history_text({"outputs": {"n": {"text": ["a", "b"]}}}, "n"), "a\nb")
        self.assertEqual(history_text({}, "n"), "")


def project_payload(groups, chain, status="review_ready", unit=(1, 1)):
    return {
        "run_storage_schema_version": 1,
        "canonical_storage_revision_id": "aa",
        "canonical_head_revision_id": chain[-1] if chain else None,
        "canonical_chain": list(chain),
        "group_revisions": [
            {"revision_id": rid, "created_utc": f"2026-10-18T00:00:0{i}", "group": {"start": g, "end": g, "physical_group": g}}
            for i, (g, rid) in enumerate(groups)
        ],
        "revisions": [{"revision_id": "aa", "status": status, "review_unit": {"start": unit[0], "end": unit[1]}}],
    }


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        root = Path(self.directory.name)
        self.workspace = WorkspaceStore(root / "state")
        self.store = ContinuumStore(self.workspace, root / "output")
        self.source = root / "upload.png"
        self.source.write_bytes(b"image-bytes")

    def tearDown(self):
        self.directory.cleanup()

    def test_create_retains_sources_and_scopes_owner(self):
        request = make_request(mode="First / last frame", first_image=str(self.source))
        run_id = self.store.create_run("alice", "My run", request)
        version, saved = self.store.request("alice", run_id)
        self.assertEqual(version, 1)
        self.assertNotEqual(saved.first_image, str(self.source))
        self.assertTrue(Path(saved.first_image).is_file())
        self.source.unlink()  # The upload expiring must not break resume.
        self.assertTrue(Path(self.store.request("alice", run_id)[1].first_image).is_file())
        with self.assertRaises(ValueError):
            self.store.run("mallory", run_id)
        self.assertEqual([r["id"] for r in self.store.runs("alice")], [run_id])
        self.assertEqual(self.store.runs("mallory"), [])

    def test_reserve_compares_epoch_and_dedupes_transport_retry(self):
        run_id = self.store.create_run("alice", "r", make_request())
        first = self.store.reserve("alice", run_id, ContinuumAction(NEXT), expected_epoch=0, key="k1")
        again = self.store.reserve("alice", run_id, ContinuumAction(NEXT), expected_epoch=0, key="k1")
        self.assertEqual(first["id"], again["id"])
        with self.assertRaises(StaleRunError):  # second tab still holds epoch 0
            self.store.reserve("alice", run_id, ContinuumAction(RETRY), expected_epoch=0, key="k2")
        with self.assertRaises(ValueError):
            self.store.reserve("alice", run_id, ContinuumAction(RETRY), expected_epoch=1, key="k1")
        with self.assertRaises(ValueError):
            self.store.reserve("mallory", run_id, ContinuumAction(NEXT), expected_epoch=1)

    def test_revisions_are_immutable_versions(self):
        run_id = self.store.create_run("alice", "r", make_request())
        _, request = self.store.request("alice", run_id)
        self.assertEqual(self.store.revise("alice", run_id, request, expected_epoch=0), 1)
        version = self.store.revise("alice", run_id, request.with_actions(("a", "b")), expected_epoch=0)
        self.assertEqual(version, 2)
        self.assertEqual(self.store.request("alice", run_id, 1)[1].chunk_count, 1)
        self.assertEqual(self.store.request("alice", run_id)[1].chunk_count, 2)
        with self.assertRaises(StaleRunError):
            self.store.revise("alice", run_id, request.with_actions(("c",)), expected_epoch=0)

    def test_reconcile_settles_unknown_operations_without_resubmitting(self):
        run_id = self.store.create_run("alice", "r", make_request())
        done = self.store.reserve("alice", run_id, ContinuumAction(NEXT), expected_epoch=0)
        self.store.update_operation(done["id"], state="submitted", prompt_id="p1")
        lost = self.store.reserve("alice", run_id, ContinuumAction(NEXT), expected_epoch=1)
        histories = {"p1": {"status": {"completed": True}, "outputs": {"x": {}}}}
        self.assertEqual(self.store.reconcile(histories.get), 2)
        states = {op["id"]: op["state"] for op in self.store.operations("alice", run_id)}
        self.assertEqual(states[done["id"]], "completed_unrecorded")
        self.assertEqual(states[lost["id"]], "interrupted")
        self.assertEqual(self.store.open_operations(), [])

    def test_project_projection_and_containment(self):
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b")))
        run = self.store.run("alice", run_id)
        self.assertIsNone(self.store.project(run))
        directory = self.store.upstream_directory(run)
        directory.mkdir(parents=True)
        (directory / "project.json").write_text(
            json.dumps(project_payload([(1, "aa11"), (1, "bb22"), (2, "cc33")], ["bb22", "cc33"], status="complete", unit=(2, 2)))
        )
        project = self.store.project(run)
        self.assertEqual(project.accepted_chunks, 2)
        self.assertEqual([t.revision_id for t in project.takes_for(1)], ["aa11", "bb22"])
        self.assertEqual(project.active_take(1).revision_id, "bb22")
        self.assertEqual((project.status, project.review_unit), ("complete", (2, 2)))
        self.assertTrue(self.store.owns_path(directory / "project.json"))
        self.store.delete_run("alice", run_id)
        self.assertFalse(directory.exists())
        with self.assertRaises(ValueError):
            self.store.run("alice", run_id)

    def test_parse_project_skips_forged_entries(self):
        payload = project_payload([(1, "../evil"), (1, "aa11")], ["aa11"])
        payload["group_revisions"].append({"revision_id": "bb", "group": {"start": 2, "end": 1, "physical_group": 2}})
        project = parse_project(payload)
        self.assertEqual([t.revision_id for t in project.takes], ["aa11"])
        with self.assertRaises(ValueError):
            parse_project({"canonical_chain": []})

    def test_run_lock_rejects_concurrent_dispatch(self):
        with run_lock("feedfeed"):
            with self.assertRaises(RunBusyError):
                with run_lock("feedfeed"):
                    pass
        with run_lock("feedfeed"):
            pass


class FakeBackend:
    """Imitates Continuum's review state and its project.json catalog."""

    def __init__(self, store: ContinuumStore, output: Path, reuse_shortfall=0):
        self.store = store
        self.output = output
        self.submitted = []
        self.groups: list[tuple[int, str]] = []
        self.chain: list[str] = []
        self.reuse_shortfall = reuse_shortfall

    def submit(self, graph, client_id):
        self.submitted.append(graph)
        inputs = graph[SAMPLER_ID]["inputs"]
        project_dir = self.output / "h3_continuum" / "runs" / f"run_{uuid.UUID(inputs['project_id']).hex}"
        accepted = len(self.chain)
        if inputs["take_action"] == "Use This Take":
            reused, generated = accepted, 0
        elif inputs["review_action"] == "Regenerate Current":
            group = len(self.chain)
            self.chain.pop()
            revision = uuid.uuid4().hex[:12]
            self.groups.append((group, revision))
            self.chain.append(revision)
            reused, generated = accepted - 1, 1
        else:
            group = len(self.chain) + 1
            revision = uuid.uuid4().hex[:12]
            self.groups.append((group, revision))
            self.chain.append(revision)
            reused, generated = accepted - self.reuse_shortfall, 1 + self.reuse_shortfall
        project_dir.mkdir(parents=True, exist_ok=True)
        status = "complete" if len(self.chain) == inputs["chunks"] else "review_ready"
        (project_dir / "project.json").write_text(json.dumps(project_payload(self.groups, self.chain, status=status, unit=(len(self.chain),) * 2)))
        video = self.output / f"sequence_{len(self.submitted)}.mp4"
        video.write_bytes(b"mp4")
        self.status = (
            f"Run Storage: run_x / revision r; {reused} reused, {generated} generated, "
            f"{inputs['chunks']} total; resume={len(self.chain) + 1}; nonce=0 (inherit_branch)."
        )
        self.video = video
        return f"prompt-{len(self.submitted)}"

    def poll(self, prompt_id, graph):
        yield ("Generating video and audio", 1, len(graph), 1, 2)

    def history(self, prompt_id):
        return {"outputs": {"continuum_status": {"text": [self.status]}, "continuum_report": {"text": ["ok"]}}}


def fake_services(store, backend):
    models = SimpleNamespace(
        load_model_config=lambda: SimpleNamespace(
            profile_key=lambda name: "pruned",
            profiles={"pruned": SimpleNamespace(fl2va="fl2va.safetensors", ref2va="ref2va.safetensors", label="Pruned")},
            video_vae="video.safetensors",
            audio_vae="audio.safetensors",
        ),
        ensure_profile_model=lambda *args: False,
        ensure_h3_text_encoder=lambda models, choice: ("te.safetensors", False),
        ensure_base_video_vae=lambda models: False,
        ensure_audio_vae=lambda models: False,
    )
    request = make_request()
    execution = SimpleNamespace(
        object_info=lambda: {name: {} for name in required_continuum_nodes(request) | {"LoadImage"}},
        submit_prompt=backend.submit,
        poll_comfy_progress=backend.poll,
        wait_for_history=backend.history,
    )
    media = SimpleNamespace(resolve_output=lambda history, queued_at: backend.video)
    generation = SimpleNamespace(models=models, execution=execution, media=media)
    checkpoints = []
    services = orchestrator.ContinuumServices(
        store=store,
        generation=generation,
        runtime=SimpleNamespace(output_dir=backend.output),
        stage=lambda path: "h3_gradio/continuum/" + Path(path).name,
        checkpoint=checkpoints.append,
    )
    return services, checkpoints


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        root = Path(self.directory.name)
        self.output = root / "output"
        self.output.mkdir()
        self.store = ContinuumStore(WorkspaceStore(root / "state"), self.output)

    def tearDown(self):
        self.directory.cleanup()

    def run_action(self, services, run_id, action, epoch):
        return list(orchestrator.execute("alice", run_id, action, services, expected_epoch=epoch))

    def test_finish_runs_one_group_per_submission_and_checkpoints_each(self):
        backend = FakeBackend(self.store, self.output)
        services, checkpoints = fake_services(self.store, backend)
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b", "c")))
        updates = self.run_action(services, run_id, ContinuumAction(FINISH), 0)
        self.assertEqual(len(backend.submitted), 3)
        self.assertEqual(len(checkpoints), 3)
        self.assertTrue(all(g[SAMPLER_ID]["inputs"]["review_action"] == "Continue / Next" for g in backend.submitted))
        run = self.store.run("alice", run_id)
        self.assertEqual(run["state"], "complete")
        self.assertEqual(run["output"], str(backend.video))
        self.assertIn("3/3 chunk(s) saved", updates[-1].status)
        ops = self.store.operations("alice", run_id)
        self.assertEqual(sum(op["parent_id"] is None for op in ops), 1)

    def test_partial_previews_stay_out_of_media_until_complete(self):
        backend = FakeBackend(self.store, self.output)
        services, _ = fake_services(self.store, backend)
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b")))
        self.run_action(services, run_id, ContinuumAction(NEXT), 0)
        prefix = backend.submitted[-1][SAVE_ID]["inputs"]["filename_prefix"]
        self.assertTrue(prefix.startswith(f"h3_continuum/previews/{run_id}/"))
        self.run_action(services, run_id, ContinuumAction(NEXT), 1)
        self.assertTrue(backend.submitted[-1][SAVE_ID]["inputs"]["filename_prefix"].startswith("h3/sequence/"))
        self.assertTrue(Path(str(backend.video) + ".settings.json").is_file())

    def test_prompt_edit_caps_expected_reuse(self):
        backend = FakeBackend(self.store, self.output)
        services, _ = fake_services(self.store, backend)
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b", "c")))
        self.run_action(services, run_id, ContinuumAction(NEXT), 0)
        self.run_action(services, run_id, ContinuumAction(NEXT), 1)
        backend.reuse_shortfall = 1  # upstream regenerates the edited chunk 2
        updates = list(orchestrator.execute(
            "alice", run_id, ContinuumAction(NEXT), services, expected_epoch=2, reuse_cap=1,
        ))
        self.assertNotIn("⚠", updates[-1].status)

    def test_stale_epoch_is_rejected_before_submission(self):
        backend = FakeBackend(self.store, self.output)
        services, _ = fake_services(self.store, backend)
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b")))
        self.run_action(services, run_id, ContinuumAction(NEXT), 0)
        with self.assertRaises(StaleRunError):
            self.run_action(services, run_id, ContinuumAction(NEXT), 0)
        self.assertEqual(len(backend.submitted), 1)

    def test_retry_then_use_take_without_sampling(self):
        backend = FakeBackend(self.store, self.output)
        services, _ = fake_services(self.store, backend)
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b")))
        self.run_action(services, run_id, ContinuumAction(NEXT), 0)
        updates = self.run_action(services, run_id, ContinuumAction(RETRY), 1)
        self.assertNotIn("⚠", updates[-1].status)
        project = self.store.project(self.store.run("alice", run_id))
        first, second = project.takes_for(1)
        action = ContinuumAction(USE_TAKE, take_group=1, take_revision_id=first.revision_id)
        updates = self.run_action(services, run_id, action, 2)
        self.assertNotIn("⚠", updates[-1].status)
        self.assertEqual(backend.submitted[-1][SAMPLER_ID]["inputs"]["take_revision_id"], first.revision_id)

    def test_regenerated_prefix_is_surfaced_and_stops_finish(self):
        backend = FakeBackend(self.store, self.output)
        services, _ = fake_services(self.store, backend)
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b", "c")))
        self.run_action(services, run_id, ContinuumAction(NEXT), 0)
        backend.reuse_shortfall = 1
        updates = self.run_action(services, run_id, ContinuumAction(FINISH), 1)
        self.assertEqual(len(backend.submitted), 2)  # stopped after the warning
        self.assertIn("Expected to reuse 1 saved chunk(s)", updates[-1].status)

    def test_failure_keeps_committed_state_and_releases_lock(self):
        backend = FakeBackend(self.store, self.output)
        services, _ = fake_services(self.store, backend)
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b")))
        self.run_action(services, run_id, ContinuumAction(NEXT), 0)

        def broken(prompt_id):
            raise H3Error("decode failed")

        services.generation.execution.wait_for_history = broken
        with self.assertRaises(H3Error):
            self.run_action(services, run_id, ContinuumAction(NEXT), 1)
        run = self.store.run("alice", run_id)
        self.assertEqual(run["state"], "failed")
        self.assertEqual(self.store.project(run).accepted_chunks, 2)
        self.assertEqual(self.store.operations("alice", run_id)[0]["state"], "failed")
        with run_lock(run_id):  # released
            pass

    def test_concurrent_action_on_same_run_is_rejected(self):
        backend = FakeBackend(self.store, self.output)
        services, _ = fake_services(self.store, backend)
        run_id = self.store.create_run("alice", "r", make_request(actions=("a", "b")))
        entered, release = threading.Event(), threading.Event()
        original = backend.poll

        def slow(prompt_id, graph):
            entered.set()
            release.wait(5)
            yield from original(prompt_id, graph)

        backend.poll = slow
        services.generation.execution.poll_comfy_progress = slow
        worker = threading.Thread(target=self.run_action, args=(services, run_id, ContinuumAction(NEXT), 0))
        worker.start()
        self.assertTrue(entered.wait(5))
        with self.assertRaises(RunBusyError):
            self.run_action(services, run_id, ContinuumAction(NEXT), 1)
        release.set()
        worker.join(5)


if __name__ == "__main__":
    unittest.main()


class BindingTests(unittest.TestCase):
    def setUp(self):
        from h3_ui.continuum_bindings import SEQUENCE_INPUT_NAMES, ContinuumActions

        self.names = SEQUENCE_INPUT_NAMES
        self.actions = ContinuumActions(
            store=lambda: None, services=lambda: None, owner=lambda request: "alice",
            interrupt=lambda *args, **kwargs: "",
        )

    def captured(self, **overrides):
        values = {name: None for name in self.names}
        values.update(
            sequence_action=NEXT, run_id="", epoch="0", name="", shared_prompt="Scene",
            chunk_prompt_1="a", chunk_prompt_2="b", chunk_count=2, chunk_seconds=5,
            continuity="Balanced", mode="Reference media", ref_image_4="/r4.png",
            size="512×512 · square", steps=20, seed=-1,
        )
        values.update(overrides)
        return values

    def test_admission_pins_only_the_media_sequence_uses(self):
        self.assertEqual(self.actions.admit(self.captured()), {f"ref_image_{i}" for i in range(1, 10)})
        self.assertEqual(self.actions.admit(self.captured(mode="Text to video")), set())
        self.assertEqual(self.actions.admit(self.captured(run_id="ab" * 16)), set())

    def test_admission_rejects_unsupported_requests_before_queueing(self):
        with self.assertRaisesRegex(H3Error, "reference images only"):
            self.actions.admit(self.captured(ref_video_1="/v.mp4"))
        with self.assertRaisesRegex(H3Error, "Save the sequence"):
            self.actions.admit(self.captured(sequence_action=USE_TAKE))
        with self.assertRaisesRegex(H3Error, "no prompt"):
            self.actions.admit(self.captured(shared_prompt="", chunk_prompt_2=""))
        with self.assertRaisesRegex(H3Error, "size"):
            self.actions.admit(self.captured(size="huge"))

    def test_turbo_is_a_note_not_a_blocker(self):
        from h3_app.continuum_policy import sequence_notes

        notes = sequence_notes(
            generation_mode="Turbo", attention_mode="SLA", cache_mode="Off",
            latent_upscale=False, postprocess="None", alternate_decoder=False, semantic_bridge=False,
        )
        self.assertTrue(any("Turbo is not applied" in note for note in notes))
        self.assertEqual(self.actions.admit(self.captured(mode="Text to video")), set())

    def test_primary_label_tracks_review_state(self):
        from h3_ui.continuum_bindings import primary_label

        self.assertEqual(primary_label(2, 0, saved=False), "Save sequence & generate")
        self.assertEqual(primary_label(2, 1, saved=True), "Accept & generate chunk 2")
        self.assertEqual(primary_label(2, 2, saved=True), "Complete · add a chunk")
        self.assertEqual(primary_label(3, 3, saved=True, edited_from=2), "Regenerate chunk 2")
