"""Storyboard documents, storage, the take runner and joining, without a GPU."""

from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from h3_app import storyboard as sb
from h3_app.catalog import UI_DEFAULTS
from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS
from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate
from h3_app.workspace_store import WorkspaceStore
from h3_ui.storyboard import handlers as handlers_module
from h3_ui.storyboard.handlers import StoryboardHandlers, StoryboardServices
from h3_ui.storyboard.presentation import strip_html, timeline_html

PROFILE_VALUES = {name: UI_DEFAULTS.get(name, f"default-{name}") for name in GENERATION_FIELDS}


def profile():
    return sb.make_profile({k: v for k, v in PROFILE_VALUES.items() if k in sb.PROFILE_FIELDS}, PROFILE_VALUES)


def board(*links, takes=True):
    scenes = [sb.new_scene(f"S{i + 1}", f"prompt {i + 1}", link=link) for i, link in enumerate(("flow", *links))]
    document = sb.model.new_storyboard("Test", profile(), scenes)
    if takes:
        for index, scene in enumerate(scenes):
            add_take(document, index)
    return document


def add_take(document, index, take_id=None):
    scene = document["scenes"][index]
    take_id = take_id or sb.new_id()
    document["takes"][take_id] = {"id": take_id, "inputs": sb.scene_inputs(document, index), "video": "",
                                  "latent": f"{take_id}.safetensors", "stage_latent": None, "prefix_frames": 39,
                                  "seed": 1}
    scene["takes"].append(take_id)
    scene["in_use"] = take_id
    return take_id


class StoryboardRuleTests(unittest.TestCase):
    def test_flow_scenes_follow_the_take_they_continue_from_until_a_cut(self):
        document = board("flow", "cut", "flow")
        self.assertEqual(sb.scene_states(document), ["done"] * 4)
        add_take(document, 0)
        self.assertEqual(sb.scene_states(document), ["done", "outdated", "done", "done"])
        document["scenes"][2]["prompt"] = "changed"
        self.assertEqual(sb.scene_states(document), ["done", "outdated", "outdated", "outdated"])
        self.assertEqual(sb.affected_by(document, 0), [1])
        self.assertEqual(sb.affected_by(document, 2), [3])

    def test_edits_outdate_only_their_scene_and_profile_outdates_all(self):
        document = board("flow")
        document["scenes"][1]["seconds"] = 10
        self.assertEqual(sb.scene_states(document), ["done", "outdated"])
        document["scenes"][1]["seconds"] = 5
        self.assertEqual(sb.scene_states(document), ["done", "done"])
        document["profile"] = {**document["profile"], "width": 1}
        self.assertEqual(sb.scene_states(document), ["outdated", "outdated"])

    def test_switching_a_link_outdates_that_scene_and_its_flow_run(self):
        document = board("flow", "flow")
        document["scenes"][1]["link"] = "cut"
        self.assertEqual(sb.scene_states(document), ["done", "outdated", "outdated"])

    def test_new_seconds_subtract_the_overlap_of_flow_scenes(self):
        document = board("flow", "cut", takes=False)
        for scene in document["scenes"]:
            scene["seconds"] = 10
        self.assertEqual([round(sb.new_seconds(document, i), 3) for i in range(3)], [10, 8.375, 10])
        self.assertAlmostEqual(sb.total_seconds(document), 28.375)

    def test_ready_cut_scene_runs_ahead_of_a_waiting_flow(self):
        document = board("flow", "cut", takes=False)
        queue = [scene["id"] for scene in document["scenes"]]
        self.assertEqual(sb.next_runnable(document, queue), 0)
        self.assertEqual(sb.next_runnable(document, queue[1:]), 2)
        add_take(document, 0)
        self.assertEqual(sb.next_runnable(document, queue[1:]), 1)

    def test_scene_requests_use_the_profile_and_tag_the_cast(self):
        document = board(takes=False)
        document["cast"] = [{"name": "Mei", "image": "/cast/mei.png"}, {"name": "Officer", "image": "/cast/o.png"}]
        scene = document["scenes"][0]
        values = sb.scene_request_values(document, 0, PROFILE_VALUES, 7)
        self.assertEqual((values["mode"], values["prompt"], values["ref_image_1"]), ("Text to video", "prompt 1", None))
        self.assertEqual(values["postprocess"], "None")
        self.assertEqual((values["duration"], values["seed"], values["result_format"]), (5.0, 7, "Video"))
        scene["cast"] = ["Officer", "Mei"]
        values = sb.scene_request_values(document, 0, PROFILE_VALUES, 7)
        self.assertEqual(values["mode"], "Reference media")
        self.assertEqual(values["prompt"], "Officer is <Picture 1>. Mei is <Picture 2>. prompt 1")
        self.assertEqual((values["ref_image_1"], values["ref_image_2"]), ("/cast/o.png", "/cast/mei.png"))

    def test_profile_keeps_h3_settings_and_takes_finishing_from_defaults(self):
        values = {k: v for k, v in PROFILE_VALUES.items() if k in sb.PROFILE_FIELDS and k not in sb.DEFAULTED_FIELDS}
        result = sb.make_profile(values, PROFILE_VALUES)
        self.assertEqual(set(result), set(sb.PROFILE_FIELDS))
        self.assertNotIn("prompt", result)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            sb.make_profile({}, {})

    def test_strip_and_timeline_show_state_links_and_new_seconds(self):
        document = board("flow", "cut")
        add_take(document, 0)
        html = strip_html(document, document["scenes"][1]["id"])
        self.assertIn('aria-pressed="true"', html)
        self.assertIn("h3-sb-outdated", html)
        self.assertIn("3.4 s new", html)
        self.assertEqual(html.count('data-sb-action="link"'), 2)
        timeline = timeline_html(document, None)
        self.assertEqual(timeline.count("h3-sb-segment-cut"), 1)


def write_latent(path: Path, prefix_frames: int):
    header = json.dumps({"__metadata__": {"h3_prefix_frames": str(prefix_frames)}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header)


class StoryboardServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig(script_dir=self.root, comfy_url="http://comfy", comfy_dir=self.root / "comfy",
                                     models_config=self.root / "models.json", outputs_dir=self.root / "outputs")
        self.service = sb.StoryboardService(self.store, lambda: self.runtime)
        picture = self.root / "mei.png"
        picture.write_bytes(b"png")
        self.picture = picture

    def tearDown(self):
        self.directory.cleanup()

    def test_schema_two_adds_storyboards_to_existing_workspaces(self):
        legacy = self.root / "legacy"
        legacy.mkdir()
        with closing(sqlite3.connect(legacy / "workspace.sqlite3")) as db:
            db.execute("PRAGMA user_version=1")
        store = WorkspaceStore(legacy)
        with closing(sqlite3.connect(legacy / "workspace.sqlite3")) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)
        self.assertEqual(store.storyboards("a"), [])

    def test_storyboards_are_owner_scoped_and_validated(self):
        storyboard_id = self.service.create("a", "Chase", profile(), cast=[("Mei", self.picture)])
        self.assertEqual([row["name"] for row in self.service.list("a")], ["Chase"])
        self.assertEqual(self.service.list("b"), [])
        with self.assertRaises(ValueError):
            self.service.load("b", storyboard_id)
        with self.assertRaises(ValueError):
            self.store.save_storyboard("b", storyboard_id, {"name": "stolen"})
        document = self.service.load("a", storyboard_id)
        self.assertTrue(Path(document["cast"][0]["image"]).is_relative_to(self.store.root))
        scene_id = document["scenes"][0]["id"]
        with self.assertRaisesRegex(ValueError, "5, 10 or 15"):
            self.service.update_scene("a", storyboard_id, scene_id, seconds=7)
        with self.assertRaisesRegex(H3Error, "at least one scene"):
            self.service.delete_scene("a", storyboard_id, scene_id)

    def test_scene_editing_and_removing_cast_cleans_scene_selection(self):
        storyboard_id = self.service.create("a", "Chase", profile(), cast=[("Mei", self.picture)])
        first = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        self.service.update_scene("a", storyboard_id, first, cast=["Mei", "Ghost"], seconds=10)
        _, second = self.service.add_scene("a", storyboard_id)
        document = self.service.load("a", storyboard_id)
        self.assertEqual(document["scenes"][0]["cast"], ["Mei"])
        self.assertEqual((document["scenes"][1]["seconds"], document["scenes"][1]["cast"]), (10, ["Mei"]))
        self.service.move_scene("a", storyboard_id, second, -1)
        self.service.remove_cast("a", storyboard_id, "Mei")
        document = self.service.load("a", storyboard_id)
        self.assertEqual([scene["id"] for scene in document["scenes"]], [second, first])
        self.assertTrue(all(scene["cast"] == [] for scene in document["scenes"]))

    def test_branches_share_takes_and_deleting_one_keeps_shared_files(self):
        storyboard_id = self.service.create("a", "Chase", profile())
        document = self.service.load("a", storyboard_id)
        scene_id = document["scenes"][0]["id"]
        video = self.runtime.output_dir / "h3/.storyboard/x/take.mp4"
        latent = "h3/.storyboard/x/take.safetensors"
        write_latent(self.runtime.output_dir / latent, 0)
        video.write_bytes(b"mp4")
        self.service.record_take("a", storyboard_id, scene_id, "t1", video=video, latents={"latent": latent},
                                 seed=3, inputs=sb.scene_inputs(document, 0))
        self.assertEqual(sb.scene_states(self.service.load("a", storyboard_id)), ["done"])
        _, version = self.service.save_version("a", storyboard_id, "First")
        branch = self.service.branch("a", storyboard_id, version)
        self.assertEqual(self.service.load("a", branch)["scenes"][0]["in_use"], "t1")
        self.service.delete("a", storyboard_id)
        self.assertTrue(video.is_file())
        self.service.delete("a", branch)
        self.assertFalse(video.is_file())
        self.assertFalse((self.runtime.output_dir / latent).is_file())

    def test_backup_restore_keeps_storyboards_and_relocates_cast_pictures(self):
        from h3_app.workspace_admin import backup_workspace, restore_workspace

        storyboard_id = self.service.create("a", "Chase", profile(), cast=[("Mei", self.picture)])
        backup = backup_workspace(self.store, self.root / "backup.zip")
        restored = restore_workspace(backup, self.root / "restored")
        image = Path(restored.storyboard("a", storyboard_id)["cast"][0]["image"])
        self.assertTrue(image.is_relative_to(restored.root))
        self.assertEqual(image.read_bytes(), b"png")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
    def test_join_drops_the_repeated_overlap_of_flow_scenes(self):
        document = board("flow", "cut")
        clips = self.root / "clips"
        clips.mkdir()
        for index, scene in enumerate(document["scenes"]):
            path = clips / f"{index}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24",
                            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "2",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)
            document["takes"][scene["in_use"]]["video"] = str(path)
        target = self.service.render(document, self.root / "joined.mp4")
        frames = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_frames", "-of", "csv=p=0", str(target)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(int(frames), 3 * 48 - 39)
        document["scenes"][1]["prompt"] = "changed"
        with self.assertRaisesRegex(H3Error, "outdated"):
            self.service.clips(document, include_outdated=False)


class StoryboardRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig(script_dir=self.root, comfy_url="http://comfy", comfy_dir=self.root / "comfy",
                                     models_config=self.root / "models.json", outputs_dir=self.root / "outputs")
        self.service = sb.StoryboardService(self.store, lambda: self.runtime)
        self.graphs = []
        self.fail_on = None
        self.handlers = StoryboardHandlers(StoryboardServices(
            service=lambda: self.service, generate_take=self.fake_take,
            request_defaults=lambda: dict(PROFILE_VALUES), publish=lambda path, metadata: None,
        ))
        patcher = patch.object(handlers_module, "require_owner", lambda request: "a")
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.directory.cleanup()

    def fake_take(self, values, graph, outputs):
        self.graphs.append((values["prompt"], graph))
        if self.fail_on == values["prompt"]:
            yield GenerationUpdate(None, "Error: out of memory")
            return
        yield GenerationUpdate(None, "Queued")
        video = self.runtime.output_dir / f"{graph.output_prefix}_00001_.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"mp4")
        latent = f"{graph.latent_output_prefix}_00001_.safetensors"
        write_latent(self.runtime.output_dir / latent, 39 if graph.continuation_source else 0)
        outputs["latent"] = latent
        yield GenerationUpdate(str(video), "Completed")

    def make(self, *links):
        storyboard_id = self.service.create("a", "Chase", profile(), scenes=[
            sb.new_scene(f"S{i + 1}", f"prompt {i + 1}", link=link) for i, link in enumerate(("flow", *links))])
        return storyboard_id

    def test_generate_remaining_chains_flows_from_the_take_in_use(self):
        storyboard_id = self.make("flow", "cut")
        statuses = [status for status, _ in self.handlers.generate_remaining(storyboard_id, SimpleNamespace())]
        self.assertEqual(statuses[-1], "All queued scenes generated.")
        document = self.service.load("a", storyboard_id)
        self.assertEqual(sb.scene_states(document), ["done"] * 3)
        self.assertEqual(document["queue"], [])
        prompts = [prompt for prompt, _ in self.graphs]
        self.assertEqual(prompts, ["prompt 1", "prompt 2", "prompt 3"])
        first_take = document["takes"][document["scenes"][0]["in_use"]]
        self.assertEqual(self.graphs[1][1].continuation_source, first_take["latent"])
        self.assertIsNone(self.graphs[2][1].continuation_source)
        second = document["takes"][document["scenes"][1]["in_use"]]
        self.assertEqual((second["prefix_frames"], first_take["prefix_frames"]), (39, 0))
        self.assertTrue(self.graphs[0][1].output_prefix.startswith("h3/.storyboard/"))

    def test_failed_take_keeps_the_rest_of_the_queue_paused(self):
        storyboard_id = self.make("flow")
        self.fail_on = "prompt 2"
        with self.assertRaisesRegex(H3Error, "out of memory"):
            list(self.handlers.generate_remaining(storyboard_id, SimpleNamespace()))
        document = self.service.load("a", storyboard_id)
        self.assertEqual(sb.scene_states(document), ["done", "draft"])
        self.assertEqual(document["queue"], [document["scenes"][1]["id"]])
        rendered = self.handlers.render("a", storyboard_id)
        self.assertIn("Generate remaining", rendered[17])

    def test_flow_scene_waits_for_its_predecessor(self):
        storyboard_id = self.make("flow")
        scene_id = self.service.load("a", storyboard_id)["scenes"][1]["id"]
        with self.assertRaisesRegex(H3Error, "Scene 1 must be done"):
            list(self.handlers.generate_scene(storyboard_id, scene_id, SimpleNamespace()))

    def test_more_takes_per_run_use_consecutive_seeds_and_keep_the_first_in_use(self):
        storyboard_id = self.make()
        scene_id = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        self.service.update_scene("a", storyboard_id, scene_id, seed=10, takes_per_run=3)
        list(self.handlers.generate_scene(storyboard_id, scene_id, SimpleNamespace()))
        document = self.service.load("a", storyboard_id)
        scene = document["scenes"][0]
        self.assertEqual([document["takes"][t]["seed"] for t in scene["takes"]], [10, 11, 12])
        self.assertEqual(scene["in_use"], scene["takes"][0])


if __name__ == "__main__":
    unittest.main()
