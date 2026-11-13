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
from h3_ui.storyboard.payload import board_payload

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

    def test_runnable_scenes_leave_out_scenes_that_need_input_and_their_continuations(self):
        document = board("flow", "cut", "flow", takes=False)
        self.assertEqual(sb.runnable_scenes(document), [0, 1, 2, 3])
        document["scenes"][0]["prompt"] = ""
        self.assertEqual(sb.runnable_scenes(document), [2, 3])
        document["scenes"][2]["prompt"] = ""
        self.assertEqual(sb.runnable_scenes(document), [])
        self.assertEqual(len(sb.pending_scenes(document)), 4)

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
        self.assertEqual((values["mode"], values["ref_image_1"]), ("Text to video", None))  # Mode is explicit.
        scene.update(mode=sb.REFERENCES, ref_images=["/scene/extra.png"], ref_audios=["/scene/voice.wav"])
        values = sb.scene_request_values(document, 0, PROFILE_VALUES, 7)
        self.assertEqual(values["mode"], "Reference media")
        self.assertEqual(values["prompt"], "Officer is <Picture 1>. Mei is <Picture 2>. prompt 1")
        self.assertEqual([values[f"ref_image_{n}"] for n in (1, 2, 3)],
                         ["/cast/o.png", "/cast/mei.png", "/scene/extra.png"])
        self.assertEqual(values["ref_audio_1"], "/scene/voice.wav")
        self.assertIsNone(values["fl2va_audio_1"])
        self.assertEqual(sb.picture_tags(document, scene),
                         ["Officer = <Picture 1>", "Mei = <Picture 2>", "image 1 = <Picture 3>"])

    def test_a_continuing_scene_never_sends_a_first_frame(self):
        document = board("flow", takes=False)
        add_take(document, 0)
        scene = document["scenes"][1]
        scene.update(mode=sb.FRAMES, first_image="/scene/first.png")
        self.assertIn("Add the last-frame picture", sb.scene_issues(document, 1))
        scene["last_image"] = "/scene/last.png"
        self.assertEqual(sb.scene_issues(document, 1), [])
        values = sb.scene_request_values(document, 1, PROFILE_VALUES, 1)
        self.assertEqual((values["mode"], values["first_image"], values["last_image"]),
                         ("First / last frame", None, "/scene/last.png"))
        self.assertEqual(sb.runs_as(document, 1), "Last frame")
        scene["ref_audios"] = ["/scene/v1.wav", "/scene/v2.wav"]
        values = sb.scene_request_values(document, 1, PROFILE_VALUES, 1)
        self.assertEqual([values[f"fl2va_audio_{n}"] for n in (1, 2, 3)], ["/scene/v1.wav", "/scene/v2.wav", None])
        self.assertIsNone(values["ref_audio_1"])  # Frames voices never reach the Ref2VA slots.
        scene["link"] = "cut"
        self.assertEqual(sb.scene_request_values(document, 1, PROFILE_VALUES, 1)["first_image"], "/scene/first.png")

    def test_issues_explain_what_each_mode_still_needs(self):
        document = board("flow", takes=False)
        first, second = document["scenes"]
        first.update(prompt="", mode=sb.FRAMES)
        self.assertEqual(sb.scene_issues(document, 0), ["Write a prompt", "Add a first or last frame"])
        first.update(prompt="x", mode=sb.REFERENCES)
        self.assertEqual(sb.scene_issues(document, 0), ["Pick at least one reference"])
        first["ref_images"] = [f"/i{n}.png" for n in range(10)]
        self.assertIn("Use at most 9 pictures", sb.scene_issues(document, 0)[0])
        self.assertEqual(sb.scene_issues(document, 1), ["Generate scene 1 first: this scene continues from it"])
        with self.assertRaisesRegex(ValueError, "Scene 2: Generate scene 1 first"):
            sb.check_scene(document, 1)

    def test_profile_keeps_h3_settings_and_takes_finishing_from_defaults(self):
        values = {k: v for k, v in PROFILE_VALUES.items() if k in sb.PROFILE_FIELDS and k not in sb.DEFAULTED_FIELDS}
        result = sb.make_profile(values, PROFILE_VALUES)
        self.assertEqual(set(result), set(sb.PROFILE_FIELDS))
        self.assertNotIn("prompt", result)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            sb.make_profile({}, {})

    def test_payload_says_how_scenes_connect_and_what_they_need(self):
        document = board("flow", "cut")
        add_take(document, 0)
        document["scenes"].append(sb.new_scene("Empty", "", link="cut"))
        scenes = board_payload("id", document)["scenes"]
        self.assertTrue(scenes[1]["flow"])
        self.assertEqual(scenes[1]["summary"], "Continues scene 1 · text only")
        self.assertEqual(scenes[2]["summary"], "Fresh start · text only")
        self.assertEqual(scenes[-1]["state_label"], "Needs input")
        self.assertEqual(scenes[0]["number"], 1)


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

    def test_storyboards_are_shared_between_browsers_and_validated(self):
        # The Workshop and the GPU endpoint are different sites with different browser owners.
        storyboard_id = self.service.create("workshop", "Chase", profile(), cast=[("Mei", self.picture)])
        self.assertEqual([row["name"] for row in self.service.list("gpu")], ["Chase"])
        self.service.rename("gpu", storyboard_id, "Chase 2")
        self.assertEqual(self.service.load("workshop", storyboard_id)["name"], "Chase 2")
        with self.assertRaisesRegex(ValueError, "no longer exists"):
            self.service.load("a", "missing")
        document = self.service.load("a", storyboard_id)
        self.assertTrue(Path(document["cast"][0]["image"]).is_relative_to(self.store.root))
        scene_id = document["scenes"][0]["id"]
        with self.assertRaisesRegex(ValueError, "5 to 15 whole seconds"):
            self.service.update_scene("a", storyboard_id, scene_id, seconds=16)
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
        self.assertFalse(self.service.list("a")[0]["generated"])
        document = self.service.load("a", storyboard_id)
        scene_id = document["scenes"][0]["id"]
        video = self.runtime.output_dir / "h3/.storyboard/x/take.mp4"
        latent = "h3/.storyboard/x/take.safetensors"
        write_latent(self.runtime.output_dir / latent, 0)
        video.write_bytes(b"mp4")
        self.service.record_take("a", storyboard_id, scene_id, "t1", video=video, latents={"latent": latent},
                                 seed=3, inputs=sb.scene_inputs(document, 0))
        self.assertEqual(sb.scene_states(self.service.load("a", storyboard_id)), ["done"])
        self.assertTrue(self.service.list("a")[0]["generated"])
        _, version = self.service.save_version("a", storyboard_id, "First")
        branch = self.service.branch("a", storyboard_id, version)
        self.assertEqual(self.service.load("a", branch)["scenes"][0]["in_use"], "t1")
        self.service.delete("a", storyboard_id)
        self.assertTrue(video.is_file())
        self.service.delete("a", branch)
        self.assertFalse(video.is_file())
        self.assertFalse((self.runtime.output_dir / latent).is_file())

    def test_shared_scene_file_survives_while_only_a_branch_version_uses_it(self):
        original = self.service.create("a", "Chase", profile())
        scene_id = self.service.load("a", original)["scenes"][0]["id"]
        self.service.set_scene_media("a", original, scene_id, "ref_images", [self.picture])
        branch = self.service.branch("a", original)
        _, version = self.service.save_version("a", branch)
        self.service.set_scene_media("a", branch, scene_id, "ref_images", [])
        self.service.delete("a", original)
        restored = self.service.branch("a", branch, version)
        image = Path(self.service.load("a", restored)["scenes"][0]["ref_images"][0])
        self.assertEqual(image.read_bytes(), self.picture.read_bytes())
        self.service.delete("a", restored)
        self.assertTrue(image.exists())
        self.service.delete("a", branch)
        self.assertFalse(image.exists())

    def test_reference_take_tracks_cast_order_and_picture_across_branches(self):
        scene = sb.new_scene("Meeting", "prompt", mode=sb.REFERENCES, cast=["A", "B"])
        original = self.service.create("a", "Chase", profile(), scenes=[scene],
                                       cast=[("A", self.picture), ("B", self.picture)])
        document = self.service.load("a", original)
        inputs = sb.scene_inputs(document, 0)
        self.service.record_take("a", original, scene["id"], "t", video="v", latents={"latent": "l"},
                                 seed=1, inputs=inputs)
        branch = self.service.branch("a", original)
        self.assertEqual(sb.scene_states(self.service.load("a", branch)), [sb.DONE])
        changed = self.service.update_scene("a", branch, scene["id"], cast=["B", "A"])
        self.assertEqual(sb.scene_states(changed), [sb.OUTDATED])
        self.service.remove_cast("a", branch, "B")
        self.picture.write_bytes(b"replacement")
        self.service.add_cast("a", branch, "B", self.picture)
        changed = self.service.update_scene("a", branch, scene["id"], cast=["A", "B"])
        self.assertEqual(sb.scene_states(changed), [sb.OUTDATED])

    def test_delete_take_removes_unused_take_and_protects_take_in_use(self):
        storyboard_id = self.service.create("a", "Chase", profile())
        document = self.service.load("a", storyboard_id)
        scene_id = document["scenes"][0]["id"]
        inputs = sb.scene_inputs(document, 0)
        v1 = self.runtime.output_dir / "h3/.storyboard/x/t1.mp4"
        l1 = "h3/.storyboard/x/t1.safetensors"
        v2 = self.runtime.output_dir / "h3/.storyboard/x/t2.mp4"
        l2 = "h3/.storyboard/x/t2.safetensors"
        write_latent(self.runtime.output_dir / l1, 0)
        write_latent(self.runtime.output_dir / l2, 0)
        v1.write_bytes(b"mp4-1")
        v2.write_bytes(b"mp4-2")
        self.service.record_take("a", storyboard_id, scene_id, "t1", video=v1, latents={"latent": l1},
                                 seed=1, inputs=inputs)
        self.service.record_take("a", storyboard_id, scene_id, "t2", video=v2, latents={"latent": l2},
                                 seed=2, inputs=inputs)
        with self.assertRaisesRegex(H3Error, "Switch to another take"):
            self.service.delete_take("a", storyboard_id, scene_id, "t1")
        self.service.delete_take("a", storyboard_id, scene_id, "t2")
        after = self.service.load("a", storyboard_id)
        self.assertEqual(after["scenes"][0]["takes"], ["t1"])
        self.assertNotIn("t2", after["takes"])
        self.assertFalse(v2.is_file())
        self.assertTrue(v1.is_file())

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
        self.on_take = None
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
        if self.on_take is not None:
            self.on_take(values["prompt"])
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
        completed = []
        statuses = []
        for status, pulse in self.handlers.generate_remaining(storyboard_id, SimpleNamespace()):
            statuses.append(status)
            if isinstance(pulse, str):
                document = self.service.load("a", storyboard_id)
                completed.append(sb.scene_states(document).count(sb.DONE))
                self.assertEqual(len(self.graphs), completed[-1])
        self.assertEqual(completed, [1, 2, 3])
        self.assertRegex(statuses[-1], r"Total generation time \d+:\d{2}")
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

    def test_payload_carries_every_scene_so_switching_needs_no_request(self):
        storyboard_id = self.make("cut")
        first, second = self.service.load("a", storyboard_id)["scenes"]
        image = self.root / "portrait.png"
        image.write_bytes(b"cast")
        self.service.add_cast("a", storyboard_id, "Mei", image)
        self.service.set_mode("a", storyboard_id, first["id"], sb.REFERENCES)
        self.service.update_scene("a", storyboard_id, first["id"], cast=["Mei"])
        for field, filename in (("ref_images", "prop.png"), ("ref_videos", "clip.mp4"),
                                ("ref_audios", "voice.wav")):
            path = self.root / filename
            path.write_bytes(filename.encode())
            self.service.set_scene_media("a", storyboard_id, first["id"], field, [str(path)])
        document = self.service.load("a", storyboard_id)
        stored = document["scenes"][0]
        shown, plain = board_payload(storyboard_id, document)["scenes"]
        self.assertEqual([(item["path"], item["tag"]) for item in shown["pictures"]], [
            (document["cast"][0]["image"], "Mei = <Picture 1>"), (stored["ref_images"][0], "image 1 = <Picture 2>")])
        self.assertEqual(shown["ref_videos"][0]["path"], stored["ref_videos"][0])
        self.assertEqual(shown["ref_audios"][0]["path"], stored["ref_audios"][0])
        self.assertEqual(plain["id"], second["id"])
        self.assertEqual((plain["pictures"], plain["ref_videos"]), ([], []))
        self.service.set_mode("a", storyboard_id, first["id"], sb.FRAMES)
        frames = board_payload(storyboard_id, self.service.load("a", storyboard_id))["scenes"][0]
        self.assertEqual(frames["ref_audios"][0]["path"], stored["ref_audios"][0])
        self.assertEqual(frames["pictures"], [])

    def test_failed_take_keeps_the_rest_of_the_queue_paused(self):
        storyboard_id = self.make("flow")
        self.fail_on = "prompt 2"
        with self.assertRaisesRegex(H3Error, "out of memory"):
            list(self.handlers.generate_remaining(storyboard_id, SimpleNamespace()))
        document = self.service.load("a", storyboard_id)
        self.assertEqual(sb.scene_states(document), ["done", "draft"])
        self.assertEqual(document["queue"], [document["scenes"][1]["id"]])
        self.assertEqual(board_payload(storyboard_id, document)["queue"], [document["scenes"][1]["id"]])

    def test_flow_scene_waits_for_its_predecessor(self):
        storyboard_id = self.make("flow")
        scene_id = self.service.load("a", storyboard_id)["scenes"][1]["id"]
        with self.assertRaisesRegex(ValueError, "Generate scene 1 first"):
            list(self.handlers.generate_scene(storyboard_id, scene_id, SimpleNamespace()))

    def test_generate_remaining_skips_scenes_that_still_need_input(self):
        storyboard_id = self.make("cut", "flow")
        document = self.service.load("a", storyboard_id)
        self.service.update_scene("a", storyboard_id, document["scenes"][1]["id"], prompt="")
        statuses = [status for status, _ in self.handlers.generate_remaining(storyboard_id, SimpleNamespace())]
        self.assertEqual([prompt for prompt, _ in self.graphs], ["prompt 1"])
        self.assertIn("Skipped scene 2", statuses[-1])
        self.assertIn("wait for a scene that needs input", statuses[-1])

    def test_scenes_edited_during_a_run_leave_its_queue_instead_of_failing_it(self):
        storyboard_id = self.make("cut", "cut", "cut")
        scenes = self.service.load("a", storyboard_id)["scenes"]

        def edit(prompt):
            if prompt == "prompt 1":
                self.service.update_scene("a", storyboard_id, scenes[1]["id"], prompt="")
                self.service.delete_scene("a", storyboard_id, scenes[2]["id"])

        self.on_take = edit
        statuses = [status for status, _ in self.handlers.generate_remaining(storyboard_id, SimpleNamespace())]
        self.assertEqual([prompt for prompt, _ in self.graphs], ["prompt 1", "prompt 4"])
        self.assertIn("Skipped scene 2", statuses[-1])
        self.assertNotIn("wait for a scene", statuses[-1])
        self.assertEqual(self.service.load("a", storyboard_id)["queue"], [])

    def test_deleting_a_queued_scene_drops_it_from_the_queue(self):
        storyboard_id = self.make("cut")
        scenes = self.service.load("a", storyboard_id)["scenes"]
        self.service.set_queue("a", storyboard_id, [scene["id"] for scene in scenes])
        self.service.delete_scene("a", storyboard_id, scenes[1]["id"])
        self.assertEqual(self.service.load("a", storyboard_id)["queue"], [scenes[0]["id"]])

    def test_replaced_scene_media_is_removed_unless_something_still_uses_it(self):
        storyboard_id = self.make()
        scene_id = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        first, second = self.root / "first.png", self.root / "second.png"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        self.service.set_scene_media("a", storyboard_id, scene_id, "first_image", str(first))
        kept = Path(self.service.load("a", storyboard_id)["scenes"][0]["first_image"])
        self.service.save_version("a", storyboard_id)
        self.service.set_scene_media("a", storyboard_id, scene_id, "first_image", str(second))
        self.assertTrue(kept.is_file())  # The saved version still shows it.
        stored = Path(self.service.load("a", storyboard_id)["scenes"][0]["first_image"])
        self.service.set_scene_media("a", storyboard_id, scene_id, "first_image", None)
        self.assertFalse(stored.exists())

    def test_refused_scene_media_leaves_no_copied_files(self):
        storyboard_id = self.make()
        scene_id = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        clips = []
        for number in range(4):
            path = self.root / f"clip{number}.mp4"
            path.write_bytes(f"clip {number}".encode())
            clips.append(str(path))
        with self.assertRaisesRegex(ValueError, "at most 3 reference videos"):
            self.service.set_scene_media("a", storyboard_id, scene_id, "ref_videos", clips)
        media = self.service.media_dir(storyboard_id)
        self.assertEqual(list(media.iterdir()) if media.is_dir() else [], [])

    def test_deleting_a_storyboard_removes_its_media_folder(self):
        storyboard_id = self.make()
        scene_id = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        picture = self.root / "frame.png"
        picture.write_bytes(b"frame")
        self.service.set_scene_media("a", storyboard_id, scene_id, "first_image", str(picture))
        list(self.handlers.generate_remaining(storyboard_id, SimpleNamespace()))
        folder = self.service.media_dir(storyboard_id).parent
        self.service.delete("a", storyboard_id)
        self.assertFalse(folder.exists())

    def test_switching_a_take_asks_first_when_later_scenes_continue_from_it(self):
        storyboard_id = self.make("flow", "cut")
        list(self.handlers.generate_remaining(storyboard_id, SimpleNamespace()))
        first = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        list(self.handlers.generate_scene(storyboard_id, first, SimpleNamespace()))
        document = self.service.load("a", storyboard_id)
        new_take = document["scenes"][0]["takes"][1]
        self.assertEqual(sb.scene_states(document), ["done"] * 3)  # A new take doesn't replace the one in use.
        asked = self.handlers.use_take("a", storyboard_id, first, new_take)
        self.assertIn("Scene 2 continues", asked)
        self.assertIn("Scene 3 starts fresh", asked)
        self.assertEqual(sb.scene_states(self.service.load("a", storyboard_id)), ["done"] * 3)  # Only asked.
        self.assertIsNone(self.handlers.use_take("a", storyboard_id, first, new_take, confirmed=True))
        self.assertEqual(sb.scene_states(self.service.load("a", storyboard_id)), ["done", "outdated", "done"])

    def test_switch_regenerates_only_named_continuations(self):
        storyboard_id = self.make("flow", "flow", "flow", "cut")
        scenes = self.service.load("a", storyboard_id)["scenes"]
        for scene in scenes[:3]:
            list(self.handlers.generate_scene(storyboard_id, scene["id"]))
        first = scenes[0]["id"]
        list(self.handlers.generate_scene(storyboard_id, first))
        take = self.service.load("a", storyboard_id)["scenes"][0]["takes"][-1]
        self.graphs.clear()
        list(self.handlers.switch_and_regenerate(storyboard_id, first, take))
        document = self.service.load("a", storyboard_id)
        self.assertEqual([prompt for prompt, _ in self.graphs], ["prompt 2", "prompt 3"])
        self.assertEqual(sb.scene_states(document), [sb.DONE] * 3 + [sb.DRAFT] * 2)
        self.assertEqual(document["queue"], [])

    def test_prompt_writer_gets_valid_arguments_and_failures_keep_the_prompt(self):
        import inspect
        from h3_ui.prompt_controller import PromptController

        signature = inspect.signature(PromptController.enhance_h3_prompt)
        replies = iter([("Better prompt", "Prompt enhanced."), ("prompt 1", "Prompt enhancement failed: no key")])

        def writer(**kwargs):
            signature.bind(None, **kwargs)  # Regression: ref_image_size was passed and rejected.
            return next(replies)

        handlers = StoryboardHandlers(StoryboardServices(
            service=lambda: self.service, generate_take=self.fake_take,
            request_defaults=lambda: dict(PROFILE_VALUES), publish=lambda path, metadata: None,
            enhance_prompt=writer,
        ))
        storyboard_id = self.make()
        scene_id = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        settings = {"prompt_writer_backend": "Gemini", "gemini_prompt_model": "model", "gemini_api_key": "key",
                    "openai_api_key": "", "openai_model": "", "openai_base_url": None}
        handlers.enhance("a", storyboard_id, scene_id, settings)
        self.assertEqual(self.service.load("a", storyboard_id)["scenes"][0]["prompt"], "Better prompt")
        with self.assertRaisesRegex(H3Error, "Prompt unchanged"):
            handlers.enhance("a", storyboard_id, scene_id, settings)
        self.assertEqual(self.service.load("a", storyboard_id)["scenes"][0]["prompt"], "Better prompt")

    def test_scene_media_is_copied_once_and_matched_by_name_afterwards(self):
        storyboard_id = self.make()
        scene_id = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        picture = self.root / "upload" / "frame.png"
        picture.parent.mkdir()
        picture.write_bytes(b"png")
        self.assertTrue(self.service.set_scene_media("a", storyboard_id, scene_id, "last_image", str(picture)))
        stored = Path(self.service.load("a", storyboard_id)["scenes"][0]["last_image"])
        self.assertTrue(stored.is_relative_to(self.runtime.output_dir.resolve()))
        cached = self.root / "gradio-cache" / stored.name
        cached.parent.mkdir()
        cached.write_bytes(b"png")
        self.assertFalse(self.service.set_scene_media("a", storyboard_id, scene_id, "last_image", str(cached)))
        self.assertTrue(self.service.set_scene_media("a", storyboard_id, scene_id, "last_image", None))
        self.service.set_mode("a", storyboard_id, scene_id, sb.REFERENCES)
        self.assertEqual(self.service.load("a", storyboard_id)["scenes"][0]["mode"], sb.REFERENCES)

    def test_more_takes_per_run_use_consecutive_seeds_and_keep_the_first_in_use(self):
        storyboard_id = self.make()
        scene_id = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        self.service.update_scene("a", storyboard_id, scene_id, seed=10, takes_per_run=3)
        list(self.handlers.generate_scene(storyboard_id, scene_id, SimpleNamespace()))
        document = self.service.load("a", storyboard_id)
        scene = document["scenes"][0]
        self.assertEqual([document["takes"][t]["seed"] for t in scene["takes"]], [10, 11, 12])
        self.assertEqual(scene["in_use"], scene["takes"][0])

    def test_regenerating_an_outdated_scene_uses_the_new_take_and_unblocks_flows(self):
        storyboard_id = self.make("flow")
        list(self.handlers.generate_remaining(storyboard_id, SimpleNamespace()))
        first = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        self.service.update_scene("a", storyboard_id, first, prompt="prompt 1 changed")
        statuses = [status for status, _ in self.handlers.generate_remaining(storyboard_id, SimpleNamespace())]
        self.assertRegex(statuses[-1], r"Total generation time \d+:\d{2}")
        document = self.service.load("a", storyboard_id)
        self.assertEqual(sb.scene_states(document), ["done", "done"])
        self.assertEqual(document["scenes"][0]["in_use"], document["scenes"][0]["takes"][-1])

    def test_branch_media_matched_by_name_stays_current(self):
        storyboard_id = self.make()
        scene_id = self.service.load("a", storyboard_id)["scenes"][0]["id"]
        picture = self.root / "upload" / "frame.png"
        picture.parent.mkdir()
        picture.write_bytes(b"png")
        self.service.set_scene_media("a", storyboard_id, scene_id, "last_image", str(picture))
        stored = Path(self.service.load("a", storyboard_id)["scenes"][0]["last_image"])
        branch = self.service.branch("a", storyboard_id)
        cached = self.root / "gradio-cache" / stored.name
        cached.parent.mkdir()
        cached.write_bytes(b"png")
        # Rendering the branch hands back Gradio's copy of the original's file.
        self.assertFalse(self.service.set_scene_media("a", branch, scene_id, "last_image", str(cached)))
        self.assertEqual(self.service.load("a", branch)["scenes"][0]["last_image"], str(stored))


if __name__ == "__main__":
    unittest.main()
