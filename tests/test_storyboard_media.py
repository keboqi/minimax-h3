"""Storyboard scenes from Media: model rules, preparing the clip with ffmpeg, continuing it, the graph and page."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from h3_app import storyboard as sb
from h3_app.catalog import H3_CONTINUATION_PREFIX_NODE, H3_ENCODE_AV_TAIL_NODE, H3_LOAD_AV_LATENT_NODE, H3_TAIL_FRAMES
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate
from h3_app.workflows import h3
from h3_app.workspace_store import WorkspaceStore
from h3_ui.storyboard import handlers as handlers_module
from h3_ui.storyboard import page as page_module
from h3_ui.storyboard.handlers import StoryboardHandlers, StoryboardServices
from h3_ui.storyboard.payload import board_payload
from tests.test_storyboard import PROFILE_VALUES, profile, write_latent
from tests.test_storyboard_graph import STORYBOARD, nodes
from tests.workflow_cases import H3_DEFAULTS


def probe(path, entries, stream="v:0"):
    return subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", stream, "-show_entries", entries,
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True,
    ).stdout.strip()


def make_video(path: Path, *, width=640, height=360, seconds=4.0, audio=True, rate=30):
    """A test pattern, at a frame rate other than 24 so conforming has work to do."""
    command = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
               f"testsrc=size={width}x{height}:rate={rate}:duration={seconds}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate=44100"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", *(["-c:a", "aac"] if audio else []), "-shortest", str(path)]
    subprocess.run(command, check=True)
    return path


def media_board(*links, size=(320, 192)):
    """A scene from Media first, then generated scenes with ``links``."""
    document = sb.model.new_storyboard("Test", {**profile(), "width": size[0], "height": size[1]},
                                       [sb.new_media_scene(), *[sb.new_scene(f"S{i + 2}", f"prompt {i + 2}", link=link)
                                                                for i, link in enumerate(links)]])
    return document


class MediaSceneModelTests(unittest.TestCase):
    def test_a_media_scene_never_continues_but_the_next_scene_can(self):
        document = media_board("flow")
        document["scenes"].insert(0, sb.new_scene("Opening", "prompt"))
        document["scenes"][1]["link"] = sb.FLOW
        self.assertFalse(sb.flows_from_previous(document, 1))
        self.assertTrue(sb.flows_from_previous(document, 2))
        self.assertEqual(sb.affected_by(document, 0), [])
        self.assertEqual(sb.affected_by(document, 1), [2])
        self.assertEqual(sb.runs_as(document, 1), sb.VIDEO_FROM_MEDIA)

    def test_issues_ask_for_a_video_and_enough_of_it_for_a_continuing_scene(self):
        document = media_board("flow")
        self.assertEqual(sb.scene_issues(document, 0), ["Choose a video from Media or upload one"])
        self.assertEqual(sb.scene_issues(document, 1), ["Prepare scene 1 first: this scene continues its video"])
        self.assertEqual(sb.own_issues(document, 1), [])
        scene = document["scenes"][0]
        scene.update(video="clip.mp4", video_info={"width": 640, "height": 360, "seconds": 2.5, "audio": True})
        self.assertRegex(sb.scene_issues(document, 0)[0], "at least 3.0 s")
        document["scenes"][1]["link"] = sb.CUT
        self.assertEqual(sb.scene_issues(document, 0), [])

    def test_inputs_follow_range_fit_and_canvas_and_length_is_the_range(self):
        document = media_board()
        scene = document["scenes"][0]
        scene.update(video="/m/clip.mp4", video_info={"width": 640, "height": 360, "seconds": 10.0, "audio": False},
                     video_in=2.0, video_out=7.5)
        inputs = sb.scene_inputs(document, 0)
        self.assertEqual(inputs, {"kind": sb.MEDIA, "video": "clip.mp4", "in": 2.0, "out": 7.5, "fit": "resize",
                                  "size": [320, 192], "flow": False, "source_take": None})
        self.assertEqual(sb.new_seconds(document, 0), 5.5)
        scene["video_out"] = None
        self.assertEqual(sb.video_range(scene), (2.0, 10.0))
        self.assertEqual(sb.canvas({"width": 830, "height": 470}), (832, 480))
        with self.assertRaisesRegex(ValueError, "isn't generated"):
            sb.scene_request_values(document, 0, PROFILE_VALUES, 1)
        scene["fit"] = "stretch"
        with self.assertRaisesRegex(ValueError, "resize or crop"):
            sb.model.validate_scene(scene)

    def test_reshaping_keeps_about_the_pixel_count(self):
        self.assertEqual(sb.model.shaped_like({"width": 832, "height": 480}, 1080, 1920), (480, 832))
        width, height = sb.model.shaped_like({"width": 832, "height": 480}, 1000, 1000)
        self.assertEqual((width, height), (640, 640))

    def test_tail_matches_the_encode_node(self):
        self.assertEqual(sb.TAIL_FRAMES, H3_TAIL_FRAMES)
        self.assertEqual((sb.TAIL_FRAMES - 5) % 17, 0)
        self.assertGreaterEqual(sb.TAIL_FRAMES, max(sb.OVERLAP_FRAMES))


class MediaSceneServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig(script_dir=self.root, comfy_url="http://comfy", comfy_dir=self.root / "comfy",
                                     models_config=self.root / "models.json", outputs_dir=self.root / "outputs")
        self.service = sb.StoryboardService(self.store, lambda: self.runtime)
        self.graphs = []
        self.handlers = StoryboardHandlers(StoryboardServices(
            service=lambda: self.service, generate_take=self.fake_take,
            request_defaults=lambda: dict(PROFILE_VALUES), publish=lambda path, metadata: None,
        ))
        patcher = patch.object(handlers_module, "require_owner", lambda request: "a")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.storyboard_id = self.service.create("a", "Extend", {**profile(), "width": 320, "height": 192},
                                                 scenes=[sb.new_scene("Next", "what happens next")])
        _, self.media_id = self.service.add_media_scene("a", self.storyboard_id)
        self.service.move_scene("a", self.storyboard_id, self.media_id, -1)

    def tearDown(self):
        self.directory.cleanup()

    def fake_take(self, values, graph, outputs):
        self.graphs.append(graph)
        video = self.runtime.output_dir / f"{graph.output_prefix}_00001_.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        make_video(video, width=320, height=192, seconds=3, rate=24)
        latent = f"{graph.latent_output_prefix}_00001_.safetensors"
        write_latent(self.runtime.output_dir / latent, 39)
        outputs["latent"] = latent
        yield GenerationUpdate(str(video), "Completed")

    def board(self):
        return self.service.load("a", self.storyboard_id)

    def test_preparing_fits_the_range_to_the_canvas_and_cuts_its_tail(self):
        source = make_video(self.root / "wide.mp4", seconds=5, audio=False)
        self.service.set_scene_video("a", self.storyboard_id, self.media_id, str(source))
        self.service.update_scene("a", self.storyboard_id, self.media_id, video_in=1.0, video_out=4.5, fit=sb.CROP)
        document = self.board()
        scene = document["scenes"][0]
        self.assertEqual(scene["video_info"], {"width": 640, "height": 360, "seconds": 5.0, "audio": False})
        self.assertTrue(Path(scene["video"]).is_relative_to(self.service.media_dir(self.storyboard_id)))
        self.assertEqual(sb.scene_states(document), [sb.DRAFT, sb.DRAFT])

        self.service.prepare_media_scene("a", self.storyboard_id, self.media_id)
        document = self.board()
        take = document["takes"][document["scenes"][0]["in_use"]]
        self.assertEqual(probe(take["video"], "stream=width,height,nb_read_frames"), "320,192,84")
        self.assertEqual(probe(take["video"], "stream=codec_type", "a:0"), "audio")
        self.assertEqual(probe(take["tail"], "stream=width,height,nb_read_frames"), f"320,192,{sb.TAIL_FRAMES}")
        self.assertEqual(sb.scene_states(document)[0], sb.DONE)

        # Fit makes it out of date; preparing again keeps only the new clip.
        self.service.update_scene("a", self.storyboard_id, self.media_id, fit=sb.RESIZE)
        self.assertEqual(sb.scene_states(self.board())[0], sb.OUTDATED)
        self.service.prepare_media_scene("a", self.storyboard_id, self.media_id)
        document = self.board()
        self.assertEqual(len(document["scenes"][0]["takes"]), 1)
        self.assertNotIn(take["id"], document["takes"])
        self.assertFalse(Path(take["video"]).exists() or Path(take["tail"]).exists())

    def test_generate_remaining_prepares_the_video_then_continues_its_tail(self):
        source = make_video(self.root / "clip.mp4", seconds=4)
        self.service.set_scene_video("a", self.storyboard_id, self.media_id, str(source))
        staged = []

        def stage(path, category, reuse=False, *, runtime):
            staged.append((Path(path).name, category))
            return f"h3_gradio/{category}/tail.mp4"

        with patch.object(handlers_module, "stage_file", stage):
            statuses = [status for status, _ in self.handlers.generate_remaining(self.storyboard_id, SimpleNamespace())]
        document = self.board()
        self.assertEqual(sb.scene_states(document), [sb.DONE, sb.DONE])
        self.assertIn("Scene 1 prepared · 1 left.", statuses)
        media_take = document["takes"][document["scenes"][0]["in_use"]]
        self.assertEqual(staged, [(Path(media_take["tail"]).name, handlers_module.TAIL_CATEGORY)])
        (graph,) = self.graphs
        self.assertEqual(graph.continuation_video, "h3_gradio/storyboard_tails/tail.mp4")
        self.assertIsNone(graph.continuation_source)
        self.assertIsNone(graph.continuation_stage_source)
        next_take = document["takes"][document["scenes"][1]["in_use"]]
        self.assertEqual(next_take["inputs"]["source_take"], media_take["id"])

        # Joined at the canvas: all of the video, then the new footage after the shared frames.
        joined = self.service.render(document, self.root / "joined.mp4")
        self.assertEqual(probe(joined, "stream=width,height,nb_read_frames"), f"320,192,{96 + 72 - 39}")

    def test_payload_carries_the_clip_canvas_fit_and_preview_frames(self):
        source = make_video(self.root / "clip.mp4", seconds=4)
        self.service.set_scene_video("a", self.storyboard_id, self.media_id, str(source))
        data = board_payload(self.storyboard_id, self.board())
        scene = data["scenes"][0]
        self.assertEqual((scene["kind"], scene["runs_as"], scene["seconds"]), (sb.MEDIA, sb.VIDEO_FROM_MEDIA, 4.0))
        video = scene["video"]
        self.assertEqual((video["fit"], video["canvas"], video["continued"]), ("resize", [320, 192], True))
        self.assertEqual((video["clip"]["width"], video["clip"]["height"], video["clip"]["audio"]), (640, 360, True))
        self.assertIn("which=last", video["stills"]["last"])
        self.assertIn("end=4.000", video["stills"]["last"])
        self.assertEqual(data["scenes"][1]["kind"], "generated")
        self.assertIsNone(data["scenes"][1]["video"])

    def test_matching_the_video_shape_only_before_anything_is_generated(self):
        source = make_video(self.root / "tall.mp4", width=360, height=640, seconds=4)
        self.service.set_scene_video("a", self.storyboard_id, self.media_id, str(source))
        self.service.match_video_shape("a", self.storyboard_id, self.media_id)
        self.assertEqual(sb.canvas(self.board()["profile"]), (192, 320))
        with patch.object(handlers_module, "stage_file", lambda *args, **kwargs: "h3_gradio/tail.mp4"):
            list(self.handlers.generate_remaining(self.storyboard_id, SimpleNamespace()))
        self.assertFalse(board_payload(self.storyboard_id, self.board())["scenes"][0]["video"]["can_reshape"])
        with self.assertRaisesRegex(H3Error, "already generated"):
            self.service.match_video_shape("a", self.storyboard_id, self.media_id)

    def test_only_a_media_scene_takes_a_video_or_a_range(self):
        board = self.board()
        generated = board["scenes"][1]["id"]
        with self.assertRaisesRegex(H3Error, "Only a scene from Media"):
            self.service.update_scene("a", self.storyboard_id, generated, fit=sb.CROP)
        with self.assertRaisesRegex(H3Error, "isn't a video"):
            broken = self.root / "broken.mp4"
            broken.write_bytes(b"not a video")
            self.service.set_scene_video("a", self.storyboard_id, self.media_id, str(broken))


class MediaScenePageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig(script_dir=self.root, comfy_url="http://comfy", comfy_dir=self.root / "comfy",
                                     models_config=self.root / "models.json", outputs_dir=self.root / "outputs")
        self.service = sb.StoryboardService(self.store, lambda: self.runtime)
        self.handlers = StoryboardHandlers(StoryboardServices(
            service=lambda: self.service, generate_take=None,
            request_defaults=lambda: dict(PROFILE_VALUES), publish=lambda path, metadata: None,
        ))
        app = FastAPI()
        patcher = patch.object(page_module, "request_owner", lambda request, store: "a")
        patcher.start()
        self.addCleanup(patcher.stop)
        page_module.install_storyboard_page(app, self.handlers, gpu=False)
        self.client = TestClient(app)
        self.storyboard_id = self.service.create("a", "Extend", {**profile(), "width": 320, "height": 192})

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def wait(self):
        for _ in range(200):
            run = self.client.get(f"/storyboard/api/boards/{self.storyboard_id}/run").json()["run"]
            if not run["active"]:
                return run
            import time
            time.sleep(0.05)
        self.fail("The run never finished.")

    def test_add_upload_preview_and_prepare_without_a_gpu(self):
        api = f"/storyboard/api/boards/{self.storyboard_id}"
        data = self.client.post(f"{api}/scenes", json={"kind": sb.MEDIA}).json()
        scene_id = data["scene_id"]
        self.assertEqual(data["scenes"][-1]["kind"], sb.MEDIA)
        source = make_video(self.root / "clip.mp4", seconds=4)
        with source.open("rb") as handle:
            data = self.client.post(f"{api}/scenes/{scene_id}/video", files={"file": ("clip.mp4", handle, "video/mp4")}).json()
        video = data["scenes"][-1]["video"]
        self.assertEqual(video["clip"]["seconds"], 4.0)
        still = self.client.get(video["stills"]["last"])
        self.assertEqual((still.status_code, still.headers["content-type"]), (200, "image/jpeg"))
        self.assertEqual(self.client.get(video["stills"]["last"].replace("which=last", "which=middle")).status_code, 404)
        data = self.client.patch(f"{api}/scenes/{scene_id}", json={"fit": "crop", "video_in": 0.5}).json()
        self.assertEqual((data["scenes"][-1]["video"]["fit"], data["scenes"][-1]["video"]["in"]), ("crop", 0.5))
        self.assertIn("start=0.500", data["scenes"][-1]["video"]["stills"]["first"])
        clip_url = data["scenes"][-1]["video"]["clip"]["url"]
        self.assertEqual(self.client.get(clip_url).status_code, 200)

        started = self.client.post(f"{api}/scenes/{scene_id}/prepare").json()["run"]
        self.assertEqual(started["kind"], "prepare")
        run = self.wait()
        self.assertIsNone(run["error"])
        board = self.service.load("a", self.storyboard_id)
        self.assertEqual(sb.scene_states(board)[-1], sb.DONE)
        # A generated scene isn't prepared.
        first = board["scenes"][0]["id"]
        self.assertIn("Only a scene from Media", self.client.post(f"{api}/scenes/{first}/prepare").json()["error"])


class MediaSceneGraphTests(unittest.TestCase):
    def options(self, **extra):
        return {**H3_DEFAULTS, **STORYBOARD, "available_nodes": {*H3_DEFAULTS["available_nodes"], H3_ENCODE_AV_TAIL_NODE},
                "continuation_video": "h3_gradio/storyboard_tails/tail.mp4", **extra}

    def test_single_stage_encodes_the_clip_tail_at_the_canvas_and_continues_it(self):
        graph = h3.build_fl2va_graph(**self.options())
        self.assertEqual(nodes(graph, H3_LOAD_AV_LATENT_NODE), [])
        (load_id, load), = nodes(graph, "LoadVideo")
        self.assertEqual(load["file"], "h3_gradio/storyboard_tails/tail.mp4")
        (components_id, _), = [(node_id, inputs) for node_id, inputs in nodes(graph, "GetVideoComponents")
                               if inputs["video"] == [load_id, 0]]
        (encode_id, encode), = nodes(graph, H3_ENCODE_AV_TAIL_NODE)
        self.assertEqual((encode["width"], encode["height"], encode["frames"]), (864, 480, H3_TAIL_FRAMES))
        self.assertEqual((encode["images"], encode["audio"]), ([components_id, 0], [components_id, 1]))
        self.assertEqual(graph[encode["vae"][0]]["class_type"], "VAELoader")
        (_, prefix), = nodes(graph, H3_CONTINUATION_PREFIX_NODE)
        self.assertEqual(prefix["source"], [encode_id, 0])

    def test_latent_upscale_encodes_the_tail_at_both_stages(self):
        graph = h3.build_fl2va_graph(**self.options(latent_upscale_model_name="upscale.pth"))
        sizes = sorted((inputs["width"], inputs["height"]) for _, inputs in nodes(graph, H3_ENCODE_AV_TAIL_NODE))
        self.assertEqual(len(sizes), 2)
        self.assertEqual((sizes[0][0] * 2, sizes[0][1] * 2), sizes[1])
        self.assertEqual(len(nodes(graph, H3_CONTINUATION_PREFIX_NODE)), 2)

    def test_needs_the_node_and_only_one_kind_of_source(self):
        with self.assertRaisesRegex(H3Error, H3_ENCODE_AV_TAIL_NODE):
            h3.build_fl2va_graph(**self.options(available_nodes=H3_DEFAULTS["available_nodes"]))
        with self.assertRaisesRegex(H3Error, "not both"):
            h3.build_fl2va_graph(**self.options(continuation_source="h3/.storyboard/sb/prev.safetensors"))


if __name__ == "__main__":
    unittest.main()
