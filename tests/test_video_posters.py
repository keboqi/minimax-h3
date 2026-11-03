"""Output videos have a first-frame poster, so a player can show a picture before loading."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

import gradio as gr
from fastapi.testclient import TestClient

from h3_app.server import ServerConfig, build_server


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
class VideoPosterTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.video = self.output / "take.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24",
             "-t", "1", "-pix_fmt", "yuv420p", str(self.video)],
            check=True,
        )
        with gr.Blocks() as demo:
            gr.Markdown()
        config = ServerConfig(
            comfy_url="http://127.0.0.1:9", output_dir=self.output, outputs_dir=self.root / "outputs",
            workflows={}, workflow_dir=self.root, video_extensions=frozenset({".mp4"}),
            image_extensions=frozenset(), audio_extensions=frozenset(), css="",
            thumbnail_dir=self.root / "thumbnails",
        )
        self.client = TestClient(build_server(demo, [str(self.output)], config))

    def test_an_output_video_has_a_cached_first_frame(self):
        response = self.client.get("/video-posters", params={"path": str(self.video)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(len(list((self.root / "thumbnails").glob("*.jpg"))), 1)

    def test_only_videos_in_the_output_folders_have_posters(self):
        outside = self.root / "elsewhere.mp4"
        shutil.copy(self.video, outside)
        picture = self.output / "frame.png"
        picture.write_bytes(b"png")
        for path in (outside, picture, self.output / "missing.mp4", self.output / ".." / "elsewhere.mp4"):
            with self.subTest(path=path.name):
                response = self.client.get("/video-posters", params={"path": str(path)})
                self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
