"""View-only dub links: what they expose, how they end, and what the page contains."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import gradio as gr

from h3_app.dubbing import session as sessions, shares
from h3_app.dubbing.results import Player, player_folder, save_player
from h3_app.errors import H3Error
from h3_ui.dubbing.handlers import DubbingHandlers, DubbingServices
from h3_ui.dubbing.player import PLAYER_JS, player_html
from h3_ui.dubbing.share import install_share_routes, page_html, share_path
from h3_ui.dubbing.workshop import build_dubbing_workshop

LINES = [{
    "start": 0.5, "end": 2.0, "shown_start": 0.5, "shown_end": 2.2, "speaker": "speaker1", "source": "Hello there",
    "translation": "Bonjour <b>toi</b>", "source_words": [],
    "translation_words": [{"text": "Bonjour", "start": 0.5, "end": 1.0}, {"text": "b", "start": 1.0, "end": 1.1},
                          {"text": "toi", "start": 1.1, "end": 1.5}, {"text": "b", "start": 1.5, "end": 1.6}],
    "take_problem": "SECRET take failure",
}]


class ShareTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runtime = SimpleNamespace(outputs_dir=self.root / "outputs", output_dir=self.root / "outputs")
        env = patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(self.root / "workspace")})
        env.start()
        self.addCleanup(env.stop)
        self.output = self.dub("talk")
        app = FastAPI()
        install_share_routes(app, runtime=self.runtime)
        self.client = TestClient(app)

    def dub(self, stem: str, *, music: bool = True) -> Path:
        output = self.runtime.outputs_dir / "dubbing" / f"{stem}.mp4"
        folder = player_folder(output)
        folder.mkdir(parents=True)
        output.write_bytes(b"video bytes")
        tracks = {name: folder / f"{name}.mp3" for name in ("dub", "original", "music")}
        for name, path in tracks.items():
            path.write_bytes(name.encode() + b"-audio-bytes")
        save_player(folder, Player(output, tracks["dub"], tracks["original"], tracks["music"] if music else None, 4.0,
                                   LINES, "French", "English", True))
        return output

    def test_a_dub_has_one_link_and_stop_sharing_ends_it(self):
        token = shares.create(self.runtime, self.output, "Talk")
        self.assertRegex(token, shares.TOKEN)
        self.assertEqual(shares.create(self.runtime, self.output, "Talk"), token)
        self.assertEqual(shares.find(self.runtime, self.output), token)
        self.assertEqual(self.client.get(share_path(token)).status_code, 200)
        self.assertTrue(shares.revoke(self.runtime, self.output))
        self.assertFalse(shares.revoke(self.runtime, self.output))
        page = self.client.get(share_path(token))
        self.assertEqual(page.status_code, 404)
        self.assertIn("no longer available", page.text)
        self.assertEqual(self.client.get(f"{share_path(token)}/dub").status_code, 404)
        self.assertNotEqual(shares.create(self.runtime, self.output, "Talk"), token)

    def test_a_deleted_dub_or_a_bad_token_shows_nothing(self):
        token = shares.create(self.runtime, self.output, "Talk")
        secret = self.root / "workspace" / "dubbing-shares" / f"{token}.json"
        self.assertTrue(secret.is_file())
        for bad in ("x", "." * 32, "%2e%2e%2f" * 4, "a" * 32):
            self.assertEqual(self.client.get(f"/dub-share/{bad}").status_code, 404, bad)
        self.assertIsNone(shares.load(self.runtime, "../" * 10 + "xx"))
        (player_folder(self.output) / "dub.mp3").unlink()
        self.assertEqual(self.client.get(share_path(token)).status_code, 404)
        with self.assertRaisesRegex(H3Error, "no saved player"):
            shares.create(self.runtime, self.root / "missing.mp4", "")

    def test_tracks_are_served_by_name_with_range_requests(self):
        token = shares.create(self.runtime, self.output, "Talk")
        whole = self.client.get(f"{share_path(token)}/dub")
        self.assertEqual(whole.content, b"dub-audio-bytes")
        self.assertEqual(whole.headers["x-robots-tag"], "noindex, nofollow")
        partial = self.client.get(f"{share_path(token)}/dub", headers={"Range": "bytes=4-8"})
        self.assertEqual(partial.status_code, 206)
        self.assertEqual(partial.content, b"audio")
        self.assertEqual(self.client.head(f"{share_path(token)}/video").status_code, 200)
        self.assertEqual(self.client.get(f"{share_path(token)}/video").headers["content-type"], "video/mp4")
        for other in ("lines", "player.json", "..%2Fplayer.json", "__class__"):
            self.assertEqual(self.client.get(f"{share_path(token)}/{other}").status_code, 404, other)
        plain = self.dub("plain", music=False)
        other = shares.create(self.runtime, plain, "Plain")
        self.assertEqual(self.client.get(f"{share_path(other)}/music").status_code, 404)

    def test_the_page_is_self_contained_and_view_only(self):
        token = shares.create(self.runtime, self.output, 'Talk <script>"')
        response = self.client.get(share_path(token))
        page = response.text
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        self.assertIn("default-src 'none'", response.headers["content-security-policy"])
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn('Talk <script>', page)
        self.assertEqual(page.count("</script>"), 1)
        self.assertNotIn("</script>", PLAYER_JS)
        self.assertIn("Dubbed into French", page)
        for track in ("video", "dub", "original", "music"):
            self.assertIn(f"{share_path(token)}/{track}", page)
        # No server path, app file route or editor-only note reaches a viewer.
        for private in ("gradio_api", str(self.root), self.root.as_posix(), "SECRET", "Review lines", "Incomplete dub"):
            self.assertNotIn(private, page)
        self.assertIn('class="h3-word"', page)  # word-level sync
        self.assertIn("Bonjour &lt;b&gt;", page.replace('</span><span class="h3-word" data-start="1.000" '
                                                         'data-end="1.100">', ""))
        self.assertIn('data-shown-start="0.500"', page)
        self.assertIn("viewport-fit=cover", page)

    def test_a_dub_without_music_or_word_timing_has_a_plain_page(self):
        output = self.dub("plain", music=False)
        share = shares.load(self.runtime, shares.create(self.runtime, output, "Plain"))
        share.player.lines = [{**LINES[0], "translation_words": []}]
        page = page_html(share, "t" * 32)
        self.assertNotIn("data-music", page.split("<script>")[0])
        self.assertNotIn("vocal separation", page)
        self.assertNotIn('class="h3-word"', page)
        self.assertNotIn("Word timing unavailable", page)
        self.assertIn("Word timing unavailable", player_html(share.player, key="app", url=str))

    def test_workshop_makes_shows_and_ends_a_link(self):
        actions = DubbingHandlers(DubbingServices(lambda: self.runtime, None, lambda mode: [], gpu=False))
        source = self.root / "source.wav"
        source.write_bytes(b"x")
        session = sessions.create(self.runtime, source, video=False)
        request = SimpleNamespace(headers={"host": "shop.modal.run", "x-forwarded-proto": "https"})
        link, create, stop, _status = actions.share_state(session.id, request)
        self.assertFalse(link["visible"])
        self.assertFalse(create["interactive"])
        self.assertIn("Studio", actions.create_share(session.id, request)[3])
        session = sessions.modify(session, lambda latest: latest.data.update(last_output=str(self.output)))
        self.assertTrue(actions.share_state(session.id, request)[1]["interactive"])
        link, create, stop, status = actions.create_share(session.id, request)
        token = shares.find(self.runtime, self.output)
        self.assertEqual(link["value"], f"https://shop.modal.run/dub-share/{token}")
        self.assertTrue(link["visible"] and stop["visible"])
        self.assertFalse(create["visible"])
        self.assertIn("Link ready", status)
        self.assertEqual(shares.load(self.runtime, token).title, "source")
        self.assertEqual(actions.share_state(session.id, request)[0]["value"], link["value"])
        link, create, stop, status = actions.stop_share(session.id, request)
        self.assertFalse(link["visible"] or stop["visible"])
        self.assertTrue(create["visible"])
        self.assertIsNone(shares.find(self.runtime, self.output))

    def test_routes_are_installed_on_workshop_only(self):
        with gr.Blocks() as ui:
            view = build_dubbing_workshop(gr.Tab("Dub"), ui, self.runtime)
        self.assertTrue(view.share.visible)
        app = FastAPI()
        for install in ui.h3_routes:
            install(app)
        self.assertTrue(any(getattr(route, "path", "").startswith("/dub-share") for route in app.routes))
        self.assertTrue(any((view.create_share._id, "click") in fn.targets for fn in ui.fns.values()))
