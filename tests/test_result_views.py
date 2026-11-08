"""Soundstage and Dub & Translate results keep their rich view: in their own flow and in Media."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from h3_app.dubbing.results import PLAYER_DIR, PLAYER_FILE, load_player, prune_players
from h3_app.provenance import write_snapshot
from h3_ui.result_views import result_view_values

ROOT = Path(__file__).resolve().parents[1]
RESULT_ID = "0123456789abcdef0123456789abcdef"


class MediaResultViewTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_soundstage_result_shows_timeline_subtitles_and_stage(self):
        audio = self.root / f"story_{RESULT_ID}.mp3"
        audio.write_bytes(b"")
        write_snapshot(audio, {"family": "Drama 3", "settings": {
            "duration": 6.0,
            "chunks": [{"name": "Mia", "text": "Hello <laugh> there.", "start": 0.0, "end": 2.0, "take": "a"},
                       {"name": "Leo", "text": "Hi.", "start": 2.5, "end": 3.5, "take": "b"}],
            "sounds": [{"name": "rain", "loop": True, "start": 0.0, "end": 6.0}],
        }}, record_job=False)
        soundstage, dub, stage, switch = result_view_values(str(audio))
        self.assertEqual((dub, stage, switch), ("", "", False))
        self.assertIn(f'data-media="{RESULT_ID}"', soundstage)
        self.assertIn("Hello there.", soundstage)
        self.assertIn("h3-drama3-stage-map", soundstage)
        self.assertIn("rain", soundstage)

    def test_illustrated_video_shows_with_its_mix_and_on_its_own(self):
        from h3_ui.result_views import result_view_updates

        audio = self.root / f"story_{RESULT_ID}.mp3"
        audio.write_bytes(b"")
        write_snapshot(audio, {"family": "Drama 3", "settings": {
            "duration": 4.0, "chunks": [{"name": "Mia", "text": "Hello.", "start": 0.0, "end": 2.0, "take": "a"}],
        }}, record_job=False)
        video = self.root / f"story_{RESULT_ID}_illustrated_1a2b3c4d.mp4"
        video.write_bytes(b"")
        write_snapshot(video, {"family": "Sound Stage illustrated video",
                               "settings": {"source": audio.name}}, record_job=False)
        # The mix offers its newest video behind the switch and starts on Audio.
        soundstage, dub, stage, switch = result_view_values(str(audio))
        self.assertIn("Hello.", soundstage)
        self.assertIn(video.name, stage)
        self.assertTrue(switch)
        *_, video_preview, audio_preview, stage_update, switch_update = result_view_updates(str(audio))
        self.assertEqual((stage_update["visible"], switch_update["visible"], switch_update["value"]),
                         (False, True, "Audio"))
        self.assertNotIn("visible", video_preview)
        # The video itself plays captioned, with its mix's subtitles, instead of the plain preview.
        soundstage, dub, stage, switch = result_view_values(str(video))
        self.assertIn("Hello.", soundstage)
        self.assertIn(f'data-media="{RESULT_ID}"', stage)
        self.assertFalse(switch)
        *_, video_preview, audio_preview, stage_update, switch_update = result_view_updates(str(video))
        self.assertEqual((video_preview["visible"], stage_update["visible"], switch_update["visible"]),
                         (False, True, False))
        self.assertNotIn("visible", audio_preview)
        # A video whose mix is gone still plays, without subtitles.
        audio.unlink()
        self.assertEqual(result_view_values(str(video))[0], "")
        self.assertIn(video.name, result_view_values(str(video))[2])

    def test_media_without_a_rich_view_hides_the_panels(self):
        audio = self.root / "song.mp3"
        audio.write_bytes(b"")
        self.assertEqual(result_view_values(str(audio)), ("", "", "", False))
        write_snapshot(audio, {"family": "Drama 3", "settings": {"chunks": [{"text": "No timing."}]}},
                       record_job=False)
        self.assertEqual(result_view_values(str(audio)), ("", "", "", False))
        self.assertEqual(result_view_values(None), ("", "", "", False))

    def dub(self, *, saved_root="/elsewhere/output/dubbing", file=PLAYER_FILE, key=None):
        """A dub's outputs and player folder, its paths recorded under another mount."""
        output_dir = self.root / "dubbing"
        folder = output_dir / PLAYER_DIR / "talk.fr"
        folder.mkdir(parents=True)
        for path in (output_dir / "talk.fr.mp3", output_dir / "talk.fr.mp4",
                     folder / "dub.mp3", folder / "original.mp3", folder / "music.mp3"):
            path.write_bytes(b"")
        player = {"video": f"{saved_root}/talk.fr.mp4", "dub": f"{saved_root}/{PLAYER_DIR}/talk.fr/dub.mp3",
                  "original": f"{saved_root}/{PLAYER_DIR}/talk.fr/original.mp3",
                  "music": f"{saved_root}/{PLAYER_DIR}/talk.fr/music.mp3", "duration": 3.0,
                  "lines": [{"start": 0.0, "end": 1.0, "shown_start": 0.0, "shown_end": 1.2, "speaker": "A",
                             "source": "Hello.", "translation": "Bonjour."}],
                  "target_language": "French", "source_language": "English", "word_timing": False}
        (folder / file).write_text(json.dumps({key: player} if key else player), encoding="utf-8")
        for output in (output_dir / "talk.fr.mp3", output_dir / "talk.fr.mp4"):
            write_snapshot(output, {"family": "Dub & Translate", "settings": {}}, record_job=False)
        return output_dir, folder

    def test_dub_result_plays_in_its_saved_player_from_another_mount(self):
        output_dir, folder = self.dub()
        for output in ("talk.fr.mp3", "talk.fr.mp4"):
            player = load_player(output_dir / output)
            self.assertEqual((player.video, player.dub, player.music),
                             (output_dir / "talk.fr.mp4", folder / "dub.mp3", folder / "music.mp3"))
            soundstage, dub, *_ = result_view_values(str(output_dir / output))
            self.assertEqual(soundstage, "")
            self.assertIn("Dub (French)", dub)
            self.assertIn("Bonjour.", dub)
            self.assertIn("data-music", dub)

    def test_dub_and_media_show_the_same_player_with_separate_controls(self):
        from h3_ui.result_views import dub_html

        output_dir, _folder = self.dub()
        player = load_player(output_dir / "talk.fr.mp3")
        self.assertIn('name="h3-dub-voice-dub-talk.fr"', dub_html(player, scope="dub"))
        self.assertIn('name="h3-dub-voice-media-talk.fr"', result_view_values(str(output_dir / "talk.fr.mp3"))[1])
        self.assertEqual(dub_html(None, scope="dub"), "")

    def test_dub_player_replaces_the_plain_preview_in_media(self):
        from h3_ui.result_views import result_view_updates

        output_dir, _folder = self.dub()
        for output in ("talk.fr.mp4", "talk.fr.mp3"):
            _soundstage, dub, video, audio, *_ = result_view_updates(str(output_dir / output))
            self.assertTrue(dub["visible"])
            self.assertEqual((video["visible"], audio["visible"]), (False, False))
        # Other media keep the preview their selection showed.
        plain = self.root / "song.mp3"
        plain.write_bytes(b"")
        _soundstage, dub, video, audio, stage, switch = result_view_updates(str(plain))
        self.assertEqual((stage["visible"], switch["visible"]), (False, False))
        self.assertFalse(dub["visible"])
        self.assertNotIn("visible", video)
        self.assertNotIn("visible", audio)

    def test_dubs_from_before_player_json_use_the_alignment_record(self):
        output_dir, _folder = self.dub(file="alignment.json", key="player")
        self.assertIn("Bonjour.", result_view_values(str(output_dir / "talk.fr.mp3"))[1])

    def test_dub_without_its_tracks_has_no_player(self):
        output_dir, folder = self.dub()
        (folder / "original.mp3").unlink()
        self.assertIsNone(load_player(output_dir / "talk.fr.mp3"))
        self.assertEqual(result_view_values(str(output_dir / "talk.fr.mp3")), ("", "", "", False))

    def test_player_folder_stays_while_either_output_remains(self):
        output_dir, folder = self.dub()
        (output_dir / "talk.fr.mp3").unlink()
        prune_players(output_dir)
        self.assertTrue(folder.exists())
        (output_dir / "talk.fr.mp4").unlink()
        prune_players(output_dir)
        self.assertFalse(folder.exists())


class LibraryConsoleImportTests(unittest.TestCase):
    def test_result_views_import_without_the_generation_stack(self):
        # The Library Console renders these without the dubbing pipeline or Drama 3 serving.
        code = ("import sys\n"
                "import h3_ui.result_views\n"
                "h3_ui.result_views.result_view_values('missing.mp3')\n"
                "print(sorted(name for name in sys.modules if name.startswith(('h3_app.dubbing.pipeline', "
                "'h3_app.drama3.serving', 'h3_ui.drama3.view'))))")
        completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
