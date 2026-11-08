import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from h3_app.drama3.visual_plan import compile_plan, plan_shots
from h3_app.drama3.visuals import FPS, Shot, render_video
from h3_app.errors import H3Error
from h3_ui.drama3.visuals import generate_shot, illustrate_shots, shot_references


class IllustratedMixTests(unittest.TestCase):
    def test_returning_subject_uses_its_original_image_after_a_scene_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = [Path(tmp) / f"{i}.png" for i in range(3)]
            for image in images:
                Image.new("RGB", (4, 4)).save(image)
            controller = Mock()
            controller.generate.side_effect = [[([str(image)], "Complete")] for image in images]
            shots = [Shot(i * FPS, FPS, False, "Paint the shot", references=(subject,))
                     for i, subject in enumerate(("Mara", "tower", "Mara"))]
            list(illustrate_shots(controller, "qwen", shots, {"qwen__turbo_variant": "Off"}, True))
            calls = [call.args for call in controller.generate.call_args_list]
            self.assertEqual([args[1] for args in calls], ["Text to image", "Text to image", "Image edit"])
            self.assertEqual(calls[-1][3], [str(images[0])])

    def test_image_error_after_partial_output_stops_following_shots(self):
        controller = Mock()
        controller.generate.return_value = [([__file__], "Preview"), ([__file__], "Error: backend failed")]
        shot = Shot(0, FPS, False, "Street", references=("Mara",))
        with self.assertRaisesRegex(H3Error, "Shot 1 of 2 failed.*backend failed"):
            list(illustrate_shots(controller, "qwen", [shot, shot], {"qwen__turbo_variant": "Off"}, True))
        controller.generate.assert_called_once()

    def test_retry_resumes_after_the_shots_a_failed_export_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = [Path(tmp) / f"{i}.png" for i in range(3)]
            for image in images:
                Image.new("RGB", (4, 4)).save(image)
            shots = [Shot(i * FPS, FPS, False, "Paint", references=(subject,))
                     for i, subject in enumerate(("Mara", "tower", "Mara"))]
            captured = {"qwen__turbo_variant": "Off"}
            finished = []
            controller = Mock()
            controller.generate.side_effect = [[([str(images[0])], "Complete")],
                                               [([str(images[1])], "Complete")],
                                               [(None, "Error: out of memory")]]
            with self.assertRaisesRegex(H3Error, "Shot 3 of 3"):
                list(illustrate_shots(controller, "qwen", shots, captured, True, finished))
            self.assertEqual(finished, [str(images[0]), str(images[1])])
            controller = Mock()
            controller.generate.side_effect = [[([str(images[2])], "Complete")]]

            def drain(iterator):
                try:
                    while True:
                        next(iterator)
                except StopIteration as stop:
                    return stop.value

            result, records = drain(illustrate_shots(controller, "qwen", shots, captured, True, finished))
            self.assertEqual(result, [str(image) for image in images])
            controller.generate.assert_called_once()
            # The resumed shot still edits from Mara's first appearance.
            self.assertEqual(controller.generate.call_args.args[1:4:2], ("Image edit", [str(images[0])]))
            self.assertEqual(records[2]["references"], [{"shot": 1, "subjects": ["Mara"]}])

    def test_plan_is_saved_beside_the_mix_for_its_art_direction(self):
        from h3_ui.drama3.visuals import plan_rows, read_plan, write_plan

        plan = {"style": "ink", "continuity": {}, "shots": [
            {"start": 0, "transition": "cut", "beat": "Open", "references": [], "prompt": "A door"},
            {"start": 1.5, "transition": "dissolve", "beat": "Rain", "references": [], "prompt": "Rain"}]}
        with tempfile.TemporaryDirectory() as tmp:
            mix = Path(tmp) / f"mix_{'b' * 32}.mp3"
            self.assertIsNone(read_plan(mix))
            write_plan(mix, "Comic panels ", 16.0, plan)
            self.assertEqual(read_plan(mix, "Comic panels", 16)["plan"], plan)
            self.assertIsNone(read_plan(mix, "Oil paint", 16))
            self.assertIsNone(read_plan(mix, "Comic panels", 8))
            rows = plan_rows(read_plan(mix), 3.0)
            self.assertEqual([row[:3] for row in rows], [[0.0, 1.5, "Cut"], [1.5, 3.0, "Dissolve"]])
            # The plan file is hidden, so it never shows up as a result beside the mix.
            self.assertEqual([p.name for p in Path(tmp).iterdir() if not p.name.startswith(".")], [])

    def test_cut_off_writer_reply_explains_itself(self):
        settings = {"duration": 5, "chunks": [{"name": "Ana", "text": "Hi", "start": 0, "end": 2}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        with patch("h3_app.drama3.visual_plan.write_prompt", return_value=('{"style": "ink", "shots": [', "ok")):
            with self.assertRaisesRegex(H3Error, "cut off"):
                plan_shots(settings, "comic", 16, "model", "", "Gemini", "", "", "", runtime=runtime)

    def test_edit_reuses_first_appearances_and_respects_reference_priority(self):
        first_seen = {"Mara": 0, "bell": 0, "key": 2, "courtyard": 4}
        keys = ["key", "Mara", "courtyard", "bell", "Mara", "newcomer"]
        sources = shot_references(keys, first_seen, 2)
        self.assertEqual(list(sources.items()), [(2, ["key"]), (0, ["Mara", "bell"])])
        images = ["mara.png", "other.png", "key.png", "other2.png", "courtyard.png"]
        controller = Mock()
        for engine, tags in (("qwen", ("<image1>", "<image2>")), ("flux3", ("<ref_image_0>", "<ref_image_1>")),
                             ("h3", ("<Picture 1>", "<Picture 2>")), ("fibo2", ("<image_1>", "<image_2>"))):
            with self.subTest(engine=engine):
                generate_shot(controller, engine, "Close-up in the rain.", sources, images, {})
                _, mode, prompt, paths, _ = controller.generate.call_args.args
                self.assertEqual(mode, "Image edit")
                self.assertEqual(paths, ["key.png", "mara.png"])
                self.assertIn(f"key from {tags[0]}", prompt)
                self.assertIn(f"Mara, bell from {tags[1]}", prompt)
                self.assertIn("Close-up in the rain.", prompt)
        for keys, limit in ((["offscreen"], 3), (["Mara"], 0), ([], 10)):
            sources = shot_references(keys, first_seen, limit)
            generate_shot(controller, "moss_image", "Empty street", sources, images, {})
            controller.generate.assert_called_with("moss_image", "Text to image", "Empty street", [], {})

    def test_overlaps_boundaries_and_legacy_chunks(self):
        settings = {"duration": 12.2, "chunks": [
            {"name": "Ana", "text": "Wait here", "start": 0, "end": 5},
            {"name": "Jo", "lines": [{"text": "I see it", "start": 9, "end": 12}]},
        ], "sounds": [{"name": "rain", "loop": True, "start": 0, "end": 12},
                      {"name": "bell", "start": 4, "end": 5}]}
        plan = {"style": "ink wash", "continuity": {"Ana": "yellow coat", "bell": "brass handbell"},
                "shots": [{"start": 0, "transition": "cut", "beat": "Waiting", "references": ["Ana"],
                           "prompt": "Ana waits in the doorway."},
                          {"start": 2.25, "transition": "cut", "beat": "Recognition",
                           "references": ["Ana", "bell"], "prompt": "Ana notices the handbell."},
                          {"start": 9, "transition": "dissolve", "beat": "Answer", "references": [],
                           "prompt": "A window lights up across the street."}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        for backend in ("Gemini", "OpenAI-compatible"):
            with self.subTest(backend=backend), patch("h3_app.drama3.visual_plan.write_prompt",
                    return_value=(json.dumps(plan), "Enhanced")) as writer:
                _, shots = plan_shots(settings, "comic", 16, "model", "", backend, "", "oai", "",
                                     runtime=runtime)
                args = writer.call_args.kwargs
                self.assertEqual(args["backend"], backend)
                cues = json.loads(args["request_text"])["cues"]
                self.assertEqual([c["start"] for c in cues], [0, 0, 4, 9])
                self.assertEqual(cues[0]["text"], "Wait here")
                self.assertTrue(cues[1]["loop"])
                self.assertEqual(cues[-1]["speaker"], "Jo")
                self.assertEqual([s.first for s in shots], [0, 54, 216])
                self.assertEqual([s.dissolve for s in shots], [False, False, True])
                self.assertEqual(sum(s.frames for s in shots), 293)
                self.assertEqual(shots[1].references, ("Ana", "bell"))
                self.assertEqual(shots[1].beat, "Recognition")
                self.assertTrue(all("ink wash" in s.prompt for s in shots))
                self.assertTrue(all("yellow coat" in s.prompt for s in shots[:2]))
                self.assertNotIn("yellow coat", shots[-1].prompt)

    def test_invalid_plans_stop_before_render(self):
        row = {"start": 0, "transition": "cut", "beat": "Reveal", "prompt": "Open door", "references": []}
        for shots in ([], [row, row], [dict(row, start=1)], [row, dict(row, start=10)],
                      [dict(row, references=["missing"])], [dict(row, prompt="")],
                      [dict(row, transition="other")]):
            with self.subTest(shots=shots), self.assertRaises(H3Error):
                compile_plan({"style": "ink", "continuity": {}, "shots": shots}, 10, "", 16)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_render_preserves_timeline_and_soundtrack(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            images = []
            for color in ("red", "green", "blue"):
                image = folder / f"{color}.png"
                Image.new("RGB", (64, 48), color).save(image)
                images.append(image)
            audio = folder / "mix.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", ""))
                wav.writeframes(bytes(32000 * 3))
            # A one-frame reveal must show its own illustration, even when the writer asks for a dissolve.
            shots = [Shot(0, FPS, False, ""), Shot(FPS, 1, True, ""),
                     Shot(FPS + 1, 2 * FPS - 1, True, "")]
            video = render_video(audio, shots, images, folder)
            info = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                            "-of", "json", str(video)])
            visual, sound = json.loads(info)["streams"]
            self.assertEqual((visual["codec_name"], int(visual["nb_frames"])), ("h264", FPS * 3))
            self.assertEqual(sound["codec_name"], "aac")
            self.assertAlmostEqual(float(sound["duration"]), 3, delta=1 / FPS)
            pixels = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video),
                                              "-vf", "crop=2:2:640:360,scale=1:1", "-f", "rawvideo",
                                              "-pix_fmt", "rgb24", "-an", "-"])
            red, green, blue = pixels[FPS * 3:(FPS + 1) * 3]
            self.assertGreater(green, max(red, blue) + 60)
            red, green, blue = pixels[-3:]
            self.assertGreater(blue, max(red, green) + 60)
