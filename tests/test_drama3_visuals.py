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

from h3_app.drama3.visual_plan import compile_plan, expand_prompt, plan_shots, sheet_prompt, subjects
from h3_app.drama3.visuals import FPS, Shot, render_video
from h3_app.errors import H3Error
from h3_ui.drama3.visuals import generate_shot, illustrate_shots


def drain(iterator):
    try:
        while True:
            next(iterator)
    except StopIteration as stop:
        return stop.value


SHEETS = {"mara": ("characters", "Mara sheet"), "tower": ("locations", "Tower sheet"),
          "radio": ("props", "Radio sheet")}


def png(folder, name):
    path = Path(folder) / f"{name}.png"
    Image.new("RGB", (4, 4)).save(path)
    return str(path)


class IllustratedMixTests(unittest.TestCase):
    def test_reference_sheets_come_first_and_scenes_edit_from_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = [png(tmp, name) for name in ("mara", "tower", "s1", "s2", "s3")]
            controller = Mock()
            controller.generate.side_effect = [[([path], "Complete")] for path in out]
            shots = [Shot(0, FPS, False, "Wide", references=("mara", "tower")),
                     Shot(FPS, FPS, False, "Sky", references=()),
                     Shot(2 * FPS, FPS, False, "Close", references=("mara",))]
            captured = {"qwen__turbo_variant": "Off", "qwen__width": 1024, "qwen__height": 1024}
            images, records, sheets = drain(illustrate_shots(controller, "qwen", shots, captured, True,
                                                             None, SHEETS))
            calls = [call.args for call in controller.generate.call_args_list]
            # The unused radio gets no sheet; sheets are drawn from text before any scene.
            self.assertEqual([(c[1], c[2]) for c in calls[:2]],
                             [("Text to image", "Mara sheet"), ("Text to image", "Tower sheet")])
            self.assertEqual([c[1] for c in calls[2:]], ["Image edit", "Text to image", "Image edit"])
            self.assertEqual(calls[2][3], [out[0], out[1]])
            self.assertEqual(calls[4][3], [out[0]])
            self.assertIn("<image1> shows mara: keep the same face", calls[2][2])
            self.assertIn("<image2> shows tower: keep the same architecture", calls[2][2])
            self.assertEqual(images, out[2:])
            self.assertEqual([r["references"] for r in records], [["mara", "tower"], [], ["mara"]])
            self.assertEqual(sheets, {"mara": "mara.png", "tower": "tower.png"})
            # Every image is widescreen, edits included.
            self.assertEqual((captured["qwen__width"], captured["qwen__height"]), (1664, 928))
            self.assertEqual(captured["qwen__edit_size"], "Use width and height above")

    def test_text_only_models_and_disabled_references_skip_sheets(self):
        shot = Shot(0, FPS, False, "Street", references=("mara",))
        for engine, use in (("moss_image", True), ("qwen", False)):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as tmp:
                controller = Mock()
                controller.generate.return_value = [([png(tmp, "s")], "Complete")]
                _, _, sheets = drain(illustrate_shots(controller, engine, [shot], {}, use, None, SHEETS))
                controller.generate.assert_called_once()
                self.assertEqual(controller.generate.call_args.args[1], "Text to image")
                self.assertEqual(sheets, {})

    def test_reference_limit_keeps_the_focal_subjects(self):
        controller = Mock()
        sources = {"mara": ("characters", "m.png"), "radio": ("props", "r.png")}
        for engine, tags in (("qwen", ("<image1>", "<image2>")), ("flux3", ("<ref_image_0>", "<ref_image_1>")),
                             ("h3", ("<Picture 1>", "<Picture 2>")), ("fibo2", ("<image_1>", "<image_2>"))):
            with self.subTest(engine=engine):
                generate_shot(controller, engine, "Close-up in the rain.", sources, {})
                _, mode, prompt, paths, _ = controller.generate.call_args.args
                self.assertEqual((mode, paths), ("Image edit", ["m.png", "r.png"]))
                self.assertIn(f"{tags[0]} shows mara", prompt)
                self.assertIn(f"{tags[1]} shows radio: keep the same shape", prompt)
                self.assertTrue(prompt.endswith("Close-up in the rain."))
        with tempfile.TemporaryDirectory() as tmp:
            out = [png(tmp, n) for n in ("mara", "tower", "radio", "scene")]
            controller = Mock()
            controller.generate.side_effect = [[([path], "Complete")] for path in out]
            shot = Shot(0, FPS, False, "Busy", references=("radio", "mara", "tower"))
            # With fewer input slots than subjects, the first-mentioned subjects win.
            with patch("h3_ui.drama3.visuals.reference_limit", return_value=2):
                drain(illustrate_shots(controller, "qwen", [shot], {}, True, None, SHEETS))
            self.assertEqual(controller.generate.call_args.args[3], [out[0], out[1]])

    def test_image_error_after_partial_output_stops_following_shots(self):
        controller = Mock()
        controller.generate.return_value = [([__file__], "Preview"), ([__file__], "Error: backend failed")]
        shot = Shot(0, FPS, False, "Street", references=("mara",))
        with self.assertRaisesRegex(H3Error, "Shot 1 of 2 failed.*backend failed"):
            list(illustrate_shots(controller, "qwen", [shot, shot], {}, False))
        controller.generate.assert_called_once()

    def test_retry_resumes_after_the_sheets_and_shots_a_failed_export_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = [png(tmp, n) for n in ("mara", "s1", "s2")]
            shots = [Shot(0, FPS, False, "A", references=("mara",)),
                     Shot(FPS, FPS, False, "B", references=("mara",))]
            finished = {"sheets": {}, "shots": []}
            controller = Mock()
            controller.generate.side_effect = [[([out[0]], "Complete")], [([out[1]], "Complete")],
                                               [(None, "Error: out of memory")]]
            with self.assertRaisesRegex(H3Error, "Shot 2 of 2"):
                list(illustrate_shots(controller, "qwen", shots, {}, True, finished, SHEETS))
            self.assertEqual(finished, {"sheets": {"mara": out[0]}, "shots": [out[1]]})
            controller = Mock()
            controller.generate.side_effect = [[([out[2]], "Complete")]]
            images, _, _ = drain(illustrate_shots(controller, "qwen", shots, {}, True, finished, SHEETS))
            self.assertEqual(images, out[1:])
            controller.generate.assert_called_once()
            # The resumed shot still follows the saved sheet.
            self.assertEqual(controller.generate.call_args.args[1:4:2], ("Image edit", [out[0]]))

    def test_mentions_inline_anchors_and_become_references(self):
        plan = {"art_style": "ink wash, sepia", "characters": {"mara": "Mara, cropped black hair",
                                                              "mara_kid": "young Mara, pigtails"},
                "props": {"radio": "bakelite radio"}, "locations": {"alley": "narrow rain alley"},
                "shots": [{"start": 0, "transition": "cut", "beat": "Return",
                           "prompt": "Low angle. @mara_kid(soaked) hands @radio to @mara（smiling） "
                                     "in @alley, @nobody watches."}]}
        known = subjects(plan)
        self.assertEqual(known["alley"], ("locations", "narrow rain alley"))
        text, mentioned = expand_prompt(plan["shots"][0]["prompt"], known)
        self.assertEqual(mentioned, ("mara_kid", "radio", "mara", "alley"))
        self.assertIn("mara_kid (young Mara, pigtails; soaked) hands radio (bakelite radio) "
                      "to mara (Mara, cropped black hair; smiling) in alley (narrow rain alley)", text)
        self.assertIn("nobody watches", text)
        shot, = compile_plan(plan, 4, "watercolor", 8)
        self.assertEqual(shot.references, ("mara_kid", "radio", "mara", "alley"))
        self.assertTrue(shot.prompt.startswith("Single widescreen 16:9 illustration of one moment."))
        self.assertIn("Art style: ink wash, sepia\nwatercolor", shot.prompt)
        self.assertIn("Character reference sheet of mara: Mara, cropped black hair",
                      sheet_prompt("characters", "mara", "Mara, cropped black hair", "ink", ""))
        self.assertNotIn("grey background", sheet_prompt("locations", "alley", "narrow", "ink", ""))

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
                _, shots = plan_shots(settings, "comic", 3, "model", "", backend, "", "oai", "",
                                     runtime=runtime)
                args = writer.call_args.kwargs
                self.assertEqual(args["backend"], backend)
                request = json.loads(args["request_text"])
                # The setting is a floor; the cap comes from the mix length (one shot per 2 seconds).
                self.assertEqual((request["min_shots"], request["max_shots"]), (3, 7))
                cues = request["cues"]
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

    def test_style_presets_fill_art_direction_and_edits_become_custom(self):
        from h3_ui.drama3.visuals import CUSTOM_STYLE, DEFAULT_STYLE, STYLE_PRESETS, preset_for

        self.assertIn(DEFAULT_STYLE, STYLE_PRESETS)
        self.assertNotIn(CUSTOM_STYLE, STYLE_PRESETS)
        for name, text in STYLE_PRESETS.items():
            self.assertEqual(preset_for(f" {text} "), name)
        self.assertEqual(preset_for(STYLE_PRESETS["Film noir"] + " Set in 1940s Shanghai."), CUSTOM_STYLE)
        self.assertEqual(preset_for(""), CUSTOM_STYLE)

    def test_plan_below_the_minimum_asks_to_plan_again(self):
        settings = {"duration": 30, "chunks": [{"name": "Ana", "text": "Hi", "start": 0, "end": 2}]}
        plan = {"art_style": "ink", "shots": [{"start": 0, "transition": "cut", "beat": "Hi", "prompt": "Ana"}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        with patch("h3_app.drama3.visual_plan.write_prompt", return_value=(json.dumps(plan), "ok")):
            with self.assertRaisesRegex(H3Error, "planned 1 shots, fewer than the minimum of 4"):
                plan_shots(settings, "ink", 4, "model", "", "Gemini", "", "", "", runtime=runtime)
            _, shots = plan_shots(settings, "ink", 1, "model", "", "Gemini", "", "", "", runtime=runtime)
        self.assertEqual(len(shots), 1)

    def test_invalid_plans_stop_before_render(self):
        row = {"start": 0, "transition": "cut", "beat": "Reveal", "prompt": "Open door", "references": []}
        for shots in ([], [row, row], [dict(row, start=1)], [row, dict(row, start=10)],
                      [dict(row, prompt="")],
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
