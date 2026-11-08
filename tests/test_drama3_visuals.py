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

from h3_app.drama3.visual_plan import (compile_plan, expand_prompt, fill_tags, plan_shots, sheet_prompt,
                                      subjects)
from h3_app.drama3.visuals import FPS, Shot, render_video
from h3_app.errors import H3Error
from h3_ui.drama3.stage_video import latest_export
from h3_ui.drama3.visuals import illustrate_shots, match_characters


def drain(iterator):
    try:
        while True:
            next(iterator)
    except StopIteration as stop:
        return stop.value


PORTRAITS = {"mara": "Mara portrait", "jo": "Jo portrait", "ana": "Ana portrait"}


def png(folder, name):
    path = Path(folder) / f"{name}.png"
    Image.new("RGB", (4, 4)).save(path)
    return str(path)


class IllustratedMixTests(unittest.TestCase):
    def test_scenes_come_from_text_then_characters_are_matched_to_portraits(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ("p_mara", "p_jo", "d1", "f1", "d2", "d3", "f3")
            out = dict(zip(names, (png(tmp, n) for n in names)))
            controller = Mock()
            controller.generate.side_effect = [[([out[n]], "Complete")] for n in names]
            # Locations and props never become image inputs; only characters do.
            shots = [Shot(0, FPS, False, "Wide", "Arrival", references=("mara", "station", "jo")),
                     Shot(FPS, FPS, False, "Sky", references=("station",)),
                     Shot(2 * FPS, FPS, False, "Close", "Goodbye", references=("mara",))]
            captured = {"qwen__turbo_variant": "Off", "qwen__width": 1024, "qwen__height": 1024}
            images, records, portraits = drain(illustrate_shots(controller, "qwen", shots, captured, True,
                                                                None, PORTRAITS))
            calls = [call.args for call in controller.generate.call_args_list]
            self.assertEqual([(c[1], c[2]) for c in calls[:2]],
                             [("Text to image", "Mara portrait"), ("Text to image", "Jo portrait")])
            # Portraits are square, single views; scenes are widescreen.
            self.assertEqual((calls[0][4]["qwen__width"], calls[0][4]["qwen__height"]), (1024, 1024))
            self.assertEqual((captured["qwen__width"], captured["qwen__height"]), (1664, 928))
            self.assertEqual([c[1] for c in calls[2:]],
                             ["Text to image", "Image edit", "Text to image", "Text to image", "Image edit"])
            self.assertEqual(calls[2][2], "Wide")
            # The drawn scene is the first input, so the edit keeps its composition.
            self.assertEqual(calls[3][3], [out["d1"], out["p_mara"], out["p_jo"]])
            self.assertEqual(calls[6][3], [out["d3"], out["p_mara"]])
            self.assertEqual(images, [out["f1"], out["d2"], out["f3"]])
            self.assertEqual([r["characters"] for r in records], [["mara", "jo"], [], ["mara"]])
            self.assertEqual([r["draft"] for r in records], ["d1.png", None, "d3.png"])
            self.assertEqual(portraits, {"mara": "p_mara.png", "jo": "p_jo.png"})

    def test_match_prompt_keeps_the_scene_and_names_each_portrait(self):
        controller = Mock()
        for engine, tags in (("qwen", ("<image1>", "<image2>", "<image3>")),
                             ("flux3", ("<ref_image_0>", "<ref_image_1>", "<ref_image_2>")),
                             ("h3", ("<Picture 1>", "<Picture 2>", "<Picture 3>")),
                             ("fibo2", ("<image_1>", "<image_2>", "<image_3>"))):
            with self.subTest(engine=engine):
                shot = Shot(0, FPS, False, "plain", "A goodbye", ("mara", "station", "jo"),
                            "mara⟦mara⟧ (Mara, ochre coat) hugs jo⟦jo⟧ (Jo) at station⟦station⟧ (old station).")
                match_characters(controller, engine, "scene.png", {"mara": "m.png", "jo": "j.png"}, shot, {})
                _, mode, prompt, paths, _ = controller.generate.call_args.args
                self.assertEqual((mode, paths), ("Image edit", ["scene.png", "m.png", "j.png"]))
                self.assertIn(f"{tags[0]} is the finished scene. Keep its composition", prompt)
                self.assertIn(f"mara {tags[1]}: the same face, hairstyle and outfit.", prompt)
                self.assertIn(f"jo {tags[2]}: the same face", prompt)
                self.assertIn("Never add the references as extra figures", prompt)
                # Tags sit inline after each matched name; a location gets no tag.
                self.assertTrue(prompt.endswith(f"Scene: mara {tags[1]} (Mara, ochre coat) hugs jo {tags[2]} (Jo) "
                                                "at station (old station)."))

    def test_text_only_models_and_disabled_matching_draw_scenes_from_text(self):
        shot = Shot(0, FPS, False, "Street", references=("mara",))
        for engine, use in (("moss_image", True), ("qwen", False)):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as tmp:
                controller = Mock()
                controller.generate.return_value = [([png(tmp, "s")], "Complete")]
                _, records, portraits = drain(illustrate_shots(controller, engine, [shot], {}, use, None,
                                                               PORTRAITS))
                controller.generate.assert_called_once()
                self.assertEqual(controller.generate.call_args.args[1], "Text to image")
                self.assertEqual((portraits, records[0]["characters"]), ({}, []))

    def test_input_slots_keep_the_focal_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ("p_ana", "p_mara", "p_jo", "d", "f")
            out = dict(zip(names, (png(tmp, n) for n in names)))
            controller = Mock()
            controller.generate.side_effect = [[([out[n]], "Complete")] for n in names]
            shot = Shot(0, FPS, False, "Busy", references=("ana", "mara", "jo"))
            # Three inputs: the scene plus the two characters mentioned first.
            with patch("h3_ui.drama3.visuals.reference_limit", return_value=3):
                _, records, _ = drain(illustrate_shots(controller, "qwen", [shot], {}, True, None, PORTRAITS))
            self.assertEqual(controller.generate.call_args.args[3], [out["d"], out["p_ana"], out["p_mara"]])
            self.assertEqual(records[0]["characters"], ["ana", "mara"])

    def test_image_error_after_partial_output_stops_following_shots(self):
        controller = Mock()
        controller.generate.return_value = [([__file__], "Preview"), ([__file__], "Error: backend failed")]
        shot = Shot(0, FPS, False, "Street", references=("mara",))
        with self.assertRaisesRegex(H3Error, "Shot 1 of 2 failed.*backend failed"):
            list(illustrate_shots(controller, "qwen", [shot, shot], {}, False))
        controller.generate.assert_called_once()

    def test_retry_resumes_after_the_portraits_and_shots_a_failed_export_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ("p_mara", "d1", "f1", "d2", "f2")
            out = dict(zip(names, (png(tmp, n) for n in names)))
            shots = [Shot(0, FPS, False, "A", references=("mara",)),
                     Shot(FPS, FPS, False, "B", references=("mara",))]
            finished = {"portraits": {}, "shots": []}
            controller = Mock()
            controller.generate.side_effect = [[([out["p_mara"]], "Complete")], [([out["d1"]], "Complete")],
                                               [([out["f1"]], "Complete")], [(None, "Error: out of memory")]]
            with self.assertRaisesRegex(H3Error, "Shot 2 of 2"):
                list(illustrate_shots(controller, "qwen", shots, {}, True, finished, PORTRAITS))
            self.assertEqual(finished, {"portraits": {"mara": out["p_mara"]}, "shots": [out["f1"]]})
            controller = Mock()
            controller.generate.side_effect = [[([out["d2"]], "Complete")], [([out["f2"]], "Complete")]]
            images, _, _ = drain(illustrate_shots(controller, "qwen", shots, {}, True, finished, PORTRAITS))
            self.assertEqual(images, [out["f1"], out["f2"]])
            # The resumed shot is still matched to the saved portrait.
            self.assertEqual(controller.generate.call_args.args[3], [out["d2"], out["p_mara"]])

    def test_stage_video_carries_the_mix_id_for_its_subtitles(self):
        from h3_ui.drama3.stage_video import stage_video_html

        self.assertEqual(stage_video_html(None), "")
        name = f"Rain_{'e' * 32}_illustrated_12345678.mp4"
        page = stage_video_html(f"/data/output/drama3/{name}", url=lambda path: f"/file={path.name}")
        # The subtitle panel of the same mix has this id, so the overlay reads its lines.
        self.assertIn(f'data-media="{"e" * 32}"', page)
        self.assertIn(f'<video controls playsinline preload="metadata" src="/file={name}">', page)
        self.assertIn('<span data-caption></span>', page)
        self.assertIn(f'download="{name}"', page)

    def test_latest_export_is_the_newest_video_made_from_this_mix(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            mix = Path(tmp) / f"Rain [1]_{'c' * 32}.mp3"
            self.assertIsNone(latest_export(mix))
            old, new, other = (Path(tmp) / f"{mix.stem}_illustrated_aaaa1111.mp4",
                               Path(tmp) / f"{mix.stem}_illustrated_bbbb2222.mp4",
                               Path(tmp) / f"Other_{'d' * 32}_illustrated_cccc3333.mp4")
            for index, path in enumerate((old, new, other)):
                path.write_bytes(b"v")
                os.utime(path, (1000 + index, 1000 + index))
            self.assertEqual(latest_export(mix), new)

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
        portrait = sheet_prompt("characters", "mara", "Mara, cropped black hair", "ink", "")
        self.assertIn("Portrait of mara: Mara, cropped black hair", portrait)
        self.assertIn("Only this one view", portrait)
        self.assertNotIn("⟦", shot.prompt)
        self.assertIn("mara⟦mara⟧ (Mara, cropped black hair; smiling) in alley⟦alley⟧", shot.tagged)
        self.assertEqual(fill_tags("mara⟦mara⟧ and alley⟦alley⟧.", {"mara": "<image2>"}), "mara <image2> and alley.")
        nested, keys = expand_prompt("@mara(holding @radio) waits.", known)
        self.assertEqual(nested, "mara (Mara, cropped black hair; holding radio (bakelite radio)) waits.")
        self.assertEqual(keys, ("mara", "radio"))
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

    def test_late_planner_reply_never_replaces_a_newer_plan(self):
        from h3_ui.drama3.visuals import claim_plan, read_plan, save_plan

        with tempfile.TemporaryDirectory() as tmp:
            mix = Path(tmp) / f"mix_{'c' * 32}.mp3"
            slow = claim_plan(mix)
            pasted = claim_plan(mix)  # A pasted plan, or a second Plan click, while the first still runs.
            self.assertTrue(save_plan(mix, "ink", 16, {"shots": "pasted"}, pasted))
            self.assertFalse(save_plan(mix, "ink", 16, {"shots": "late"}, slow))
            self.assertEqual(read_plan(mix)["plan"], {"shots": "pasted"})
            # Another mix's runs are independent.
            other = Path(tmp) / f"mix_{'d' * 32}.mp3"
            self.assertTrue(save_plan(other, "ink", 16, {"shots": "own"}, claim_plan(other)))

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

    def test_copied_prompt_and_pasted_reply_follow_the_built_in_planner(self):
        from h3_app.drama3.visual_plan import check_plan_reply, full_plan_prompt

        settings = {"duration": 10, "title": "Rain", "chunks": [{"name": "Ana", "lines": [
            {"text": "Hi", "start": 0, "end": 2, "words": [{"text": "Hi", "start": 0.1, "end": 0.4}]}]}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        prompt = full_plan_prompt(settings, "ink wash", 2, runtime=runtime)
        system, request = prompt.split("\n\n----- USER REQUEST -----\n\n")
        self.assertTrue(system.startswith("You are the visual director"))
        request = json.loads(request)
        self.assertEqual((request["art_direction"], request["min_shots"], request["max_shots"]), ("ink wash", 2, 5))
        # Line times only: word timings would bloat the request without helping the planner.
        self.assertEqual(request["cues"], [{"start": 0, "end": 2, "speaker": "Ana", "text": "Hi"}])
        plan = {"art_style": "ink", "characters": {"ana": "Ana, red scarf"},
                "shots": [{"start": 0, "transition": "cut", "beat": "Hello", "prompt": "@ana waves"},
                          {"start": 4, "transition": "dissolve", "beat": "Rain", "prompt": "Rain falls"}]}
        # Chat models wrap JSON in a fence or a sentence; both are accepted.
        for reply in (json.dumps(plan), f"Here is the plan:\n```json\n{json.dumps(plan)}\n```",
                      f"Sure! {json.dumps(plan)} Enjoy."):
            with self.subTest(reply=reply[:20]):
                parsed, shots = check_plan_reply(reply, settings, "ink wash", 2)
                self.assertEqual(parsed, plan)
                self.assertEqual([s.references for s in shots], [("ana",), ()])
        for reply, error in (("not json", "isn't valid JSON"), (json.dumps({**plan, "shots": plan["shots"][:1]}),
                                                                  "fewer than the minimum of 2")):
            with self.subTest(error=error), self.assertRaisesRegex(H3Error, error):
                check_plan_reply(reply, settings, "ink wash", 2)

    def test_plan_below_the_minimum_asks_to_plan_again(self):
        settings = {"duration": 30, "chunks": [{"name": "Ana", "text": "Hi", "start": 0, "end": 2}]}
        plan = {"art_style": "ink", "shots": [{"start": 0, "transition": "cut", "beat": "Hi", "prompt": "Ana"}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        with patch("h3_app.drama3.visual_plan.write_prompt", return_value=(json.dumps(plan), "ok")):
            with self.assertRaisesRegex(H3Error, "has 1 shots, fewer than the minimum of 4"):
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
