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
from h3_app.provenance import write_snapshot
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
            write_snapshot(out["f1"], {"family": "Qwen Image 2.1", "job_id": "match-1",
                                       "settings": {"seed": 42, "effective_match_input_size": True}})
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
            self.assertIn("facial identity", calls[3][2])
            self.assertEqual(records[0]["matching"]["prompt"], calls[3][2])
            self.assertEqual(records[0]["matching"]["references"], ["d1.png", "p_mara.png", "p_jo.png"])
            self.assertEqual(records[0]["matching"]["job_id"], "match-1")
            self.assertEqual(records[0]["matching"]["effective_settings"]["seed"], 42)
            self.assertEqual(images, [out["f1"], out["d2"], out["f3"]])
            self.assertEqual([r["characters"] for r in records], [["mara", "jo"], [], ["mara"]])
            self.assertEqual([Path(r["draft"]).name if r["draft"] else None for r in records],
                             ["d1.png", None, "d3.png"])
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
                self.assertTrue(prompt.startswith(f"In {tags[0]}, change only"))
                self.assertIn(f"For Mara, ochre coat, use {tags[1]}", prompt)
                self.assertIn(f"For Jo, use {tags[2]}", prompt)
                self.assertIn("No added figures", prompt)
                self.assertIn("level of surface detail unchanged", prompt)
                # No unchanged scene narrative or style brief to trigger a full-frame repaint.
                self.assertNotIn("old station", prompt)
                self.assertNotIn("hugs", prompt)
                self.assertNotIn("Scene:", prompt)

    def test_qwen_identity_edit_preserves_source_size_without_changing_sampling(self):
        from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH
        from h3_app.generation.qwen import resolve_qwen_output_dimensions
        from h3_ui.drama3.visuals import character_match_request

        with tempfile.TemporaryDirectory() as tmp:
            scene = str(Path(tmp) / "scene.png")
            Image.new("RGB", (1664, 928)).save(scene)
            captured = {"qwen__width": 1024, "qwen__height": 1024,
                        "qwen__reference_resolution": 1024, "qwen__edit_size": QWEN_EDIT_SIZE_MANUAL,
                        "qwen__output_resolution_1k": "a saved preset", "qwen__prompt_enhancer": "3B",
                        "qwen__steps": 8, "qwen__cfg": 1.0, "qwen__cache_dtype": "default"}
            before = dict(captured)
            _, references, options = character_match_request("qwen", scene, {"mara": "m.png"},
                                                             Shot(0, FPS, False, "Full scene"), captured)
            self.assertEqual(captured, before)
            self.assertEqual(references, [scene, "m.png"])
            self.assertEqual(options["qwen__edit_size"], QWEN_EDIT_SIZE_MATCH)
            self.assertEqual(options["qwen__reference_resolution"], 0)
            self.assertEqual(options["qwen__prompt_enhancer"], "Off")
            self.assertIsNone(options["qwen__output_resolution_1k"])
            for name in ("steps", "cfg", "cache_dtype"):
                self.assertEqual(options[f"qwen__{name}"], before[f"qwen__{name}"])
            request = SimpleNamespace(width=1024, height=1024, max_resolution=False,
                                      match_input_size=options["qwen__edit_size"] == QWEN_EDIT_SIZE_MATCH)
            self.assertTrue(resolve_qwen_output_dimensions(request, (scene,), True)[2])

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
            self.assertEqual(finished["portraits"], {"mara": out["p_mara"]})
            self.assertEqual(finished["shots"], [out["f1"]])
            first_details = finished["shot_details"][0]
            self.assertEqual(first_details["draft"], out["d1"])
            controller = Mock()
            controller.generate.side_effect = [[([out["d2"]], "Complete")], [([out["f2"]], "Complete")]]
            images, records, _ = drain(illustrate_shots(controller, "qwen", shots, {}, True, finished, PORTRAITS))
            self.assertEqual(images, [out["f1"], out["f2"]])
            self.assertEqual(records[0]["draft"], out["d1"])
            self.assertEqual(records[0]["matching"], first_details["matching"])
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
        shot, = compile_plan(plan, 4, "watercolor", 8, art_style="ink wash, sepia")
        self.assertEqual(shot.references, ("mara_kid", "radio", "mara", "alley"))
        self.assertTrue(shot.prompt.startswith("Single widescreen 16:9 illustration of one moment"))
        self.assertIn("Art style: ink wash, sepia\nwatercolor", shot.prompt)
        # A plan carries no style of its own: the art direction and frame are added when drawing.
        tall, = compile_plan(plan, 4, "watercolor", 8, orientation="portrait")
        self.assertTrue(tall.prompt.startswith("Single vertical 9:16 illustration of one moment"))
        self.assertNotIn("Art style:", tall.prompt)
        self.assertEqual(tall.prompt.split("\n")[1], "watercolor")
        self.assertTrue(shot.prompt.index("watercolor") < shot.prompt.index("Low angle."))
        self.assertEqual(tall.prompt.split("\n")[-1], shot.prompt.split("\n")[-1])
        portrait = sheet_prompt("characters", "mara", "Mara, cropped black hair", "ink", "")
        self.assertIn("Portrait of mara: Mara, cropped black hair", portrait)
        self.assertIn("Only this one view", portrait)
        self.assertNotIn("⟦", shot.prompt)
        self.assertIn("mara⟦mara⟧ (Mara, cropped black hair; smiling) in alley⟦alley⟧", shot.tagged)
        self.assertEqual(fill_tags("mara⟦mara⟧ and alley⟦alley⟧.", {"mara": "<image2>"}), "mara <image2> and alley.")
        nested, keys = expand_prompt("@mara(holding @radio) waits.", known)
        self.assertEqual(nested, "mara (Mara, cropped black hair; holding radio (bakelite radio)) waits.")
        self.assertEqual(keys, ("mara", "radio"))

        from h3_ui.drama3.visuals import visible_face_references
        from dataclasses import replace
        tagged, _ = expand_prompt("@mara(holding @radio(from behind); smiling) waits.", known, slots=True)
        self.assertEqual(visible_face_references(replace(shot, tagged=tagged, references=("mara",))), ("mara",))
        tagged, _ = expand_prompt("@mara(holding @radio(close up) with back turned) waits.", known, slots=True)
        self.assertEqual(visible_face_references(replace(shot, tagged=tagged, references=("mara",))), ())
        self.assertNotIn("grey background", sheet_prompt("locations", "alley", "narrow", "ink", ""))

    def test_plan_is_saved_beside_the_mix_for_its_minimum_shots_in_any_style_and_frame(self):
        from h3_ui.drama3.visuals import plan_shots_for, read_plan, write_plan

        plan = {"style": "ink", "continuity": {}, "shots": [
            {"start": 0, "transition": "cut", "beat": "Open", "references": [], "prompt": "A door"},
            {"start": 1.5, "transition": "dissolve", "beat": "Rain", "references": [], "prompt": "Rain"}]}
        with tempfile.TemporaryDirectory() as tmp:
            mix = Path(tmp) / f"mix_{'b' * 32}.mp3"
            self.assertIsNone(read_plan(mix))
            write_plan(mix, "Comic panels ", 16.0, plan)
            self.assertEqual(read_plan(mix, 16)["plan"], plan)
            self.assertIsNone(read_plan(mix, 8))
            shots = plan_shots_for(read_plan(mix), 3.0, "Comic panels")
            self.assertEqual([(shot.first, shot.frames, shot.dissolve) for shot in shots],
                             [(0, round(1.5 * FPS), False), (round(1.5 * FPS), round(1.5 * FPS), True)])
            # An older plan's own style ("ink") goes with the art direction it was planned for only.
            self.assertIn("Art style: ink", shots[0].prompt)
            other = plan_shots_for(read_plan(mix), 3.0, "Oil paint", "portrait")
            self.assertNotIn("Art style: ink", other[0].prompt)
            self.assertTrue(other[0].prompt.startswith("Single vertical 9:16"))
            self.assertTrue(other[0].prompt.index("Oil paint") < other[0].prompt.index("A door"))
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

    def test_request_carries_the_voice_of_each_speaker_in_the_mix(self):
        from h3_app.drama3.visual_plan import plan_request

        settings = {"duration": 6, "chunks": [{"name": "Ana", "text": "Hi", "start": 0, "end": 2},
                                              {"name": "Old Tom", "text": "Hm", "start": 3, "end": 4}],
                    "voices": {"old tom": "An elderly man, gravelly and slow.", "Ana": " A girl of about ten. ",
                               "Nobody": "A woman in her forties.", "Speaker 2": ""}}
        request = json.loads(plan_request(settings, 3))
        self.assertEqual(request["speakers"], [{"name": "Ana", "voice": "A girl of about ten."},
                                               {"name": "Old Tom", "voice": "An elderly man, gravelly and slow."}])
        self.assertNotIn("speakers", json.loads(plan_request({**settings, "voices": {}}, 3)))

    def test_story_presets_ask_for_finished_settings(self):
        from h3_ui.drama3.visuals import SETTING_DETAIL, SLIDE_STYLES, STYLE_PRESETS

        for name, text in STYLE_PRESETS.items():
            self.assertEqual(text.endswith(SETTING_DETAIL), name not in SLIDE_STYLES)
            self.assertNotIn("sharper detail around faces", text)

    def test_portraits_leave_out_the_setting_but_keep_older_portraits_current(self):
        from h3_ui.drama3.visuals import (SETTING_DETAIL, STYLE_PRESETS, portrait_keys, portrait_prompts,
                                          prompt_key)

        style = STYLE_PRESETS["Watercolor"] + " Set in 1940s Shanghai."
        saved = {"style": style, "plan": {"art_style": "ink", "characters": {"mara": "Mara, cropped hair"}}}
        prompt = portrait_prompts(saved, style)["mara"]
        self.assertNotIn(SETTING_DETAIL.strip(), prompt)
        self.assertIn("Plain light grey background", prompt)
        self.assertIn("transparent layered washes", prompt)
        self.assertIn("Set in 1940s Shanghai.", prompt)
        self.assertNotIn("  ", prompt)
        old = portrait_prompts(saved, style, with_setting=True)["mara"]
        self.assertIn(SETTING_DETAIL.strip(), old)
        self.assertTrue({prompt_key(prompt), prompt_key(old)} <= portrait_keys(saved, style)["mara"])
        # Portraits follow the chosen style, without an older plan's style written for another one.
        anime = portrait_prompts(saved, STYLE_PRESETS["Anime"])["mara"]
        self.assertIn("cel shading", anime)
        self.assertNotIn("Art style: ink", anime)

    def test_portrait_keys_accept_the_original_body_first_prompts(self):
        from h3_ui.drama3.visuals import portrait_keys, prompt_key

        body = ("Portrait of mara: Mara, cropped hair. One single figure from the waist up, facing the viewer, "
                "centered, calm expression. Only this one view of this one person. "
                "Plain light grey background, even soft lighting. No text, labels or captions.")
        for art_style in ("ink", None):
            with self.subTest(art_style=art_style):
                plan = {"characters": {"mara": "Mara, cropped hair"}}
                if art_style:
                    plan["art_style"] = art_style
                saved = {"style": "comic", "plan": plan}
                legacy = f"{body}\nArt style: {art_style}\ncomic"
                self.assertIn(prompt_key(legacy), portrait_keys(saved, "comic")["mara"])

    def test_presets_are_grouped_by_use_with_a_preview_scene(self):
        from h3_ui.drama3.visuals import (SLIDE_STYLES, STYLE_GROUPS, STYLE_PRESETS, STYLE_PREVIEWS, STYLE_USES,
                                          preview_prompt)

        grouped = [name for group in STYLE_GROUPS for name in group["presets"]]
        self.assertEqual(grouped, list(STYLE_PRESETS))
        self.assertEqual(set(STYLE_USES), set(STYLE_PRESETS))
        self.assertEqual(set(STYLE_PREVIEWS), set(STYLE_PRESETS))
        self.assertEqual([group["presets"] for group in STYLE_GROUPS if group["name"] == "Explainers & slides"],
                         [list(SLIDE_STYLES)])
        self.assertIn("Chinese ink painting", STYLE_PRESETS)
        for name in STYLE_PRESETS:
            prompt = preview_prompt(name)
            self.assertTrue(prompt.startswith("Single widescreen 16:9 illustration of one moment."))
            self.assertTrue(prompt.index(STYLE_PRESETS[name]) < prompt.index(STYLE_PREVIEWS[name]))

    def test_earlier_preset_wording_still_counts_as_the_preset(self):
        from h3_ui.drama3.visuals import (SETTING_DETAIL, SLIDE_STYLES, STYLE_PRESETS, _PRESETS,
                                          preset_for, reference_direction)

        earlier = (" Paint the setting with the same finish and focus as the figures: legible architecture, "
                   "objects and landscape, never a blurred, smeared or sketchy backdrop.")
        self.assertEqual(preset_for(STYLE_PRESETS["Anime"].replace(SETTING_DETAIL, earlier)), "Anime")
        previous = earlier.replace(" Paint", " Render", 1)
        # Refining a preset must not turn saved projects into Custom, including projects saved
        # before either of the shared setting instructions changed. User additions stay Custom.
        for _group, name, _use, _preview, original in _PRESETS:
            suffixes = ("",) if name in SLIDE_STYLES else ("", earlier, previous, SETTING_DETAIL)
            for suffix in suffixes:
                with self.subTest(preset=name, setting=suffix):
                    self.assertEqual(preset_for(original + suffix), name)
            self.assertEqual(preset_for(original + " Use a muted autumn palette."), "Custom")
        self.assertEqual(preset_for(STYLE_PRESETS["Anime"].replace(SETTING_DETAIL, previous)), "Anime")
        self.assertNotIn(previous.strip(), reference_direction(STYLE_PRESETS["Anime"].replace(SETTING_DETAIL, previous)))
        manga = STYLE_PRESETS["Japanese comic"].replace(SETTING_DETAIL, earlier).replace(
            "gestures. One", "gestures. Full color throughout. One")
        self.assertEqual(preset_for(manga), "Japanese comic")
        self.assertNotIn("setting with the same finish", reference_direction(manga))

    def test_style_previews_are_kept_per_model_until_their_prompt_changes(self):
        from h3_ui.drama3 import visuals

        with tempfile.TemporaryDirectory() as tmp:
            controller = Mock(fields={"seed": None})
            controller.generate.side_effect = lambda *args: iter([([png(tmp, "drawn")], "Complete")])
            deps = SimpleNamespace(runtime_config=lambda: SimpleNamespace(output_dir=Path(tmp)))
            self.assertEqual(visuals.chosen_previews(tmp, "qwen", "missing"), list(visuals.STYLE_PRESETS))
            updates = list(visuals.draw_previews(controller, deps, "qwen", ["Anime", "Pixel art"], 7))
            self.assertEqual(updates[-1][1], "Drew 2 style previews.")
            self.assertEqual(controller.generate.call_args.args[2], visuals.preview_prompt("Pixel art"))
            self.assertEqual(controller.generate.call_args.args[1], "Text to image")
            kept = visuals.preview_images(tmp, "qwen")
            self.assertEqual(set(kept), {"Anime", "Pixel art"})
            self.assertEqual(visuals.preview_images(tmp, "flux3"), {})
            self.assertNotIn("Anime", visuals.chosen_previews(tmp, "qwen", "missing"))
            # Drawing again replaces the file; a preview from an older prompt no longer shows.
            old = kept["Anime"]
            list(visuals.draw_previews(controller, deps, "qwen", ["Anime"], 7))
            self.assertFalse((visuals.previews_dir(tmp, "qwen") / old).exists())
            with patch.dict(visuals.STYLE_PREVIEWS, {"Anime": "Another scene."}):
                self.assertNotIn("Anime", visuals.preview_images(tmp, "qwen"))
            with self.assertRaises(H3Error):
                visuals.chosen_previews(tmp, "qwen", ["Nope"])
            with self.assertRaises(H3Error):
                visuals.previews_dir(tmp, "../x")

    def test_portrait_frame_draws_tall_and_renders_a_tall_video(self):
        from h3_ui.drama3 import visuals

        captured = {}
        visuals.widescreen("qwen", captured, "portrait")
        self.assertEqual((captured["qwen__width"], captured["qwen__height"]), (928, 1664))
        visuals.widescreen("flux3", captured)
        self.assertEqual((captured["flux3__width"], captured["flux3__height"]), (1344, 768))
        self.assertEqual(visuals.orientation_of("square"), "landscape")
        with tempfile.TemporaryDirectory() as tmp:
            mix = Path(tmp) / f"mix_{'e' * 32}.mp3"
            mix.write_bytes(b"ID3")
            visuals.write_snapshot(mix, {"family": "drama3", "settings": {
                "duration": 2.0, "chunks": [{"name": "Ana", "text": "Hi", "start": 0, "end": 1}]}})
            visuals.keep_stills(mix, tmp, [Shot(0, 48, False, "p", "Hi")], [png(tmp, "tall")], {}, orientation="portrait")
            self.assertEqual(visuals.read_stills(mix, tmp)["orientation"], "portrait")
            rendered = {}

            def fake_render(audio, shots, images, output_dir, subtitles=None, size=None):
                rendered.update(size=size, subtitles=subtitles)
                return Path(output_dir) / "video.mp4"

            with patch.object(visuals, "render_video", fake_render), patch.object(visuals, "which", lambda name: name), \
                    patch.object(visuals, "write_snapshot"):
                visuals.render_stills(mix, tmp, subtitles=100)
            self.assertEqual(rendered["size"], (720, 1280))
            self.assertIn("PlayResX: 720\nPlayResY: 1280", rendered["subtitles"])

    def test_planner_writes_style_and_frame_neutral_shots(self):
        text = Path("prompts/prompt_drama3_visuals.txt").read_text(encoding="utf-8")
        self.assertIn("STYLE AND FRAME STAY OUT", text)
        self.assertIn("reads in both a wide and a tall frame", text)
        self.assertIn("rather than a story", text)  # explainers keep simple settings
        self.assertNotIn("art_style", text)
        self.assertNotIn("art_direction", text)

    def test_cut_off_writer_reply_explains_itself(self):
        settings = {"duration": 5, "chunks": [{"name": "Ana", "text": "Hi", "start": 0, "end": 2}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        with patch("h3_app.drama3.visual_plan.write_prompt", return_value=('{"style": "ink", "shots": [', "ok")):
            with self.assertRaisesRegex(H3Error, "cut off"):
                plan_shots(settings, 16, "model", "", "Gemini", "", "", "", runtime=runtime)

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
                _, shots = plan_shots(settings, 3, "model", "", backend, "", "oai", "",
                                     runtime=runtime)
                args = writer.call_args.kwargs
                self.assertEqual(args["backend"], backend)
                request = json.loads(args["request_text"])
                # The setting is a floor; the cap comes from the mix length (one shot per 2 seconds).
                self.assertEqual((request["min_shots"], request["max_shots"]), (3, 7))
                cues = request["cues"]
                self.assertEqual(request["timing"], "anchored")
                self.assertEqual([c["start"] for c in cues], [0, 9])
                self.assertEqual(cues[0]["text"], "Wait here")
                self.assertTrue(request["sounds"][0]["loop"])
                self.assertEqual(cues[-1]["speaker"], "Jo")
                self.assertEqual([s.first for s in shots], [0, 54, 216])
                self.assertEqual([s.dissolve for s in shots], [False, False, True])
                self.assertEqual(sum(s.frames for s in shots), 293)
                self.assertEqual(shots[1].references, ("Ana", "bell"))
                self.assertEqual(shots[1].beat, "Recognition")
                # Checked without a style: that is added when drawing.
                self.assertTrue(all(s.prompt.startswith("Single widescreen 16:9") and "ink wash" not in s.prompt for s in shots))
                self.assertTrue(all("yellow coat" in s.prompt for s in shots[:2]))
                self.assertNotIn("yellow coat", shots[-1].prompt)

    def test_style_presets_fill_art_direction_and_edits_become_custom(self):
        from h3_ui.drama3.visuals import CUSTOM_STYLE, DEFAULT_STYLE, STYLE_PRESETS, preset_for

        self.assertIn(DEFAULT_STYLE, STYLE_PRESETS)
        self.assertNotIn(CUSTOM_STYLE, STYLE_PRESETS)
        for name, text in STYLE_PRESETS.items():
            self.assertEqual(preset_for(f" {text} "), name)
        self.assertEqual(preset_for(STYLE_PRESETS[DEFAULT_STYLE] + " Set in 1940s Shanghai."), CUSTOM_STYLE)
        self.assertEqual(preset_for(""), CUSTOM_STYLE)

    def test_preset_texts_saved_before_setting_detail_still_match(self):
        from h3_ui.drama3.visuals import SETTING_DETAIL, STYLE_PRESETS, preset_for

        for name, text in STYLE_PRESETS.items():
            self.assertEqual(preset_for(text.removesuffix(SETTING_DETAIL)), name)
        old = STYLE_PRESETS["Digital painting"].removesuffix(SETTING_DETAIL).replace(
            "clear detail on faces and hands", "sharper detail around faces and hands")
        self.assertIn("sharper detail around faces and hands", old)
        self.assertEqual(preset_for(old), "Digital painting")

    def test_copied_prompt_and_pasted_reply_follow_the_built_in_planner(self):
        from h3_app.drama3.visual_plan import check_plan_reply, full_plan_prompt

        settings = {"duration": 10, "title": "Rain", "chunks": [{"name": "Ana", "lines": [
            {"text": "Hi", "start": 0, "end": 2, "words": [{"text": "Hi", "start": 0.1, "end": 0.4}]}]}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        prompt = full_plan_prompt(settings, 2, runtime=runtime)
        system, request = prompt.split("\n\n----- USER REQUEST -----\n\n")
        self.assertTrue(system.startswith("You are the visual director"))
        request = json.loads(request)
        self.assertEqual((request["min_shots"], request["max_shots"]), (2, 5))
        self.assertNotIn("art_direction", request)
        # Line times only: word timings would bloat the request without helping the planner.
        self.assertEqual(request["cues"], [{"start": 0, "end": 2, "speaker": "Ana", "text": "Hi"}])
        plan = {"art_style": "ink", "characters": {"ana": "Ana, red scarf"},
                "shots": [{"start": 0, "transition": "cut", "beat": "Hello", "prompt": "@ana waves"},
                          {"start": 4, "transition": "dissolve", "beat": "Rain", "prompt": "Rain falls"}]}
        # Chat models wrap JSON in a fence or a sentence; both are accepted.
        for reply in (json.dumps(plan), f"Here is the plan:\n```json\n{json.dumps(plan)}\n```",
                      f"Sure! {json.dumps(plan)} Enjoy."):
            with self.subTest(reply=reply[:20]):
                parsed, shots = check_plan_reply(reply, settings, 2)
                self.assertEqual(parsed["characters"], plan["characters"])
                self.assertEqual([row["prompt"] for row in parsed["shots"]],
                                 [row["prompt"] for row in plan["shots"]])
                self.assertTrue(all("cue" in row and "start" not in row and row["id"]
                                    for row in parsed["shots"]))
                self.assertEqual(parsed["shots"][1]["phase"], "after")
                self.assertEqual([shot.first for shot in shots], [0, 96])
                self.assertEqual([s.references for s in shots], [("ana",), ()])
        for reply, error in (("not json", "isn't valid JSON"), (json.dumps({**plan, "shots": plan["shots"][:1]}),
                                                                  "fewer than the minimum of 2")):
            with self.subTest(error=error), self.assertRaisesRegex(H3Error, error):
                check_plan_reply(reply, settings, 2)

    def test_plan_below_the_minimum_asks_to_plan_again(self):
        settings = {"duration": 30, "chunks": [{"name": "Ana", "text": "Hi", "start": 0, "end": 2}]}
        plan = {"art_style": "ink", "shots": [{"start": 0, "transition": "cut", "beat": "Hi", "prompt": "Ana"}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        with patch("h3_app.drama3.visual_plan.write_prompt", return_value=(json.dumps(plan), "ok")):
            with self.assertRaisesRegex(H3Error, "has 1 shots, fewer than the minimum of 4"):
                plan_shots(settings, 4, "model", "", "Gemini", "", "", "", runtime=runtime)
            _, shots = plan_shots(settings, 1, "model", "", "Gemini", "", "", "", runtime=runtime)
        self.assertEqual(len(shots), 1)

    def test_invalid_plans_stop_before_render(self):
        row = {"start": 0, "transition": "cut", "beat": "Reveal", "prompt": "Open door", "references": []}
        for shots in ([], [row, row], [dict(row, start=1)], [row, dict(row, start=10)],
                      [dict(row, prompt="")],
                      [dict(row, transition="other")]):
            with self.subTest(shots=shots), self.assertRaises(H3Error):
                compile_plan({"style": "ink", "continuity": {}, "shots": shots}, 10, "", 16)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_saved_video_is_rendered_from_the_kept_stills(self):
        """Drawing keeps JPEG stills out of Media; saving renders them beside the mix and records the video."""
        from h3_app.provenance import read_snapshot
        from h3_ui.drama3 import visuals

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            mixes = folder / "h3" / "drama3"
            mixes.mkdir(parents=True)
            audio = mixes / "night.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", ""))
                wav.writeframes(bytes(32000 * 2))
            images = []
            for color in ("red", "blue"):
                image = folder / f"{color}.png"
                Image.new("RGB", (64, 36), color).save(image)
                images.append(image)
            root = folder / "h3"
            with self.assertRaisesRegex(H3Error, "Draw the images first"):
                visuals.render_stills(audio, root)
            shots = [Shot(0, FPS, False, "red"), Shot(FPS, FPS, True, "blue")]
            visuals.keep_stills(audio, root, shots, images, {"model": "qwen"})
            stills = visuals.stills_dir(audio, root)
            self.assertIn(".soundstage", stills.parts)  # Media leaves .soundstage alone.
            self.assertEqual(sorted(path.suffix for path in stills.iterdir()), [".jpg", ".jpg", ".json"])
            video = visuals.render_stills(audio, root)
            self.assertEqual(video.parent, mixes)
            self.assertEqual(latest_export(audio), video)
            settings = read_snapshot(video)["settings"]
            self.assertEqual((settings["source"], settings["model"]), ("night.wav", "qwen"))
            self.assertEqual(settings["stills"], visuals.read_stills(audio, root)["revision"])
            info = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)])
            self.assertEqual(int(json.loads(info)["streams"][0]["nb_frames"]), 2 * FPS)

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

    def test_prompts_and_presets_encode_depth_aware_shot_design(self):
        from h3_app.drama3 import effect_enhancement
        from h3_app.drama3.visual_plan import full_plan_prompt

        visuals_prompt = Path("prompts/prompt_drama3_visuals.txt").read_text(encoding="utf-8")
        self.assertIn("three readable depth planes", visuals_prompt)
        self.assertIn("on any shot using", visuals_prompt)
        self.assertIn("rack_focus", visuals_prompt)
        self.assertIn("visible warm practical light source", visuals_prompt)
        self.assertIn("bright upper sky opening", visuals_prompt)
        self.assertIn("keeping the air clear", visuals_prompt)
        self.assertNotIn("Do not routinely\nrequest deep focus", visuals_prompt)
        self.assertNotIn("art_style", visuals_prompt)
        self.assertNotIn("art_direction", visuals_prompt)

        effects_prompt = Path("prompts/prompt_soundstage_effects.txt").read_text(encoding="utf-8")
        for token in ("depth_summary", "depth_m", "depth_band", '"behind"', '"inside"', '"outside"', "max_layers_per_shot"):
            self.assertIn(token, effects_prompt)

        settings = {"duration": 8, "title": "Night", "chunks": [{"name": "Ana", "text": "Look.", "start": 0, "end": 2}]}
        runtime = SimpleNamespace(prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        plan_req = json.loads(full_plan_prompt(settings, 1, runtime=runtime).split("\n\n----- USER REQUEST -----\n\n")[1])
        self.assertIn("depth parallax", plan_req["effect_presets"]["push_in"]["description"])
        self.assertIn("Optical focus pull", plan_req["effect_presets"]["rack_focus"]["description"])
        self.assertIn("metric depth", effect_enhancement.MODES["behind"])

        effects_req = json.loads(effect_enhancement.request_text([], "rev-1"))
        self.assertIn("metric depth", effects_req["application_modes"]["behind"])
        self.assertIn("Depth-occluded", effects_req["presets"]["sunbeams"]["description"])
