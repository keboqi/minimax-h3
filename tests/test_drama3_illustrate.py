"""The Illustrate step's plan edits, per-shot drawing and the looks-against-voice check."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from h3_app.drama3 import voice_looks
from h3_app.drama3.results import DRAMA3_FAMILY
from h3_app.drama3.visual_plan import edit_shot, edit_subject, mentions
from h3_app.drama3.visuals import FPS, Shot
from h3_app.errors import H3Error
from h3_app.provenance import write_snapshot
from h3_ui.drama3 import visuals

PLAN = {"art_style": "ink wash", "characters": {"mia": "Mia, girl of ten, yellow coat"},
        "props": {"radio": "bakelite radio"}, "locations": {},
        "shots": [{"start": 0, "transition": "cut", "beat": "Stall", "prompt": "@mia(smiling) at her stall."},
                  {"start": 2, "transition": "dissolve", "beat": "Radio", "prompt": "@mia lifts @radio."},
                  {"start": 3, "transition": "cut", "beat": "Rain", "prompt": "Rain on the awning."}]}


def drain(iterator):
    try:
        while True:
            next(iterator)
    except StopIteration as stop:
        return stop.value


class PlanEditTests(unittest.TestCase):
    def test_edit_shot_and_subject_return_new_plans(self):
        edited = edit_shot(PLAN, 1, prompt="  @mia  drops @radio. ", transition="cut")
        self.assertEqual((edited["shots"][1]["prompt"], edited["shots"][1]["transition"]), ("@mia drops @radio.", "cut"))
        self.assertEqual(PLAN["shots"][1]["prompt"], "@mia lifts @radio.")  # the original is untouched
        with self.assertRaisesRegex(H3Error, "not in the current"):
            edit_shot(PLAN, 3, prompt="x")
        with self.assertRaisesRegex(H3Error, "cut or a dissolve"):
            edit_shot(PLAN, 0, transition="wipe")
        looks = edit_subject(PLAN, "characters", "mia", "Mia, woman in her forties")
        self.assertEqual(looks["characters"]["mia"], "Mia, woman in her forties")
        legacy = edit_subject({"continuity": {"jo": "red hat"}, "shots": []}, "characters", "jo", "blue hat")
        self.assertEqual(legacy["continuity"], {"jo": "blue hat"})
        with self.assertRaisesRegex(H3Error, "not in the current"):
            edit_subject(PLAN, "props", "lamp", "brass")

    def test_mentions_lists_each_shots_subjects(self):
        self.assertEqual(mentions(PLAN), [("mia",), ("mia", "radio"), ()])


class VoiceLooksTests(unittest.TestCase):
    def test_conflicts_flag_only_clear_disagreements(self):
        self.assertEqual(voice_looks.conflicts("Tom, boy of about sixteen", "An elderly man, gravelly"),
                         ["Looks say about 16; the voice sounds 60+."])
        self.assertEqual(voice_looks.conflicts("Mara, a woman in her thirties", "A man in his thirties"),
                         ["Looks say female; the voice is male."])
        # Neighbouring ages agree, and a description that states nothing is never flagged.
        self.assertEqual(voice_looks.conflicts("woman in her thirties", "a woman in her forties"), [])
        self.assertEqual(voice_looks.conflicts("tall, red scarf", "warm and calm"), [])
        self.assertEqual(voice_looks.conflicts("老王，满头白发的老人", "年轻的男声"),
                         ["Looks say about 60+; the voice sounds 18–35."])
        self.assertEqual(voice_looks.age("16岁的少年"), (16, 16))
        self.assertEqual(voice_looks.gender("其他人都走了"), None)

    def test_aged_and_number_words_need_an_age_beside_them(self):
        self.assertEqual(voice_looks.age("Mara, a woman aged about 30, red coat"), (30, 30))
        self.assertEqual(voice_looks.age("Mara, young woman, aged brass locket"), (18, 30))
        self.assertEqual(voice_looks.age("Li, a man in a ten-gallon hat"), None)
        self.assertEqual(voice_looks.age("a girl, twelve years old"), (12, 12))
        self.assertEqual(voice_looks.age("a middle-aged man"), (40, 60))
        self.assertEqual(voice_looks.conflicts("Mara, a woman aged about 30", "a woman in her thirties"), [])


class StillsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.audio = self.folder / "mix.mp3"
        self.audio.write_bytes(b"mp3")
        self.root = self.folder / "out"

    def png(self, name, color="red"):
        path = self.folder / f"{name}.png"
        Image.new("RGB", (32, 18), color).save(path)
        return path

    def test_partial_stills_hold_until_the_next_and_cover_the_mix(self):
        record = {"total_frames": 5 * FPS, "shots": [
            {"first_frame": FPS, "frames": FPS, "dissolve": True, "image": "a.jpg"},
            {"first_frame": 3 * FPS, "frames": FPS, "dissolve": True, "image": "b.jpg"}]}
        shots, names = visuals.stills_shots(record)
        self.assertEqual([(shot.first, shot.frames, shot.dissolve) for shot in shots],
                         [(0, 3 * FPS, False), (3 * FPS, 2 * FPS, True)])
        self.assertEqual(names, ["a.jpg", "b.jpg"])

    def test_rebinding_partial_stills_preserves_images_portraits_and_changed_prompts(self):
        old = [Shot(0, FPS, False, "one", id="first"), Shot(FPS, 2 * FPS, True, "two", id="second")]
        image = self.folder / "pattern.png"
        Image.frombytes("RGB", (32, 18), bytes((i * 17) % 256 for i in range(32 * 18 * 3))).save(image)
        visuals.keep_stills(self.audio, self.root, [old[1]], [image], {}, total_frames=3 * FPS,
                            portraits={"mia": (self.png("portrait"), "original")})
        record = visuals.read_stills(self.audio, self.root)
        image_bytes = (visuals.stills_dir(self.audio, self.root) / record["shots"][0]["image"]).read_bytes()
        newer = self.folder / "updated.mp3"
        newer.write_bytes(b"mp3")
        shots = [Shot(0, 2 * FPS, False, "one", id="first"),
                 Shot(2 * FPS, 3 * FPS, True, "updated prompt", id="second")]
        visuals.rebind_stills(newer, self.root, shots, previous_audio=self.audio, previous_shots=old)
        moved = visuals.read_stills(newer, self.root)
        self.assertEqual(len(moved["shots"]), 1)
        self.assertEqual((moved["shots"][0]["first_frame"], moved["shots"][0]["frames"]), (2 * FPS, 3 * FPS))
        self.assertEqual(moved["total_frames"], 5 * FPS)
        self.assertEqual((visuals.stills_dir(newer, self.root) / moved["shots"][0]["image"]).read_bytes(), image_bytes)
        self.assertEqual(moved["portraits"], record["portraits"])
        self.assertEqual([visuals.shot_status(shot, item) for shot, item in zip(shots, visuals.match_stills(shots, moved))],
                         ["todo", "changed"])

    def test_rebinding_does_not_assign_another_shots_image_at_the_same_time(self):
        old = [Shot(0, FPS, False, "same", id="first"), Shot(FPS, FPS, False, "same", id="second")]
        visuals.keep_stills(self.audio, self.root, [old[1]], [self.png("second")], {}, total_frames=2 * FPS)
        newer = self.folder / "new.mp3"
        newer.write_bytes(b"mp3")
        shots = [Shot(0, FPS // 2, False, "same", id="first"),
                 Shot(FPS // 2, FPS // 2, False, "same", id="second"),
                 Shot(FPS, FPS, False, "same", id="new-third")]
        visuals.rebind_stills(newer, self.root, shots, previous_audio=self.audio, previous_shots=old)
        record = visuals.read_stills(newer, self.root)
        self.assertEqual([item["shot_id"] for item in record["shots"]], ["second"])
        self.assertEqual([item is not None for item in visuals.match_stills(shots, record)], [False, True, False])

    def test_kept_stills_are_copied_and_portraits_kept(self):
        shots = [Shot(0, FPS, False, "one"), Shot(FPS, FPS, True, "two")]
        visuals.keep_stills(self.audio, self.root, shots, [self.png("a"), self.png("b")], {},
                            portraits={"mia": (self.png("p"), "abc")})
        first = visuals.read_stills(self.audio, self.root)
        folder = visuals.stills_dir(self.audio, self.root)
        kept = folder / first["shots"][0]["image"]
        before = kept.read_bytes()
        visuals.keep_stills(self.audio, self.root, [shots[0]], [kept], {},
                            portraits=visuals.kept_portraits(first, self.root, self.audio))
        second = visuals.read_stills(self.audio, self.root)
        self.assertEqual(len(second["shots"]), 1)
        # A kept still is copied byte for byte, never re-encoded.
        self.assertEqual((folder / second["shots"][0]["image"]).read_bytes(), before)
        self.assertEqual(second["portraits"], first["portraits"])
        self.assertEqual(second["portraits"]["mia"]["prompt"], "abc")
        self.assertEqual(sorted(path.name for path in folder.glob("*.*") if path.suffix != ".json"),
                         sorted([second["shots"][0]["image"], second["portraits"]["mia"]["image"]]))

    def test_changed_portrait_pixels_get_a_new_immutable_filename(self):
        shots = [Shot(0, FPS, False, "one")]
        visuals.keep_stills(self.audio, self.root, shots, [self.png("a")], {},
                            portraits={"mia": (self.png("old-portrait", "blue"), "same-prompt")})
        before = visuals.read_stills(self.audio, self.root)["portraits"]["mia"]
        visuals.keep_stills(self.audio, self.root, shots, [self.png("b")], {},
                            portraits={"mia": (self.png("new-portrait", "red"), "same-prompt")})
        after = visuals.read_stills(self.audio, self.root)["portraits"]["mia"]
        self.assertEqual(before["prompt"], after["prompt"])
        self.assertNotEqual(before["image"], after["image"])
        self.assertNotEqual(before["version"], after["version"])

    def test_portrait_identity_includes_transparency(self):
        opaque = self.folder / "opaque.png"
        transparent = self.folder / "transparent.png"
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(opaque)
        Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(transparent)
        self.assertNotEqual(visuals.portrait_version(opaque), visuals.portrait_version(transparent))

    def test_interrupted_all_shots_redraw_marks_remaining_old_faces_changed(self):
        write_snapshot(self.audio, {"family": DRAMA3_FAMILY, "settings": {"duration": 4.0}})
        visuals.write_plan(self.audio, "comic", 3, PLAN)
        saved = visuals.read_plan(self.audio)
        shots = visuals.plan_shots_for(saved, 4.0, "comic")
        fingerprint = visuals.prompt_key(visuals.portrait_prompts(saved, "comic")["mia"])
        visuals.keep_stills(self.audio, self.root, shots, [self.png(f"old-{i}") for i in range(3)], {},
                            extras=[{"characters": ["mia"]}, {"characters": ["mia"]}, {"characters": []}],
                            portraits={"mia": (self.png("old-face", "blue"), fingerprint)})
        deps = SimpleNamespace(resolve_result=lambda _name: str(self.audio),
                               runtime_config=lambda: SimpleNamespace(output_dir=self.root))
        controller = Mock(fields={})
        controller.generate.side_effect = [[([str(self.png("new-face"))], "Complete")],
            [([str(self.png("scene"))], "Complete")], [([str(self.png("matched"))], "Complete")],
            [(None, "Error: interrupted")]]
        with self.assertRaisesRegex(H3Error, "interrupted"):
            drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, "landscape", only="all"))
        record = visuals.read_stills(self.audio, self.root)
        statuses = [visuals.shot_status(shot, item) for shot, item in zip(shots, visuals.match_stills(shots, record))]
        self.assertEqual(statuses, ["drawn", "changed", "drawn"])
        self.assertEqual(visuals.chosen_shots(shots, visuals.match_stills(shots, record), "missing"), [1])

    def test_redrawing_an_unrelated_shot_preserves_an_edited_characters_old_portrait(self):
        write_snapshot(self.audio, {"family": DRAMA3_FAMILY, "settings": {"duration": 4.0}})
        visuals.write_plan(self.audio, "comic", 3, PLAN)
        saved = visuals.read_plan(self.audio)
        shots = visuals.plan_shots_for(saved, 4.0, "comic")
        fingerprint = visuals.prompt_key(visuals.portrait_prompts(saved, "comic")["mia"])
        visuals.keep_stills(self.audio, self.root, shots, [self.png(f"old-{i}") for i in range(3)], {},
                            portraits={"mia": (self.png("old-face"), fingerprint)})
        before = visuals.read_stills(self.audio, self.root)["portraits"]
        changed = edit_subject(PLAN, "characters", "mia", "Mia, short silver hair and a green coat")
        visuals.write_plan(self.audio, "comic", 3, changed)
        deps = SimpleNamespace(resolve_result=lambda _name: str(self.audio),
                               runtime_config=lambda: SimpleNamespace(output_dir=self.root))
        controller = Mock(fields={})
        controller.generate.return_value = [([str(self.png("rain"))], "Complete")]
        drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, "landscape", only=[2]))
        self.assertEqual(visuals.read_stills(self.audio, self.root)["portraits"], before)
        self.assertTrue((visuals.stills_dir(self.audio, self.root) / before["mia"]["image"]).is_file())

    def test_matching_draft_and_request_survive_partial_redraw_without_recompression(self):
        shot = Shot(0, FPS, False, "one")
        draft, final = self.png("draft", "blue"), self.png("final")
        matching = {"prompt": "Change only the face.", "references": [draft.name, "portrait.png"],
                    "settings": {"reference_resolution": 0, "steps": 8}}
        visuals.keep_stills(self.audio, self.root, [shot], [final], {},
                            extras=[{"draft": str(draft), "characters": ["mia"], "matching": matching}])
        folder = visuals.stills_dir(self.audio, self.root)
        first = visuals.read_stills(self.audio, self.root)["shots"][0]
        old_draft = folder / first["draft"]
        self.assertEqual(old_draft.read_bytes(), draft.read_bytes())
        self.assertEqual(first["matching"], matching)
        # Keeping this shot while another is drawn must retain both its before-image and request.
        extras = {key: first[key] for key in ("draft", "characters", "matching")}
        visuals.keep_stills(self.audio, self.root, [shot], [folder / first["image"]], {}, extras=[extras])
        second = visuals.read_stills(self.audio, self.root)["shots"][0]
        self.assertEqual((folder / second["draft"]).read_bytes(), draft.read_bytes())
        self.assertEqual(second["matching"], matching)
        self.assertEqual(second["draft"], first["draft"])
        self.assertTrue(old_draft.exists())

    def test_a_new_transition_applies_to_kept_stills_without_redrawing(self):
        shots = [Shot(0, FPS, False, "one"), Shot(FPS, FPS, True, "two")]
        visuals.keep_stills(self.audio, self.root, shots, [self.png("a"), self.png("b")], {})
        before = visuals.read_stills(self.audio, self.root)
        visuals.retime_stills(self.audio, self.root, shots)  # nothing changed: same revision
        self.assertEqual(visuals.read_stills(self.audio, self.root)["revision"], before["revision"])
        visuals.retime_stills(self.audio, self.root, [shots[0], Shot(FPS, FPS, False, "two")])
        after = visuals.read_stills(self.audio, self.root)
        self.assertEqual([item["dissolve"] for item in after["shots"]], [False, False])
        self.assertNotEqual(after["revision"], before["revision"])
        self.assertEqual([item["image"] for item in after["shots"]], [item["image"] for item in before["shots"]])

    def test_status_and_which_shots_to_draw(self):
        shots = [Shot(0, FPS, False, "one"), Shot(FPS, FPS, False, "two"), Shot(2 * FPS, FPS, False, "three")]
        record = {"shots": [{"first_frame": 0, "frames": FPS, "prompt": "one", "image": "a.jpg"},
                            {"first_frame": FPS, "frames": FPS, "prompt": "old", "image": "b.jpg"}]}
        stills = visuals.match_stills(shots, record)
        self.assertEqual([visuals.shot_status(shot, still) for shot, still in zip(shots, stills)],
                         ["drawn", "changed", "todo"])
        self.assertEqual(visuals.chosen_shots(shots, stills, "missing"), [1, 2])
        self.assertEqual(visuals.chosen_shots(shots, stills, None), [0, 1, 2])
        self.assertEqual(visuals.chosen_shots(shots, stills, [2, 0, 2]), [0, 2])
        with self.assertRaisesRegex(H3Error, "not in the current"):
            visuals.chosen_shots(shots, stills, [3])

    def test_drawing_missing_shots_keeps_the_rest_and_reuses_portraits(self):
        write_snapshot(self.audio, {"family": DRAMA3_FAMILY, "settings": {"duration": 4.0}})
        visuals.write_plan(self.audio, "comic", 3, PLAN)
        deps = SimpleNamespace(resolve_result=lambda name: str(self.audio),
                               runtime_config=lambda: SimpleNamespace(output_dir=self.root))
        made = iter(range(100))
        controller = Mock(fields={})
        controller.generate.side_effect = lambda *args: [([str(self.png(f"g{next(made)}"))], "Complete")]

        drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, "landscape"))
        first = visuals.read_stills(self.audio, self.root)
        modes = [call.args[1] for call in controller.generate.call_args_list]
        # A portrait, then each shot; the two showing Mia are matched to it.
        self.assertEqual(modes, ["Text to image", "Text to image", "Image edit", "Text to image", "Image edit",
                                 "Text to image"])
        self.assertEqual(list(first["portraits"]), ["mia"])

        visuals.write_plan(self.audio, "comic", 3, edit_shot(PLAN, 1, prompt="@mia drops @radio."))
        controller.generate.reset_mock()
        drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, "landscape",
                                  only="missing"))
        second = visuals.read_stills(self.audio, self.root)
        # Only shot 2 is drawn again, matched to the kept portrait instead of a new one.
        self.assertEqual([call.args[1] for call in controller.generate.call_args_list], ["Text to image", "Image edit"])
        folder = visuals.stills_dir(self.audio, self.root)
        self.assertEqual(len(second["shots"]), 3)
        self.assertEqual(second["portraits"], first["portraits"])
        self.assertNotEqual(second["shots"][1]["prompt"], first["shots"][1]["prompt"])
        self.assertEqual(second["shots"][0]["prompt"], first["shots"][0]["prompt"])
        self.assertTrue((folder / second["shots"][0]["image"]).is_file())
        with self.assertRaisesRegex(H3Error, "already drawn"):
            drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, "landscape",
                                      only="missing"))
        # Drawing every shot again starts afresh: a new portrait, then every scene.
        controller.generate.reset_mock()
        drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, "landscape",
                                  only="all"))
        self.assertEqual([call.args[1] for call in controller.generate.call_args_list], modes)
        json.dumps(second)  # the record stays plain JSON

    def test_partial_redraw_reuses_legacy_portrait_through_every_checkpoint(self):
        from h3_app.drama3.visual_plan import sheet_prompt

        write_snapshot(self.audio, {"family": DRAMA3_FAMILY, "settings": {"duration": 4.0}})
        visuals.write_plan(self.audio, "comic", 3, PLAN)
        shots = visuals.plan_shots_for(visuals.read_plan(self.audio), 4.0, "comic")
        body = sheet_prompt("characters", "mia", PLAN["characters"]["mia"], None, "")
        legacy = f"{body}\nArt style: {PLAN.get('art_style') or PLAN.get('style')}\ncomic"
        fingerprint = visuals.prompt_key(legacy)
        visuals.keep_stills(self.audio, self.root, shots, [self.png(f"old-{i}") for i in range(3)], {},
                            portraits={"mia": (self.png("portrait"), fingerprint)})
        before = visuals.read_stills(self.audio, self.root)
        deps = SimpleNamespace(resolve_result=lambda name: str(self.audio),
                               runtime_config=lambda: SimpleNamespace(output_dir=self.root))
        controller = Mock(fields={})
        calls = []

        def generate(engine, mode, prompt, references, captured):
            self.assertTrue(all(Path(path).is_file() for path in references))
            calls.append(mode)
            return [([str(self.png(f"new-{len(calls)}"))], "Complete")]

        controller.generate.side_effect = generate
        drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3,
                                 True, "landscape", only=[0, 1]))
        self.assertEqual(calls, ["Text to image", "Image edit", "Text to image", "Image edit"])
        self.assertEqual(visuals.read_stills(self.audio, self.root)["portraits"], before["portraits"])

    def test_each_completed_shot_is_published_before_the_next_generation(self):
        write_snapshot(self.audio, {"family": DRAMA3_FAMILY, "settings": {"duration": 4.0}})
        visuals.write_plan(self.audio, "comic", 3, PLAN)
        deps = SimpleNamespace(resolve_result=lambda name: str(self.audio),
                               runtime_config=lambda: SimpleNamespace(output_dir=self.root))
        controller = Mock(fields={})
        made = []

        def generate(*args):
            if made:
                record = visuals.read_stills(self.audio, self.root)
                self.assertEqual(len(record["shots"]), len(made))
                folder = visuals.stills_dir(self.audio, self.root)
                self.assertTrue(all((folder / row["image"]).is_file() for row in record["shots"]))
            path = self.png(f"progress-{len(made)}")
            made.append(path)
            return [([str(path)], "Complete")]

        controller.generate.side_effect = generate
        updates = list(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3,
                                          False, "landscape"))
        ready = [value for value, status in updates if status.endswith(" · Ready")]
        self.assertEqual(len(ready), 3)
        self.assertEqual(len(visuals.read_stills(self.audio, self.root)["shots"]), 3)

    def test_failed_redraw_keeps_completed_shot_and_previous_undrawn_images(self):
        write_snapshot(self.audio, {"family": DRAMA3_FAMILY, "settings": {"duration": 4.0}})
        visuals.write_plan(self.audio, "comic", 3, PLAN)
        shots = visuals.plan_shots_for(visuals.read_plan(self.audio), 4.0, "comic")
        old_images = [self.png(f"old-{i}", "blue") for i in range(3)]
        visuals.keep_stills(self.audio, self.root, shots, old_images, {})
        before = visuals.read_stills(self.audio, self.root)
        deps = SimpleNamespace(resolve_result=lambda name: str(self.audio),
                               runtime_config=lambda: SimpleNamespace(output_dir=self.root))
        controller = Mock(fields={})
        controller.generate.side_effect = [[([str(self.png("new"))], "Complete")],
                                           [(None, "Error: interrupted")]]
        with self.assertRaisesRegex(H3Error, "interrupted"):
            drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3,
                                     False, "landscape"))
        after = visuals.read_stills(self.audio, self.root)
        self.assertNotEqual(after["revision"], before["revision"])
        self.assertNotEqual(after["shots"][0]["image"], before["shots"][0]["image"])
        self.assertEqual([row["image"] for row in after["shots"][1:]],
                         [row["image"] for row in before["shots"][1:]])
        folder = visuals.stills_dir(self.audio, self.root)
        self.assertTrue(all((folder / row["image"]).is_file() for row in after["shots"]))


if __name__ == "__main__":
    unittest.main()
