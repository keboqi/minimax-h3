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
        self.assertEqual(second["portraits"], {"mia": {"image": "portrait-abc.png", "prompt": "abc"}})
        self.assertEqual(sorted(path.name for path in folder.glob("*.*") if path.suffix != ".json"),
                         sorted([second["shots"][0]["image"], "portrait-abc.png"]))

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

        drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True))
        first = visuals.read_stills(self.audio, self.root)
        modes = [call.args[1] for call in controller.generate.call_args_list]
        # A portrait, then each shot; the two showing Mia are matched to it.
        self.assertEqual(modes, ["Text to image", "Text to image", "Image edit", "Text to image", "Image edit",
                                 "Text to image"])
        self.assertEqual(list(first["portraits"]), ["mia"])

        visuals.write_plan(self.audio, "comic", 3, edit_shot(PLAN, 1, prompt="@mia drops @radio."))
        controller.generate.reset_mock()
        drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, only="missing"))
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
            drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, only="missing"))
        # Drawing every shot again starts afresh: a new portrait, then every scene.
        controller.generate.reset_mock()
        drain(visuals.draw_stills(controller, deps, self.audio.name, "qwen", "comic", 3, True, only="all"))
        self.assertEqual([call.args[1] for call in controller.generate.call_args_list], modes)
        json.dumps(second)  # the record stays plain JSON


if __name__ == "__main__":
    unittest.main()
