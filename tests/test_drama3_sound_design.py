"""Scene sound design: density budget, levels, transitions, score palette and the density report."""

import json
import unittest
from pathlib import Path

import numpy as np

from h3_app.drama3.guided.compile import compile_script
from h3_app.drama3.guided.model import Draft, Scene, Sound, SoundEvent, Staging, Utterance
from h3_app.drama3.guided.schema import parse_sound_palette, parse_staging
from h3_app.drama3.serving import audio
from h3_app.drama3.serving.tree import ParallelNode, iter_sound_nodes, parse_drama3_tree
from h3_app.drama3.sound_policy import (
    DURATION_CLASSES, LEVELS, SOUND_ROLES, TRANSITIONS, event_budget, sound_volume,
)
from h3_app.drama3.sound_report import density_summary, sound_density_report
from h3_app.prompt_writers.drama3_guided import _merge_batch_sounds, _scene_listing

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
RATE = 24000


def _lines(count, start=1, speaker="Mia"):
    return [Utterance(f"u{n}", speaker if n % 2 else "Leo", f"Line number {n} is spoken here.")
            for n in range(start, start + count)]


def _two_scenes(transition):
    utterances = _lines(4)
    scenes = [Scene("c1", ("u1", "u2")), Scene("c2", ("u3", "u4"))]
    sounds = {"rain": Sound("rain", "ambience", "Steady rain on a tin roof"),
              "hum": Sound("hum", "ambience", "Quiet room tone")}
    staging = {"c1": Staging(ambience="rain"), "c2": Staging(ambience="hum", transition=transition)}
    return Draft(utterances, scenes, sounds=sounds, staging=staging)


class SoundDesignTests(unittest.TestCase):
    def test_event_budget_scales_with_scene_length(self):
        self.assertEqual(event_budget(4), 6)
        self.assertEqual(event_budget(20), 10)
        self.assertEqual(event_budget(200), 32)
        utterances = _lines(20)
        events = tuple(SoundEvent("step", u.id) for u in utterances)
        draft = Draft(utterances, [Scene("c1", tuple(u.id for u in utterances))],
                      sounds={"step": Sound("step", "effect", "One footstep on wood")},
                      staging={"c1": Staging(events=events)})
        notes = []
        script = compile_script(draft, advanced=True, notes=notes)
        self.assertEqual(len(list(iter_sound_nodes(parse_drama3_tree(script)))), 10)
        self.assertIn("c1: kept the first 10 of 20 sound events", notes)

    def test_levels_set_each_use_and_parse_from_bed_objects(self):
        self.assertEqual(sound_volume("ambience", "low"), 0.2)
        self.assertEqual(sound_volume("effect", "high"), 1.3)
        self.assertEqual(sound_volume("stinger", ""), 0.4)
        reply = json.dumps({"sounds": [{"name": "storm", "kind": "ambience", "prompt": "Howling storm"},
                                       {"name": "slam", "kind": "effect", "prompt": "Door slam"}],
                            "scenes": [{"scene": "c1", "ambience": {"name": "storm", "level": "high"},
                                        "music": None, "transition": "establish",
                                        "events": [{"sound": "slam", "at": "u1", "level": "high"},
                                                   {"sound": "slam", "at": "u2", "level": "loud"}]}]})
        sounds, staging, _ = parse_staging(reply, {"c1"}, {"u1", "u2"}, [])
        plan = staging["c1"]
        self.assertEqual((plan.ambience, plan.ambience_level, plan.transition), ("storm", "high", "establish"))
        self.assertEqual([e.level for e in plan.events], ["high", ""])
        draft = Draft(_lines(2), [Scene("c1", ("u1", "u2"))], sounds=sounds, staging=staging)
        nodes = list(iter_sound_nodes(parse_drama3_tree(compile_script(draft, advanced=True))))
        self.assertEqual([n.volume for n in nodes], [0.55, 1.3, 1.0])

    def test_establish_lengthens_the_lead_in(self):
        script = compile_script(_two_scenes("establish"), advanced=True)
        masters = [node for node in parse_drama3_tree(script).children if isinstance(node, ParallelNode)]
        self.assertEqual([(m.lead_in, m.tail) for m in masters], [(0.0, 0.0), (2.5, 0.0)])
        self.assertIn("<#0.8#>", script)

    def test_bridge_crossfades_beds_with_a_short_gap(self):
        script = compile_script(_two_scenes("bridge"), advanced=True)
        masters = [node for node in parse_drama3_tree(script).children if isinstance(node, ParallelNode)]
        self.assertEqual([(m.lead_in, m.tail) for m in masters], [(0.0, 2.5), (2.0, 0.0)])
        self.assertIn("<#0.3#>", script)
        self.assertNotIn("<#0.8#>", script)
        # The old bed's unducked tail has nearly faded when the new scene's speech starts.
        lead_in, tail, gap = TRANSITIONS["bridge"]
        self.assertGreaterEqual(gap + lead_in, 0.9 * tail)

    def test_bridge_without_beds_on_both_sides_is_a_cut(self):
        draft = _two_scenes("bridge")
        draft.staging["c2"] = Staging(transition="bridge")
        script = compile_script(draft, advanced=True)
        self.assertNotIn("tail=", script)
        self.assertIn("<#0.8#>", script)

    def test_cut_is_unchanged(self):
        script = compile_script(_two_scenes(""), advanced=True)
        self.assertNotIn("lead_in", script)
        self.assertNotIn("tail=", script)

    def test_tree_bounds_lead_in_and_tail(self):
        tree = parse_drama3_tree('<parallel lead_in="9" tail="0.05">\n<sound loop="true">Rain</sound>\n'
                                 '<sequential>\nMia: Hello.\n</sequential>\n</parallel>')
        node = tree.children[0]
        self.assertEqual((node.lead_in, node.tail), (4.0, 0.3))
        node = parse_drama3_tree('<parallel lead_in="soon" tail="">\n<sound loop="true">Rain</sound>\n'
                                 '<sequential>\nMia: Hello.\n</sequential>\n</parallel>').children[0]
        self.assertEqual((node.lead_in, node.tail), (0.0, 0.0))

    def test_mixer_uses_lead_in_and_tail(self):
        voice = np.full((2 * RATE, 2), .03, dtype=np.float32)
        bed = np.full((RATE, 2), .01, dtype=np.float32)
        tracks = lambda tail=None: [{"audio": voice, "dialogue": voice},  # noqa: E731
                                    {"audio": bed, "is_sound": True, "loop": True,
                                     **({"tail": tail} if tail else {})}]
        default = audio.mix_parallel(tracks(), RATE)
        longer = audio.mix_parallel(tracks(3.0), RATE, lead_in=2.5)
        self.assertEqual(default.speech_offset, int(audio.SCENE_LEAD_IN_SECONDS * RATE))
        self.assertEqual(longer.speech_offset, int(2.5 * RATE))
        self.assertEqual(longer.sound_lengths[1] - default.sound_lengths[1],
                         int(1.7 * RATE) + int(3.0 * RATE) - int(audio.LOOP_TAIL_SECONDS * RATE))

    def test_palette_parses_and_stays_authoritative_across_batches(self):
        style, palette = parse_sound_palette(json.dumps({"sound_palette": {
            "style": "Wuxia epic, guqin and taiko", "music": [
                {"name": "Main Theme", "kind": "music", "prompt": "Guqin and soft strings, slow, no vocals"},
                {"name": "bad", "kind": "effect", "prompt": "Not score"}]}}))
        self.assertEqual(style, "Wuxia epic, guqin and taiko")
        self.assertEqual(list(palette), ["main_theme"])
        self.assertEqual(palette["main_theme"].perspective, "editorial")
        self.assertEqual(parse_sound_palette("not json"), ("", {}))
        sounds, staging, notes = dict(palette), {}, []
        redefined = Sound("main_theme", "music", "Electric guitar riff, no vocals")
        _merge_batch_sounds(sounds, staging, {"main_theme": redefined},
                            {"c2": Staging(music="main_theme")}, notes, palette)
        self.assertEqual(sounds["main_theme"], palette["main_theme"])
        self.assertEqual(staging["c2"].music, "main_theme")
        self.assertTrue(any("story palette" in note for note in notes), notes)

    def test_scene_listing_shows_structured_beats(self):
        utterances = _lines(1)
        scene = Scene("c1", ("u1",), "a cave", "the map is found",
                      (("Line number", "Mia", "hold, then let the score enter"),))
        listing = _scene_listing([scene], {u.id: u for u in utterances})
        self.assertIn('Beats: at "Line number" (Mia): hold, then let the score enter', listing)

    def test_density_report_counts_scenes_cues_beds_and_assets(self):
        script = compile_script(_two_scenes(""), advanced=True)
        draft = _two_scenes("")
        draft.sounds["knock"] = Sound("knock", "effect", "Two knocks on a wooden door")
        draft.staging["c2"] = Staging(ambience="hum", events=(SoundEvent("knock", "u3", "before"),
                                                              SoundEvent("knock", "u4", "after")))
        report = sound_density_report(compile_script(draft, advanced=True))
        self.assertEqual(len(report["scenes"]), 2)
        self.assertEqual((report["lines"], report["events"], report["scenes_with_bed"], report["assets"]),
                         (4, 2, 2, 3))
        self.assertEqual(report["events_per_10_lines"], 5.0)
        self.assertEqual(density_summary(script), "Sound design: 0 cues (0 per 10 lines), beds in 2 of 2 scenes, 2 assets.")
        self.assertEqual(density_summary("Mia: Hello."), "")

    def test_guided_run_shares_the_palette_with_every_sound_batch(self):
        from h3_app.prompt_writers.drama3 import DRAMA3_SPEAKER_AUTO
        from h3_app.prompt_writers.drama3_guided import guided_prompt_paths, write_drama3_guided
        preview = ("We have plenty of time to walk to the station this morning. I will bring a map and "
                   "a warm coat, and we can stop for tea on the way if the rain holds off for a while.")
        theme = {"name": "main_theme", "kind": "music", "prompt": "Solo cello, slow and wistful, no vocals"}
        requests = []

        def ask(path, request):
            stage = path.stem.removeprefix("prompt_drama3_guided_")
            requests.append((stage, request))
            if stage == "preserve":
                return json.dumps({"cast": [{"name": "Mia"}], "speakers": {},
                                   "scenes": [{"from": "s1", "place": "hall"}, {"from": "s2", "place": "garden"}],
                                   "sound_palette": {"style": "quiet chamber drama", "music": [theme]}}), "ok"
            if stage == "voices":
                return json.dumps({"voices": [{"name": "Mia", "description": "A female voice: calm.",
                                               "preview_text": preview}]}), "ok"
            if stage == "sound":
                return json.dumps({"sounds": [{**theme, "prompt": "Loud rock guitar, no vocals"}], "scenes": [
                    {"scene": "c1", "music": "main_theme"}, {"scene": "c2", "music": "main_theme"}]}), "ok"
            return '{"lines":[]}', "ok"

        notes = []
        reply, _, _ = write_drama3_guided("Mia: We are in the hall.\nMia: Later we reached the garden.", ask,
                                          guided_prompt_paths(PROMPTS), speaker_mode=DRAMA3_SPEAKER_AUTO,
                                          advanced=True, notes=notes)
        preserve = next(request for stage, request in requests if stage == "preserve")
        sound = next(request for stage, request in requests if stage == "sound")
        self.assertIn("Script mode: Advanced.", preserve)
        self.assertIn("Story sound palette: quiet chamber drama.", sound)
        self.assertIn("Solo cello", sound)
        script = json.loads(reply)["script"]
        self.assertEqual(script.count("Solo cello, slow and wistful, no vocals"), 2)
        self.assertNotIn("rock guitar", script)

    def test_both_prompts_document_the_shared_vocabulary(self):
        guided = (PROMPTS / "prompt_drama3_guided_sound.txt").read_text(encoding="utf-8")
        single = (PROMPTS / "prompt_drama3.txt").read_text(encoding="utf-8")
        for prompt in (guided, single):
            for word in (*DURATION_CLASSES, *SOUND_ROLES):
                self.assertIn(word, prompt)
        for word in (*LEVELS, *TRANSITIONS):
            self.assertIn(word, guided)
        self.assertIn('lead_in="', single)
        self.assertIn('tail="', single)
        for name in ("prompt_drama3_guided_preserve.txt", "prompt_drama3_guided_brief.txt"):
            self.assertIn("sound_palette", (PROMPTS / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
