"""Soundstage scene rooms, stage positions and voice distance: parsing and mixing."""

from __future__ import annotations

import unittest

import numpy as np

from h3_app.drama3.lint import lint_drama3_script
from h3_app.drama3.serving import audio, generate, rooms
from h3_app.drama3.serving.planner import plan_turns_chunks
from h3_app.drama3.serving.speaker_effects import apply_speaker_effects, extract_speaker_effects
from h3_app.drama3.serving.tree import (
    ParallelNode,
    iter_speech_nodes,
    parse_drama3_tree,
    staging_problems,
    strip_drama3_sound_tags,
)
from h3_app.workflows.drama3 import format_script

RATE = 16000


def voice(seconds=1.0, rate=RATE):
    """Broadband like speech, so a room's response is heard across the spectrum."""
    return (0.1 * np.random.default_rng(7).standard_normal(int(seconds * rate))).astype(np.float32)


def rms(samples):
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))


class RoomResponseTests(unittest.TestCase):
    def test_responses_are_stereo_unit_energy_and_decorrelated(self):
        for name in rooms.ROOMS:
            with self.subTest(room=name):
                response = rooms.room_response(name, RATE)
                self.assertEqual(response.ndim, 2)
                self.assertEqual(response.shape[1], 2)
                self.assertTrue(np.all(np.isfinite(response)))
                np.testing.assert_allclose(np.sum(response.astype(np.float64) ** 2), 2.0, rtol=1e-3)
                if not rooms.ROOMS[name].open_air:
                    # Indoor diffuse tails surround the listener; open air has no such tail.
                    tail = response[int(0.1 * RATE):]
                    self.assertLess(abs(np.corrcoef(tail[:, 0], tail[:, 1])[0, 1]), 0.3)
                self.assertIs(rooms.room_response(name, RATE), response)  # computed once

    def test_larger_rooms_ring_longer_and_tails_decay(self):
        def ringing(name):
            energy = np.sum(rooms.room_response(name, RATE).astype(np.float64) ** 2, axis=1)
            return np.searchsorted(np.cumsum(energy), 0.99 * energy.sum()) / RATE

        self.assertLess(ringing("small_room"), ringing("medium_room"))
        self.assertLess(ringing("medium_room"), ringing("large_hall"))
        hall = rooms.room_response("large_hall", RATE)
        half = len(hall) // 2
        self.assertLess(rms(hall[half:]), 0.1 * rms(hall[:half]))

    def test_highs_die_away_before_lows(self):
        from scipy import signal

        hall = rooms.room_response("large_hall", RATE)[:, 0]
        late = slice(int(0.6 * RATE), int(1.0 * RATE))
        low = signal.sosfilt(signal.butter(4, 400 / (0.5 * RATE), "lowpass", output="sos"), hall)
        high = signal.sosfilt(signal.butter(4, 3500 / (0.5 * RATE), "highpass", output="sos"), hall)
        early = slice(int(0.05 * RATE), int(0.25 * RATE))
        self.assertLess(rms(high[late]) / rms(high[early]), rms(low[late]) / rms(low[early]))

    def test_room_wet_rings_past_the_sound_at_its_send_level(self):
        dry = voice(0.5)
        wet = rooms.room_wet(dry, RATE, "small_room")
        self.assertEqual(wet.shape, (len(dry) + len(rooms.room_response("small_room", RATE)) - 1, 2))
        self.assertGreater(rms(wet[len(dry):]), 0.0)
        half = rooms.room_wet(dry, RATE, "small_room", send=0.5)
        np.testing.assert_allclose(half, 0.5 * wet, atol=1e-6)
        self.assertEqual(rooms.room_wet(dry, RATE, "underwater").shape, (len(dry), 2))  # not a real room
        self.assertFalse(rooms.room_wet(dry, RATE, "underwater").any())

    def test_room_names_accept_reverb_prefixes_and_spaces(self):
        self.assertEqual(rooms.scene_room_name("small_room"), "small_room")
        self.assertEqual(rooms.scene_room_name("Large Hall"), "large_hall")
        self.assertEqual(rooms.scene_room_name("reverb_small_room"), "small_room")
        self.assertEqual(rooms.scene_room_name("dream-sequence"), "dream_sequence")
        self.assertEqual(rooms.scene_room_name("kitchen"), "")

    def test_everyday_rooms_keep_dialogue_dry_and_hall_has_audible_space(self):
        dry = voice(2.0)
        for name in ("small_room", "medium_room", "outdoors"):
            with self.subTest(room=name):
                wet = rooms.room_wet(dry, RATE, name)
                self.assertLess(rms(wet[:len(dry)]) / rms(dry), 0.1)
                # After a phrase stops, ordinary spaces must not leave a prominent wash.
                tail = wet[len(dry):len(dry) + RATE // 4]
                tail = np.pad(tail, ((0, max(0, RATE // 4 - len(tail))), (0, 0)))
                self.assertLess(rms(tail) / rms(dry), 0.02)
        hall = rooms.room_wet(dry, RATE, "large_hall")
        self.assertGreater(rms(hall[:len(dry)]) / rms(dry), 0.12)

    def test_outdoors_has_only_a_short_ground_reflection(self):
        response = rooms.room_response("outdoors", RATE)
        self.assertLess(len(response) / RATE, 0.03)
        self.assertGreater(rms(response), 0.0)

    def test_stage_position_shifts_early_room_reflections_toward_the_speaker_side(self):
        early_left, _ = rooms._early(rooms.ROOMS["small_room"], RATE, position=-0.7)
        early_right, _ = rooms._early(rooms.ROOMS["small_room"], RATE, position=0.7)
        first_reflections = slice(0, int(0.01 * RATE))
        self.assertGreater(rms(early_left[first_reflections, 0]), rms(early_left[first_reflections, 1]))
        self.assertGreater(rms(early_right[first_reflections, 1]), rms(early_right[first_reflections, 0]))
        np.testing.assert_allclose(early_left[:, 0], early_right[:, 1], rtol=1e-5, atol=1e-6)
        predelay = int(rooms.ROOMS["small_room"].predelay * RATE)
        onset = slice(predelay, predelay + int(0.008 * RATE))
        left_ir = rooms.room_response("small_room", RATE, position=-0.7)
        right_ir = rooms.room_response("small_room", RATE, position=0.7)
        self.assertGreater(rms(left_ir[onset, 0]), rms(left_ir[onset, 1]))
        self.assertGreater(rms(right_ir[onset, 1]), rms(right_ir[onset, 0]))


class StagingParseTests(unittest.TestCase):
    def test_parallel_carries_room_and_stage(self):
        names = {"Mia": "Mia", "Leo": "Leo"}
        tree = parse_drama3_tree(
            '<parallel room="small_room" stage="Mia=-0.5, Leo=0.4, Ann=3">\nMia: Hello.\nLeo: Hi.\n</parallel>',
            resolve=names.get)
        (scene,) = tree.children
        self.assertIsInstance(scene, ParallelNode)
        self.assertEqual(scene.room, "small_room")
        self.assertEqual(scene.stage, (("Mia", -0.5), ("Leo", 0.4), ("Ann", 1.0)))  # clamped

    def test_an_opening_tag_on_a_line_with_speech_is_still_a_block(self):
        tree = parse_drama3_tree('<parallel room="outdoors">Mia: Hello.</parallel>')
        (scene,) = tree.children
        self.assertEqual(scene.room, "outdoors")
        self.assertEqual(len(iter_speech_nodes(scene)), 1)

    def test_unknown_room_and_bad_stage_are_left_out_and_reported(self):
        tree = parse_drama3_tree('<parallel room="kitchen" stage="left of Mia">\nMia: Hello.\n</parallel>')
        (scene,) = tree.children
        self.assertEqual((scene.room, scene.stage), ("", ()))
        problems = staging_problems('<parallel room="kitchen" stage="left of Mia">')
        self.assertEqual(len(problems), 2)
        self.assertIn('unknown room "kitchen"', problems[0])
        self.assertEqual(staging_problems('<parallel room="large_hall" stage="Mia=-0.5; Leo=0.2">'), [])
        _script, notes = lint_drama3_script('<parallel room="kitchen">\n<|speaker:1|> Hello.\n</parallel>')
        self.assertTrue(any('unknown room "kitchen"' in note for note in notes))

    def test_distance_tags_are_extracted_like_effects(self):
        self.assertEqual(extract_speaker_effects("<far> Mia? Are you up there?"),
                         ("Mia? Are you up there?", ("far",)))
        self.assertEqual(extract_speaker_effects("<fairy_tiny><near> Hush."), ("Hush.", ("fairy_tiny", "near")))
        script, _notes = lint_drama3_script("<|speaker:1|> <far> [shouting] Mia, come down!")
        self.assertEqual(script, "<|speaker:1|> <far> Mia, [shouting] come down!")  # the cue moves, the tag stays

    def test_formatting_and_basic_mode_keep_or_drop_attributes(self):
        script = '<parallel room="small_room">\nMia: Hi.\n<sequential>\nLeo: Yes.\n</sequential>\n</parallel>'
        self.assertEqual(format_script(script).split("\n"),
                         ['<parallel room="small_room">', "  Mia: Hi.", "  <sequential>", "    Leo: Yes.",
                          "  </sequential>", "</parallel>"])
        self.assertEqual(strip_drama3_sound_tags(script), "Mia: Hi.\nLeo: Yes.")


class StagedMixTests(unittest.TestCase):
    def render(self, script, positions=None):
        names = {name: name for name in ("narrator", "Mia", "Leo", "Caller")}
        tree = parse_drama3_tree(script, resolve=names.get)
        node_chunks, takes = {}, {}
        for speech in iter_speech_nodes(tree):
            chunks = plan_turns_chunks(speech.turns, start_index=len(takes))
            node_chunks[id(speech)] = chunks
            for chunk in chunks:
                takes[chunk.index] = generate.Take(chunk, voice(), RATE, 1, None)
        spans, placements = {}, {}
        mixed = generate._render_ast(tree, node_chunks, takes, {}, RATE, spans=spans,
                                     positions=positions or {}, placements=placements)
        return mixed, spans, placements, takes

    def take_audio(self, mixed, spans, index):
        start, end, _lines = spans[index]
        return mixed[start:end]

    def test_scene_room_reaches_characters_but_not_the_narrator_or_a_phone(self):
        script = ('<parallel room="medium_room">\n<sequential>\nnarrator: She waited.\n<#0.5#>\nMia: Hello?\n'
                  '<#0.5#>\nCaller: <phone_call> It is me.\n<#0.5#>\n</sequential>\n</parallel>')
        mixed, spans, _placements, _takes = self.render(script)
        dry, _ = self.render(script.replace(' room="medium_room"', ""))[:2]

        def ringing(audio_, index):
            # What plays in the half second after the take: the room's tail, or the pause's silence.
            end = spans[index][1]
            return rms(audio_[end:end + RATE // 4])

        self.assertEqual(ringing(mixed, 0), 0.0)  # the narrator stays outside the room
        self.assertGreater(ringing(mixed, 1), 0.0)  # Mia has a subtle tail into the pause
        self.assertLess(ringing(mixed, 1), 0.02 * rms(voice()))
        self.assertEqual(ringing(dry, 1), 0.0)
        self.assertEqual(ringing(mixed, 2), 0.0)  # the caller is on the phone, not in the room

    def test_line_place_tag_uses_the_real_room_and_rings_past_the_take(self):
        mixed, spans, _placements, _takes = self.render("Mia: <reverb_large_hall> Hello?")
        start, end, _lines = spans[0]
        self.assertGreater(len(mixed), end + RATE // 2)  # the hall's tail is kept, not cut at the take's end
        self.assertGreater(rms(mixed[end:]), 0.001)

    def test_stage_places_named_speakers_for_its_scene_only(self):
        script = ('<parallel stage="Mia=-0.6, narrator=0.9">\n<sequential>\nnarrator: Inside.\nMia: Here.\n'
                  '</sequential>\n</parallel>\nMia: Outside.')
        mixed, spans, placements, _takes = self.render(script, positions={"Mia": 0.3})
        self.assertEqual(placements, {0: 0.0, 1: -0.6, 2: 0.3})  # the narrator stays centred
        inside = self.take_audio(mixed, spans, 1)
        self.assertGreater(rms(inside[:, 0]), 2 * rms(inside[:, 1]))  # left

    def test_far_voice_is_quieter_and_duller_but_keeps_its_room(self):
        near_mix, near_spans, *_ = self.render('<parallel room="small_room">\nMia: Hello.\n</parallel>')
        far_mix, far_spans, *_ = self.render('<parallel room="small_room">\nMia: <far> Hello.\n</parallel>')
        near_take, far_take = self.take_audio(near_mix, near_spans, 0), self.take_audio(far_mix, far_spans, 0)
        self.assertLess(rms(far_take), 0.6 * rms(near_take))
        end = near_spans[0][1]
        # The room's tail is the same size: the voice is further away inside the room.
        self.assertGreater(rms(far_mix[end:]) / rms(near_mix[end:]), 0.8)

    def test_far_voice_outside_a_room_and_near_voice(self):
        from scipy import signal

        voice_ = voice()
        far = audio.voice_distance(voice_, RATE, "far", in_room=False)
        self.assertLess(rms(far), rms(voice_))
        self.assertEqual(len(far), len(voice_))
        near = audio.voice_distance(voice_, RATE, "near", in_room=True)
        self.assertAlmostEqual(rms(near) / rms(voice_), audio.DISTANCE_GAINS["near"], places=3)
        self.assertEqual(len(near), len(voice_))
        # Proximity effect warms the low chest (<220 Hz) and adds breath air (>4.5 kHz) relative to mid-band.
        low = signal.butter(2, 220 / (0.5 * RATE), "lowpass", output="sos")
        mid = signal.butter(2, [400 / (0.5 * RATE), 2500 / (0.5 * RATE)], "bandpass", output="sos")
        high = signal.butter(2, 4500 / (0.5 * RATE), "highpass", output="sos")
        self.assertGreater(rms(signal.sosfilt(low, near)) / rms(signal.sosfilt(mid, near)),
                           rms(signal.sosfilt(low, voice_)) / rms(signal.sosfilt(mid, voice_)))
        self.assertGreater(rms(signal.sosfilt(high, near)) / rms(signal.sosfilt(mid, near)),
                           rms(signal.sosfilt(high, voice_)) / rms(signal.sosfilt(mid, voice_)))
        np.testing.assert_array_equal(audio.voice_distance(voice_, RATE, "", in_room=True), voice_)

    def test_sequential_sound_cue_in_a_room_advances_at_dry_end_and_keeps_subtitle_sync(self):
        from pathlib import Path
        from unittest.mock import patch

        script = (
            '<parallel room="large_hall">\n'
            '  <sequential>\n'
            '    <sound name="door_slam" duration="0.5">Door slam</sound>\n'
            '    Mia: Who is there?\n'
            '  </sequential>\n'
            '</parallel>'
        )
        names = {"Mia": "Mia"}
        tree = parse_drama3_tree(script, resolve=names.get)
        speech = iter_speech_nodes(tree)[0]
        chunks = plan_turns_chunks(speech.turns, start_index=0)
        line_audio = voice(0.8)
        takes = {0: generate.Take(chunks[0], line_audio, RATE, 1, None)}
        hit = np.ones((int(0.5 * RATE), 2), dtype=np.float32) * 0.2
        spans = {}
        with patch.object(audio, "load_audio_file", return_value=hit):
            mixed = generate._render_ast(
                tree, {id(speech): chunks}, takes, {"door_slam": Path("door_slam.wav")}, RATE, spans=spans
            )
        start, end, line_spans = spans[0]
        # Mia's line starts right when the dry 0.5s door slam finishes (8000 samples, not waiting
        # for the hall's 1.8s reverb tail), and the subtitle span matches the exact slice where Mia speaks.
        self.assertEqual(start, int(0.5 * RATE))
        self.assertEqual(end - start, len(line_audio))
        self.assertEqual([(start + s, start + e) for s, e in line_spans], [(start, end)])
        self.assertGreater(rms(mixed[start:end]), 0.05)


    def test_vocal_effect_room_tags_keep_the_take_length(self):
        voice_ = voice()
        for effects in (("reverb_small_room",), ("reverb_large_hall", "phone_call"), ("far",)):
            with self.subTest(effects=effects):
                processed = apply_speaker_effects(voice_, RATE, effects)
                self.assertEqual(len(processed), len(voice_))
                self.assertTrue(np.all(np.isfinite(processed)))
        np.testing.assert_array_equal(apply_speaker_effects(voice_, RATE, ("far",)), voice_)  # placement only


class StackedSpaceTests(unittest.TestCase):
    """A voice is heard in one space: stacked reverbs wash the words out."""

    def boards(self, effects, in_room):
        """The plugins the effects chain builds."""
        from unittest.mock import patch

        import pedalboard

        from h3_app.drama3.serving import speaker_effects

        seen = []

        class Board:
            def __init__(self, plugins):
                seen.append(plugins)

            def __call__(self, samples, rate):
                return samples

        with patch.object(pedalboard, "Pedalboard", Board):
            speaker_effects.apply_speaker_effects(voice(0.2), RATE, effects, in_room=in_room)
        return [type(plugin).__name__ for plugins in seen for plugin in plugins]

    def setUp(self):
        try:
            import pedalboard  # noqa: F401
        except ImportError:
            self.skipTest("pedalboard not installed")

    def test_a_room_replaces_an_effects_own_reverb(self):
        self.assertIn("Reverb", self.boards(("ghost_spirit",), in_room=False))
        self.assertNotIn("Reverb", self.boards(("ghost_spirit",), in_room=True))
        self.assertNotIn("Reverb", self.boards(("reverb_large_hall", "ghost_spirit"), in_room=False))
        self.assertIn("PitchShift", self.boards(("ghost_spirit",), in_room=True))  # the character stays

    def test_stacked_effects_keep_one_reverb(self):
        self.assertEqual(self.boards(("ghost_spirit", "ancient_god", "dream_sequence"), in_room=False)
                         .count("Reverb"), 1)

    def test_fallback_drops_the_reverb_a_room_or_later_effect_replaces(self):
        from unittest.mock import patch

        from h3_app.drama3.serving import speaker_effects

        calls = []
        real = speaker_effects._reverb
        with patch.object(speaker_effects, "_try_apply_pedalboard", lambda *args: None), \
                patch.object(speaker_effects, "_reverb", lambda *args: calls.append(1) or real(*args)):
            speaker_effects.apply_speaker_effects(voice(0.2), RATE, ("ghost_spirit", "ancient_god"))
            self.assertEqual(len(calls), 1)
            speaker_effects.apply_speaker_effects(voice(0.2), RATE, ("ghost_spirit",), in_room=True)
            self.assertEqual(len(calls), 1)

    def test_a_ghost_in_a_hall_keeps_its_words(self):
        # Syllables at ~4 Hz: the rhythm that carries the words must survive the effect and the room.
        from scipy import signal

        t = np.arange(3 * RATE) / RATE
        dry = (voice(3.0) * np.clip(np.sin(2 * np.pi * 4.3 * t), 0, None) ** 2).astype(np.float32)
        processed = apply_speaker_effects(dry, RATE, ("ghost_spirit",), in_room=True)
        processed = processed + rooms.room_wet(processed, RATE, "large_hall")[: len(dry)].mean(axis=1)
        sos = signal.butter(2, 20, "lowpass", fs=RATE, output="sos")
        envelope = [signal.sosfiltfilt(sos, np.abs(x)) for x in (dry, processed)]
        self.assertGreater(np.corrcoef(*envelope)[0, 1], 0.85)


if __name__ == "__main__":
    unittest.main()
