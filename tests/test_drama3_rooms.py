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
                # Independent tails per ear: the room surrounds the listener instead of sitting in the middle.
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
        self.assertGreater(ringing(mixed, 1), 0.001)  # Mia's line rings on into the pause
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
        voice_ = voice()
        far = audio.voice_distance(voice_, RATE, "far", in_room=False)
        self.assertLess(rms(far), rms(voice_))
        self.assertEqual(len(far), len(voice_))
        near = audio.voice_distance(voice_, RATE, "near", in_room=True)
        self.assertAlmostEqual(rms(near) / rms(voice_), audio.DISTANCE_GAINS["near"], places=3)
        np.testing.assert_array_equal(audio.voice_distance(voice_, RATE, "", in_room=True), voice_)

    def test_vocal_effect_room_tags_keep_the_take_length(self):
        voice_ = voice()
        for effects in (("reverb_small_room",), ("reverb_large_hall", "phone_call"), ("far",)):
            with self.subTest(effects=effects):
                processed = apply_speaker_effects(voice_, RATE, effects)
                self.assertEqual(len(processed), len(voice_))
                self.assertTrue(np.all(np.isfinite(processed)))
        np.testing.assert_array_equal(apply_speaker_effects(voice_, RATE, ("far",)), voice_)  # placement only


if __name__ == "__main__":
    unittest.main()
