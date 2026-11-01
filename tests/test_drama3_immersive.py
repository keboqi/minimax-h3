"""Behavioral checks for the shared production controls used by both writers."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from h3_app.drama3.guided.schema import parse_staging
from h3_app.drama3.guided.schema import parse_content_preserve
from h3_app.drama3.guided.segment import segment_source
from h3_app.drama3.guided.compile import sound_line
from h3_app.drama3.serving import audio, generate
from h3_app.drama3.serving.cue_timing import phrase_time, resolve_anchor, insert_holds, shifted_time
from h3_app.drama3.serving.planner import plan_turns_chunks
from h3_app.drama3.serving.tree import parse_drama3_tree, parse_sound_tag, iter_speech_nodes, sound_placement_attrs
from h3_app.prompt_writers.drama3_guided import _ordered_effects

RATE = 16000


class ImmersiveTests(unittest.TestCase):
    def test_sequential_sound_receives_the_same_automation_as_parallel_sound(self):
        tag = '<sound name="hit" envelope="0:0,0.5:1,1:0">Hit</sound>'
        sequential, _ = self.render(tag)
        parallel, _ = self.render('<parallel>\n' + tag + '\n</parallel>')
        np.testing.assert_array_equal(sequential, parallel)
        self.assertEqual(float(sequential[0, 0]), 0)
        self.assertEqual(float(sequential[-1, 0]), 0)

    def test_identical_holds_have_independent_scene_scopes(self):
        samples = np.ones(500, dtype=np.float32)
        consumed = set()
        args = (samples, 100, [(0, 500)], [[('door', 0, 200)]], ['door'])
        first = insert_holds(*args, [(1, 'door', 1)], consumed)
        second = insert_holds(*args, [(2, 'door', 1)], consumed)
        repeated = insert_holds(*args, [(1, 'door', 1)], consumed)
        self.assertEqual(len(first[0]), 600)
        self.assertEqual(len(second[0]), 600)
        self.assertEqual(len(repeated[0]), 500)

    def test_unmatched_anchors_are_omitted_instead_of_playing_at_scene_start(self):
        plain, _ = self.render('Mia: Open the door.')
        anchored, _ = self.render('<parallel>\n<sound name="hit" anchor="missing">Hit</sound>\n'
                                 'Mia: Open the door.\n</parallel>')
        np.testing.assert_array_equal(plain, anchored)

    def test_anchor_fallback_respects_occurrence_and_english_word_boundaries(self):
        self.assertIsNone(phrase_time('he', [('the', 0, 1)]))
        self.assertIsNone(resolve_anchor('he', [], [('the door', 0, 10)]))
        self.assertEqual(resolve_anchor('door', [], [('door', 0, 10), ('door', 20, 30)], occurrence=2), 30)
        self.assertIsNone(resolve_anchor('door', [], [('door', 0, 10)], occurrence=2))

    def test_hold_mapping_is_piecewise_not_uniform_time_stretching(self):
        words = [[('Open', 0, 100), ('door', 100, 200), ('wait', 300, 400)]]
        samples, bounds, timed, holds = insert_holds(np.ones(500, dtype=np.float32), 100,
            [(0, 500)], words, ['Open door wait'], [('door', 1)], set())
        self.assertEqual(len(samples), 600)
        self.assertEqual(timed[0], [('Open', 0, 100), ('door', 100, 200), ('wait', 400, 500)])
        self.assertEqual(shifted_time(100, holds), 100)
        self.assertEqual(shifted_time(300, holds), 400)
        self.assertEqual(bounds, [(0, 600)])

    def test_resumed_bed_matches_the_crossfaded_stream(self):
        samples = np.random.default_rng(3).normal(size=(RATE, 2)).astype(np.float32)
        continuous = audio.loop_to_length(samples, 4 * RATE, RATE)
        cursor = RATE + 1200
        resumed = audio.continued_loop(samples, RATE, RATE, cursor)
        np.testing.assert_allclose(resumed, continuous[cursor:cursor + RATE], atol=1e-6)
        first = audio.continued_loop(samples, RATE - 100, RATE)
        np.testing.assert_allclose(first, continuous[:RATE - 100], atol=1e-6)

    def test_gain_automation_changes_energy_and_rejects_invalid_points(self):
        node = parse_sound_tag('<sound envelope="0:0,0.5:1,1:0">Swell</sound>')
        result = audio.mix_parallel([{'audio': np.ones((RATE, 2), dtype=np.float32),
                                     'is_sound': True, 'envelope': node.envelope}], RATE)
        self.assertEqual(float(result.audio[0, 0]), 0)
        self.assertGreater(float(result.audio[RATE // 2, 0]), .99)
        self.assertEqual(float(result.audio[-1, 0]), 0)
        self.assertEqual(parse_sound_tag('<sound envelope="0:1,0:nan">Swell</sound>').envelope, ())

    def test_preserve_profiles_and_shared_beats_survive_parsing(self):
        source = segment_source('Mia: Hello.')
        unit = source.units[0].id
        result = parse_content_preserve(json.dumps({"cast": [{"name": "Mia", "role": "elderly, low voice"}],
            "speakers": {}, "scenes": [{"from": unit, "place": "market", "change": "discovery",
            "beats": [{"at": "Hello", "intent": "sincere, leave score quiet"}]}]}), source)
        self.assertEqual(result['roles']['Mia'], 'elderly, low voice')
        self.assertIn('leave score quiet', result['scenes'][0][2])

    def test_music_span_withdraws_at_anchor_and_preserves_controls(self):
        node = parse_sound_tag('<sound name="score" loop="true" anchor="opens" end_anchor="waits" '
                               'resume="true" fade="0.3" perspective="editorial">Sparse strings</sound>')
        rewritten = parse_sound_tag('<sound ' + ' '.join(sound_placement_attrs(node)) + '>Sparse strings</sound>')
        self.assertEqual((rewritten.anchor, rewritten.end_anchor, rewritten.resume, rewritten.fade),
                         ('opens', 'waits', True, .3))
        voice = np.full((3 * RATE, 2), .03, dtype=np.float32)
        bed = np.full((RATE, 2), .01, dtype=np.float32)
        mixed = audio.mix_parallel([{'audio': voice, 'dialogue': voice},
            {'audio': bed, 'is_sound': True, 'loop': True, 'speech_relative': True,
             'offset': RATE, 'end': 2 * RATE, 'fade': .1}], RATE)
        self.assertEqual(mixed.speech_offset, 0)
        np.testing.assert_array_equal(mixed.audio[:RATE], voice[:RATE])
        self.assertGreater(np.max(mixed.audio[RATE:2*RATE] - voice[RATE:2*RATE]), 0)

    def test_continuing_bed_does_not_add_another_scene_lead_in(self):
        voice = np.full((RATE, 2), .03, dtype=np.float32)
        result = audio.mix_parallel([{'audio': voice}, {'audio': voice, 'loop': True,
                                    'is_sound': True, 'cursor': RATE // 2}], RATE)
        self.assertEqual(result.speech_offset, 0)

    def test_roles_duration_and_editorial_routing_survive_guided_compilation(self):
        sounds, _, _ = parse_staging(json.dumps({"sounds": [{"name": "awkward", "kind": "stinger",
            "role": "comic", "duration_class": "accent", "prompt": "Single dry pluck"}]}), set(), set(), [])
        node = parse_sound_tag(sound_line(sounds["awkward"]))
        self.assertEqual((node.duration, node.perspective), (1, "editorial"))
        self.assertIn('perspective="editorial"', sound_placement_attrs(node))

    def test_legacy_routing_and_explicit_duration_are_preserved(self):
        node = parse_sound_tag('<sound duration="4" duration_class="accent">Knock</sound>')
        self.assertEqual((node.duration, node.perspective), (4, ""))

    def test_phrase_occurrences_and_cjk_tokenization(self):
        words = [("Open", 0, 2), ("it", 2, 3), ("open", 5, 7), ("it", 7, 8)]
        self.assertEqual(phrase_time("open it", words, occurrence=2), 8)
        self.assertEqual(phrase_time("打开门", [("打开", 1, 2), ("门", 2, 3)]), 3)
        self.assertIsNone(phrase_time("missing", words))

    def test_distance_does_not_discard_device(self):
        self.assertEqual(_ordered_effects(("robotic_voice", "far", "walkie_talkie")),
                         ("robotic_voice", "far", "walkie_talkie"))

    def render(self, script, *, words=None):
        tree = parse_drama3_tree(script)
        chunks_by_node, takes = {}, {}
        for speech in iter_speech_nodes(tree):
            chunks = plan_turns_chunks(speech.turns, start_index=len(takes))
            chunks_by_node[id(speech)] = chunks
            for chunk in chunks:
                samples = np.random.default_rng(1).normal(0, .05, RATE).astype(np.float32)
                takes[chunk.index] = generate.Take(chunk, samples, RATE, 1, None,
                    line_words=words, line_bounds=[(0, RATE)] if words else None)
        spans = {}
        hit = np.ones((RATE // 10, 2), dtype=np.float32) * .1
        with patch.object(audio, 'load_audio_file', return_value=hit):
            result = generate._render_ast(tree, chunks_by_node, takes, {'hit': Path('hit.wav')}, RATE, spans=spans)
        return result, spans

    def test_editorial_hit_bypasses_hall_while_physical_hit_rings(self):
        prefix = '<parallel room="large_hall">\n'
        physical, _ = self.render(prefix + '<sound name="hit">Hit</sound>\n</parallel>')
        editorial, _ = self.render(prefix + '<sound name="hit" perspective="editorial">Hit</sound>\n</parallel>')
        self.assertEqual(len(editorial), RATE // 10)
        self.assertGreater(len(physical), len(editorial))

    def test_actual_word_time_moves_the_hit(self):
        script = '<parallel>\n<sound name="hit" anchor="door" edge="end">Hit</sound>\nMia: Open the door.\n</parallel>'
        early, _ = self.render(script, words=[[("Open", 0, .1), ("the", .1, .2), ("door", .2, .3)]])
        late, _ = self.render(script, words=[[("Open", 0, .2), ("the", .2, .5), ("door", .5, .7)]])
        self.assertFalse(np.array_equal(early, late))
        # The deterministic speech is identical; only the effect's location changes.
        self.assertGreater(np.max(np.abs(early[int(.3*RATE):int(.4*RATE)] - late[int(.3*RATE):int(.4*RATE)])), .01)

    def test_hold_extends_continuous_speech_without_another_take(self):
        plain, _ = self.render('Mia: Open the door.')
        held, spans = self.render('<parallel holds="door~1.2">\nMia: Open the door.\n</parallel>',
                                 words=[[("Open", 0, .1), ("the", .1, .2), ("door", .2, .3)]])
        self.assertEqual(len(held) - len(plain), int(1.2 * RATE))
        self.assertEqual(len([k for k in spans if isinstance(k, int)]), 1)

    def test_nested_effect_is_absent_from_ducking_sidechain(self):
        observed = []
        original = audio.speech_duck_gain
        def capture(samples, rate, **kw):
            observed.append(samples.copy())
            return original(samples, rate, **kw)
        with patch.object(audio, 'speech_duck_gain', side_effect=capture):
            self.render('<parallel>\n<sound name="hit" loop="true">Bed</sound>\n<sequential>\n'
                        '<sound name="hit">Hit</sound>\nMia: Wait.\n</sequential>\n</parallel>')
        self.assertTrue(observed)
        # Outer bed lead-in and foreground hit contain no dialogue.
        self.assertTrue(np.all(observed[-1][:int(.9 * RATE)] == 0))


if __name__ == '__main__':
    unittest.main()
