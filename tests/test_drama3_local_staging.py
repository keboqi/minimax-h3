"""Readable local cue scopes, precise repeated phrases and unchanged passage takes."""

import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import numpy as np

from h3_app.drama3.guided.compile import compile_script
from h3_app.drama3.guided.model import Draft, Scene, Sound, SoundEvent, Staging, Utterance
from h3_app.drama3.guided.schema import parse_staging
from h3_app.drama3.serving import audio, generate
from h3_app.drama3.serving.planner import plan_turns_chunks, spoken_text
from h3_app.drama3.serving.tree import (
    ParallelNode, SoundNode, iter_sound_nodes, iter_speech_nodes, parse_drama3_tree,
    parse_sound_tag, sound_placement_attrs,
)


def owner(tree, name):
    if isinstance(tree, ParallelNode) and any(isinstance(child, SoundNode) and child.name == name
                                             for child in tree.children):
        return tree
    for child in getattr(tree, 'children', ()):
        found = owner(child, name)
        if found is not None:
            return found
    return None


def turns(tree):
    return [turn for speech in iter_speech_nodes(tree) for turn in speech.turns]


def render(script):
    """Deterministic word-aligned stand-ins; no model or downloaded audio."""
    rate = 1000
    tree = parse_drama3_tree(script)
    node_chunks, takes = {}, {}
    for speech in iter_speech_nodes(tree):
        chunks = plan_turns_chunks(speech.turns, start_index=len(takes))
        node_chunks[id(speech)] = chunks
        for chunk in chunks:
            cursor, bounds, word_lines = 0, [], []
            for line in chunk.lines:
                start = cursor
                words = []
                for word in re.findall(r'\w+', spoken_text(line)):
                    words.append((word, cursor / rate, (cursor + 100) / rate))
                    cursor += 100
                bounds.append((start, cursor))
                word_lines.append(words)
            takes[chunk.index] = generate.Take(chunk, np.full(cursor, .02, dtype=np.float32), rate, 1, None,
                                                line_bounds=bounds, line_words=word_lines)
    clips = {sound.name: Path(sound.name + '.wav') for sound in iter_sound_nodes(tree)}
    spans = {}
    with patch.object(audio, 'load_audio_file', return_value=np.full((50, 2), .1, dtype=np.float32)):
        mixed = generate._render_ast(tree, node_chunks, takes, clips, rate, spans=spans)
    return mixed, spans, takes


class LocalStagingTests(unittest.TestCase):
    def test_legacy_overlap_keeps_chronological_anchor_scope(self):
        from h3_app.drama3.guided.local_staging import localize_existing_cues
        script = ('<parallel>\n<sound name="hit" anchor="door" edge="start">Dry impact</sound>\n'
                  '<sequential>\n<parallel>\nMia: Please take your time before opening the door.\n'
                  'Tom: Door!\n</parallel>\nMia: Now we can leave.\n</sequential>\n</parallel>')
        localized = localize_existing_cues(script)
        # The direct cue must stay on the enclosing scope: source order says Mia,
        # whereas actual speech reaches Tom's "Door!" first.
        self.assertEqual(localized, script)
        rate = 1000
        def cue_time(version):
            tree = parse_drama3_tree(version)
            nodes, takes = {}, {}
            for speech in iter_speech_nodes(tree):
                chunks = plan_turns_chunks(speech.turns, start_index=len(takes))
                nodes[id(speech)] = chunks
                for chunk in chunks:
                    tokens = re.findall(r'\w+', chunk.text)
                    words = [[(word, .02, .08) if chunk.speaker == 'Tom' else (word, i * .1, (i + 1) * .1)
                              for i, word in enumerate(tokens)]]
                    takes[chunk.index] = generate.Take(chunk, np.full(rate, .02, dtype=np.float32), rate, 1, None,
                                                       line_bounds=[(0, rate)], line_words=words)
            spans = {}
            with patch.object(audio, 'load_audio_file', return_value=np.full((50, 2), .1, dtype=np.float32)):
                generate._render_ast(tree, nodes, takes, {'hit': Path('hit.wav')}, rate, spans=spans)
            return next(value[0] for key, value in spans.items() if not isinstance(key, int))
        self.assertEqual(cue_time(script), 20)
        self.assertEqual(cue_time(localized), 20)

    def test_mixed_case_containers_are_valid_and_malformed_tags_fail_cleanly(self):
        from h3_app.errors import H3Error
        from h3_app.prompt_writers.drama3 import parse_drama3_reply, DRAMA3_SCRIPT_ADVANCED
        from h3_app.drama3.guided.local_staging import localize_existing_cues
        from h3_app.drama3.guided.segment import is_control_line
        script = ('<Parallel>\n<SOUND name="hit" anchor="door">Dry impact</SOUND>\n'
                  '<Sequential>\nMia: Open the door.\n</Sequential>\n</Parallel>')
        self.assertTrue(is_control_line('<SOUND name="hit">Dry impact</SOUND>'))
        reply = {'script': script, 'speakers': [{'speaker_id': 0, 'name': 'Mia',
                 'description': 'A female voice: calm.', 'preview_text': 'Hello there.'}]}
        result, _ = parse_drama3_reply(json.dumps(reply), 'Mia: Open the door.', None, 'Auto',
                                      require_cast=True, script_mode=DRAMA3_SCRIPT_ADVANCED)
        self.assertIn('Open the door.', result)
        for bad in ['</Parallel>\nMia: Open the door.',
                    '<Parallel>\nMia: Open the door.\n</Sequential>',
                    '<Parallel>\nMia: Open the door.']:
            with self.subTest(bad=bad), self.assertRaises(H3Error):
                localize_existing_cues(bad)

    def draft(self, lines, events, *, bed=False):
        utterances = [Utterance(f'u{i + 1}', speaker, text) for i, (speaker, text) in enumerate(lines)]
        sounds = {'hit': Sound('hit', 'effect', 'One dry impact'),
                  'score': Sound('score', 'music', 'Quiet cello pulse, no vocals'),
                  'rain': Sound('rain', 'ambience', 'Steady diffuse rain')}
        return Draft(utterances, [Scene('c1', tuple(u.id for u in utterances))], sounds=sounds,
                     staging={'c1': Staging(ambience='rain' if bed else '', events=tuple(events))})

    def test_event_stays_beside_its_line_and_ignores_earlier_duplicate(self):
        draft = self.draft([('Mia', 'I heard the door.'), ('Tom', 'Please wait here.'),
                            ('Mia', 'Now open the door.'), ('Tom', 'We can go.')],
                           [SoundEvent('hit', 'u3', phrase='door', when='with')], bed=True)
        script = compile_script(draft, advanced=True)
        tree = parse_drama3_tree(script)
        scope = owner(tree, 'hit')
        self.assertEqual([t.text for t in turns(scope)], ['Now open the door.'])
        self.assertNotIn(scope, tree.children[0].children[:1])
        self.assertLess(script.index('Please wait here.'), script.index('name="hit"'))
        self.assertLess(script.index('name="hit"'), script.index('Now open the door.'))
        _, spans, _ = render(script)
        hit = next(value for key, value in spans.items() if not isinstance(key, int) and value[2]['name'] == 'hit')
        target = next(value for key, value in spans.items() if isinstance(key, int) and key == 2)
        self.assertEqual(hit[0], target[0] + 300)

    def test_same_speaker_passage_remains_one_take_and_occurrence_is_adjusted(self):
        draft = self.draft([('narrator', 'He touched the door.'), ('narrator', 'Then he opened the door.'),
                            ('Mia', 'Come inside.')], [SoundEvent('hit', 'u2', phrase='door', when='with')])
        script = compile_script(draft, advanced=True)
        tree = parse_drama3_tree(script)
        scope = owner(tree, 'hit')
        self.assertEqual(len(turns(scope)), 2)
        self.assertEqual(next(sound for sound in iter_sound_nodes(scope) if sound.name == 'hit').occurrence, 2)
        _, spans, takes = render(script)
        narrator = [take for take in takes.values() if take.chunk.speaker == 'narrator']
        self.assertEqual(len(narrator), 1)
        self.assertEqual(len(narrator[0].chunk.lines), 2)
        hit = next(value for key, value in spans.items() if not isinstance(key, int))
        self.assertEqual(hit[0], 800)

    def test_music_wraps_its_span_and_action_keeps_a_smaller_scope(self):
        draft = self.draft([('Mia', 'We should wait.'), ('Tom', 'Open the box.'),
                            ('Mia', 'It is empty.'), ('Tom', 'Leave it there.'), ('Mia', 'Let us go.')],
            [SoundEvent('score', 'u2', phrase='Open', when='with', end_phrase='empty', end_at='u3'),
             SoundEvent('hit', 'u2', phrase='box', when='with')], bed=True)
        script = compile_script(draft, advanced=True)
        tree = parse_drama3_tree(script)
        self.assertEqual([t.text for t in turns(owner(tree, 'score'))], ['Open the box.', 'It is empty.'])
        self.assertEqual([t.text for t in turns(owner(tree, 'hit'))], ['Open the box.'])
        self.assertLess(script.index('We should wait.'), script.index('name="score"'))
        self.assertLess(script.index('name="score"'), script.index('Open the box.'))
        _, spans, _ = render(script)
        score = next(value for key, value in spans.items() if not isinstance(key, int) and value[2]['name'] == 'score')
        entrance = spans[1]
        self.assertEqual(score[0], entrance[0])

    def test_music_end_line_selects_later_repeated_phrase(self):
        draft = self.draft([('Mia', 'Start now.'), ('Tom', 'Wait here.'),
                            ('Mia', 'A little longer.'), ('Tom', 'Wait here.')],
            [SoundEvent('score', 'u1', phrase='Start', when='with', end_phrase='Wait here', end_at='u4')])
        script = compile_script(draft, advanced=True)
        cue = next(sound for sound in iter_sound_nodes(parse_drama3_tree(script)) if sound.name == 'score')
        self.assertEqual(cue.end_occurrence, 2)
        self.assertIn('end_occurrence="2"', script)
        serialized = parse_sound_tag('<sound ' + ' '.join(sound_placement_attrs(cue)) + '>Score</sound>')
        self.assertEqual(serialized.end_occurrence, 2)
        scored, spans, _ = render(script)
        plain, _, _ = render(compile_script(self.draft(
            [('Mia', 'Start now.'), ('Tom', 'Wait here.'), ('Mia', 'A little longer.'), ('Tom', 'Wait here.')],
            []), advanced=True))
        start, end, _ = spans[2]  # the score must survive beyond the first "Wait here"
        self.assertGreater(np.max(np.abs(scored[start:end] - plain[start:end])), .0001)

    def test_schema_uses_line_local_start_and_end_occurrences(self):
        _, plans, _ = parse_staging(json.dumps({'sounds': [], 'scenes': [{'scene': 'c1', 'events': [
            {'sound': 'score', 'at': 'u2', 'phrase': 'wait', 'occurrence': 2,
             'end_at': 'u4', 'end_phrase': 'done', 'end_occurrence': 2}]}]}),
            {'c1'}, {'u2', 'u4'}, [])
        cue = plans['c1'].events[0]
        self.assertEqual((cue.occurrence, cue.end_at, cue.end_occurrence), (2, 'u4', 2))

    def test_wrong_line_phrase_does_not_fall_back_to_another_line(self):
        draft = self.draft([('Mia', 'Open the door.'), ('Tom', 'Wait here.')],
                           [SoundEvent('hit', 'u2', phrase='door')])
        notes = []
        script = compile_script(draft, advanced=True, notes=notes)
        self.assertNotIn('name="hit"', script)
        self.assertTrue(any('unmatched sound anchor' in note for note in notes))

    def test_existing_hoisted_cues_move_locally_without_changing_their_targets(self):
        from h3_app.drama3.guided.local_staging import localize_existing_cues
        script = ('<parallel room="large_hall">\n'
                  '<sound name="hit" duration="1" anchor="door" edge="start" occurrence="2">Dry impact</sound>\n'
                  '<sequential>\nMia: Close the door.\nTom: Wait here.\nMia: Open the door.\n'
                  '</sequential>\n</parallel>')
        result = localize_existing_cues(script)
        scope = owner(parse_drama3_tree(result), 'hit')
        self.assertEqual([turn.text for turn in turns(scope)], ['Open the door.'])
        self.assertLess(result.index('Tom: Wait here.'), result.index('name="hit"'))
        cue = iter_sound_nodes(scope)[0]
        self.assertEqual(cue.occurrence, 1)
        self.assertIn('room="large_hall"', result)
        self.assertEqual(localize_existing_cues(result), result)
        original_audio, original_spans, original_takes = render(script)
        localized_audio, localized_spans, localized_takes = render(result)
        self.assertEqual(len(original_audio), len(localized_audio))
        for key, span in original_spans.items():
            if isinstance(key, int):
                self.assertEqual(localized_spans[key], span)
        original_hit = next(value for key, value in original_spans.items() if not isinstance(key, int))
        localized_hit = next(value for key, value in localized_spans.items() if not isinstance(key, int))
        self.assertEqual(original_hit[0], localized_hit[0])
        self.assertEqual(len(original_takes), len(localized_takes))

    def test_shared_reply_validation_localizes_single_pass_tagged_output(self):
        from h3_app.prompt_writers.drama3 import parse_drama3_reply, DRAMA3_SCRIPT_ADVANCED
        source = 'Mia: Close the door.\nTom: Wait here.\nMia: Open the door.'
        reply = json.dumps({'script': '<parallel>\n'
            '<sound name="hit" anchor="door" occurrence="2">Dry impact</sound>\n'
            '<sequential>\n' + source + '\n</sequential>\n</parallel>',
            'speakers': [{'speaker_id': index, 'name': name, 'description': 'A female voice: calm.',
                          'preview_text': 'We have time to walk to the station.'}
                         for index, name in enumerate(['Mia', 'Tom'])]})
        script, _ = parse_drama3_reply(reply, source, None, 'Auto', require_cast=True,
                                      script_mode=DRAMA3_SCRIPT_ADVANCED)
        scope = owner(parse_drama3_tree(script), 'hit')
        self.assertEqual([turn.text for turn in turns(scope)], ['Open the door.'])
        self.assertLess(script.index('<|speaker:1|> Wait here.'), script.index('name="hit"'))


if __name__ == '__main__':
    unittest.main()
