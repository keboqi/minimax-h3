"""Regressions for source fidelity and executable Guided acting choices."""

import json
from pathlib import Path
import unittest

from h3_app.drama3.guided.compile import compile_script
from h3_app.drama3.guided.fidelity import validate_source_fidelity
from h3_app.drama3.guided.model import Direction, Draft, Scene, Sound, SoundEvent, Staging, Utterance
from h3_app.drama3.guided.schema import StageError, parse_content_preserve, parse_direction, parse_staging, parse_voices
from h3_app.drama3.guided.segment import BRIEF, PROSE, SCRIPT, input_kind, segment_source, source_request
from h3_app.errors import H3Error
from h3_app.prompt_writers.drama3 import DRAMA3_SPEAKER_AUTO, DRAMA3_SPEAKER_SINGLE, parse_drama3_reply
from h3_app.prompt_writers.drama3_guided import guided_prompt_paths, write_drama3_guided


PREVIEW = ('We have plenty of time to walk to the station this morning. I will bring a map and '
           'check the route before we leave. We can find a quiet place to sit along the way.')
PROMPTS = guided_prompt_paths(Path(__file__).resolve().parents[1] / 'prompts')


class WriterIntegrityTests(unittest.TestCase):
    def test_long_english_attributions_stay_spoken_narration(self):
        source = 'She pressed the button and said: "Run!"\nHe shook his head and replied: "No."'
        self.assertEqual(input_kind(source), PROSE)
        units = segment_source(source).units
        self.assertEqual(units[0].text, 'She pressed the button and said:')
        self.assertEqual(units[2].text, 'He shook his head and replied:')
        with self.assertRaises(H3Error):
            validate_source_fidelity(source, '<|speaker:0|> Run!\n<|speaker:1|> No.')
        validate_source_fidelity(source, '<|speaker:0|> She pressed the button and said:\n<|speaker:1|> Run!\n'
                                 '<|speaker:0|> He shook his head and replied:\n<|speaker:2|> No.')
        self.assertEqual(input_kind('Commander Alex Morgan Reyes: Stay here.'), SCRIPT)
        self.assertEqual(input_kind('Said: Stay here.'), SCRIPT)

    def test_spoken_expand_is_not_a_request_to_insert_or_remove_words(self):
        source = 'Expand your chest and breathe in.\nRelax your shoulders and breathe out.'
        self.assertEqual(source_request(source), (source, ''))
        with self.assertRaises(H3Error):
            validate_source_fidelity(source, '<|speaker:0|> Relax your shoulders and breathe out.')
        source = 'The instructor began.\nExpand with more dialogue.\nNobody moved.'
        self.assertEqual(source_request(source), (source, ''))
        self.assertEqual(source_request('Expand with more dialogue.\nMia: Stay here.'),
                         ('Mia: Stay here.', 'Expand with more dialogue.'))
        self.assertEqual(source_request('Mia: Stay here.\n请扩写以下故事。'),
                         ('Mia: Stay here.', '请扩写以下故事。'))

    def test_similar_real_words_and_names_are_not_spelling_repairs(self):
        for old, new in [('dead', 'deaf'), ('hate', 'have'), ('united', 'untied'), ('Alex', 'Axel'), ('Teh', 'the')]:
            with self.subTest(old=old, new=new), self.assertRaises(H3Error):
                validate_source_fidelity('Mia: I said ' + old + '.', '<|speaker:0|> I said ' + new + '.')
        source = segment_source('Mia: He is dead.')
        answer = {'cast': [{'name': 'Mia'}], 'speakers': {}, 'scenes': [],
                  'fixes': [{'unit': 's1', 'from': 'dead', 'to': 'deaf'}]}
        with self.assertRaises(StageError):
            parse_content_preserve(json.dumps(answer), source)

    def test_duplicate_quotes_require_explicit_joint_attribution_and_distinct_voices(self):
        source = '"Run now," Mia said.'
        for block in ['<|speaker:0|> Run now.\n<|speaker:0|> Run now.',
                      '<parallel>\n<|speaker:0|> Run now.\n<|speaker:1|> Run now.\n</parallel>']:
            with self.subTest(block=block), self.assertRaises(H3Error):
                validate_source_fidelity(source, block + '\n<|speaker:2|> Mia said.')
        source = '"Run now," they said together.'
        unison = '<parallel>\n<|speaker:0|> Run now.\n<|speaker:1|> Run now.\n</parallel>\n'
        validate_source_fidelity(source, unison + '<|speaker:2|> they said together.')
        with self.assertRaises(H3Error):
            validate_source_fidelity(source, unison.replace('speaker:1', 'speaker:0') +
                                     '<|speaker:2|> they said together.')
        source = '"Run now," they said together. "Keep going," Mia added.'
        with self.assertRaises(H3Error):
            validate_source_fidelity(source, unison + '<|speaker:2|> they said together.\n'
                '<parallel>\n<|speaker:0|> Keep going.\n<|speaker:1|> Keep going.\n</parallel>\n'
                '<|speaker:2|> Mia added.')

    def test_spoken_values_units_and_abbreviations_have_exact_equivalents(self):
        examples = [('$15', 'fifteen dollars'), ('$1.50', 'one dollar and fifty cents'),
                    ('£115', 'one hundred and fifteen pounds'), ('€2', 'two euros'),
                    ('15%', 'fifteen percent'), ('1kg', 'one kilogram'), ('5kg', 'five kilograms'),
                    ('-5', 'minus five'), ('1.25', 'one point two five'), ('007', 'zero zero seven'),
                    ('Dr. Smith', 'doctor Smith'), ('e.g. apples', 'for example apples')]
        for literal, spoken in examples:
            with self.subTest(literal=literal):
                validate_source_fidelity('Mia: It says ' + literal + '.', '<|speaker:0|> It says ' + spoken + '.')
                validate_source_fidelity('Mia: It says ' + literal + '.', '<|speaker:0|> It says ' + literal + '.')
        for literal, changed in [('$15', 'sixteen dollars'), ('$15', 'fifteen pounds'), ('$15', 'fifteen'),
                                 ('15%', 'fifteen'), ('-5', 'five'), ('007', 'seven')]:
            with self.subTest(literal=literal, changed=changed), self.assertRaises(H3Error):
                validate_source_fidelity('Mia: It says ' + literal + '.', '<|speaker:0|> It says ' + changed + '.')

    def test_short_supplied_lines_and_long_names_are_preserved(self):
        self.assertEqual(input_kind('The door was locked. Nobody answered.'), PROSE)
        self.assertEqual(input_kind('Mia: Stay here.'), SCRIPT)
        self.assertEqual(input_kind('Commander Alex Morgan Reyes: Stay here.'), SCRIPT)
        self.assertEqual(input_kind('I waited for 30 seconds.'), PROSE)
        self.assertEqual(input_kind('Two friends arguing about pizza, 30 seconds.'), BRIEF)

    def test_unknown_mode_cannot_disable_source_validation(self):
        reply = {'title': 'Door', 'source_mode': 'brief', 'script': 'Mia: The room was empty.',
                 'speakers': [{'speaker_id': 0, 'name': 'Mia', 'description': 'A female voice: calm.',
                               'preview_text': PREVIEW}]}
        with self.assertRaises(H3Error):
            parse_drama3_reply(json.dumps(reply), 'Mia: The room was empty. The key lay under the table.',
                               None, DRAMA3_SPEAKER_AUTO, require_cast=True)

    def test_order_numbers_and_known_speakers_are_validated(self):
        source = '<|speaker:0|> Bring the silver key.\n<|speaker:1|> Wait here for 30 seconds.'
        for bad in ['<|speaker:1|> Wait here for 30 seconds.\n<|speaker:0|> Bring the silver key.',
                    '<|speaker:1|> Bring the silver key.\n<|speaker:0|> Wait here for 30 seconds.',
                    '<|speaker:0|> Bring the silver key.\n<|speaker:1|> Wait here for 40 seconds.']:
            with self.subTest(bad=bad), self.assertRaises(H3Error):
                validate_source_fidelity(source, bad)
        validate_source_fidelity(source, source)
        validate_source_fidelity(source, source.replace('speaker:1', 'speaker:0'), single=True)

    def test_legitimate_controls_and_small_repairs_keep_source_words(self):
        validate_source_fidelity('Mia: Bring teh key to Siobhan.',
                                '<|speaker:0|> Bring [softly] the key to Siobhan{SH IH0 V AO1 N}.')
        with self.assertRaises(H3Error):
            validate_source_fidelity('Mia: Bring the silver key.', '<|speaker:0|> Bring the golden key.')
        with self.assertRaises(H3Error):
            validate_source_fidelity('Mia: No.\nMia: No.', '<|speaker:0|> No.')

    def test_chinese_homophone_slips_and_short_names_are_repairs(self):
        source = ('韩立走进去。众人自然知道韩竟是一名元婴级修士，都是不小地商号。\n'
                  '“怎么看起来这般年青？”韩立问道。')
        repaired = ('narrator: 韩立走进去。众人自然知道韩立竟是一名元婴级修士，都是不小的商号。\n'
                    '韩立: 怎么看起来这般年轻？\nnarrator: 韩立问道。')
        validate_source_fidelity(source, repaired)
        for changed in (repaired.replace('年轻', '年清'),      # same sound, but not a slip group
                        repaired.replace('韩立竟', '韩又竟'),  # an insertion that is no cast name
                        repaired.replace('不小的', '不大的')):  # a changed word
            with self.subTest(changed=changed), self.assertRaises(H3Error):
                validate_source_fidelity(source, changed)

    def test_important_short_narrator_action_keeps_requested_direction(self):
        lines = [Utterance('u1', 'Mia', 'We are safe now.'), Utterance('u2', 'narrator', 'The gun fired.')]
        draft = Draft(lines, [Scene('c1', ('u1', 'u2'))], directions={'u2': Direction(cue='sudden, stark')})
        self.assertIn('[sudden, stark]', compile_script(draft, advanced=False))
        draft.utterances[1] = Utterance('u2', 'narrator', 'she said.')
        self.assertNotIn('[sudden, stark]', compile_script(draft, advanced=False))

    def test_repeated_cues_survive_for_different_actors(self):
        lines = [Utterance('u1', 'Mia', 'Please stay here.'), Utterance('u2', 'Tom', 'I will stay here.')]
        draft = Draft(lines, [Scene('c1', ('u1', 'u2'))],
                      directions={u.id: Direction(cue='quiet, reassuring') for u in lines})
        self.assertEqual(compile_script(draft, advanced=False).count('[quiet, reassuring]'), 2)

    def test_missing_stage_envelopes_fail_but_explicit_restraint_is_valid(self):
        with self.assertRaises(StageError):
            parse_direction('{}', {'u1'}, [])
        with self.assertRaises(StageError):
            parse_staging('{}', {'c1'}, {'u1'}, [])
        with self.assertRaises(StageError):
            parse_staging('{"sounds":[],"scenes":[]}', {'c1'}, {'u1'}, [])
        self.assertEqual(parse_direction('{"lines":[]}', {'u1'}, []), {})
        sounds, scenes, _ = parse_staging('{"sounds":[],"scenes":[{"scene":"c1"}]}', {'c1'}, {'u1'}, [])
        self.assertFalse(sounds)
        self.assertIn('c1', scenes)

    def test_one_word_english_preview_requires_repair(self):
        voice = {'name': 'Mia', 'description': 'A female voice: calm.', 'preview_text': 'Okay.'}
        with self.assertRaises(StageError):
            parse_voices(json.dumps({'voices': [voice]}), ['Mia'])
        voice['preview_text'] = PREVIEW
        self.assertIn('Mia', parse_voices(json.dumps({'voices': [voice]}), ['Mia']))

    def test_source_linked_edits_and_expansion_require_valid_authorization(self):
        source = segment_source('Mia: Bring teh key to Siobhan.')
        answer = {'cast': [{'name': 'Mia'}], 'speakers': {}, 'scenes': [{'from': 's1'}],
                  'fixes': [{'unit': 's1', 'from': 'teh', 'to': 'the'}],
                  'pronunciations': [{'word': 'Siobhan', 'reading': 'SH IH0 V AO1 N'}],
                  'insertions': [{'after': 's1', 'speaker': 'Mia', 'text': 'I will wait here.'}]}
        with self.assertRaises(StageError):
            parse_content_preserve(json.dumps(answer), source)
        parsed = parse_content_preserve(json.dumps(answer), source, expansion=True)
        self.assertEqual(parsed['fixes'], [('s1', 'teh', 'the')])
        self.assertEqual(len(parsed['insertions']), 1)

    def test_advanced_round_trip_retains_controls_and_avoids_reference_redesign(self):
        source = ('<parallel room="large_hall">\n'
                  '<sound name="draft" loop="true" duration="25">Steady dry air hiss</sound>\n'
                  '<sequential>\nMia: Bring the key.\nMia: I will wait here.\n</sequential>\n</parallel>')
        requested = []
        def ask(path, request):
            stage = path.stem.removeprefix('prompt_drama3_guided_')
            requested.append(stage)
            if stage == 'preserve':
                return json.dumps({'title': 'Hall', 'cast': [{'name': 'Mia'}], 'speakers': {},
                                   'scenes': [{'from': 's1'}]}), 'ok'
            if stage == 'direction':
                return '{"lines":[{"id":"u1","cue":"quiet, urgent"}]}', 'ok'
            self.fail(f'unexpected stage {stage}')
        context = 'Slots:\n' + json.dumps([{'speaker_id': 3, 'name': 'Mia', 'has_reference': True,
                                            'description': 'A female voice: elderly, gravelly.'}])
        reply, mode, calls = write_drama3_guided(source, ask, PROMPTS, voices=((3, 'Mia'),),
            speaker_mode=DRAMA3_SPEAKER_SINGLE, advanced=True, cast_context=context)
        data = json.loads(reply)
        self.assertEqual((mode, calls), ('preserve', 2))
        self.assertIn('room="large_hall"', data['script'])
        self.assertIn('<sound name="draft" loop="true" duration="25">', data['script'])
        self.assertIn('[quiet, urgent]', data['script'])
        self.assertEqual(data['speakers'][0]['description'], 'A female voice: elderly, gravelly.')
        self.assertEqual(set(requested), {'preserve', 'direction'})

    def test_performance_annotation_is_control_while_prose_aside_stays_spoken(self):
        units = segment_source('Mia: (sighs) I am fine (for now).').units
        self.assertEqual(units[0].text, '[sighs] I am fine (for now).')

    def test_unanchored_effect_layers_are_simultaneous(self):
        from h3_app.drama3.serving.tree import ParallelNode, parse_drama3_tree
        draft = Draft([Utterance('u1', 'Mia', 'The window broke.')], [Scene('c1', ('u1',))],
                      sounds={'hit': Sound('hit', 'effect', 'Metal collision'),
                              'glass': Sound('glass', 'effect', 'Falling glass')},
                      staging={'c1': Staging(events=(SoundEvent('hit', 'u1'), SoundEvent('glass', 'u1')))})
        tree = parse_drama3_tree(compile_script(draft, advanced=True))
        self.assertIsInstance(tree.children[0], ParallelNode)
        self.assertEqual(len(tree.children[0].children), 3)

    def test_expansion_corrections_and_pronunciations_compile_together(self):
        source = 'Expand with more dialogue.\nMia: Bring teh key to Siobhan.\nMia: Wait here.'
        def ask(path, request):
            stage = path.stem.removeprefix('prompt_drama3_guided_')
            if stage == 'preserve':
                self.assertIn('Explicit expansion instructions:', request)
                return json.dumps({'title': 'Key', 'cast': [{'name': 'Mia'}], 'speakers': {},
                    'scenes': [{'from': 's1'}], 'fixes': [{'unit': 's1', 'from': 'teh', 'to': 'the'}],
                    'pronunciations': [{'word': 'Siobhan', 'reading': 'SH IH0 V AO1 N'}],
                    'insertions': [{'after': 's1', 'speaker': 'Mia', 'text': 'I will be back soon.'}]}), 'ok'
            if stage == 'voices':
                return json.dumps({'voices': [{'name': 'Mia', 'description': 'A female voice: calm.',
                                               'preview_text': PREVIEW}]}), 'ok'
            return '{"lines":[]}', 'ok'
        reply, _, _ = write_drama3_guided(source, ask, PROMPTS, speaker_mode=DRAMA3_SPEAKER_AUTO, advanced=False)
        script, _ = parse_drama3_reply(reply, source, None, DRAMA3_SPEAKER_AUTO, require_cast=True)
        self.assertIn('the key to Siobhan{SH IH0 V AO1 N}', script)
        self.assertLess(script.index('Siobhan'), script.index('I will be back soon.'))
        self.assertLess(script.index('I will be back soon.'), script.index('Wait here.'))

    def test_malformed_optional_stage_retries_and_reports_degradation(self):
        counts = {}
        notes = []
        def ask(path, request):
            stage = path.stem.removeprefix('prompt_drama3_guided_')
            counts[stage] = counts.get(stage, 0) + 1
            if stage == 'preserve':
                return '{"cast":[{"name":"Mia"}],"speakers":{},"scenes":[{"from":"s1"}]}', 'ok'
            if stage == 'voices':
                return json.dumps({'voices': [{'name': 'Mia', 'description': 'A female voice: calm.',
                                               'preview_text': PREVIEW}]}), 'ok'
            return '{}', 'ok'
        reply, _, _ = write_drama3_guided('Mia: Stay right here.', ask, PROMPTS,
            speaker_mode=DRAMA3_SPEAKER_AUTO, advanced=True, notes=notes)
        self.assertEqual(counts['direction'], 2)
        self.assertEqual(counts['sound'], 2)
        self.assertTrue(any('left undirected' in note for note in notes))
        self.assertTrue(any('without added sound' in note for note in notes))
        self.assertIn('Stay right here.', json.loads(reply)['script'])

    def test_single_reference_keeps_scene_analysis(self):
        seen = []
        def ask(path, request):
            stage = path.stem.removeprefix('prompt_drama3_guided_')
            seen.append((stage, request))
            if stage == 'preserve':
                return json.dumps({'cast': [{'name': 'Mia'}], 'speakers': {},
                    'scenes': [{'from': 's1', 'place': 'hall'}, {'from': 's2', 'place': 'garden'}]}), 'ok'
            if stage == 'sound':
                return '{"sounds":[],"scenes":[{"scene":"c1"},{"scene":"c2"}]}', 'ok'
            return '{"lines":[]}', 'ok'
        reply, _, _ = write_drama3_guided('Mia: We are in the hall.\nMia: Later we reached the garden.', ask, PROMPTS,
            voices=((0, 'Mia'),), speaker_mode=DRAMA3_SPEAKER_SINGLE, advanced=True)
        self.assertIn('<#0.8#>', json.loads(reply)['script'])
        self.assertIn('Scene c2. Place: garden', dict(seen)['direction'])
        self.assertNotIn('voices', dict(seen))

    def test_conflicting_batch_assets_keep_their_own_descriptions(self):
        def ask(path, request):
            stage = path.stem.removeprefix('prompt_drama3_guided_')
            if stage == 'preserve':
                return json.dumps({'cast': [{'name': 'Mia'}], 'speakers': {},
                    'scenes': [{'from': 's1'}, {'from': 's11'}]}), 'ok'
            if stage == 'sound':
                scene = 'c1' if 'Scene c1.' in request else 'c2'
                return json.dumps({'sounds': [{'name': 'wind', 'kind': 'ambience',
                    'prompt': 'Steady soft wind' if scene == 'c1' else 'Harsh howling wind'}],
                    'scenes': [{'scene': scene, 'ambience': 'wind'}]}), 'ok'
            return '{"lines":[]}', 'ok'
        notes = []
        source = '\n'.join(f'Mia: This is line number {i}.' for i in range(20))
        reply, _, _ = write_drama3_guided(source, ask, PROMPTS, voices=((0, 'Mia'),),
            speaker_mode=DRAMA3_SPEAKER_SINGLE, advanced=True, max_batch_lines=10, notes=notes)
        script = json.loads(reply)['script']
        self.assertIn('name="wind"', script)
        self.assertIn('name="wind_2"', script)
        self.assertIn('Steady soft wind', script)
        self.assertIn('Harsh howling wind', script)
        self.assertTrue(any('Renamed conflicting sound' in note for note in notes))

    def test_overlapping_pronunciation_mappings_keep_existing_notes(self):
        from h3_app.drama3.guided.enhance import add_pronunciations
        text = '单于{chan2 yu2}与单先生。单于走了。'
        self.assertEqual(add_pronunciations(text, {'单于': 'chan2 yu2', '单': 'shan4'}),
                         '单于{chan2 yu2}与单{shan4}先生。单于{chan2 yu2}走了。')

    def test_chinese_word_repair_and_invalid_edit_types(self):
        source = segment_source('Mia: 我以经到了。')
        answer = {'cast': [{'name': 'Mia'}], 'speakers': {}, 'scenes': [],
                  'fixes': [{'unit': 's1', 'from': '以经', 'to': '已经'}]}
        self.assertEqual(parse_content_preserve(json.dumps(answer), source)['fixes'], [('s1', '以经', '已经')])
        answer['fixes'][0]['unit'] = []
        with self.assertRaises(StageError):
            parse_content_preserve(json.dumps(answer), source)


if __name__ == '__main__':
    unittest.main()
