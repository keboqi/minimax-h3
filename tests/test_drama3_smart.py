"""Smart Guided planning, bounded requests and continuity contracts; no remote calls."""

import json
import re
import unittest
from pathlib import Path

from h3_app.drama3.guided.model import Scene, SoundEvent, Staging, Utterance
from h3_app.drama3.guided.schema import StageError, parse_staging
from h3_app.drama3.guided.segment import SCRIPT, segment_source
from h3_app.drama3.guided.smart import parse_plan
from h3_app.drama3.guided.batching import merge_staging, scene_batches, windows
from h3_app.prompt_writers.drama3_guided import guided_prompt_paths, write_drama3_guided
from h3_app.prompt_writers.drama3 import DRAMA3_SPEAKER_AUTO

ROOT = Path(__file__).resolve().parents[1]


class SmartContracts(unittest.TestCase):
    def test_large_scene_is_bounded_without_changing_scene_identity(self):
        lines = [Utterance(f'u{i}', 'Mia', f'Line {i}.') for i in range(120)]
        batches = scene_batches([Scene('c1', tuple(u.id for u in lines))], {u.id: u for u in lines}, 16)
        self.assertEqual([len(b[0].utterances) for b in batches], [16] * 7 + [8])
        self.assertEqual({b[0].id for b in batches}, {'c1'})
        self.assertEqual([uid for b in batches for uid in b[0].utterances], [u.id for u in lines])

    def test_character_budget_and_indivisible_source(self):
        values = ['x' * 3000, 'y' * 3000, 'z' * 6000]
        self.assertEqual(list(windows(values, lambda s: s)), [[v] for v in values])

    def test_scene_map_requires_order_coverage_and_real_ids(self):
        source = segment_source('Mia: Hello.\nMia: Goodbye.', SCRIPT)
        for starts in (['s2'], ['s1', 's1'], ['s2', 's1'], ['s1', 'missing'], [['s1']]):
            with self.subTest(starts=starts), self.assertRaises(StageError):
                parse_plan(json.dumps({'cast': [{'name': 'Mia'}],
                                      'scenes': [{'from': s, 'change': 'leave'} for s in starts]}), source)

    def test_staging_rejects_cross_scene_and_context_only_references(self):
        for field, item in [('events', {'at': 'u2'}), ('events', {'at': 'u1', 'end_at': 'u2'}),
                            ('pauses', {'before': 'u2'}), ('voices', {'id': 'u2'})]:
            with self.subTest(field=field), self.assertRaises(StageError):
                parse_staging(json.dumps({'sounds': [], 'scenes': [{'scene': 'c1', field: [item]}, {'scene': 'c2'}]}),
                              {'c1', 'c2'}, {'u1', 'u2'}, [], scene_lines={'c1': {'u1'}, 'c2': {'u2'}})

    def test_merge_preserves_scene_bed_and_all_window_events(self):
        first = Staging(ambience='rain', room='large_hall', transition='establish',
                        events=(SoundEvent('bell', 'u1'),), pauses=(('u2', 'short'),))
        later = Staging(events=(SoundEvent('door', 'u17'),), pauses=(('u18', 'long'),))
        result = merge_staging(first, later)
        self.assertEqual(result.ambience, 'rain')
        self.assertEqual(result.transition, 'establish')
        self.assertEqual([event.at for event in result.events], ['u1', 'u17'])
        self.assertEqual(result.pauses, (('u2', 'short'), ('u18', 'long')))

    def test_duplicate_scene_plans_cannot_overwrite_events(self):
        with self.assertRaises(StageError):
            parse_staging(json.dumps({'sounds': [], 'scenes': [
                {'scene': 'c1', 'events': [{'sound': 'bell', 'at': 'u1'}]}, {'scene': 'c1'}]}),
                {'c1'}, {'u1'}, [], scene_lines={'c1': {'u1'}})

    def test_later_window_cannot_reposition_earlier_speech(self):
        first = Staging(positions=(('Mia', -0.5),))
        later = Staging(positions=(('Mia', 0.8), ('Tom', 0.3)))
        result = merge_staging(first, later)
        self.assertEqual(dict(result.positions), {'Mia': -0.5, 'Tom': 0.3})

    def test_brief_retains_beats_from_each_part(self):
        from h3_app.drama3.guided.smart import brief_content
        from h3_app.prompt_writers.drama3_guided import _Calls
        plan = {'cast': [{'name': 'Mia'}], 'scenes': [
            {'place': 'station', 'change': 'departure', 'parts': ['Wait.', 'Leave.']}]}
        replies = iter([{'scenes': [{'lines': [{'speaker': 'Mia', 'text': phrase}],
                                     'beats': [{'at': phrase, 'focus': 'Mia', 'intent': intent}]}]}
                        for phrase, intent in [('Wait.', 'uncertain'), ('Leave.', 'resolved')]])
        _, _, scenes, _ = brief_content(plan, _Calls(lambda *_: (json.dumps(next(replies)), 'ok')),
                                        Path('unused'), '', 16, [])
        self.assertEqual(scenes[0].beats, (('Wait.', 'Mia', 'uncertain'), ('Leave.', 'Mia', 'resolved')))

    def run_smart(self, text, plan, *, reply_hook=None, **options):
        requests = []
        def ask(path, request):
            stage = path.stem.removeprefix('prompt_drama3_guided_')
            requests.append((stage, request))
            if stage == 'plan':
                answer = plan
            elif stage == 'preserve':
                answer = {'speakers': {}}
            elif stage == 'brief':
                target = request.split('Write ONLY this part: ', 1)[1].split('\n', 1)[0]
                answer = {'scenes': [{'lines': [{'speaker': 'Mia', 'text': target}]}]}
            elif stage == 'direction':
                answer = {'lines': []}
            elif stage == 'voices':
                names = re.findall(r'^- ([^(:]+)', request, re.M)
                answer = {'voices': [{'name': name.strip(),
                    'description': 'A female voice, calm and clear. Clean studio recording.',
                    'preview_text': 'We have plenty of time to get there this morning. I will bring the map and check '
                                    'the route before we leave. We can stop somewhere quiet along the way.'}
                                    for name in names]}
            elif stage == 'sound':
                target = request.split('TARGET:\n', 1)[1]
                targets = re.findall(r'Scene (c\d+)\.[^\n]*\n(u\d+) ', target)
                answer = {'sounds': [{'name': 'bell', 'kind': 'effect', 'prompt': 'One dry bell strike'}]}
                answer['scenes'] = [{'scene': sid, 'events': [{'sound': 'bell', 'at': uid, 'when': 'before'}]}
                                    for sid, uid in targets]
            else:
                self.fail(f'Unexpected stage {stage}')
            if reply_hook is not None:
                answer = reply_hook(stage, request, answer)
                if answer is None:
                    return '', 'Prompt enhancement failed: HTTP 500'
            return json.dumps(answer), 'ok'
        notes = []
        kwargs = dict(speaker_mode=DRAMA3_SPEAKER_AUTO, advanced=True, smart=True, max_batch_lines=16,
                      cast_context='[{"speaker_id":0,"name":"Mia","has_reference":true}]', notes=notes)
        kwargs.update(options)
        result = write_drama3_guided(text, ask, guided_prompt_paths(ROOT / 'prompts'), **kwargs)
        return json.loads(result[0]), requests, notes

    def test_preserve_splits_before_casting_and_keeps_all_lines(self):
        source = '\n'.join(f'Mia: This is line {i}.' for i in range(35))
        result, requests, notes = self.run_smart(source, {
            'title': 'A long scene', 'cast': [{'name': 'Mia', 'role': 'calm traveler'}],
            'scenes': [{'from': 's1', 'place': 'station', 'change': 'waiting ends'}]})
        self.assertEqual([s for s, _ in requests].count('preserve'), 3)
        self.assertEqual([s for s, _ in requests].count('direction'), 3)
        self.assertEqual([s for s, _ in requests].count('sound'), 3)
        for i in range(35):
            self.assertIn(f'This is line {i}.', result['script'])
        for stage, request in requests:
            if stage == 'preserve':
                target = request.split('TARGET units:\n')[1].split('\nReturn only')[0]
                self.assertLessEqual(len(target.splitlines()), 16)
                self.assertIn('Shared story plan', request)
        self.assertEqual(result['script'].count('name="bell"'), 3)
        self.assertEqual(notes, [])

    def test_brief_writes_only_one_planned_part_at_a_time(self):
        result, requests, notes = self.run_smart('Write a scene about Mia leaving a station.', {
            'title': 'Departure', 'cast': [{'name': 'Mia'}],
            'scenes': [{'place': 'station', 'change': 'Mia leaves',
                        'parts': ['The train has arrived.', 'It is time to leave.']}]})
        writing = [r for s, r in requests if s == 'brief']
        self.assertEqual(len(writing), 2)
        self.assertIn('Previous spoken lines (do not repeat):\nMia: The train has arrived.', writing[1])
        self.assertIn('The train has arrived.', result['script'])
        self.assertIn('It is time to leave.', result['script'])
        self.assertEqual(result['source_mode'], 'brief')

    def test_brief_oversized_output_retries_only_the_current_part(self):
        from h3_app.drama3.guided.smart import brief_content
        from h3_app.prompt_writers.drama3_guided import _Calls
        plan = {'title': 'Departure', 'cast': [{'name': 'Mia'}],
                'scenes': [{'place': 'station', 'change': 'leave', 'parts': ['Say goodbye.']}]}
        requests = []
        def ask(path, request):
            requests.append(request)
            lines = [{'speaker': 'Mia', 'text': 'Goodbye.'}] * (17 if len(requests) == 1 else 1)
            return json.dumps({'scenes': [{'lines': lines}]}), 'ok'
        _, lines, scenes, _ = brief_content(plan, _Calls(ask), Path('unused'), '', 16, [])
        self.assertEqual(len(requests), 2)
        self.assertIn('1–16 lines', requests[1])
        self.assertEqual(len(lines), 1)
        self.assertEqual(scenes[0].utterances, ('u1',))

    def test_brief_part_can_open_with_an_interrupt(self):
        from h3_app.drama3.guided.smart import brief_content
        from h3_app.prompt_writers.drama3_guided import _Calls
        plan = {'cast': [{'name': 'Mia'}, {'name': 'Tom'}], 'scenes': [
            {'change': 'argument', 'parts': ['Mia starts.', 'Tom cuts in.']}]}
        replies = iter([{'scenes': [{'lines': [{'speaker': 'Mia', 'text': 'I was going to say'}]}]},
                        {'scenes': [{'lines': [{'speaker': 'Tom', 'text': 'No, listen.', 'type': 'interrupt'}]}]}])
        _, lines, scenes, _ = brief_content(plan, _Calls(lambda *_: (json.dumps(next(replies)), 'ok')),
                                            Path('unused'), '', 16, [])
        self.assertEqual([(u.id, u.speaker, u.kind) for u in lines],
                         [('u1', 'Mia', 'line'), ('u2', 'Tom', 'interrupt')])
        self.assertEqual(scenes[0].utterances, ('u1', 'u2'))

    def test_regular_guided_drops_a_cross_scene_event_instead_of_failing(self):
        sounds, staging, _ = parse_staging(json.dumps({'sounds': [], 'scenes': [
            {'scene': 'c1', 'events': [{'sound': 'bell', 'at': 'u2'}]}, {'scene': 'c2'}]}),
            {'c1', 'c2'}, {'u1', 'u2'}, [])
        self.assertEqual(set(staging), {'c1', 'c2'})

    def test_existing_staging_is_retained_without_sound_requests(self):
        text = '<parallel room="large_hall">\nMia: We are home.\n</parallel>'
        result, requests, _ = self.run_smart(text, {
            'title': 'Home', 'cast': [{'name': 'Mia'}],
            'scenes': [{'from': 's1', 'place': 'hall', 'change': 'arrival'}]})
        self.assertNotIn('sound', [stage for stage, _ in requests])
        self.assertIn('room="large_hall"', result['script'])

    def test_smart_selection_uses_guided_path_and_retains_failure_contract(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from h3_app.prompt_writers import drama3 as writer
        runtime = SimpleNamespace(prompt_systems={'Drama 3': ROOT / 'prompts' / 'prompt_drama3.txt'})
        with patch.object(writer.common, 'write_prompt', return_value=('{}', 'ok')) as remote:
            script, status = writer.enhance_drama3_text('Mia: Hello.', 'test', '', runtime=runtime,
                                                       workflow=writer.DRAMA3_WORKFLOW_SMART)
        self.assertEqual(script, 'Mia: Hello.')
        self.assertTrue(status.startswith(writer.DRAMA3_PROMPT_FAILED), status)
        self.assertEqual(remote.call_count, 2)
        self.assertTrue(all(c.kwargs['system_path'].name == 'prompt_drama3_guided_plan.txt'
                            for c in remote.call_args_list))

    def test_budget_auto_uses_active_model_and_override_wins(self):
        from h3_app.drama3.guided.budget import resolve_budget, REQUESTS, TOKENS
        self.assertEqual(resolve_budget(model='gemini-3.5-flash-lite').name, REQUESTS)
        self.assertEqual(resolve_budget(model='gemini-3.5-flash-lite', backend='OpenAI-compatible',
                                        openai_model='gpt-6-luna').name, TOKENS)
        self.assertEqual(resolve_budget(REQUESTS, backend='OpenAI-compatible', openai_model='gpt-6-luna').name, REQUESTS)
        self.assertEqual(resolve_budget(TOKENS, model='gemini-3.5-flash-lite').name, TOKENS)
        self.assertEqual(resolve_budget(model='custom-model').name, TOKENS)
        self.assertEqual(resolve_budget(backend='OpenAI-compatible', openai_model='google/gemini-3.5-flash-lite').name,
                         REQUESTS)

    def test_profiles_group_short_scenes_and_preserve_scene_identity(self):
        from h3_app.drama3.guided.budget import REQUEST_BUDGET, TOKEN_BUDGET
        source = '\n'.join(f'Mia: This is line {i}.' for i in range(30))
        plan = {'title': 'Journey', 'cast': [{'name': 'Mia'}],
                'scenes': [{'from': f's{i + 1}', 'place': f'Place {i}', 'change': 'Mia moves'} for i in (0, 10, 20)]}
        runs = [self.run_smart(source, plan, max_batch_lines=40, smart_budget=b) for b in (REQUEST_BUDGET, TOKEN_BUDGET)]
        counts = [[s for s, _ in calls].count('sound') for _, calls, _ in runs]
        self.assertEqual(counts, [1, 2])
        for result, calls, notes in runs:
            self.assertEqual(result['script'].count('name="bell"'), 3)
            self.assertEqual(notes, [])
            for i in range(30):
                self.assertIn(f'This is line {i}.', result['script'])

    def test_failed_opening_window_does_not_promote_later_scene_settings(self):
        def reply(stage, request, answer):
            if stage == 'sound':
                if answer['scenes'][0]['events'][0]['at'] == 'u1':
                    return None
                answer['scenes'][0]['room'] = 'large_hall'
                answer['scenes'][0]['transition'] = 'establish'
            return answer
        result, _, notes = self.run_smart('\n'.join(f'Mia: Line number {i}.' for i in range(20)),
            {'cast': [{'name': 'Mia'}], 'scenes': [{'from': 's1', 'change': 'waiting'}]}, reply_hook=reply)
        self.assertNotIn('room="large_hall"', result['script'])
        self.assertIn('name="bell"', result['script'])
        self.assertTrue(any('ignored scene-wide' in note for note in notes))

    def test_grouped_continuation_does_not_suppress_new_scene_settings(self):
        def reply(stage, request, answer):
            if stage == 'sound':
                for scene in answer['scenes']:
                    scene['room'] = 'large_hall'
                if 'Continuation windows:' in request:
                    self.assertIn('For these scene IDs ONLY', request)
            return answer
        result, calls, notes = self.run_smart('\n'.join(f'Mia: Line number {i}.' for i in range(26)),
            {'cast': [{'name': 'Mia'}], 'scenes': [{'from': 's1', 'change': 'waiting'},
                                                  {'from': 's21', 'change': 'arriving'}]}, reply_hook=reply)
        self.assertEqual([stage for stage, _ in calls].count('sound'), 2)
        self.assertEqual(result['script'].count('room="large_hall"'), 2)
        self.assertEqual(sum('ignored scene-wide' in note for note in notes), 1)

    def test_token_context_is_selective_without_truncating_relevant_profiles(self):
        from h3_app.drama3.guided.budget import REQUEST_BUDGET, TOKEN_BUDGET
        from h3_app.drama3.guided.smart import context
        plan = {'title': 'Journey', 'summary': 'Mia returns home.',
                'cast': [{'name': 'Mia', 'role': 'A quiet traveler with a gravelly voice.'},
                         {'name': 'Other', 'role': 'Unrelated profile'}],
                'scenes': [{'place': 'other', 'change': 'UNRELATED DETAIL' * 200}]}
        small = context(plan, TOKEN_BUDGET, scenes=[{'place': 'home', 'change': 'arrival'}], names={'Mia'})
        rich = context(plan, REQUEST_BUDGET)
        self.assertLess(len(small), len(rich))
        self.assertIn('A quiet traveler with a gravelly voice.', small)
        self.assertNotIn('UNRELATED DETAIL', small)
        self.assertNotIn('Unrelated profile', small)

    def test_dense_content_gets_smaller_windows(self):
        from h3_app.drama3.guided.budget import REQUEST_BUDGET
        lines = [Utterance(f'u{i}', 'Mia', 'Long source text. ' * 30) for i in range(32)]
        batches = scene_batches([Scene('c1', tuple(u.id for u in lines))], {u.id: u for u in lines}, 32, REQUEST_BUDGET)
        self.assertGreater(len(batches), 1)
        self.assertEqual([uid for b in batches for s in b for uid in s.utterances], [u.id for u in lines])

    def test_profiles_batch_voice_design(self):
        from h3_app.drama3.guided.budget import REQUEST_BUDGET, TOKEN_BUDGET
        names = ['Mia', 'Tom', 'Nora', 'Kai']
        plan = {'title': 'Meeting', 'cast': [{'name': n} for n in names],
                'scenes': [{'from': 's1', 'place': 'hall', 'change': 'a meeting'}]}
        for budget, expected in ((REQUEST_BUDGET, 1), (TOKEN_BUDGET, 2)):
            with self.subTest(profile=budget.name):
                result, calls, _ = self.run_smart('\n'.join(f'{n}: Hello there.' for n in names), plan,
                                                 cast_context='', smart_budget=budget)
                self.assertEqual([s for s, _ in calls].count('voices'), expected)
                self.assertEqual([s['name'] for s in result['speakers']], names)

    def test_public_writer_passes_resolved_budget_and_reports_it(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from h3_app.prompt_writers import drama3 as writer
        from h3_app.drama3.guided.budget import REQUESTS, TOKENS
        reply = json.dumps({'title': '', 'script': 'Mia: Hello.', 'source_mode': 'preserve',
                            'speakers': [{'speaker_id': 0, 'name': 'Mia', 'description': 'A female voice, calm.',
                                          'preview_text': 'We have plenty of time to get there this morning. '
                                                          'I will bring the map and check the route before we leave.'}]})
        runtime = SimpleNamespace(prompt_systems={'Drama 3': ROOT / 'prompts' / 'prompt_drama3.txt'})
        for override, expected in [('Auto', TOKENS), (REQUESTS, REQUESTS)]:
            with patch('h3_app.prompt_writers.drama3_guided.write_drama3_guided', return_value=(reply, 'preserve', 3)) as run:
                _, status = writer.enhance_drama3_text('Mia: Hello.', 'gemini-3.5-flash-lite', '', runtime=runtime,
                    backend='OpenAI-compatible', openai_model='gpt-6-luna', workflow=writer.DRAMA3_WORKFLOW_SMART,
                    smart_budget_profile=override)
            self.assertEqual(run.call_args.kwargs['smart_budget'].name, expected)
            self.assertIn(f'budget: {expected}', status)

    def test_ui_handler_preserves_old_arguments_and_accepts_override(self):
        from unittest.mock import Mock, patch
        from h3_ui.drama3.writer import WriterHandlers
        from h3_app.prompt_writers import drama3 as writer
        handler = WriterHandlers()
        handler._drama3_writer_group = Mock(return_value=None)
        handler._drama3_writer_slots = Mock(return_value=([], {}, '', 4))
        handler._drama3_writer_mode_and_sounds = Mock(return_value=('Basic', ''))
        handler._runtime_config = Mock(return_value=None)
        handler._drama3_apply_cast = Mock(return_value=('applied',))
        for suffix, expected in [((), 'Auto'), (('Request optimized',), 'Request optimized')]:
            with patch.object(writer, 'enhance_drama3_text', return_value=('Mia: Hi.', 'ok', [], '')) as call:
                handler.enhance_drama3_text('Mia: Hi.', 'Auto', 'gemini-3.5-flash-lite', '',
                    'OpenAI-compatible', '', 'gpt-6-luna', '', 'slot-marker', writer.DRAMA3_WORKFLOW_SMART,
                    20, True, *suffix)
            handler._drama3_writer_slots.assert_called_with(('slot-marker',))
            self.assertEqual(call.call_args.kwargs['smart_budget_profile'], expected)
            self.assertEqual(call.call_args.kwargs['workflow'], writer.DRAMA3_WORKFLOW_SMART)
            self.assertEqual(call.call_args.kwargs['max_batch_lines'], 20)
            self.assertTrue(call.call_args.kwargs['strict_source'])


if __name__ == '__main__':
    unittest.main()
