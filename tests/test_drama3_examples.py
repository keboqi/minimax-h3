"""Verify that all three Soundstage examples are ready, locally staged and executable."""

import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from h3_app.drama3.cast import parse_script_cast_and_sounds, voices_without_sex
from h3_app.drama3.example_story import example_story, example_story_reply_json
from h3_app.drama3.guided.local_staging import localize_existing_cues, phrase_count
from h3_app.drama3.lint import lint_drama3_script, needs_direction
from h3_app.drama3.serving import audio, generate
from h3_app.drama3.serving.planner import plan_turns_chunks, spoken_text
from h3_app.drama3.serving.tree import (
    ParallelNode,
    iter_sound_nodes,
    iter_speech_nodes,
    parse_drama3_tree,
)
from h3_app.drama3.sound_policy import PERSPECTIVES, SOUND_ROLES
from h3_app.drama3.sound_report import sound_density_report

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _iter_parallels(node):
    if isinstance(node, ParallelNode):
        yield node
    for child in getattr(node, "children", ()):
        yield from _iter_parallels(child)


def _is_scene_master(node):
    return isinstance(node, ParallelNode) and (
        bool(node.room or node.stage)
        or any(
            child in node.children and getattr(child, "loop", False) and not getattr(child, "anchor", "")
            for child in node.children
        )
    )


def _node_speech(node):
    return "\n".join(turn.text for speech in iter_speech_nodes(node) for turn in speech.turns)


def _render_example(raw_script, *, rate=8000):
    tree = parse_drama3_tree(raw_script)
    chunks_by_node, takes = {}, {}
    for speech in iter_speech_nodes(tree):
        chunks = plan_turns_chunks(speech.turns, start_index=len(takes))
        chunks_by_node[id(speech)] = chunks
        for chunk in chunks:
            bounds, words, cursor = [], [], 0
            for line in chunk.lines:
                spoken = spoken_text(line)
                tokens = list("".join(spoken.split())) if _CJK_RE.search(spoken) else spoken.split()
                start = cursor
                timed = []
                for token in tokens:
                    step = 320 if _CJK_RE.search(token) else 1600
                    timed.append((token, cursor / rate, (cursor + step) / rate))
                    cursor += step
                bounds.append((start, cursor))
                words.append(timed)
            samples = 0.03 * np.sin(2 * np.pi * 220 * np.arange(cursor) / rate)
            takes[chunk.index] = generate.Take(
                chunk,
                samples.astype(np.float32),
                rate,
                1,
                None,
                line_bounds=bounds,
                line_words=words,
            )

    sounds = {node.name: node for node in iter_sound_nodes(tree)}
    clips = {name: Path(f"{name}.wav") for name in sounds}

    def load_sound(path, _rate, *, stereo=False):
        length = sounds[Path(path).stem].duration * rate
        tone = 0.01 * np.sin(2 * np.pi * 110 * np.arange(length) / rate)
        return np.column_stack((tone, tone)).astype(np.float32)

    spans = {}
    with patch.object(audio, "load_audio_file", side_effect=load_sound):
        mixed = generate._render_ast(tree, chunks_by_node, takes, clips, rate, spans=spans)
    return mixed, spans, takes, sounds


class SoundstageExamplesTests(unittest.TestCase):
    EXAMPLES = (
        ("en", 6, 12, 4),
        ("zh", 6, 12, 4),
        ("zh-latest", 5, 9, 3),
    )

    def test_all_examples_are_ready_for_generation(self):
        for lang, speaker_count, sound_count, _scene_count in self.EXAMPLES:
            with self.subTest(lang=lang):
                _title, raw_script, _speakers = example_story(lang)
                script, cast, sounds = parse_script_cast_and_sounds(example_story_reply_json(lang))
                self.assertEqual(len(cast), speaker_count)
                self.assertEqual(len(sounds), sound_count)
                self.assertEqual(voices_without_sex(cast), [])
                for entry in cast:
                    preview = entry["preview_text"]
                    self.assertFalse(any(ch in preview for ch in "<>[]{}"), entry["name"])
                    if lang == "en":
                        self.assertGreaterEqual(len(preview.split()), 30, entry["name"])
                    else:
                        self.assertGreaterEqual(len(_CJK_RE.findall(preview)), 55, entry["name"])
                self.assertEqual(lint_drama3_script(script), (script, []))
                self.assertFalse(needs_direction(script))
                normalized_lines = "\n".join(line.strip() for line in raw_script.splitlines() if line.strip())
                self.assertEqual(localize_existing_cues(raw_script), normalized_lines)
                tree = parse_drama3_tree(raw_script)
                names = {turn.speaker for node in iter_speech_nodes(tree) for turn in node.turns}
                self.assertEqual(names, {entry["name"] for entry in cast})

    def test_sound_roles_local_anchors_holds_and_reused_assets(self):
        for lang, _speaker_count, _sound_count, scene_count in self.EXAMPLES:
            with self.subTest(lang=lang):
                _title, raw_script, _speakers = example_story(lang)
                tree = parse_drama3_tree(raw_script)
                report = sound_density_report(raw_script)
                self.assertEqual(sum(s["roles"].get("unlabelled", 0) for s in report["scenes"]), 0)

                masters = [node for node in tree.children if _is_scene_master(node)]
                self.assertEqual(len(masters), scene_count)

                assets = {}
                for sound in iter_sound_nodes(tree):
                    self.assertIn(sound.role, SOUND_ROLES, sound.name)
                    self.assertIn(sound.perspective, PERSPECTIVES, sound.name)
                    identity = (sound.description, sound.duration, sound.loop)
                    self.assertEqual(assets.setdefault(sound.name, identity), identity)

                for scene in masters:
                    loops = [node for node in scene.children if getattr(node, "loop", False)]
                    self.assertLessEqual(len(loops), 2)
                    for child in scene.children:
                        if hasattr(child, "anchor"):
                            self.assertFalse(child.anchor, f"anchored cue {child.name} belongs in a local block")
                    self.assertEqual(scene.holds, ())

                for parallel in _iter_parallels(tree):
                    speech = _node_speech(parallel)
                    plain = "".join(spoken_text(turn.text) for s in iter_speech_nodes(parallel) for turn in s.turns)
                    for phrase, seconds in parallel.holds:
                        self.assertGreater(seconds, 0)
                        self.assertEqual(phrase_count(phrase, speech), 1, phrase)
                        self.assertEqual(plain.count(phrase), 1, phrase)
                    for child in parallel.children:
                        if not getattr(child, "anchor", ""):
                            continue
                        self.assertFalse(_is_scene_master(parallel), child.name)
                        self.assertEqual(phrase_count(child.anchor, speech), 1, child.anchor)
                        if child.end_anchor:
                            self.assertEqual(phrase_count(child.end_anchor, speech), 1, child.end_anchor)
                            self.assertLess(plain.index(child.anchor), plain.index(child.end_anchor))

    def test_zh_latest_keeps_dry_dialogue_and_reused_score(self):
        _title, raw_script, _speakers = example_story("zh-latest")
        tree = parse_drama3_tree(raw_script)
        masters = [node for node in tree.children if _is_scene_master(node)]
        self.assertTrue(all(not scene.room for scene in masters))
        scores = [node for node in iter_sound_nodes(tree) if node.name == "homeward_piano"]
        self.assertEqual(len(scores), 2)
        self.assertTrue(scores[0].end_anchor)
        self.assertTrue(scores[1].resume)
        self.assertTrue(all(node.envelope and node.perspective == "editorial" for node in scores))

    def test_full_mix_zh_latest_keeps_confession_unscored_and_withdraws_music_before_clock(self):
        rate = 8000
        _title, raw_script, _speakers = example_story("zh-latest")
        mixed, spans, takes, sounds = _render_example(raw_script, rate=rate)
        self.assertEqual(mixed.shape[1], 2)
        self.assertTrue(np.isfinite(mixed).all())
        sound_spans = [(start, end, info) for key, (start, end, info) in spans.items() if not isinstance(key, int)]
        self.assertEqual({info["name"] for _, _, info in sound_spans}, set(sounds))
        scores = sorted((start, end) for start, end, info in sound_spans if info["name"] == "homeward_piano")
        self.assertEqual(len(scores), 2)
        confession = next(t for t in takes.values() if "我没敢答应" in t.chunk.text)
        self.assertLess(spans[confession.chunk.index][1], scores[0][0])
        self.assertLess(scores[0][1], scores[1][0])
        clock_start = next(start for start, _, info in sound_spans if info["name"] == "clock_restart")
        self.assertLessEqual(scores[1][1], clock_start)

    def test_full_mix_en_and_zh_render_scoped_scores_and_stingers(self):
        rate = 8000
        cases = (
            ("en", "noir_cello", "Look at the voltmeter", "Listen to the casing", "revelation_sting", "Ignition sequence"),
            ("zh", "guqin_mystery", "石盘上的刻度", "底座下面", "starlight_sting", "浑天星图校准完毕"),
        )
        for lang, score_name, entry_phrase, after_phrase, sting_name, ignite_phrase in cases:
            with self.subTest(lang=lang):
                _title, raw_script, _speakers = example_story(lang)
                mixed, spans, takes, sounds = _render_example(raw_script, rate=rate)
                self.assertEqual(mixed.shape[1], 2)
                self.assertTrue(np.isfinite(mixed).all())
                sound_spans = [
                    (start, end, info) for key, (start, end, info) in spans.items() if not isinstance(key, int)
                ]
                self.assertEqual({info["name"] for _, _, info in sound_spans}, set(sounds))
                score_start, score_end = next(
                    (start, end) for start, end, info in sound_spans if info["name"] == score_name
                )
                entry_take = next(t for t in takes.values() if entry_phrase in t.chunk.text)
                after_take = next(t for t in takes.values() if after_phrase in t.chunk.text)
                self.assertGreaterEqual(score_start, spans[entry_take.chunk.index][0])
                # The bed withdraws at end_anchor before the whispered line, fading out over LOOP_TAIL_SECONDS.
                self.assertLessEqual(score_end - int(audio.LOOP_TAIL_SECONDS * rate), spans[after_take.chunk.index][0])
                self.assertLess(score_end, spans[after_take.chunk.index][1])
                sting_start = next(start for start, _, info in sound_spans if info["name"] == sting_name)
                ignite_take = next(t for t in takes.values() if ignite_phrase in t.chunk.text)
                self.assertGreater(sting_start, spans[ignite_take.chunk.index][1])


if __name__ == "__main__":
    unittest.main()
