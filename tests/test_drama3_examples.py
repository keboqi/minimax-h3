"""Check that the Chinese production showcase has executable scene-local cues."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from h3_app.drama3.cast import parse_script_cast_and_sounds, voices_without_sex
from h3_app.drama3.example_story import example_story_reply_json
from h3_app.drama3.example_story_zh_latest import EXAMPLE_STORY_ZH_LATEST_SCRIPT
from h3_app.drama3.lint import lint_drama3_script, needs_direction
from h3_app.drama3.serving import audio, generate
from h3_app.drama3.serving.planner import plan_turns_chunks, spoken_text
from h3_app.drama3.serving.tree import (
    ParallelNode, iter_sound_nodes, iter_speech_nodes, parse_drama3_tree,
)


class ChineseProductionExampleTests(unittest.TestCase):
    def test_cast_and_direction_are_ready_for_generation(self):
        script, cast, sounds = parse_script_cast_and_sounds(example_story_reply_json("zh-latest"))
        self.assertEqual(len(cast), 5)
        self.assertEqual(len(sounds), 9)
        self.assertEqual(voices_without_sex(cast), [])
        self.assertEqual(lint_drama3_script(script), (script, []))
        self.assertFalse(needs_direction(script))
        tree = parse_drama3_tree(EXAMPLE_STORY_ZH_LATEST_SCRIPT)
        names = {turn.speaker for node in iter_speech_nodes(tree) for turn in node.turns}
        self.assertEqual(names, {entry["name"] for entry in cast})

    def test_anchors_holds_and_reused_assets_resolve_in_their_own_scenes(self):
        tree = parse_drama3_tree(EXAMPLE_STORY_ZH_LATEST_SCRIPT)
        scenes = [node for node in tree.children if isinstance(node, ParallelNode)]
        self.assertEqual(len(scenes), 3)
        assets = {}
        for scene in scenes:
            speech = "".join(spoken_text(turn.text) for node in iter_speech_nodes(scene) for turn in node.turns)
            self.assertFalse(scene.room)  # Intimate dialogue is deliberately dry.
            for phrase, seconds in scene.holds:
                self.assertEqual(speech.count(phrase), 1, phrase)
                self.assertGreater(seconds, 0)
            loops = [node for node in iter_sound_nodes(scene) if node.loop]
            self.assertLessEqual(len(loops), 2)
            for sound in iter_sound_nodes(scene):
                # Anchors belong directly to the scene master beside its speech sequence.
                if sound.anchor:
                    self.assertIn(sound, scene.children)
                for phrase in (sound.anchor, sound.end_anchor):
                    if phrase:
                        self.assertEqual(speech.count(phrase), 1, phrase)
                if sound.end_anchor:
                    self.assertLess(speech.index(sound.anchor), speech.index(sound.end_anchor))
                identity = (sound.description, sound.duration, sound.loop)
                self.assertEqual(assets.setdefault(sound.name, identity), identity)
        scores = [node for node in iter_sound_nodes(tree) if node.name == "homeward_piano"]
        self.assertEqual(len(scores), 2)
        self.assertTrue(scores[0].end_anchor)
        self.assertTrue(scores[1].resume)
        self.assertTrue(all(node.envelope and node.perspective == "editorial" for node in scores))

    def test_full_mix_keeps_confession_unscored_and_withdraws_music_before_clock(self):
        """Exercise the prepared script's timing with deterministic, aligned stand-in takes."""
        rate = 8000
        tree = parse_drama3_tree(EXAMPLE_STORY_ZH_LATEST_SCRIPT)
        chunks_by_node, takes = {}, {}
        for speech in iter_speech_nodes(tree):
            chunks = plan_turns_chunks(speech.turns, start_index=len(takes))
            chunks_by_node[id(speech)] = chunks
            for chunk in chunks:
                bounds, words, cursor = [], [], 0
                for line in chunk.lines:
                    characters = "".join(spoken_text(line).split())
                    start = cursor
                    timed = []
                    for character in characters:
                        timed.append((character, cursor / rate, (cursor + 320) / rate))
                        cursor += 320
                    bounds.append((start, cursor))
                    words.append(timed)
                samples = .03 * np.sin(2 * np.pi * 220 * np.arange(cursor) / rate)
                takes[chunk.index] = generate.Take(
                    chunk, samples.astype(np.float32), rate, 1, None,
                    line_bounds=bounds, line_words=words,
                )

        sounds = {node.name: node for node in iter_sound_nodes(tree)}
        clips = {name: Path(f"{name}.wav") for name in sounds}

        def load_sound(path, _rate, *, stereo=False):
            length = sounds[Path(path).stem].duration * rate
            tone = .01 * np.sin(2 * np.pi * 110 * np.arange(length) / rate)
            return np.column_stack((tone, tone)).astype(np.float32)

        spans = {}
        with patch.object(audio, "load_audio_file", side_effect=load_sound):
            mixed = generate._render_ast(tree, chunks_by_node, takes, clips, rate, spans=spans)
        self.assertEqual(mixed.shape[1], 2)
        self.assertTrue(np.isfinite(mixed).all())
        sound_spans = [(start, end, info) for key, (start, end, info) in spans.items()
                       if not isinstance(key, int)]
        self.assertEqual({info["name"] for _, _, info in sound_spans}, set(sounds))
        scores = sorted((start, end) for start, end, info in sound_spans
                        if info["name"] == "homeward_piano")
        self.assertEqual(len(scores), 2)
        confession = next(t for t in takes.values() if "我没敢答应" in t.chunk.text)
        self.assertLess(spans[confession.chunk.index][1], scores[0][0])
        self.assertLess(scores[0][1], scores[1][0])
        clock_start = next(start for start, _, info in sound_spans if info["name"] == "clock_restart")
        self.assertLessEqual(scores[1][1], clock_start)


if __name__ == "__main__":
    unittest.main()
