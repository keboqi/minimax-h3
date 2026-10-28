"""Script alignment, cache invalidation and render timing regression tests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from h3_app.dubbing.tools_worker import Worker, aligned_line_bounds
from h3_app.drama3.serving.alignment import align_take
from h3_app.drama3.serving import generate
from h3_app.drama3.serving.planner import Chunk
from h3_app.drama3.serving.tree import parse_drama3_tree, iter_speech_nodes


class AlignmentTests(unittest.TestCase):
    def test_uneven_delivery_uses_aligned_words_and_ignores_internal_pause(self):
        items = [{"text": "Wait", "start": .1, "end": 1},
                 {"text": "please", "start": 1.3, "end": 4.3},
                 {"text": "Yes", "start": 4.6, "end": 5.6}]
        self.assertEqual(aligned_line_bounds(["Wait, please!", "Yes."], items), [(.1, 4.3), (4.6, 5.6)])

    def test_cjk_and_repeated_lines(self):
        items = [{"text": c, "start": i, "end": i + .5} for i, c in enumerate("你好你好")]
        self.assertEqual(aligned_line_bounds(["你好！", "你好。"], items), [(0, 1.5), (2, 3.5)])

    def test_invalid_alignment_is_rejected(self):
        for items in ([{"text": "wrong", "start": 0, "end": 1}],
                      [{"text": "yes", "start": float("nan"), "end": 1}]):
            with self.assertRaises(ValueError):
                aligned_line_bounds(["yes"], items)

    def test_worker_aligns_supplied_script_instead_of_recognized_paraphrase(self):
        from unittest.mock import MagicMock
        worker = Worker(Path("."))
        model = MagicMock()
        model.transcribe.return_value = [SimpleNamespace(language="English", text="incorrect")]
        model.forced_aligner.align.return_value = [[SimpleNamespace(text="Yes", start_time=.2, end_time=.8)]]
        with patch.object(worker, "_asr", return_value=model):
            self.assertEqual(worker.handle({"op": "align_lines", "audio": "take.wav", "lines": ["Yes!"]}),
                             {"bounds": [(.2, .8)]})
        self.assertEqual(model.forced_aligner.align.call_args.kwargs["text"], "Yes!")

    def test_cache_tracks_audio_and_line_boundaries(self):
        take = SimpleNamespace(chunk=SimpleNamespace(lines=("Hello.", "Yes.")),
                               samples=np.full(2000, .2, dtype=np.float32), rate=1000)
        with tempfile.TemporaryDirectory() as root, patch("h3_app.dubbing.tools.run", return_value={
                "bounds": [[.1, .8], [1.2, 1.8]]}) as run:
            self.assertEqual(align_take(take, root, None, free_gpu=lambda: None), [(100, 800), (1200, 1800)])
            align_take(take, root, None, free_gpu=lambda: None)
            self.assertEqual(run.call_count, 1)
            take.samples = take.samples * .5
            align_take(take, root, None, free_gpu=lambda: None)
            self.assertEqual(run.call_count, 2)
        with tempfile.TemporaryDirectory() as root, patch("h3_app.dubbing.tools.run", return_value={
                "bounds": [[.1, .8], [2.005, 2.015]]}):
            with self.assertRaises(ValueError):
                align_take(take, root, None, free_gpu=lambda: None)

    def test_render_uses_aligned_offsets_after_resampling(self):
        tree = parse_drama3_tree("<|speaker:0|> Hello.")
        speech = list(iter_speech_nodes(tree))[0]
        chunk = Chunk(0, "Speaker 0", "Hello. Yes.", (), ("Hello.", "Yes."))
        take = generate.Take(chunk, np.full(2000, .2, dtype=np.float32), 1000, 1, None,
                             line_bounds=[(100, 800), (1200, 1800)])
        spans = {}
        generate._render_ast(tree, {id(speech): [chunk]}, {0: take}, {}, 2000, spans=spans)
        self.assertEqual(spans[0][2], [(200, 1600), (2400, 3600)])
