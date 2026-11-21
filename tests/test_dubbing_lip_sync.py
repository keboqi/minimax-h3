"""Frozen dialogue and graph wiring for a prepared H3 lip-sync shot."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from h3_app.dubbing.alignment import FITTED_AUDIO_RATE as RATE
from h3_app.dubbing.lip_sync import fitted_dialogue, render_lip_sync_preview
from h3_app.dubbing.results import player_folder
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from h3_app.workflows.lip_sync import build_lip_sync_graph
from tests.test_face_refine import by_class, link_id, models


class LipSyncTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dub = Path(temporary.name) / "finished.mp4"
        self.dub.touch()
        folder = player_folder(self.dub)
        folder.mkdir(parents=True)
        (folder / "alignment.json").write_text(json.dumps({
            "lines": [{"speaker": "A"}, {"speaker": "B"}, {"speaker": "A"}],
            "spoken": {"0": [0.125, 0.25], "1": [0.0, 0.25], "2": [0.375, 0.5]},
        }), encoding="utf-8")
        samples = lambda start, end: round(end * RATE) - round(start * RATE)
        np.savez_compressed(folder / "alignment.npz", **{
            "0": np.full(samples(0.125, 0.25), 0.25, dtype=np.float32),
            "1": np.full(samples(0.0, 0.25), 0.75, dtype=np.float32),
            "2": np.full(samples(0.375, 0.5), 0.5, dtype=np.float32),
        })
        self.models = models()
        self.models.turbo_lora_for = lambda _mode, _variant: "pika.safetensors"

    def test_dialogue_starts_at_the_fitted_time_and_excludes_other_speakers(self):
        dialogue = fitted_dialogue(self.dub, "A", first_frame=4, frames=8)
        self.assertEqual(len(dialogue), round(RATE / 2) - round(4 * RATE / 24))
        first_interval = round(0.25 * RATE) - round(4 * RATE / 24)
        next_interval = round(0.375 * RATE) - round(4 * RATE / 24)
        np.testing.assert_array_equal(dialogue[:first_interval], 0.25)
        np.testing.assert_array_equal(dialogue[first_interval:next_interval], 0)
        np.testing.assert_array_equal(dialogue[next_interval:], 0.5)
        with self.assertRaisesRegex(H3Error, "no audible fitted dialogue"):
            fitted_dialogue(self.dub, "A", first_frame=24, frames=5)

    def test_graph_keeps_dialogue_locked_and_composites_with_the_original_mask(self):
        graph = build_lip_sync_graph(crop="crop.mp4", mask="mask.mp4", dialogue="dialogue.wav",
                                    width=640, height=640, frames=124, models=self.models,
                                    refinement={}, output_prefix="preview")
        held = by_class(graph, "H3SourceAudioLatent")
        mask = by_class(graph, "H3LipSyncMask")
        self.assertEqual(held["audio"], [link_id(graph, "LoadAudio"), 0])
        self.assertEqual(mask["latent"], [link_id(graph, "H3SourceAudioLatent"), 0])
        sample = by_class(graph, "SamplerCustomAdvanced")
        self.assertEqual(sample["latent_image"], [link_id(graph, "H3LipSyncMask"), 0])
        composite = by_class(graph, "ImageCompositeMasked")
        self.assertEqual(composite["mask"], [link_id(graph, "H3LipSyncMask"), 1])
        self.assertEqual(composite["destination"], mask["source_frames"])
        self.assertEqual(by_class(graph, "KSamplerSelect")["sampler_name"], "lcm")

    def test_prepared_crop_can_submit_with_fitted_audio(self):
        def stage(path, category):
            self.assertEqual(category, "lip_sync")
            if path.endswith(".wav"):
                with wave.open(path, "rb") as audio:
                    self.assertEqual((audio.getframerate(), audio.getnchannels()), (RATE, 1))
                    self.assertEqual(audio.getnframes(), round(22 * RATE / 24))
            return Path(path).name

        def submit(graph, client_id):
            self.assertTrue(by_class(graph, "SaveVideo")["filename_prefix"].endswith(client_id))
            return client_id

        services = SimpleNamespace(
            models=SimpleNamespace(load_model_config=lambda: self.models),
            media=SimpleNamespace(probe_video_metadata=Mock(return_value=SimpleNamespace(
                fps=24, width=640, height=640, frame_count=22)), stage_file=stage,
                resolve_output=lambda history, _: Path(f"{history}_00001_.mp4")),
            execution=SimpleNamespace(submit_prompt=Mock(side_effect=submit), wait_for_history=lambda prompt_id: prompt_id,
                                      poll_comfy_progress=lambda *_: []),
        )
        options = dict(dub=self.dub, speaker="A", first_frame=0, crop=Path("crop.mp4"), mask=Path("mask.mp4"),
                       services=services)
        result = list(render_lip_sync_preview(**options))
        self.assertIsInstance(result[-1], Path)

        services.execution.submit_prompt.reset_mock()
        preview = render_lip_sync_preview(**options)
        self.assertEqual(next(preview), "Checking the lip-sync window")
        self.assertEqual(next(preview), "Rendering the H3 mouth patch")
        with patch("h3_app.dubbing.lip_sync.check_cancelled", side_effect=JobCancelled("Stopped")):
            with self.assertRaises(JobCancelled):
                next(preview)
        services.execution.submit_prompt.assert_not_called()

    def test_invalid_crop_is_rejected_before_staging(self):
        media = SimpleNamespace(probe_video_metadata=Mock(return_value=SimpleNamespace(
            fps=24, width=641, height=640, frame_count=22)), stage_file=Mock())
        with self.assertRaisesRegex(H3Error, "multiples of 32"):
            list(render_lip_sync_preview(dub=self.dub, speaker="A", first_frame=0, crop=Path("crop.mp4"),
                                         mask=Path("mask.mp4"), services=SimpleNamespace(media=media)))
        media.stage_file.assert_not_called()
