"""Soundstage's combined preparation, panel actions, and streamed slot updates."""

import inspect
import unittest
from unittest.mock import Mock

import gradio as gr

from h3_app.jobs import JobCancelled
from h3_app.prompt_writers.drama3 import DRAMA3_SCRIPT_ADVANCED
from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS
from h3_ui.drama3.handlers import Drama3Handlers, Drama3Services


class PrepareCastTests(unittest.TestCase):
    def setUp(self):
        self.handlers = Drama3Handlers(Drama3Services(
            runtime_config=lambda: None, stage_file=lambda *_a: "", generation_services=lambda: None,
            free_gpu=lambda: None))
        self.voice_inputs = tuple(range(len(inspect.signature(self.handlers.generate_drama3_voices).parameters)))
        self.sound_inputs = tuple(range(1000, 1000 + len(inspect.signature(self.handlers.generate_drama3_sounds).parameters)))

    def frames(self, stride, count, path, summary):
        frame = [gr.skip()] * (stride * count)
        frame[0] = path
        yield (*frame, "Working")
        yield (*[gr.skip()] * len(frame), summary)

    def test_prepares_voices_then_sounds_and_keeps_audio_in_later_frames(self):
        order = []

        def voices(*args):
            self.assertEqual(args, self.voice_inputs)
            order.append("voices")
            yield from self.frames(3, DRAMA3_VOICE_SLOTS, "voice.wav", "Generated 1 of 1 speaker voices.")

        def sounds(*args):
            self.assertEqual(args, self.sound_inputs)
            order.append("sounds")
            yield from self.frames(2, DRAMA3_SOUND_SLOTS, "sound.wav", "Generated 1 of 1 sound clips.")

        self.handlers.generate_drama3_voices = voices
        self.handlers.generate_drama3_sounds = sounds
        updates = list(self.handlers.prepare_drama3_cast(DRAMA3_SCRIPT_ADVANCED,
                                                        *self.voice_inputs, *self.sound_inputs))
        self.assertEqual(order, ["voices", "sounds"])
        sound_start = 3 * DRAMA3_VOICE_SLOTS + 1
        self.assertTrue(all(frame[0] == "voice.wav" for frame in updates))
        self.assertTrue(all(frame[sound_start] == "sound.wav" for frame in updates[2:]))
        self.assertIn("Generated 1 of 1 speaker voices", updates[-1][-1])
        self.assertIn("Generated 1 of 1 sound clips", updates[-1][-1])
        self.assertEqual(len(updates[-1]), 3 * DRAMA3_VOICE_SLOTS + 2 * DRAMA3_SOUND_SLOTS + 3)

    def test_basic_mode_prepares_only_voices(self):
        self.handlers.generate_drama3_voices = Mock(return_value=self.frames(
            3, DRAMA3_VOICE_SLOTS, "voice.wav", "Generated 1 of 1 speaker voices."))
        self.handlers.generate_drama3_sounds = Mock()
        updates = list(self.handlers.prepare_drama3_cast("Basic", *self.voice_inputs, *self.sound_inputs))
        self.handlers.generate_drama3_sounds.assert_not_called()
        self.assertIn("Basic scripts do not use sounds", updates[-1][-1])

    def test_voice_failure_is_visible_and_does_not_prevent_independent_sound_preparation(self):
        self.handlers.generate_drama3_voices = Mock(side_effect=ValueError("voice backend failed"))
        self.handlers.generate_drama3_sounds = Mock(return_value=self.frames(
            2, DRAMA3_SOUND_SLOTS, "sound.wav", "Generated 1 of 1 sound clips."))
        updates = list(self.handlers.prepare_drama3_cast(DRAMA3_SCRIPT_ADVANCED,
                                                        *self.voice_inputs, *self.sound_inputs))
        self.assertIn("Voice generation failed: voice backend failed", updates[-1][-1])
        self.assertIn("Generated 1 of 1 sound clips", updates[-1][-1])

    def test_interrupt_stops_before_sounds(self):
        self.handlers.generate_drama3_voices = Mock(side_effect=JobCancelled("interrupted"))
        self.handlers.generate_drama3_sounds = Mock()
        with self.assertRaises(JobCancelled):
            list(self.handlers.prepare_drama3_cast(DRAMA3_SCRIPT_ADVANCED,
                                                  *self.voice_inputs, *self.sound_inputs))
        self.handlers.generate_drama3_sounds.assert_not_called()



if __name__ == "__main__":
    unittest.main()
