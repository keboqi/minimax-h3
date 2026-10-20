"""Repeated audio outputs reuse Gradio's ffmpeg lookup and playability probe."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from gradio import processing_utils

from h3_ui import audio_probe_cache


class AudioProbeCacheTests(unittest.TestCase):
    def install(self, probe, lookup):
        for name in ("audio_is_playable", "ffmpeg_installed", "_h3_probe_cache"):
            self.addCleanup(setattr, processing_utils, name, getattr(processing_utils, name, False))
        processing_utils._h3_probe_cache = False
        processing_utils.audio_is_playable, processing_utils.ffmpeg_installed = probe, lookup
        audio_probe_cache.install()
        return processing_utils.audio_is_playable, processing_utils.ffmpeg_installed

    def test_a_file_is_probed_once_per_version(self):
        probe = Mock(return_value=True)
        playable, _ = self.install(probe, Mock(return_value=True))
        with TemporaryDirectory() as directory:
            clip = Path(directory) / "voice.wav"
            clip.write_bytes(b"RIFF1")
            for _ in range(3):
                self.assertTrue(playable(str(clip)))
            self.assertEqual(probe.call_count, 1)
            clip.write_bytes(b"RIFF22")  # a rewritten clip is probed again
            os.utime(clip, ns=(1, 1))
            probe.return_value = False
            self.assertFalse(playable(str(clip)))
            self.assertEqual(probe.call_count, 2)
            playable(str(Path(directory) / "missing.wav"))  # unstat-able paths are not cached
            playable(str(Path(directory) / "missing.wav"))
            self.assertEqual(probe.call_count, 4)

    def test_only_a_found_ffmpeg_is_remembered(self):
        lookup = Mock(side_effect=[False, True, True])
        _, installed = self.install(Mock(return_value=True), lookup)
        self.assertFalse(installed())
        self.assertTrue(installed())
        self.assertTrue(installed())
        self.assertEqual(lookup.call_count, 2)


if __name__ == "__main__":
    unittest.main()
