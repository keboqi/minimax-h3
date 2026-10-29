"""Soundstage mix details: edge breaths, one-shot onsets and room tone."""

from __future__ import annotations

import unittest

import numpy as np

from h3_app.drama3.serving import audio

RATE = 16000


def noise(seconds, level, seed=0):
    return (level * np.random.default_rng(seed).standard_normal(int(seconds * RATE))).astype(np.float32)


class MixDetailTests(unittest.TestCase):
    def test_a_soft_breath_before_a_line_survives_the_edge_trim(self):
        silence = np.zeros(int(0.5 * RATE), dtype=np.float32)
        breath = noise(0.3, 0.006, seed=1)  # about 37 dB under the line
        take = np.concatenate([silence, breath, noise(1.0, 0.4), silence])
        tidied = audio.tidy_take(take, RATE)
        self.assertGreater(len(tidied), 1.25 * RATE)  # the breath is still there
        self.assertLess(len(tidied), 1.55 * RATE)  # the silence is not
        self.assertLess(abs(float(tidied[0])), 1e-4)  # faded in, so the cut never clicks
        self.assertEqual(len(audio.tidy_take(np.zeros(RATE, dtype=np.float32), RATE)), 0)

    def test_one_shots_lose_their_leading_dead_air_but_not_a_deliberate_wait(self):
        hit = np.concatenate([np.zeros(int(0.3 * RATE)), noise(0.5, 0.5)]).astype(np.float32)
        trimmed = audio.trim_onset(np.stack([hit, hit], axis=1), RATE)
        self.assertAlmostEqual(len(hit) - len(trimmed), int(0.3 * RATE) - int(audio.ONSET_PREROLL_SECONDS * RATE),
                               delta=2)
        late = np.concatenate([np.zeros(2 * RATE), noise(0.5, 0.5)]).astype(np.float32)
        self.assertEqual(len(late) - len(audio.trim_onset(late, RATE)), int(audio.MAX_ONSET_TRIM_SECONDS * RATE))

    def test_room_tone_fills_silence_far_under_the_mix(self):
        speech = np.concatenate([noise(2.0, 0.2), np.zeros(RATE, dtype=np.float32)])
        mix = np.stack([speech, speech], axis=1)
        toned = audio.add_room_tone(mix, RATE)
        gap = toned[-RATE // 2:]
        self.assertGreater(float(np.abs(gap).max()), 0.0)  # no digital silence
        loudness = audio.integrated_loudness(mix, RATE)
        tone_db = 20 * np.log10(float(np.sqrt(np.mean(gap.astype(np.float64) ** 2))))
        self.assertAlmostEqual(tone_db, loudness - audio.ROOM_TONE_BELOW_DB, delta=4)
        np.testing.assert_array_equal(audio.add_room_tone(np.zeros((RATE, 2), np.float32), RATE),
                                      np.zeros((RATE, 2), np.float32))  # a silent mix stays silent


if __name__ == "__main__":
    unittest.main()
