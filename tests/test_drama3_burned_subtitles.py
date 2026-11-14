import re
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from h3_app.drama3.subtitles import (LINGER_SECONDS, Line, ass_subtitles, font_size, subtitle_font)


def events(script):
    return [line.split(",", 9) for line in script.splitlines() if line.startswith("Dialogue:")]


def words(start, *pieces, step=0.5):
    return [(piece, start + index * step, start + (index + 1) * step) for index, piece in enumerate(pieces)]


class AssSubtitleTests(unittest.TestCase):
    def test_each_word_lights_up_at_its_own_start(self):
        line = Line("Ada", "“You had them”", 1.0, 3.0, ("“", words(1.2, "You ", "had ", "them”")))
        (event,) = events(ass_subtitles([line]))
        self.assertEqual(event[1:4], ["0:00:01.00", "0:00:03.60", "Caption"])
        text = event[9]
        self.assertTrue(text.startswith("{\\b1}Ada:{\\b0} “"))  # the name and lead text stay white
        # Karaoke syllables: 20 cs before the first word, then each word's length up to the next one.
        self.assertEqual(re.findall(r"\\k(\d+)", text), ["20", "50", "50", "50"])
        self.assertIn("{\\k50}You {\\k50}had {\\k50}them”", text)

    def test_lines_without_word_timing_show_whole(self):
        (event,) = events(ass_subtitles([Line("旁白", "钥匙{在}这里\\N。", 0.0, 1.0)]))
        self.assertNotIn("\\k", event[9])
        self.assertIn("旁白", event[9])
        # Braces and backslashes in the text are shown, not read as override tags or line breaks.
        self.assertNotIn("{在}", event[9])
        self.assertNotIn("\\N", event[9])
        # An empty word list shows the line whole too.
        (event,) = events(ass_subtitles([Line("Ada", "Hi.", 0.0, 1.0, ("", []))]))
        self.assertTrue(event[9].endswith("{\\b0} Hi."))

    def test_a_line_lingers_until_the_next_one_and_overlaps_show_together(self):
        lines = [Line("Ada", "One.", 0.0, 1.0), Line("Leo", "Two.", 1.3, 2.0), Line("Ada", "Three.", 5.0, 6.0),
                 Line("Leo", "Over —", 5.5, 6.5)]
        ends = [event[2] for event in events(ass_subtitles(lines))]
        self.assertEqual(ends[0], "0:00:01.30")  # cut short where the next line starts
        self.assertEqual(ends[1], f"0:00:0{2 + LINGER_SECONDS:.2f}")
        self.assertEqual(ends[2], "0:00:06.60")  # no line starts after it: it lingers, under the barge-in
        self.assertEqual(ends[3], "0:00:07.10")

    def test_size_and_font_follow_the_page(self):
        self.assertEqual(font_size(100), 28)
        self.assertEqual(font_size(200), 56)
        self.assertEqual(font_size(400), 64)  # never more than 5% of the width
        self.assertIn(",56,", ass_subtitles([], size=200))
        self.assertEqual(subtitle_font("win32"), "Microsoft YaHei")
        self.assertEqual(subtitle_font("linux"), "Noto Sans CJK SC")


def _has_libass():
    if not shutil.which("ffmpeg"):
        return False
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " ass " in filters


@unittest.skipUnless(_has_libass() and shutil.which("ffprobe"), "requires FFmpeg with libass")
class BurnedSubtitleRenderTests(unittest.TestCase):
    def test_words_turn_teal_in_the_rendered_video(self):
        from PIL import Image

        from h3_app.drama3.visuals import FPS, Shot, render_video

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            still = folder / "still.png"
            Image.new("RGB", (64, 36), (40, 40, 40)).save(still)
            audio = folder / "mix.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", ""))
                wav.writeframes(bytes(32000 * 3))
            script = ass_subtitles([Line("Ada", "You had them last", 0.0, 2.5,
                                         ("", words(1.0, "You ", "had ", "them ", "last", step=0.3)))])
            video = render_video(audio, [Shot(0, 3 * FPS, False, "")], [still], folder / "out", subtitles=script)

            def teal_pixels(seconds):
                raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(video),
                                               "-frames:v", "1", "-vf", "crop=1280:120:0:600",
                                               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                return sum(1 for i in range(0, len(raw), 3)
                           if raw[i + 1] > 180 and raw[i + 2] > 160 and raw[i] < raw[i + 1] - 40)

            self.assertEqual(teal_pixels(0.5), 0)  # shown, all white, before its first word
            self.assertGreater(teal_pixels(2.3), 200)
            info = subprocess.check_output(["ffprobe", "-v", "error", "-of", "csv=p=0",
                                            "-show_entries", "stream=codec_name", str(video)], text=True)
            self.assertEqual(info.split(), ["h264", "aac"])


if __name__ == "__main__":
    unittest.main()
