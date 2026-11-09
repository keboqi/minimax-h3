"""Workshop's image has no scipy or model packages, so its dub steps must import without them."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
CHECK = """
import sys

class Blocked:
    NAMES = {"scipy", "torch", "torchaudio", "transformers", "soundfile", "librosa", "qwen_asr", "audio_separator"}
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in self.NAMES:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, Blocked())
from h3_app.dubbing import pipeline, projects, session, shares
from h3_app.dubbing.gemini import transcribe_prompt
from h3_app.drama3.serving.speech import run_blocking
from h3_ui.dubbing import handlers, page, presenters, share, tab, workshop
line = {"start": 0.0, "end": 0.1, "speaker": "speaker1", "source": "Hi", "translation": "A long line to say", "dub": True}
assert pipeline.long_lines([line]) == [0]
assert pipeline.translation_issues([line]) == []
"""


class WorkshopImportTests(unittest.TestCase):
    def test_dub_steps_load_without_gpu_packages(self):
        done = subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-1500:])
