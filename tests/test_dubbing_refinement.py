"""Acoustic refinement contracts without model downloads or GPU inference."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from h3_app.dubbing import pipeline, refinement, segments, tools_worker
from h3_app.jobs import JobCancelled


def group():
    return {"language": "English", "start": 10, "end": 12,
            "lines": [{"start": 10, "end": 10.8, "text": "Hello!", "speaker": "speaker2"},
                      {"start": 11, "end": 12, "text": "How are you?", "speaker": "speaker2"}],
            "words": [{"text": "Hello!", "start": 10, "end": 10.8},
                      {"text": "How", "start": 11, "end": 11.2},
                      {"text": "are", "start": 11.3, "end": 11.5},
                      {"text": "you?", "start": 11.6, "end": 12}]}


class RefinementTests(unittest.TestCase):
    def test_qwen_returns_chunk_word_times_with_recording_offsets(self):
        worker = object.__new__(tools_worker.Worker)
        item = SimpleNamespace(text="Hello!", start_time=0.2, end_time=0.8)
        missing_time = SimpleNamespace(text="ignored", start_time=None, end_time=None)
        result = SimpleNamespace(text="Hello!", language="English", time_stamps=[item, missing_time])
        asr = Mock()
        asr.transcribe.return_value = [result, result]
        fake_librosa = SimpleNamespace(load=lambda *a, **k: (np.zeros(16000 * 20), 16000))
        with patch.dict(sys.modules, {"librosa": fake_librosa}), \
                patch.object(worker, "_asr", return_value=asr), \
                patch.object(worker, "_speech_spans", return_value=[(10, 12), (15, 16)]):
            reply = worker.transcribe({"audio": "speech.wav", "aligner": True, "word_details": True})
        clips = asr.transcribe.call_args.kwargs["audio"]
        self.assertEqual([len(samples) / rate for samples, rate in clips], [2, 1])
        self.assertTrue(asr.transcribe.call_args.kwargs["return_time_stamps"])
        self.assertEqual([g["words"][0]["start"] for g in reply["groups"]], [10.2, 15.2])
        self.assertEqual(reply["segments"], [line for g in reply["groups"] for line in g["lines"]])

    def test_cuts_use_first_and_last_word_without_changing_text_or_speakers(self):
        original = group()
        words = deepcopy(original["words"])
        words[0].update(start=10.1, end=10.7)
        words[-1]["end"] = 11.9
        lines, evidence = refinement.apply_group(original, {"words": words}, 20)
        self.assertEqual([(line["start"], line["end"]) for line in lines], [(10.1, 10.7), (11.0, 11.9)])
        self.assertEqual([line["text"] for line in lines], ["Hello!", "How are you?"])
        self.assertTrue(all(line["speaker"] == "speaker2" for line in lines))
        self.assertEqual(evidence["original_words"], original["words"])
        self.assertEqual(evidence["status"], "refined")

    def test_invalid_refinements_are_rejected(self):
        for field, value in (("start", float("nan")), ("start", -0.1), ("end", 21),
                             ("start", 8), ("end", 9.9), ("text", "Changed")):
            with self.subTest(field=field, value=value):
                words = deepcopy(group()["words"])
                words[0][field] = value
                with self.assertRaises(ValueError):
                    refinement.apply_group(group(), {"words": words}, 20)
        with self.assertRaises(ValueError):
            refinement.apply_group(group(), {"words": group()["words"][:-1]}, 20)
        for result in (None, "invalid", {"words": [None] * 4}, {"words": "abcd"}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                refinement.apply_group(group(), result, 20)

    def test_cjk_boundaries_map_to_unchanged_script(self):
        original = {"lines": [{"text": "你好。", "speaker": "speaker1"}], "words": [
            {"text": "你", "start": 0.1, "end": 0.3}, {"text": "好", "start": 0.4, "end": 0.7}]}
        lines, evidence = refinement.apply_group(original, {"words": original["words"]}, 1)
        self.assertEqual(lines, [{"text": "你好。", "speaker": "speaker1", "start": 0.1, "end": 0.7}])
        self.assertEqual(evidence["status"], "unchanged")

    def test_language_aliases_do_not_guess_unsupported_languages(self):
        self.assertEqual(refinement.language_code("Chinese"), "zh")
        self.assertEqual(refinement.language_code("pt-BR"), "pt")
        for language in ("Auto-detect", "Cantonese", "Russian", "Chinese,English"):
            self.assertEqual(refinement.language_code(language), "")

    def test_refinement_changes_only_qwen_transcript_cache_key(self):
        plain = pipeline.AnalyzeRequest("a.wav", source_language="English", refine_timestamps=False)
        refined = pipeline.AnalyzeRequest("a.wav", source_language="English")
        self.assertNotEqual(pipeline._transcript_key(plain), pipeline._transcript_key(refined))
        for backend in (pipeline.PIPELINE_GEMINI, pipeline.PIPELINE_MOSS):
            self.assertEqual(pipeline._transcript_key(pipeline.AnalyzeRequest("a", pipeline=backend,
                                                                            refine_timestamps=False)),
                             pipeline._transcript_key(pipeline.AnalyzeRequest("a", pipeline=backend,
                                                                            refine_timestamps=True)))

    def test_failed_group_keeps_original_cuts_and_good_group_is_applied(self):
        groups = [group(), group()]
        groups[1]["lines"] = [{**line, "speaker": "speaker3"} for line in groups[1]["lines"]]
        words = deepcopy(groups[0]["words"])
        words[0]["start"] = 10.1
        session = SimpleNamespace(speech_audio=Path("speech.wav"), data={"duration": 20})
        baseline = segments.normalize({**line, "source": line["text"]}
                                      for speech in groups for line in speech["lines"])
        with patch.object(refinement, "check_runtime"), patch.object(refinement, "refine", return_value=[
                {"words": words}, {"error": "bad boundary"}]):
            runner = pipeline._refine_timestamps(session, {"segments": baseline}, groups,
                runtime=SimpleNamespace(comfy_dir=Path("ComfyUI"), generation_timeout=60))
            while True:
                try:
                    next(runner)
                except StopIteration as finished:
                    result = finished.value
                    break
        self.assertEqual([g["status"] for g in result["refinement"]["groups"]], ["refined", "original"])
        self.assertEqual(next(s["start"] for s in result["segments"] if s["speaker"] == "speaker3"), 10)
        self.assertEqual(next(s["start"] for s in result["segments"] if s["speaker"] == "speaker2"), 10.1)

    def test_cancellation_is_never_reported_as_refinement_fallback(self):
        with patch.object(refinement, "check_runtime", side_effect=JobCancelled("stop")):
            runner = pipeline._refine_timestamps(SimpleNamespace(), {"segments": []}, [], runtime=None)
            with self.assertRaises(JobCancelled):
                next(runner)

    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_node_bridge_restores_recording_offsets_and_preserves_sdk_fallback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio, module, job, result = [root / name for name in ("audio.f32", "fake.mjs", "job.json", "out.json")]
            audio.write_bytes(np.zeros(16000 * 20, dtype="<f4").tobytes())
            module.write_text("export class Align { static async load() { return new Align(); } "
                "static isSupported(s) { return s === 'en'; } dispose() {} "
                "async refine(samples, rate, words) { return words.map(w => ({...w, "
                "start:w.start+0.1, end:w.end+0.1, refined:true})); } }", encoding="utf-8")
            speech = group()
            speech["language_code"] = "en"
            job.write_text(json.dumps({"audio": str(audio), "module": str(module), "cache": str(root),
                "groups": [speech, {**speech, "language_code": "ru"}], "output": str(result)}), encoding="utf-8")
            script = Path(refinement.__file__).with_name("refine_timestamps.mjs")
            subprocess.run([shutil.which("node"), str(script), str(job)], check=True, capture_output=True, timeout=15)
            output = json.loads(result.read_text())
            self.assertAlmostEqual(output[0]["words"][0]["start"], 10.1)
            self.assertIn("unsupported language", output[1]["error"])
