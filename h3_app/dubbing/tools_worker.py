"""Standalone entry point: run only with the isolated dubbing tools interpreter.

``serve`` keeps audio-separator, OmniVAD + Qwen3-ASR (with its ForcedAligner)
and Hy-MT resident: each loads on its first request and stays in VRAM until the
parent stops the process. qwen-asr pins Transformers 4.57.6, which conflicts with
both ComfyUI's environment and the VoiceDesign/Stable Audio one, so these models
live in their own interpreter.

Requests arrive one JSON line at a time on stdin; replies go to the original
stdout and everything the libraries print goes to stderr (the log). Weights are
fetched into ``<models>/dubbing`` on first use.

The parent imports this module for its constants, so keep heavy imports inside
functions.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback

ASR_ID = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
TRANSLATOR_ID = "tencent/Hy-MT2-1.8B"
SEPARATOR_MODELS = {
    "Fast (UVR-MDX-NET)": "UVR-MDX-NET-Inst_full_292.onnx",
    "Balanced (BS-Roformer)": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "Quality (Mel-Band-Roformer)": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
}
ASR_RATE = 16000
# OmniVAD spans closer than this are one utterance; longer ones are cut at aligned sentence ends.
VAD_MERGE_GAP = 0.35
MAX_LINE_SECONDS = 12.0
MAX_LINE_CHARS = 90
MIN_SPAN_SECONDS = 0.2
ASR_BATCH = 16
TRANSLATE_BATCH = 16
SENTENCE_END = re.compile(r"[.!?。！？…;；]$")
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
# CJK text and its full-width punctuation take no spaces around words.
CJK_OR_PUNCTUATION = re.compile(r"[　-ヿ㐀-鿿가-힯＀-￯]")


def models_root(models_dir: str | Path) -> Path:
    return Path(models_dir) / "dubbing"


# ---------------------------------------------------------------------------
# Pure helpers (also exercised by the parent's tests)
# ---------------------------------------------------------------------------


def merge_spans(spans, gap: float) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in spans if float(e) > float(s)):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def join_words(words) -> str:
    """Join aligned units: CJK characters and punctuation attach, other words take a space."""
    text = ""
    for word in (str(w).strip() for w in words):
        if not word:
            continue
        attach = (not text or CJK_OR_PUNCTUATION.match(text[-1]) or CJK.match(word)
                  or re.match(r"^[,.;:!?%)\]}、-〿！-／：-？]", word))
        text += word if attach else " " + word
    return text.strip()


def lines_from_alignment(items, offset: float, end_limit: float, fallback: str) -> list[dict]:
    """Group aligned words into subtitle-sized lines, cut at sentence ends or length limits."""
    lines: list[dict] = []
    words: list[str] = []
    start = end = None
    for item in items:
        word = str(getattr(item, "text", "") or "").strip()
        s, e = getattr(item, "start_time", None), getattr(item, "end_time", None)
        if not word or s is None or e is None:
            continue
        s, e = offset + float(s), min(offset + float(e), end_limit)
        if e <= s:
            continue
        if start is None:
            start = s
        words.append(word)
        end = e
        text = join_words(words)
        if SENTENCE_END.search(word) or len(text) >= MAX_LINE_CHARS or end - start >= MAX_LINE_SECONDS:
            lines.append({"start": round(start, 3), "end": round(end, 3), "text": text})
            words, start, end = [], None, None
    if words and start is not None:
        lines.append({"start": round(start, 3), "end": round(end, 3), "text": join_words(words)})
    if not lines and fallback.strip():
        lines.append({"start": round(offset, 3), "end": round(end_limit, 3), "text": fallback.strip()})
    return lines


def clean_translation(text: str) -> str:
    cleaned = str(text or "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return re.sub(r"^(assistant|translation|translated text)\s*[:：]\s*", "", cleaned, flags=re.I).strip()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Worker:
    def __init__(self, models_dir: Path):
        self.root = models_root(models_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.root / "hf"))
        self.models: dict[str, object] = {}

    @property
    def loaded(self) -> list[str]:
        return sorted(self.models)

    @staticmethod
    def torch():
        import torch

        return torch

    def dtype(self):
        torch = self.torch()
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def device(self) -> str:
        return "cuda:0" if self.torch().cuda.is_available() else "cpu"

    def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "separate":
            return self.separate(request)
        if op == "transcribe":
            return self.transcribe(request)
        if op == "translate":
            return self.translate(request)
        if op == "unload":
            self.unload(request.get("model"))
            return {}
        raise ValueError(f"Unknown dubbing tools request: {op}")

    def unload(self, model: str | None = None) -> None:
        for name in [model] if model else list(self.models):
            self.models.pop(name, None)
        gc.collect()
        if self.torch().cuda.is_available():
            self.torch().cuda.empty_cache()

    # -- vocal separation ---------------------------------------------------

    def separate(self, request: dict) -> dict:
        from audio_separator.separator import Separator

        model = SEPARATOR_MODELS.get(request.get("model"), SEPARATOR_MODELS["Balanced (BS-Roformer)"])
        output_dir = Path(request["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        key = f"separator:{model}"
        separator = self.models.get(key)
        if separator is None:
            # One separator at a time; another model replaces it.
            for name in [name for name in self.models if name.startswith("separator:")]:
                self.unload(name)
            separator = Separator(model_file_dir=str(self.root / "audio-separator"), output_dir=str(output_dir),
                                  output_format="WAV")
            separator.load_model(model_filename=model)
            self.models[key] = separator
        separator.output_dir = str(output_dir)
        if getattr(separator, "model_instance", None) is not None:
            separator.model_instance.output_dir = str(output_dir)
        files = separator.separate(request["input"], {"Vocals": "vocals", "Instrumental": "backing"})
        found = {}
        for name in files:
            path = Path(name) if Path(name).is_absolute() else output_dir / name
            stem = path.stem.lower()
            if "vocal" in stem:
                found["vocals"] = str(path)
            elif "backing" in stem or "instrument" in stem:
                found["backing"] = str(path)
        if set(found) != {"vocals", "backing"}:
            raise RuntimeError(f"audio-separator returned unexpected stems: {files}")
        return found

    # -- transcription ------------------------------------------------------

    def _asr(self, aligner: bool):
        # A loaded model with the aligner also serves text-only requests.
        for key in ("asr+aligner",) if aligner else ("asr", "asr+aligner"):
            if key in self.models:
                return self.models[key]
        from huggingface_hub import snapshot_download
        from qwen_asr import Qwen3ASRModel

        self.unload("asr")
        key = "asr+aligner" if aligner else "asr"
        kwargs = {"dtype": self.dtype(), "device_map": self.device()}
        model = Qwen3ASRModel.from_pretrained(
            snapshot_download(ASR_ID), max_inference_batch_size=ASR_BATCH, max_new_tokens=4096,
            forced_aligner=snapshot_download(ALIGNER_ID) if aligner else None,
            forced_aligner_kwargs=kwargs if aligner else None, **kwargs)
        self.models[key] = model
        return model

    @staticmethod
    def _speech_spans(audio_16k, duration: float) -> list[tuple[float, float]]:
        from omnivad import OmniVAD

        vad = OmniVAD()
        try:
            result = vad.detect(audio_16k, sample_rate=ASR_RATE, chunk_seconds=600 if duration > 600 else 0,
                                overlap_seconds=2.0)
        finally:
            close = getattr(vad, "close", None)
            if close is not None:
                close()
        return [(max(0.0, s), min(duration, e)) for s, e in result.get("timestamps") or [] if e > s]

    def transcribe(self, request: dict) -> dict:
        import librosa
        import numpy as np

        samples, _rate = librosa.load(request["audio"], sr=ASR_RATE, mono=True)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        duration = len(samples) / ASR_RATE
        aligner = bool(request.get("aligner", True))
        language = request.get("language") or None
        spans = merge_spans(self._speech_spans(samples, duration), float(request.get("merge_gap", VAD_MERGE_GAP)))
        spans = [(s, e) for s, e in spans if e - s >= MIN_SPAN_SECONDS]
        if not spans:
            return {"segments": [], "language": "", "spans": 0}
        asr = self._asr(aligner)
        segments, languages = [], []
        for first in range(0, len(spans), ASR_BATCH):
            batch = spans[first:first + ASR_BATCH]
            clips = [(samples[int(s * ASR_RATE):int(e * ASR_RATE)], ASR_RATE) for s, e in batch]
            try:
                results = asr.transcribe(audio=clips, language=language, return_time_stamps=aligner)
            except Exception:
                if not aligner:
                    raise
                traceback.print_exc()
                results = asr.transcribe(audio=clips, language=language, return_time_stamps=False)
            for (start, end), result in zip(batch, results):
                text = str(getattr(result, "text", "") or "").strip()
                if not text:
                    continue
                if getattr(result, "language", ""):
                    languages.append(str(result.language).split(",")[0])
                stamps = getattr(result, "time_stamps", None)
                items = list(getattr(stamps, "items", None) or stamps or [])
                segments += lines_from_alignment(items, start, end, text) if items else [
                    {"start": round(start, 3), "end": round(end, 3), "text": text}]
            print(f"Transcribed {min(first + ASR_BATCH, len(spans))}/{len(spans)} speech spans", file=sys.stderr)
        detected = max(set(languages), key=languages.count) if languages else ""
        return {"segments": segments, "language": detected, "spans": len(spans)}

    # -- translation --------------------------------------------------------

    def _translator(self):
        if "translator" in self.models:
            return self.models["translator"]
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = snapshot_download(TRANSLATOR_ID)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=self.dtype(),
                                                     device_map=self.device()).eval()
        self.models["translator"] = (tokenizer, model)
        return tokenizer, model

    def translate(self, request: dict) -> dict:
        texts = [str(text or "") for text in request.get("texts") or []]
        target = str(request.get("target_language") or "English")
        notes = str(request.get("notes") or "").strip()
        tokenizer, model = self._translator()
        torch = self.torch()
        out = [""] * len(texts)
        pending = [(index, text) for index, text in enumerate(texts) if text.strip()]
        for first in range(0, len(pending), TRANSLATE_BATCH):
            batch = pending[first:first + TRANSLATE_BATCH]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": (
                    f"Translate the following text into {target}. "
                    + (f"Context: {notes}\n" if notes else "")
                    + f"Note that you should only output the translated result without any additional "
                      f"explanation:\n\n{text}")}],
                tokenize=False, add_generation_prompt=True) for _index, text in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                                           max_new_tokens=512, do_sample=True, temperature=0.7, top_k=20,
                                           top_p=0.6, repetition_penalty=1.05,
                                           pad_token_id=tokenizer.pad_token_id)
            width = encoded["input_ids"].shape[-1]
            for (index, _text), row in zip(batch, generated):
                out[index] = clean_translation(tokenizer.decode(row[width:], skip_special_tokens=True))
            print(f"Translated {min(first + TRANSLATE_BATCH, len(pending))}/{len(pending)} lines", file=sys.stderr)
        return {"texts": out}


def serve(models_dir: str, requests=None, replies=None) -> None:
    if replies is None:
        # Keep the protocol on its own descriptor; stray prints (including C-level) go to the log.
        replies = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        # Separation and decoding scratch files stay beside the models, not on a small system disk.
        tempfile.tempdir = str(models_root(models_dir) / "tmp")
        Path(tempfile.tempdir).mkdir(parents=True, exist_ok=True)
    worker = Worker(Path(models_dir))
    for line in requests if requests is not None else sys.stdin.buffer:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            reply = {"ok": True, **worker.handle(request)}
        except Exception as exc:
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stderr.flush()
        replies.write(json.dumps({"id": request.get("id"), "loaded": worker.loaded, **reply},
                                 ensure_ascii=False) + "\n")
        replies.flush()


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(sys.argv[2])
    else:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")
