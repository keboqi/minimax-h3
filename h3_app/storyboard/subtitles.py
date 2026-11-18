"""Prompt captions and audio-aligned subtitles for storyboard takes."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from textwrap import wrap

from h3_app.burned_subtitles import word_pieces
from h3_app.dubbing.tools_worker import ASR_RATE
from h3_app.errors import H3Error
from h3_app.processes import run_media_process

from .model import FPS


def take_lines(take: dict) -> list[dict]:
    """Use the take's saved audio analysis or estimate timings from its original prompt."""
    if "subtitles" in take:
        return take["subtitles"]
    inputs = take["inputs"]
    start = take.get("prefix_frames", 0) / FPS
    end = float(inputs.get("seconds", 0))
    duration = end - start
    if duration <= 0:
        return []
    texts = []
    for block in re.findall(r"<d>(.*?)</d>", inputs.get("prompt", ""), re.I | re.S):
        body = re.sub(r"^\s*\[[^\]\n]+\]", "", block, count=1)
        body = re.sub(r"<(?:scenetrans|cutoff)>", " ", body, flags=re.I)
        texts.extend(wrap(" ".join(body.split()), width=62))
    length = sum(map(len, texts))
    lines = []
    for caption in texts:
        stop = start + duration * len(caption) / length
        lines.append({"text": caption, "start": start, "end": stop})
        start = stop
    return lines


def transcribe(video: str, runtime, *, free_gpu, report) -> list[dict]:
    from h3_app.dubbing import tools
    from h3_models import resolve_hf_token

    with TemporaryDirectory(prefix="h3-storyboard-subtitles-") as folder:
        audio = Path(folder) / "take.wav"
        process = run_media_process(
            ["ffmpeg", "-y", "-v", "error", "-i", video, "-map", "0:a:0", "-vn",
             "-ac", "1", "-ar", str(ASR_RATE), str(audio)], timeout=120,
        )
        if process.returncode != 0:
            raise H3Error(f"Could not extract the take's audio: {process.stderr.strip()[-300:]}")
        result = tools.run("transcribe", {"audio": str(audio), "aligner": True, "word_details": True},
                           runtime, free_gpu=free_gpu, hf_token=resolve_hf_token(), report=report)
    return aligned_lines(result.get("groups", []))


def aligned_lines(groups: list[dict]) -> list[dict]:
    """Attach the worker's word timings to its subtitle lines when their text matches."""
    lines = []
    for group in groups:
        for line in group["lines"]:
            # The worker rounds line bounds to milliseconds and returns raw word bounds.
            words = [word for word in group["words"]
                     if line["start"] <= round(word["start"], 3)
                     and round(word["end"], 3) <= line["end"]]
            lines.append({**line, "words": words if word_pieces(line["text"], words) else []})
    return lines


def shifted_lines(lines: list[dict], start: float, end: float, offset: float) -> list[dict]:
    """Keep the part of a take that plays in the full video and shift it onto that timeline."""
    shifted = []
    for line in lines:
        first, last = max(start, line["start"]), min(end, line["end"])
        if last <= first:
            continue
        text = line["text"]
        pieces = word_pieces(text, line.get("words"))
        words = []
        if pieces:
            lead, timed = pieces
            kept = [(piece, max(a, first), min(b, last)) for piece, a, b in timed if b > first and a < last]
            if not kept:
                continue
            text = ((lead if timed[0][2] > first else "") + "".join(piece for piece, _, _ in kept)).strip()
            words = [{"text": piece, "start": a + offset, "end": b + offset} for piece, a, b in kept]
        shifted.append({"text": text, "start": first + offset, "end": last + offset, "words": words})
    return shifted


def _timestamp(milliseconds: int, separator=".") -> str:
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{ms:03d}"


def subtitle_text(lines: list[dict], *, web: bool) -> str:
    """WebVTT with native cue timestamps for playback, or plain SRT for MP4 muxing."""
    cues = []
    sep = "." if web else ","
    for line in sorted(lines, key=lambda line: line["start"]):
        start, end = (max(0, round(line[bound] * 1000)) for bound in ("start", "end"))
        if end <= start:
            continue
        body = escape(line["text"], quote=False)
        pieces = word_pieces(line["text"], line.get("words")) if web else None
        if pieces:
            lead, timed = pieces
            chunks = [escape(lead, quote=False)]
            previous = start
            for piece, word_start, _ in timed:
                at = round(word_start * 1000)
                if previous < at < end:
                    chunks.append(f"<{_timestamp(at)}>")
                    previous = at
                chunks.append(f"<c>{escape(piece, quote=False)}</c>")
            body = "".join(chunks)
        timing = f"{_timestamp(start, sep)} --> {_timestamp(end, sep)}"
        cues.append(f"{len(cues) + 1}\n{timing}\n{body}")
    return ("WEBVTT\n\n" if web else "") + ("\n\n".join(cues) + "\n" if cues else "")
