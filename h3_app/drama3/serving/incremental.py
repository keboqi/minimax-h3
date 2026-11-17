"""Recover individual lines from older, aligned multi-line speech takes."""

from types import SimpleNamespace
import logging

from h3_app.drama3.results import result_settings
from h3_app.jobs import CURRENT_JOB, JobCancelled
from . import audio, takes
from .alignment import align_take


def _aligned_bounds(take, root, runtime, free_gpu, hf_token):
    from h3_app.dubbing import tools

    try:
        return align_take(take, root, runtime, free_gpu=free_gpu, hf_token=hf_token)[0]
    finally:
        tools.stop()  # Release alignment VRAM before speech synthesis starts.


def migrate_lines(previous_result, root, wanted, references, sampling, budget, *, runtime, free_gpu,
                  hf_token=None):
    """Populate unchanged line caches from verified older grouped takes.

    Only use a legacy take whose complete generation identity still matches the selected
    voice and sampling. Split its dry audio with verified alignment, never by guessed timing.
    A failed alignment leaves the new line uncached so normal synthesis can replace it.
    """
    previous = result_settings(previous_result) if previous_result else {}
    occurrences, line_occurrences = {}, {}
    for chunk in previous.get("chunks") or []:
        job = CURRENT_JOB.get()
        if job is not None:
            job.check()
        speaker = chunk.get("name") or chunk.get("speaker")
        if speaker not in references:
            continue
        reference, transcript = references[speaker]

        def key_for(text):
            return takes.take_key(speaker=speaker, text=text, reference=reference.stem,
                                  transcript=transcript, sampling=sampling,
                                  max_new_tokens=budget(SimpleNamespace(text=text)))

        content = key_for(chunk.get("text", ""))
        occurrence = occurrences.get(content, 0)
        occurrences[content] = occurrence + 1
        valid = chunk.get("take") == takes.occurrence_key(content, occurrence)
        lines = chunk.get("lines") or []
        entries = []
        for line in lines:
            content = key_for(line["text"])
            occurrence = line_occurrences.get(content, 0)
            line_occurrences[content] = occurrence + 1
            entries.append(takes.occurrence_key(content, occurrence))
        if not valid or len(lines) < 2 or not chunk.get("ok", True):
            continue
        missing = [key for key in entries if key in wanted and not (root / f"{key}.wav").is_file()]
        if not missing:
            continue
        saved = takes.load_take(root, chunk["take"])
        if saved is None:
            continue
        samples, rate = saved
        take = SimpleNamespace(samples=samples, rate=rate, chunk=SimpleNamespace(
            text=chunk["text"], lines=tuple(line["text"] for line in lines)))
        try:
            bounds = _aligned_bounds(take, root, runtime, free_gpu, hf_token)
        except JobCancelled:
            raise
        except Exception as exc:
            if job is not None:
                job.check()
            logging.warning("Cannot split older Drama 3 take %s; missing lines will be generated: %s",
                            chunk["take"], exc)
            continue
        tidied = audio.tidy_take(samples, rate)
        for key, (start, end) in zip(entries, bounds, strict=True):
            if key in missing and 0 <= start < end <= len(tidied):
                takes.save_take(root, key, audio.encode_wav(tidied[start:end], rate))
