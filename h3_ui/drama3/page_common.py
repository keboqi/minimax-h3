"""Pieces every step of the Soundstage page shares: runs, jobs, files, saved fields and tab answers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from functools import lru_cache
import uuid
import hashlib
import json

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import ConfigDict, ValidationError, create_model

from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, OPENAI_PROMPT_MODELS
from h3_app.errors import H3Error
from h3_app.jobs import JOBS, JobCancelled

from ..job_admission import execute_accepted
from ..page_runs import Run
from ..subtitle_words import word_pieces
from .result_view import timeline_data

SKIP = object()


def proposal_token(proposal: dict) -> str:
    """Identify one result, including proposals saved before explicit IDs existed."""
    return proposal.get("id") or hashlib.sha256(
        json.dumps(proposal, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def check_proposal(proposal: dict, expected) -> None:
    if not isinstance(expected, str) or expected != proposal_token(proposal):
        raise H3Error("The saved result changed. Review it again before applying.")


# The Speak step's defaults, for a project saved before a field existed.
SPEECH_DEFAULTS = {"temperature": 2.0, "top_p": 1.0, "top_k": 30, "max_frames": 8192, "parallel": 16, "retries": 0,
                   "word_timing": False, "script_mode": "Basic", "text": "", "title": ""}


@dataclass
class PageRun(Run):
    """A run of the page; ``family`` is its running job's family, which Interrupt names."""

    family: str | None = None
    stop_requested: bool = False

    def public(self, scene_id):
        return {**super().public(scene_id), "stop_requested": self.stop_requested}


def value(update) -> Any:
    """What a tab handler set a component to, or SKIP when it left it alone."""
    if isinstance(update, dict) and update.get("__type__") == "update":
        return update.get("value", SKIP)
    return update


def speech_value(data: dict, key: str):
    """A saved Speak setting, or its default when the project has none."""
    current = data.get(key)
    return SPEECH_DEFAULTS.get(key) if current is None else current


@lru_cache(maxsize=32)
def patch_model(fields: tuple):
    return create_model("SoundstagePatch", __config__=ConfigDict(strict=True, allow_inf_nan=False),
                        **{key: (kind, None) for key, kind in fields})


def apply_fields(data: dict, body: dict, fields: dict, ranges: dict | None = None,
                 choices: dict | None = None) -> None:
    """Copy each of ``fields`` that ``body`` sends into ``data`` as its kind, kept inside ``ranges`` (low, high, ...)."""
    try:
        values = patch_model(tuple(fields.items())).model_validate(body).model_dump(exclude_unset=True)
    except ValidationError as exc:
        issue = exc.errors()[0]
        raise H3Error(f"Invalid {issue['loc'][0]}: {issue['msg']}") from exc
    for key, current in values.items():
        kind = fields[key]
        if choices and key in choices and current not in choices[key]:
            raise H3Error(f"Unknown {key} choice.")
        data[key] = current
        if ranges and key in ranges:
            low, high = ranges[key][:2]
            data[key] = kind(min(high, max(low, data[key])))


def writer_service(body: dict) -> list[Any]:
    """The writer service inputs: model, Gemini key, backend, OpenAI key, model and base URL."""
    service = body.get("writer") or {}
    return [service.get("gemini_prompt_model") or DEFAULT_GEMINI_PROMPT_MODEL, service.get("gemini_api_key") or "",
            service.get("prompt_writer_backend") or "Gemini", service.get("openai_api_key") or "",
            service.get("openai_model") or OPENAI_PROMPT_MODELS[0], service.get("openai_base_url") or ""]


def run_job(run: PageRun, owner: str, session, family: str, values: list, callback, **options) -> Iterator:
    """Accept a job for ``run`` on the usual job path and yield each frame it makes.

    The job is accepted when the first frame is asked for, so a run refused before then leaves none.
    """
    if run.stop_requested:
        raise JobCancelled("Interrupted before job admission.")
    job = JOBS.accept(owner, family, values, callback=callback, key=uuid.uuid4().hex, **options)
    run.job_id, run.family = job.id, family
    yield from execute_accepted(job.id, session)


def not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="File not found")


@contextmanager
def found() -> Iterator[None]:
    """Answer a file request whose project, slot or run can't be read with 404, as a missing file."""
    try:
        yield
    except (H3Error, ValueError, IndexError) as exc:
        raise not_found() from exc


def file_response(path, *, cache: str = "private, max-age=3600", media_type: str | None = None,
                  filename: str | None = None) -> FileResponse:
    """Serve ``path``, or 404 when it is no file."""
    if not path or not Path(path).is_file():
        raise not_found()
    return FileResponse(path, media_type=media_type, filename=filename, headers={"Cache-Control": cache})


def timeline_payload(settings: dict) -> dict:
    """A snapshot's timed lines (with their aligned words), speakers and sounds, as the page draws them."""
    entries, words, tones, sounds = timeline_data(settings)
    lines = []
    for (speaker, text, start, end), line_words in zip(entries, words):
        pieces = word_pieces(text, line_words)
        lines.append({"speaker": speaker, "text": text, "start": start, "end": end,
                      "words": None if pieces is None else {
                          "lead": pieces[0], "items": [{"text": piece, "start": s, "end": e}
                                                       for piece, s, e in pieces[1]]}})
    return {
        "duration": float(settings.get("duration") or 0),
        "speakers": [{"name": name, "tone": tone, "pan": pan} for name, (tone, pan) in tones.items()],
        "lines": lines,
        "sounds": [{"name": name, "loop": loop, "start": start, "end": end} for name, loop, start, end in sounds],
    }
