"""What the standalone pages outside Gradio share: options as a page draws them, the prompt writer's
settings, uploads, the Media library and owner-scoped files."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping
import uuid

from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse

from h3_app import catalog
from h3_app.errors import H3Error

SECRET_FIELDS = ("gemini_api_key", "openai_api_key")
UI_LIMIT = 4096


@dataclass(frozen=True)
class FieldSpec:
    """One option as the page draws it: its kind, its default and what it may be."""

    kind: str  # choice, bool, int, num, text or area
    label: str
    default: Any
    choices: tuple = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    hint: str = ""
    placeholder: str = ""


def spec_of(component) -> FieldSpec:
    """The Gradio control a model's view declares, as the option the page shows."""
    import gradio as gr

    label, hint = component.label or "", getattr(component, "info", None) or ""
    if isinstance(component, (gr.Dropdown, gr.Radio)):
        choices = tuple((str(name), value) for name, value in component.choices)
        return FieldSpec("choice", label, component.value, choices, hint=hint)
    if isinstance(component, gr.Checkbox):
        return FieldSpec("bool", label, bool(component.value), hint=hint)
    if isinstance(component, gr.Slider):
        whole = float(component.step or 1).is_integer()
        return FieldSpec("int" if whole else "num", label, component.value, minimum=component.minimum,
                         maximum=component.maximum, step=component.step, hint=hint)
    if isinstance(component, gr.Number):
        return FieldSpec("int" if component.precision == 0 else "num", label, component.value, hint=hint)
    return FieldSpec("area" if (getattr(component, "lines", 1) or 1) > 1 else "text", label, component.value or "",
                     hint=hint, placeholder=getattr(component, "placeholder", None) or "")


def coerce(spec: FieldSpec, value: Any) -> Any:
    """``value`` as the kind ``spec`` holds; a browser sends 6 for 6.0 and may send text for a number."""
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise H3Error(f"{spec.label} must be on or off.")
        return value
    if spec.kind == "choice":
        allowed = [choice for _name, choice in spec.choices]
        if value in allowed or (value is None and spec.default is None):
            return value
        raise H3Error(f"{spec.label}: choose one of the listed options.")
    if spec.kind in ("int", "num"):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise H3Error(f"{spec.label} must be a number.") from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise H3Error(f"{spec.label} must be a number.")
        if spec.minimum is not None and not spec.minimum <= number <= spec.maximum:
            raise H3Error(f"{spec.label} must be between {spec.minimum:g} and {spec.maximum:g}.")
        if spec.step:
            steps = (number - (spec.minimum or 0)) / spec.step
            if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
                raise H3Error(f"{spec.label} must change in increments of {spec.step:g} from {spec.minimum or 0:g}.")
        return int(round(number)) if spec.kind == "int" else number
    return "" if value is None else str(value)


def option_row(name: str, spec: FieldSpec, value: Any, **extra) -> dict[str, Any]:
    """One option as the page's JSON describes it."""
    return {"name": name, "label": spec.label, "kind": spec.kind, "value": value, "hint": spec.hint,
            "choices": [list(choice) for choice in spec.choices], "minimum": spec.minimum,
            "maximum": spec.maximum, "step": spec.step, "placeholder": spec.placeholder, **extra}


# The prompt writer ---------------------------------------------------------------
def writer_defaults() -> dict[str, Any]:
    return {"backend": "Gemini", "model": catalog.DEFAULT_GEMINI_PROMPT_MODEL, "gemini_api_key": "",
            "openai_api_key": "", "openai_model": catalog.DEFAULT_OPENAI_PROMPT_MODEL, "openai_base_url": "",
            # Score each result against its prompt and tag it with Gemini Decision (h3_app.media_review).
            "review": True}


REVIEW_HINT = ("After each result is saved, Gemini Decision scores how well it fits the prompt and adds Media library tags, in the background. Uses the Gemini key above, else GEMINI_API_KEY.")


def review_row(writer: Mapping[str, Any], **extra) -> dict[str, Any]:
    """The option row for prompt-fit scores and library tags; it is shown whatever writer is chosen."""
    return {"name": "review", "label": "Score results and tag them", "kind": "bool", "value": bool(writer["review"]),
            "store": "writer", "hint": REVIEW_HINT, "choices": [], **extra}


def writer_rows(writer: Mapping[str, Any], *, backends=("OpenAI-compatible", "Gemini"), **extra) -> list[dict[str, Any]]:
    """The writer's settings as option rows; only the chosen service's own are shown."""
    row = lambda name, label, kind, **more: {
        "name": name, "label": label, "kind": kind, "value": writer[name], "store": "writer",
        "hint": "", "choices": [], **extra, **more}
    rows = [row("backend", "Prompt writer", "choice", choices=[[name, name] for name in backends])]
    if writer["backend"] == "Gemini":
        rows += [row("model", "Gemini model", "choice", choices=[[name, name] for name in catalog.GEMINI_PROMPT_MODELS]),
                 row("gemini_api_key", "Temporary Gemini API key", "password", placeholder="Uses GEMINI_API_KEY when blank",
                     hint="The server does not store keys entered here.")]
    else:
        rows += [row("openai_model", "OpenAI-compatible model", "text",
                     hint="Any model your endpoint serves, for example " + ", ".join(catalog.OPENAI_PROMPT_MODELS[:3]) + "."),
                 row("openai_api_key", "Temporary OpenAI API key", "password", placeholder="Uses OPENAI_API_KEY when blank",
                     hint="The server does not store keys entered here."),
                 row("openai_base_url", "Temporary OpenAI endpoint", "text",
                     placeholder="Uses OPENAI_BASE_URL, else the official OpenAI API, when blank")]
        if writer["review"]:
            rows.append(row("gemini_api_key", "Temporary Gemini API key", "password",
                            placeholder="Uses GEMINI_API_KEY when blank",
                            hint="Only for result scores and tags. The server does not store keys entered here."))
    return [*rows, review_row(writer, **extra)]


def change_writer(writer: dict[str, Any], changes: Mapping[str, Any]) -> None:
    defaults = writer_defaults()
    for name, value in (changes or {}).items():
        if name not in defaults:
            raise H3Error(f"Unknown writer setting: {name}")
        if isinstance(defaults[name], bool):
            if not isinstance(value, bool):
                raise H3Error(f"{name} must be on or off.")
            writer[name] = value
        else:
            writer[name] = "" if value is None else str(value)


# Pictures, the Media library and files -----------------------------------------
def spool_image(upload: UploadFile, folder: Path, rollback: ExitStack) -> Path:
    """An uploaded picture saved in ``folder``; ``rollback`` removes it if the edit fails."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in catalog.IMAGE_EXTENSIONS:
        raise H3Error(f"{upload.filename} isn't a supported image ({', '.join(sorted(catalog.IMAGE_EXTENSIONS))}).")
    stem = re.sub(r"[^\w. -]", "_", Path(upload.filename or "").stem).strip(" .")[:60] or "upload"
    path = folder / f"{uuid.uuid4().hex[:12]}-{stem}{suffix}"
    with path.open("xb") as handle:
        rollback.callback(path.unlink, missing_ok=True)
        shutil.copyfileobj(upload.file, handle)
    return path


def library_source(library, source: str) -> str:
    """A Media library picture a page may copy, or why not."""
    if library is None:
        raise H3Error("The media library isn't available here.")
    library.validate_path(source, "Image")
    if not Path(source).is_file():
        raise H3Error("This image is unavailable. Refresh the library and choose another.")
    return source


def library_items(library, query: str, limit: int) -> dict[str, Any]:
    """One page of the Media library's pictures, with their thumbnails."""
    if library is None:
        raise H3Error("The media library isn't available here.")
    limit = max(1, min(limit, 240))
    page = library.refresh_page("Image", limit, paths=library.index.inventory("Image", query, limit=limit))
    version = library.index.versions.get("Image", 0)
    return {"items": [{"path": str(path), "caption": caption,
                       "thumb": f"/media-previews/{Path(thumb).name}?v={version}"}
                      for path, (thumb, caption) in zip(page.paths, page.items)]}


def owned_file(owner_of: Callable[[], str], may_read: Callable[[str, str], bool], path: str,
               download: bool) -> FileResponse:
    """``path`` for its owner only; every other answer is a plain 404."""
    try:
        owner = owner_of()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    file = Path(path)
    if not may_read(owner, path) or not file.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file, filename=file.name if download else None,
                        headers={"Cache-Control": "private, max-age=3600"})


def update_value(update) -> tuple[bool, Any]:
    """Whether a Gradio update sets a value, and the value."""
    if isinstance(update, dict) and update.get("__type__") == "update":
        return ("value" in update, update.get("value"))
    return (True, update)
