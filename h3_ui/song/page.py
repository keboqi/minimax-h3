"""The Song page and its JSON API, outside Gradio, as the Image page is.

Music 3 and YuE2 each keep their own draft (description or style, lyrics, score), options and takes.
Every edit is checked as the Gradio views checked it, and the answer says what the page shows: the
song check, whether a take may start, the takes to compare and the settings that differ from the
defaults. Write and Generate take the paths the Gradio buttons took, in a thread here, and the page
polls them. The Gradio views stay built but unseen for their job callbacks and published API.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
from threading import RLock
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from urllib.parse import quote
import uuid

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from h3_app import song_brief
from h3_app.capabilities import engines_for_task
from h3_app.errors import H3Error
from h3_app import media_review
from h3_app.jobs import JOBS
from h3_app.provenance import render_snapshot
from h3_app.reference_metadata import describe_reference
from h3_app.workspace_store import request_owner

from ..engine_readiness import song_issue
from ..h3.page import plain
from ..job_admission import execute_accepted
from ..page_fields import (
    SECRET_FIELDS, UI_LIMIT, FieldSpec, change_writer, coerce, library_items, library_source, option_row, owned_file,
    spec_of, spool_image, update_value, writer_defaults, writer_rows,
)
from ..page_runs import Run, Runs, guarded
from ..song_controls import DURATION_PRESETS, QUICK_FIXES, SECTION_INSERTS, STARTERS, reused_score_lyrics
from ..song_takes import MAX_TAKES, describe_takes, take_label

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
ROUTE = "/song"
API = f"{ROUTE}/api"
ENGINES = ("music", "yue2")
GENERATE, WRITE = "generate", "write"
# One take at a time per owner, whichever model sings it; each model's draft has its own Write.
TAKE_KEY = "song"
BUSY = {GENERATE: "A take is already being made.", WRITE: "The writer is already working on this song."}
REFERENCE_LIMIT = 3  # Music 3's writer reads up to three pictures.
# The draft's description is Music 3's caption and YuE2's style, by the names their generations take.
DESCRIPTION = {"music": "caption", "yue2": "style"}
# The engine names the song check and the writers know.
CHECK_ENGINE = {"music": "MiniMax Music 3", "yue2": "YuE2"}
WRITE_FAILED = "Prompt enhancement failed"

# What each generation takes, in its order: draft text by name, every other value from the options.
JOB_ORDER = {
    "music": ("model", "caption", "lyrics", "duration", "seed", "steps", "cfg", "ar_cfg", "top_k", "tiled"),
    "yue2": ("model", "style", "lyrics", "abc", "mode", "duration", "seed", "steps", "cfg", "temperature", "top_p",
             "top_k", "repetition_penalty", "max_abc_tokens", "abc_temperature", "abc_top_p", "abc_top_k",
             "abc_repetition_penalty", "abc_penalty_window", "tiled"),
}
DRAFT_TEXT = ("caption", "style", "lyrics", "abc")
# What a take remembers of the draft that made it, as the Gradio take history did.
TAKE_DRAFT = {
    "music": ("caption", "lyrics", "duration", "vocals", "language", "use"),
    "yue2": ("style", "lyrics", "abc", "mode", "duration", "vocals", "language", "use"),
}
# Options the brief and the score show themselves; Generation settings shows the rest by group.
BRIEF = ("duration", "vocals", "language", "use", "lyrics_policy", "mode")
GROUPS = (
    ("Model & seed", "Which weights load, and a fixed seed to repeat a take."),
    ("Sampling", "How the song is sampled."),
    ("Score writing", "How YuE2 plans the ABC score before it sings."),
    ("Decode", "Memory use while the audio is decoded."),
)
GROUP_OF = {
    "model": 0, "seed": 0,
    "steps": 1, "cfg": 1, "ar_cfg": 1, "top_k": 1, "temperature": 1, "top_p": 1, "repetition_penalty": 1,
    "max_abc_tokens": 2, "abc_temperature": 2, "abc_top_p": 2, "abc_top_k": 2, "abc_repetition_penalty": 2,
    "abc_penalty_window": 2,
    "tiled": 3,
}


def fields_of(views: Mapping[str, Any]) -> dict[str, FieldSpec]:
    """Each model's options by job name, as its unseen Gradio view declares them."""
    return {key_of(engine, name): spec_of(getattr(views[engine], name))
            for engine in ENGINES for name in (*GROUP_OF, *BRIEF) if getattr(views[engine], name, None) is not None}


@dataclass(frozen=True)
class SongPageServices:
    """What the page needs from the application; ``root`` is where owners' drafts and pictures live."""

    root: Callable[[], Path]
    # Each model's options by job name (``music__steps``), as its Gradio view declares them.
    fields: Mapping[str, FieldSpec]
    # Each model's writer, as its Gradio Write button called it: (description, lyrics, status).
    enhance: Mapping[str, Callable[..., tuple[str, str, str]]]
    # Seed, length and score of a finished take.
    take_facts: Mapping[str, Callable[[Any], Mapping[str, Any]]] = field(default_factory=dict)
    interrupt: Callable[..., str] | None = None
    library: Callable[[], Any] = lambda: None


@dataclass
class SongRun(Run):
    """A Write or a take: what it is doing, and what it left for the page."""

    engine: str = ""
    result: str | None = None
    suggestion: dict[str, str] | None = None
    original: dict[str, str] | None = None
    decided: bool = False

    def public(self, scene_id=None) -> dict[str, Any]:
        pending = self.suggestion if self.suggestion and not self.decided else None
        return {**super().public(scene_id), "engine": self.engine, "suggestion": pending}


def key_of(engine: str, parameter: str) -> str:
    return f"{engine}__{parameter}"


def private_song_root(runtime, workspace: Path) -> Path:
    """Keep owner drafts outside the served roots."""
    publics = (runtime.outputs_dir.resolve(), runtime.output_dir.resolve())
    root = (workspace / "song-page").resolve()
    holding = [public for public in publics if root.is_relative_to(public)]
    if holding:
        # The outermost served folder that holds the drafts; they move beside it, keeping their own
        # path inside it so two workspaces never share one drafts folder.
        outer = min(holding, key=lambda public: len(public.parts))
        root = (outer.with_name(f".{outer.name}-song-drafts") / root.relative_to(outer)).resolve()
        if any(root.is_relative_to(public) for public in publics):
            raise ValueError("Choose an H3_WORKSPACE_DIR outside the output directories for private song drafts.")
    return root


def blank_draft() -> dict[str, Any]:
    return {"description": "", "lyrics": "", "abc": "", "notes": "", "references": [],
            # The score reused last and the words its take sang, for the song check.
            "score_words": None,
            # The description and lyrics before an accepted or applied Write, for Undo.
            "previous": None}


def score_and_mode(take: Mapping[str, Any], current_mode: str) -> tuple[str, str, dict[str, str]]:
    """A take's score, the composition mode to sing it in, and the words it was sung with."""
    value = take.get("score") or ""
    if not value:
        raise H3Error(f"Take {take.get('number')} planned no score to reuse.")
    words = {"score": value, "lyrics": str((take.get("draft") or {}).get("lyrics") or "")}
    # Sing a score in the mode it was planned in: a melody-only score is not a full arrangement.
    planned = take.get("score_mode")
    if planned in {"full", "melody"}:
        return value, planned, words
    return value, ("full" if current_mode == "off" else current_mode), words


def _path_of(value) -> str | None:
    if isinstance(value, dict):
        return _path_of(value.get("path") or value.get("name"))
    if isinstance(value, (list, tuple)):
        return _path_of(value[0]) if value else None
    return str(value) if value else None


def _quote(path: str) -> str:
    return quote(str(path), safe="")


def _clock(seconds) -> str:
    seconds = round(float(seconds or 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


class SongPageState:
    """Each owner's drafts, options, writer and takes, saved as one JSON file."""

    def __init__(self, services: SongPageServices):
        self.services = services
        self.lock = RLock()
        self.secrets: dict[str, dict[str, str]] = {}
        self.granted: dict[str, set[str]] = {}
        self.labels = {engine.id: engine.label for engine in engines_for_task("Audio / Music")}

    # Storage ---------------------------------------------------------------
    def owner_root(self, owner: str) -> Path:
        folder = self.services.root() / re.sub(r"[^\w-]", "_", owner)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def uploads(self, owner: str) -> Path:
        folder = self.owner_root(owner) / "uploads"
        folder.mkdir(exist_ok=True)
        return folder

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.services.fields.items()}

    def load(self, owner: str) -> dict[str, Any]:
        saved: dict[str, Any] = {}
        try:
            saved = json.loads((self.owner_root(owner) / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        saved = saved if isinstance(saved, dict) else {}
        options = self.defaults()
        for name, value in (saved.get("options") or {}).items():
            if name in options:
                try:
                    options[name] = coerce(self.services.fields[name], value)
                except H3Error:
                    continue
        writer = writer_defaults()
        writer.update({name: value for name, value in (saved.get("writer") or {}).items()
                       if name in writer and name not in SECRET_FIELDS})
        writer.update(self.secrets.get(owner, {}))
        uploads = self.owner_root(owner) / "uploads"
        drafts, takes = {}, {}
        for engine in ENGINES:
            draft = blank_draft()
            kept = (saved.get("drafts") or {}).get(engine)
            if isinstance(kept, dict):
                for name in ("description", "lyrics", "abc", "notes"):
                    draft[name] = str(kept.get(name) or "")
                draft["references"] = [str(uploads / Path(path).name) for path in kept.get("references") or []
                                       if path and (uploads / Path(path).name).is_file()][:REFERENCE_LIMIT]
                for name in ("score_words", "previous"):
                    draft[name] = kept.get(name) if isinstance(kept.get(name), dict) else None
            drafts[engine] = draft
            takes[engine] = [take for take in ((saved.get("takes") or {}).get(engine) or [])
                             if isinstance(take, dict) and take.get("path") and Path(take["path"]).is_file()]
        ui = saved.get("ui") if isinstance(saved.get("ui"), dict) else {}
        return {"engine": saved.get("engine") if saved.get("engine") in ENGINES else ENGINES[0],
                "drafts": drafts, "options": options, "writer": writer, "takes": takes, "ui": ui}

    def save(self, owner: str, state: dict[str, Any]) -> None:
        self.secrets[owner] = {name: state["writer"].get(name) or "" for name in SECRET_FIELDS}
        kept = {**state, "writer": {name: value for name, value in state["writer"].items()
                                    if name not in SECRET_FIELDS}}
        target = self.owner_root(owner) / "settings.json"
        scratch = target.with_suffix(".tmp")
        scratch.write_text(json.dumps(kept, default=str), encoding="utf-8")
        scratch.replace(target)

    def may_read(self, owner: str, path) -> bool:
        if str(path) in self.granted.get(owner, ()):
            return True
        return self.owner_root(owner).resolve() in Path(path).resolve().parents

    def grant(self, owner: str, path) -> str:
        self.granted.setdefault(owner, set()).add(str(path))
        return str(path)

    def keep(self, owner: str, source, rollback: ExitStack | None = None) -> str:
        target = self.uploads(owner) / f"{uuid.uuid4().hex[:12]}-{Path(source).name}"
        if rollback is not None:
            rollback.callback(target.unlink, missing_ok=True)
        shutil.copy2(source, target)
        return str(target)

    def receive(self, owner: str, path: str) -> str:
        """Media hands over one picture for Music 3's writer to read."""
        with self.lock:
            state = self.load(owner)
            references = state["drafts"]["music"]["references"]
            if len(references) >= REFERENCE_LIMIT:
                raise H3Error("Visual inspiration is full. Remove a picture on the Song page first.")
            references.append(self.keep(owner, path))
            state["engine"] = "music"
            self.save(owner, state)
        return "Added image to Song · Visual inspiration. Open Song to use it."

    # The draft, as each model's generation and writer take it ----------------
    def value(self, state: Mapping[str, Any], engine: str, name: str) -> Any:
        """A generation input by its name: draft text, or one of the model's options."""
        draft = state["drafts"][engine]
        if name == DESCRIPTION[engine]:
            return draft["description"]
        if name in DRAFT_TEXT:
            return draft[name]
        return state["options"][key_of(engine, name)]

    def job_values(self, state: Mapping[str, Any], engine: str) -> list[Any]:
        return [self.value(state, engine, name) for name in JOB_ORDER[engine]]

    def take_draft(self, state: Mapping[str, Any], engine: str) -> dict[str, Any]:
        return {name: self.value(state, engine, name) for name in TAKE_DRAFT[engine]}

    def issue(self, state: Mapping[str, Any], engine: str | None = None) -> str:
        engine = engine or state["engine"]
        draft = state["drafts"][engine]
        return song_issue(engine, draft["description"], draft["lyrics"])

    def writer_arguments(self, state: Mapping[str, Any], engine: str) -> tuple:
        """The writer's inputs, in the order its Gradio Write button sent them."""
        draft, writer = state["drafts"][engine], state["writer"]
        option = lambda name: state["options"][key_of(engine, name)]
        base_url = writer["openai_base_url"] or None
        if engine == "music":
            pictures = [*draft["references"], None, None, None][:REFERENCE_LIMIT]
            return (draft["description"], writer["model"], writer["gemini_api_key"], draft["lyrics"], *pictures,
                    writer["backend"], writer["openai_api_key"], writer["openai_model"], option("duration"),
                    option("vocals"), option("language"), option("use"), option("lyrics_policy"), draft["notes"],
                    base_url)
        return (draft["description"], writer["model"], writer["gemini_api_key"], draft["lyrics"], option("mode"),
                option("duration"), writer["backend"], writer["openai_api_key"], writer["openai_model"],
                option("vocals"), option("language"), option("use"), option("lyrics_policy"), draft["notes"],
                base_url)

    # Edits -----------------------------------------------------------------
    def apply(self, state: dict[str, Any], action: str, changes: Mapping[str, Any]) -> None:
        engine, options = state["engine"], state["options"]
        if action == "engine":
            if changes.get("engine") not in ENGINES:
                raise H3Error("Choose a song model.")
            state["engine"] = changes["engine"]
        elif action == "restore":
            for name in self.fine_names(engine):
                options[name] = self.services.fields[name].default
        else:
            for changed, value in changes.items():
                if changed not in self.services.fields:
                    raise H3Error(f"Unknown setting: {changed}")
                options[changed] = coerce(self.services.fields[changed], value)

    def fine_names(self, engine: str) -> list[str]:
        prefix = key_of(engine, "")
        return [name for name in self.services.fields
                if name.startswith(prefix) and name.partition("__")[2] in GROUP_OF]

    def edit_draft(self, state: dict[str, Any], engine: str, changes: Mapping[str, Any]) -> None:
        if engine not in ENGINES:
            raise H3Error("Choose a song model.")
        draft = state["drafts"][engine]
        for name in ("description", "lyrics", "abc", "notes"):
            if name in changes:
                draft[name] = "" if changes[name] is None else str(changes[name])

    def add_take(self, state: dict[str, Any], engine: str, path: str, draft: Mapping[str, Any],
                 facts: Mapping[str, Any]) -> None:
        takes = state["takes"][engine]
        if any(take["path"] == str(path) for take in takes):
            return
        number = (takes[-1]["number"] + 1) if takes else 1
        takes.append({"number": number, "path": str(path), "draft": dict(draft), **dict(facts)})
        del takes[:-MAX_TAKES]
        # The newest take is A and the one before it B, as the Gradio take history chose them.
        state["ui"][f"{engine}_a"] = len(takes) - 1
        state["ui"][f"{engine}_b"] = len(takes) - 2 if len(takes) > 1 else None

    def take_at(self, state: Mapping[str, Any], index) -> dict[str, Any]:
        takes = state["takes"][state["engine"]]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(takes):
            raise H3Error("Choose one of the takes.")
        return takes[index]

    def reuse_draft(self, state: dict[str, Any], index) -> None:
        engine = state["engine"]
        draft = self.take_at(state, index).get("draft") or {}
        for name, value in draft.items():
            if name == DESCRIPTION[engine]:
                state["drafts"][engine]["description"] = str(value or "")
            elif name in DRAFT_TEXT:
                state["drafts"][engine][name] = str(value or "")
            elif key_of(engine, name) in self.services.fields:
                try:
                    state["options"][key_of(engine, name)] = coerce(self.services.fields[key_of(engine, name)], value)
                except H3Error:
                    continue

    def use_seed(self, state: dict[str, Any], index) -> None:
        take = self.take_at(state, index)
        if take.get("seed") is None:
            raise H3Error(f"Take {take['number']} has no recorded seed.")
        state["options"][key_of(state["engine"], "seed")] = int(take["seed"])

    def reuse_score(self, state: dict[str, Any], index) -> None:
        engine = state["engine"]
        if engine != "yue2":
            raise H3Error("Only YuE2 plans a score.")
        current = state["options"][key_of(engine, "mode")]
        score, mode, words = score_and_mode(self.take_at(state, index), current)
        state["drafts"][engine]["abc"] = score
        state["options"][key_of(engine, "mode")] = mode
        state["drafts"][engine]["score_words"] = words

    # What the page shows -----------------------------------------------------
    def option(self, name: str, options: Mapping[str, Any], **extra) -> dict[str, Any]:
        return option_row(name, self.services.fields[name], options[name], **extra)

    def fine(self, engine: str, options: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for name in self.fine_names(engine):
            default = self.services.fields[name].default
            changed = options[name] != default
            rows.append(self.option(name, options, group=GROUP_OF[name.partition("__")[2]], changed=changed,
                                    preset=default if changed else None))
        return rows

    def check(self, state: Mapping[str, Any], engine: str) -> str:
        draft, options = state["drafts"][engine], state["options"]
        option = lambda name: options[key_of(engine, name)]
        words = ""
        if engine == "yue2":
            words = reused_score_lyrics(draft["abc"], option("mode"), draft["score_words"])
        return song_brief.render_check(
            engine=CHECK_ENGINE[engine], description=draft["description"], lyrics=draft["lyrics"],
            max_seconds=float(option("duration") or 1), vocals=option("vocals"), use=option("use"),
            score_lyrics=words)

    def describe(self, owner: str, state: dict[str, Any]) -> dict[str, Any]:
        engine, options = state["engine"], state["options"]
        draft, takes = state["drafts"][engine], state["takes"][engine]
        rows = self.fine(engine, options)
        counts: dict[str, int] = {}
        for row in rows:
            if row["changed"]:
                counts[str(row["group"])] = counts.get(str(row["group"]), 0) + 1
        index = lambda value: value if isinstance(value, int) and 0 <= value < len(takes) else None
        a, b = index(state["ui"].get(f"{engine}_a")), index(state["ui"].get(f"{engine}_b"))
        if a is None and takes:
            a = len(takes) - 1
        issue = self.issue(state)

        def shown(position: int, take: Mapping[str, Any]) -> dict[str, Any]:
            path = self.grant(owner, take["path"])
            first = str((take.get("draft") or {}).get(DESCRIPTION[engine]) or "").strip().splitlines()
            return {"index": position, "number": take["number"], "label": take_label(take),
                    "time": _clock(take["seconds"]) if take.get("seconds") else "",
                    "seed": take.get("seed"), "has_score": bool(take.get("score")),
                    "note": first[0] if first else "", "name": Path(path).name,
                    "url": f"{API}/file?path={_quote(path)}", "review": media_review.public(path)}

        count = len(takes)
        return {
            "engine": engine, "label": self.labels.get(engine, engine), "ui": state["ui"],
            "draft": {"description": draft["description"], "lyrics": draft["lyrics"], "abc": draft["abc"],
                      "notes": draft["notes"],
                      "references": [{"path": path, "url": f"{API}/file?path={_quote(path)}",
                                      "detail": re.sub(r"^[0-9a-f]{12}-", "", describe_reference(path, "Picture"))}
                                     for path in draft["references"]]},
            "undo": draft["previous"] is not None,
            "brief": {name: self.option(key_of(engine, name), options) for name in BRIEF
                      if key_of(engine, name) in self.services.fields},
            "options": options, "fields": rows, "changed": counts,
            "seed": options[key_of(engine, "seed")],
            "check": self.check(state, engine),
            "readiness": {"ready": not issue, "message": issue or "Ready to generate."},
            "writer": state["writer"], "writer_rows": writer_rows(state["writer"]),
            "takes": {
                "items": [shown(position, take) for position, take in enumerate(takes)],
                "a": a, "b": b if b != a else None,
                "summary": (f"{count} take{'s' if count != 1 else ''} this session. Play A and B back to back and "
                            "keep the one you like; every take is also in Media." if count else
                            "Every take from this session lands here with the draft that made it."),
                "details": describe_takes(takes, a, b),
                "settings": render_snapshot([takes[a]["path"]]) if a is not None else "",
                "last_score": (takes[-1].get("score") or "") if takes else "",
            },
        }


def install_song_page(app: FastAPI, pages: SongPageState, *, gpu: bool = True) -> SongPageState:
    """Serve the page at ``/song/`` and its API under ``/song/api``."""
    services = pages.services
    runs = {kind: Runs(busy=message) for kind, message in BUSY.items()}

    def owner_of(request: Request) -> str:
        return request_owner(request, JOBS.store)

    def session_of(request: Request):
        return SimpleNamespace(cookies=dict(request.cookies), session_hash=None, headers=dict(request.headers))

    def can_generate(engine: str) -> bool:
        return gpu and engine in JOBS.callbacks

    def run_payloads(owner: str, engine: str) -> dict[str, Any]:
        take, written = runs[GENERATE].latest(owner, TAKE_KEY), runs[WRITE].latest(owner, engine)
        return {GENERATE: take.public() if take else None, WRITE: written.public() if written else None}

    def answer(owner: str, state: dict[str, Any], **extra) -> dict[str, Any]:
        return {"state": pages.describe(owner, state), "runs": run_payloads(owner, state["engine"]),
                "gpu": can_generate(state["engine"]), **extra}

    def edit(request: Request, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        owner = owner_of(request)
        with pages.lock:
            state = pages.load(owner)
            change(state)
            pages.save(owner, state)
            return answer(owner, state)

    # The page --------------------------------------------------------------
    @app.get(ROUTE, include_in_schema=False)
    def song_redirect():
        return RedirectResponse(f"{ROUTE}/")

    @app.get(f"{ROUTE}/", include_in_schema=False)
    def song_page():
        return FileResponse(STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get(ROUTE + "/{name}", include_in_schema=False)
    def song_asset(name: str):
        if name not in STATIC_TYPES:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name, media_type=STATIC_TYPES[name], headers={"Cache-Control": "no-cache"})

    @app.get(f"{API}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner = owner_of(request)
        with pages.lock:
            described = answer(owner, pages.load(owner))
        return {
            **described,
            "engines": [{"id": engine.id, "label": engine.label, "description": engine.description}
                        for engine in sorted(engines_for_task("Audio / Music"), key=lambda item: ENGINES.index(item.id))
                        if engine.id in ENGINES],
            "groups": [list(group) for group in GROUPS],
            "starters": {engine: list(ideas) for engine, ideas in STARTERS.items()},
            "durations": {engine: [list(preset) for preset in presets] for engine, presets in DURATION_PRESETS.items()},
            "quick_fixes": list(QUICK_FIXES), "sections": list(SECTION_INSERTS),
        }

    # Draft and options -------------------------------------------------------
    @app.post(f"{API}/settings", include_in_schema=False)
    @guarded
    def change_settings(request: Request, body: dict = Body(...)):
        changes = body.get("changes") or {}
        if not isinstance(changes, dict):
            raise H3Error("Send the changed settings by name.")
        return edit(request, lambda state: pages.apply(state, str(body.get("action") or "edit"), changes))

    @app.post(f"{API}/writer", include_in_schema=False)
    @guarded
    def writer_settings(request: Request, body: dict = Body(...)):
        return edit(request, lambda state: change_writer(state["writer"], body.get("changes")))

    @app.post(f"{API}/draft", include_in_schema=False)
    @guarded
    def change_draft(request: Request, body: dict = Body(...)):
        changes = body.get("changes") or {}
        if not isinstance(changes, dict):
            raise H3Error("Send the changed draft by name.")
        return edit(request, lambda state: pages.edit_draft(state, str(body.get("engine") or ""), changes))

    @app.post(f"{API}/empty", include_in_schema=False)
    @guarded
    def empty_draft(request: Request):
        return edit(request, lambda state: state["drafts"].update({state["engine"]: blank_draft()}))

    @app.post(f"{API}/ui", include_in_schema=False)
    @guarded
    def change_ui(request: Request, body: dict = Body(...)):
        """What only the page reads: the open settings group and the takes being compared."""
        ui = body.get("ui")
        if not isinstance(ui, dict) or len(json.dumps(ui)) > UI_LIMIT:
            raise H3Error("That is too much to remember.")
        return edit(request, lambda state: state["ui"].update(ui))

    # Writer review -----------------------------------------------------------
    def pending(owner: str, engine: str) -> SongRun:
        run = runs[WRITE].latest(owner, engine)
        if run is None or run.finished_at is None or not run.suggestion or run.decided:
            raise H3Error("There is no suggestion to review.")
        return run

    @app.post(f"{API}/accept", include_in_schema=False)
    @guarded
    def accept(request: Request):
        owner = owner_of(request)

        def change(state):
            engine = state["engine"]
            run = pending(owner, engine)
            run.decided = True
            draft = state["drafts"][engine]
            if {"description": draft["description"], "lyrics": draft["lyrics"]} != run.original:
                run.status = "The draft changed while the writer worked. Write again for the current draft."
                raise H3Error(run.status)
            draft["previous"] = dict(run.original)
            draft.update(run.suggestion)
            run.status = "Suggestion accepted."

        return edit(request, change)

    @app.post(f"{API}/keep", include_in_schema=False)
    @guarded
    def keep_original(request: Request):
        owner = owner_of(request)

        def change(state):
            run = pending(owner, state["engine"])
            run.decided = True
            run.status = "Original kept."

        return edit(request, change)

    @app.post(f"{API}/undo", include_in_schema=False)
    @guarded
    def undo(request: Request):
        def change(state):
            draft = state["drafts"][state["engine"]]
            if draft["previous"] is None:
                raise H3Error("There is nothing to undo.")
            draft.update(draft["previous"])
            draft["previous"] = None

        return edit(request, change)

    # Visual inspiration (Music 3) ---------------------------------------------
    def add(owner: str, paths: list[str], rollback: ExitStack) -> dict[str, Any]:
        with pages.lock:
            state = pages.load(owner)
            if state["engine"] != "music":
                raise H3Error("Only Music 3's writer reads pictures.")
            draft = state["drafts"]["music"]
            if len(draft["references"]) + len(paths) > REFERENCE_LIMIT:
                raise H3Error(f"Add at most {REFERENCE_LIMIT} pictures.")
            draft["references"] += paths
            pages.save(owner, state)
            rollback.pop_all()
            return answer(owner, state)

    @app.post(f"{API}/references/upload", include_in_schema=False)
    @guarded
    def upload_references(request: Request, files: list[UploadFile] = File(...)):
        owner = owner_of(request)
        folder = pages.uploads(owner)
        with ExitStack() as rollback:
            paths = [str(spool_image(upload, folder, rollback)) for upload in files]
            return add(owner, paths, rollback)

    @app.post(f"{API}/references/library", include_in_schema=False)
    @guarded
    def reference_from_library(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        source = library_source(services.library(), str(body.get("source") or ""))
        with ExitStack() as rollback:
            path = pages.keep(owner, source, rollback)
            return add(owner, [path], rollback)

    @app.post(f"{API}/references/remove", include_in_schema=False)
    @guarded
    def remove_reference(request: Request, body: dict = Body(...)):
        def change(state):
            references = state["drafts"]["music"]["references"]
            index = body.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(references):
                raise H3Error("Choose one of the pictures.")
            references.pop(index)

        return edit(request, change)

    @app.get(f"{API}/library", include_in_schema=False)
    @guarded
    def library_page(request: Request, query: str = "", limit: int = 24):
        owner_of(request)
        return library_items(services.library(), query, limit)

    @app.get(f"{API}/file", include_in_schema=False)
    def read_file(path: str, request: Request, download: bool = False):
        return owned_file(lambda: owner_of(request), pages.may_read, path, download)

    # Takes -------------------------------------------------------------------
    @app.post(f"{API}/takes/select", include_in_schema=False)
    @guarded
    def select_takes(request: Request, body: dict = Body(...)):
        def change(state):
            engine = state["engine"]
            for side in ("a", "b"):
                if side in body:
                    value = body[side]
                    if value is not None:
                        pages.take_at(state, value)
                    state["ui"][f"{engine}_{side}"] = value
            if "compare" in body:
                state["ui"]["compare"] = bool(body["compare"])

        return edit(request, change)

    @app.post(f"{API}/takes/reuse-draft", include_in_schema=False)
    @guarded
    def reuse_draft(request: Request, body: dict = Body(...)):
        return edit(request, lambda state: pages.reuse_draft(state, body.get("index")))

    @app.post(f"{API}/takes/use-seed", include_in_schema=False)
    @guarded
    def use_seed(request: Request, body: dict = Body(...)):
        return edit(request, lambda state: pages.use_seed(state, body.get("index")))

    @app.post(f"{API}/takes/reuse-score", include_in_schema=False)
    @guarded
    def reuse_score(request: Request, body: dict = Body(...)):
        def change(state):
            index = body.get("index")
            if index == "latest":
                index = len(state["takes"][state["engine"]]) - 1
                if index < 0:
                    raise H3Error("No take has planned a score yet.")
            pages.reuse_score(state, index)

        return edit(request, change)

    # Long work -------------------------------------------------------------
    def begin_take(owner: str, session, engine: str) -> SongRun:
        if not can_generate(engine):
            raise H3Error("Open Song in Studio to generate on the GPU.")
        with pages.lock:
            state = pages.load(owner)
            if issue := pages.issue(state, engine):
                raise H3Error(issue)
            values = pages.job_values(state, engine)
            # The draft is captured when the take starts, so edits made while it renders are not credited to it.
            draft = pages.take_draft(state, engine)
            brief = dict(state["drafts"][engine])
            writer = dict(state["writer"])
        label = pages.labels.get(engine, engine)

        def work(run: SongRun):
            job = JOBS.accept(owner, engine, values, callback=JOBS.callbacks[engine], key=uuid.uuid4().hex,
                              media_indices=(), output_indices=(0,))
            run.job_id = job.id
            for update in execute_accepted(job.id, session):
                sets, value = update_value(update[0])
                if sets and value and (path := _path_of(value)):
                    run.result = path
                status = update_value(update[1])[1] if len(update) > 1 else None
                if isinstance(status, str) and status:
                    run.status = plain(status)
                run.revision += 1
            if run.result:
                facts = services.take_facts.get(engine, lambda _path: {})(run.result) or {}
                with pages.lock:
                    latest = pages.load(owner)
                    pages.add_take(latest, engine, run.result, draft, facts)
                    pages.save(owner, latest)
                # In the background: the take is playable before its score and tags arrive.
                media_review.schedule(run.result, brief.get("description") or "", lyrics=brief.get("lyrics") or "",
                                      api_key=writer.get("gemini_api_key") or "",
                                      enabled=bool(writer.get("review", True)))
            if not run.status.startswith(label):
                run.status = f"{label} · {run.status}"

        run = SongRun(owner, TAKE_KEY, GENERATE, engine=engine)
        run.status = "Waiting for the GPU…"
        return runs[GENERATE].start(run, work)

    @app.post(f"{API}/generate", include_in_schema=False)
    @guarded
    def generate(request: Request):
        owner = owner_of(request)
        with pages.lock:
            engine = pages.load(owner)["engine"]
        begin_take(owner, session_of(request), engine)
        return {"runs": run_payloads(owner, engine)}

    def write(request: Request, *, then_generate: bool) -> dict[str, Any]:
        owner = owner_of(request)
        session = session_of(request)
        with pages.lock:
            state = pages.load(owner)
            engine = state["engine"]
            arguments = pages.writer_arguments(state, engine)
            draft = state["drafts"][engine]
            original = {"description": draft["description"], "lyrics": draft["lyrics"]}
        enhance = services.enhance[engine]

        def work(run: SongRun):
            run.status = "Writing…"
            description, lyrics, status = enhance(*arguments)
            message = plain(status)
            if message.startswith(WRITE_FAILED):
                raise H3Error(message)
            run.original = original
            suggestion = {"description": str(description or ""), "lyrics": str(lyrics or "")}
            if suggestion == original:
                run.status = message + (" The writer kept your draft as it is." if then_generate else "")
            elif not then_generate:
                run.suggestion = suggestion
                run.status = f"{message} Preview ready; choose Accept or Keep original."
                return
            else:
                # Write & generate applies the draft without review; Undo still works.
                with pages.lock:
                    latest = pages.load(owner)
                    current = latest["drafts"][engine]
                    if {"description": current["description"], "lyrics": current["lyrics"]} != original:
                        raise H3Error("The draft changed while the writer worked. Write again for the current draft.")
                    current["previous"] = dict(original)
                    current.update(suggestion)
                    pages.save(owner, latest)
                run.status = message + " Applied without review."
                run.revision += 1
            if not then_generate:
                return
            try:
                begin_take(owner, session, engine)
                run.status += " Starting a take."
            except H3Error as error:
                run.status += f" No take started: {error}"

        run = SongRun(owner, engine, WRITE, engine=engine)
        run.status = "Writing…"
        runs[WRITE].start(run, work)
        return {"runs": run_payloads(owner, engine)}

    @app.post(f"{API}/write", include_in_schema=False)
    @guarded
    def write_song(request: Request):
        return write(request, then_generate=False)

    @app.post(f"{API}/write-generate", include_in_schema=False)
    @guarded
    def write_and_generate(request: Request):
        return write(request, then_generate=True)

    @app.get(f"{API}/run", include_in_schema=False)
    @guarded
    def read_runs(request: Request, state: bool = False):
        owner = owner_of(request)
        with pages.lock:
            current = pages.load(owner)
            found: dict[str, Any] = {"runs": run_payloads(owner, current["engine"])}
            if state:
                found["state"] = pages.describe(owner, current)
                found["gpu"] = can_generate(current["engine"])
        return found

    @app.post(f"{API}/interrupt", include_in_schema=False)
    @guarded
    def stop(request: Request):
        owner = owner_of(request)
        with pages.lock:
            engine = pages.load(owner)["engine"]
        run = runs[GENERATE].active(owner, TAKE_KEY)
        if services.interrupt is None:
            raise H3Error("Nothing can run here.")
        if run is None:
            message = "Nothing is running."
        elif run.job_id is None:
            message = "The take is still starting. Try again in a moment."
        else:
            message = services.interrupt(session_of(request), family=run.engine, job_id=run.job_id)
        return {"message": message, "runs": run_payloads(owner, engine)}

    return pages
