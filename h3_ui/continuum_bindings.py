"""Sequence-mode callbacks: compile composer state, admit, run and present.

Composer components are shared with Clip, but every callback snapshots its own
values; a queued Sequence job never reads or writes Clip state.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from typing import Any, Callable

import gradio as gr

from h3_app.continuum_policy import (
    CONTINUITY_CHOICES,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_CONTINUITY,
    DEFAULT_SEQUENCE_SIZE,
    DEFAULT_SEQUENCE_STEPS,
    MAX_CHUNKS,
    MAX_REFERENCE_IMAGES,
    MAX_SEQUENCE_STEPS,
    MIN_SEQUENCE_STEPS,
    SEQUENCE_SCHEDULER,
    SEQUENCE_SIZES,
    compile_prompts,
    sequence_issues,
    sequence_notes,
)
from h3_app.continuum_store import ContinuumStore, RunBusyError, StaleRunError
from h3_app.errors import H3Error
from h3_app.generation import continuum as orchestrator
from h3_app.generation.continuum_requests import (
    CONTINUE_FROM_TAKE,
    FINISH,
    NEXT,
    REBUILD,
    REGENERATE_FROM,
    RETRY,
    USE_TAKE,
    ContinuumAction,
    ContinuumRequest,
)
from h3_app.jobs import record_failure

CLIP = "Clip"
SEQUENCE = "Sequence (experimental)"
VIDEO_MODES = (CLIP, SEQUENCE)
NEW_SEQUENCE = ("New sequence", "")

REF_IMAGES = tuple(f"ref_image_{i}" for i in range(1, MAX_REFERENCE_IMAGES + 1))
REF_VIDEOS = tuple(f"ref_video_{i}" for i in range(1, 4))
REF_AUDIOS = tuple(f"ref_audio_{i}" for i in range(1, 4))
CHUNK_PROMPTS = tuple(f"chunk_prompt_{i}" for i in range(1, MAX_CHUNKS + 1))

# Positional contract of the Sequence job; captured at click time.
SEQUENCE_INPUT_NAMES = (
    "sequence_action",
    "run_id",
    "epoch",
    "name",
    "shared_prompt",
    *CHUNK_PROMPTS,
    "chunk_count",
    "chunk_seconds",
    "continuity",
    "take",
    "regenerate_from",
    "mode",
    "first",
    "last",
    *REF_IMAGES,
    *REF_VIDEOS,
    *REF_AUDIOS,
    "size",
    "steps",
    "seed",
    "model_profile",
    "text_encoder",
)

# Inputs of the live (unqueued) summary.
DESCRIBE_INPUT_NAMES = (
    "run_id",
    "shared_prompt",
    *CHUNK_PROMPTS,
    "chunk_count",
    "chunk_seconds",
    "mode",
    "first",
    "last",
    *REF_IMAGES,
    *REF_VIDEOS,
    *REF_AUDIOS,
    "size",
    "generation_mode",
    "attention_mode",
    "cache_mode",
    "latent_upscale",
    "postprocess",
    "use_int8_vae",
    "use_trt_vae",
    "use_lynnreal_vae",
    "lightvae_decoder",
    "semantic_bridge",
)


def _path(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    return str(value) if value else None


def _actions(values: dict[str, Any]) -> tuple[str, ...]:
    count = max(1, min(MAX_CHUNKS, int(values.get("chunk_count") or 1)))
    return tuple(str(values.get(name) or "") for name in CHUNK_PROMPTS[:count])


def _size(values: dict[str, Any]) -> tuple[int, int]:
    try:
        return SEQUENCE_SIZES[values["size"]]
    except KeyError as exc:
        raise H3Error("Choose a Sequence size.") from exc


def _issues(values: dict[str, Any]) -> list[str]:
    width, height = _size(values)
    return sequence_issues(
        mode=values["mode"],
        chunk_count=int(values["chunk_count"] or 1),
        chunk_seconds=float(values["chunk_seconds"]),
        width=width,
        height=height,
        has_first_image=bool(_path(values.get("first"))),
        has_last_image=bool(_path(values.get("last"))),
        reference_images=sum(bool(_path(values.get(n))) for n in REF_IMAGES),
        reference_videos=sum(bool(_path(values.get(n))) for n in REF_VIDEOS),
        reference_audios=sum(bool(_path(values.get(n))) for n in REF_AUDIOS),
    )


def _parse_take(value: Any) -> tuple[int, str]:
    group, _, revision = str(value or "").partition(":")
    if not group.isdigit() or not revision:
        raise H3Error("Choose a saved take first.")
    return int(group), revision


def _first_changed(saved: ContinuumRequest, draft: ContinuumRequest) -> int:
    """First chunk whose effective prompt differs (count+1 when only extended)."""
    for index, (old, new) in enumerate(zip(saved.prompts, draft.prompts), start=1):
        if old != new:
            return index
    return min(saved.chunk_count, draft.chunk_count) + 1


@dataclass(frozen=True)
class ContinuumActions:
    store: Callable[[], ContinuumStore]
    services: Callable[[], orchestrator.ContinuumServices]
    owner: Callable[[Any], str]
    interrupt: Callable[..., str]

    # Admission ----------------------------------------------------------

    def admit(self, captured: dict[str, Any]) -> set[str]:
        """Validate at click time; return the media names worth pinning."""
        action = captured["sequence_action"]
        new = not captured["run_id"]
        if new and action not in {NEXT, FINISH}:
            raise H3Error("Save the sequence with its first generation before using takes.")
        if action in {NEXT, FINISH, RETRY, REGENERATE_FROM}:
            compile_prompts(captured["shared_prompt"], _actions(captured), len(_actions(captured)))
        if action in {USE_TAKE, CONTINUE_FROM_TAKE, REBUILD}:
            _parse_take(captured["take"])
        if not new:
            return set()  # Saved sequences use their retained sources.
        issues = _issues(captured)
        if issues:
            raise H3Error(" ".join(issues))
        if captured["mode"] == "First / last frame":
            return {"first"}
        if captured["mode"] == "Reference media":
            return set(REF_IMAGES)
        return set()

    def _new_request(self, values: dict[str, Any]) -> ContinuumRequest:
        seed = int(values["seed"])
        mode = values["mode"]
        width, height = _size(values)
        steps = int(values["steps"])
        if not MIN_SEQUENCE_STEPS <= steps <= MAX_SEQUENCE_STEPS:
            raise H3Error(f"Sequence steps must be {MIN_SEQUENCE_STEPS}–{MAX_SEQUENCE_STEPS}.")
        return ContinuumRequest(
            mode=mode,
            shared_prompt=str(values["shared_prompt"] or ""),
            actions=_actions(values),
            chunk_seconds=float(values["chunk_seconds"]),
            continuity=values["continuity"],
            width=width,
            height=height,
            # Resolve a random seed once; it is part of the saved request.
            seed=random.randrange(0, 2**63 - 1) if seed < 0 else seed,
            model_profile=values["model_profile"],
            text_encoder=values["text_encoder"],
            steps=steps,
            scheduler=SEQUENCE_SCHEDULER,
            first_image=_path(values["first"]) if mode == "First / last frame" else None,
            reference_images=tuple(
                _path(values[n]) if mode == "Reference media" else None for n in REF_IMAGES
            ),
        )

    # Generation ---------------------------------------------------------

    def run(self, *args, request: gr.Request = None):
        values = dict(zip(SEQUENCE_INPUT_NAMES, args, strict=True))
        store = self.store()
        owner = self.owner(request)
        run_id = values["run_id"] or ""
        try:
            epoch = int(values["epoch"] or 0)
            kind = values["sequence_action"]
            if not run_id:
                issues = _issues(values)
                if issues:
                    raise H3Error(" ".join(issues))
                run_id = store.create_run(owner, values["name"] or "", self._new_request(values))
                epoch = 0
                yield gr.skip(), "Sequence saved. Preparing the first chunk…", run_id, "0"
            reuse_cap = None
            _, saved = store.request(owner, run_id)
            if kind in {NEXT, FINISH, RETRY, REGENERATE_FROM}:
                # Only prompts and the chunk target are editable on a saved
                # sequence; everything else is part of its reuse contract.
                draft = replace(
                    saved,
                    shared_prompt=str(values["shared_prompt"] or ""),
                    actions=_actions(values),
                )
                if draft.digest() != saved.digest():
                    store.revise(owner, run_id, draft, expected_epoch=epoch)
                    epoch += 1
                    reuse_cap = _first_changed(saved, draft) - 1
            if kind in {USE_TAKE, CONTINUE_FROM_TAKE, REBUILD}:
                group, revision = _parse_take(values["take"])
                action = ContinuumAction(kind, take_group=group, take_revision_id=revision)
            elif kind == REGENERATE_FROM:
                action = ContinuumAction(kind, regenerate_from=int(values["regenerate_from"] or 0))
            else:
                action = ContinuumAction(kind)
            for update in orchestrator.execute(
                owner,
                run_id,
                action,
                self.services(),
                expected_epoch=epoch,
                reuse_cap=reuse_cap,
            ):
                yield (update.output or gr.skip()), update.status, run_id, gr.skip()
            yield gr.skip(), gr.skip(), run_id, str(store.run(owner, run_id)["epoch"])
        except (H3Error, StaleRunError, RunBusyError, ValueError) as exc:
            record_failure(exc)
            epoch_value = str(store.run(owner, run_id)["epoch"]) if run_id else "0"
            yield gr.skip(), f"Error: {exc}", run_id, epoch_value

    # Presentation -------------------------------------------------------

    def saved_choices(self, request: gr.Request):
        runs = self.store().runs(self.owner(request))
        choices = [NEW_SEQUENCE] + [
            (f"{run['name']} · {run['state']}", run["id"]) for run in runs
        ]
        return gr.update(choices=choices)

    def load(self, run_id: str, request: gr.Request):
        """Fill the Sequence composer from a saved run (Clip state untouched)."""
        empty = ("",) * MAX_CHUNKS
        if not run_id:
            return (
                "", "0", "", "", *empty, 1,
                *(gr.update(visible=i == 0) for i in range(MAX_CHUNKS)),
                DEFAULT_CHUNK_SECONDS, DEFAULT_CONTINUITY,
                DEFAULT_SEQUENCE_SIZE, DEFAULT_SEQUENCE_STEPS, -1,
                None,
                "New sequence. It is saved when you first generate, so later "
                "chunks can continue from it.",
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                primary_update(1, 0, saved=False),
            )
        store = self.store()
        owner = self.owner(request)
        run = store.run(owner, run_id)
        _, saved = store.request(owner, run_id)
        project = store.project(run)
        prompts = tuple(saved.actions) + ("",) * (MAX_CHUNKS - saved.chunk_count)
        takes = project.takes if project else ()
        head_group = max((take.group for take in takes if take.active), default=None)
        active = project.active_take(head_group) if project and head_group else None
        accepted = project.accepted_chunks if project else 0
        return (
            run_id,
            str(run["epoch"]),
            run["name"],
            saved.shared_prompt,
            *prompts[:MAX_CHUNKS],
            saved.chunk_count,
            *(gr.update(visible=i < saved.chunk_count) for i in range(MAX_CHUNKS)),
            saved.chunk_seconds,
            saved.continuity,
            next(
                (label for label, size in SEQUENCE_SIZES.items() if size == (saved.width, saved.height)),
                DEFAULT_SEQUENCE_SIZE,
            ),
            saved.steps,
            saved.seed,
            run["output"] if run["output"] and os.path.isfile(run["output"]) else None,
            describe_run(run, saved, project),
            gr.update(
                choices=[(take.label, f"{take.group}:{take.revision_id}") for take in takes],
                value=f"{active.group}:{active.revision_id}" if active else None,
            ),
            gr.update(choices=[str(i) for i in range(1, accepted + 1)], value=None),
            primary_update(saved.chunk_count, accepted, saved=True),
        )

    def delete(self, run_id: str, request: gr.Request):
        if run_id:
            try:
                self.store().delete_run(self.owner(request), run_id)
            except (RunBusyError, ValueError) as exc:
                raise gr.Error(str(exc)) from exc
        return self.load("", request)

    def describe(self, *args, request: gr.Request = None):
        """Live summary: effective prompts, limits, notes, affected range."""
        values = dict(zip(DESCRIBE_INPUT_NAMES, args, strict=True))
        run_id = values["run_id"]
        shared = values["shared_prompt"]
        count = max(1, min(MAX_CHUNKS, int(values["chunk_count"] or 1)))
        values["chunk_count"] = count
        seconds = float(values["chunk_seconds"])
        prompts = _actions(values)
        lines = [f"**Total:** {count} chunk(s) × {seconds:g} s = {count * seconds:g} s"]
        try:
            compiled = compile_prompts(shared, prompts, count)
            preview = "\n\n".join(
                f"**Chunk {i}**\n\n```text\n{text}\n```" for i, text in enumerate(compiled, start=1)
            )
        except H3Error as exc:
            compiled = None
            preview = f"⚠ {exc}"
        accepted = 0
        edited_from = None
        if run_id:
            store = self.store()
            owner = self.owner(request)
            run = store.run(owner, run_id)
            _, saved = store.request(owner, run_id)
            project = store.project(run)
            accepted = project.accepted_chunks if project else 0
            lines.append(
                "Settings, size, seed, model and source images are locked to this "
                "saved sequence; prompts and chunk count stay editable."
            )
            if compiled is not None:
                draft = replace(saved, shared_prompt=str(shared or ""), actions=prompts)
                if draft.digest() != saved.digest():
                    changed = edited_from = _first_changed(saved, draft)
                    if changed <= accepted:
                        lines.append(
                            f"⚠ Your edits regenerate chunk {changed} onward "
                            f"(chunks {changed}–{max(accepted, count)}); chunks before it are reused."
                        )
                    else:
                        lines.append(f"Edits apply from chunk {changed}; saved chunks are reused.")
        else:
            lines += [f"⚠ {issue}" for issue in _issues(values)]
            lines += [
                f"ℹ {note}"
                for note in sequence_notes(
                    generation_mode=values["generation_mode"],
                    attention_mode=values["attention_mode"],
                    cache_mode=values["cache_mode"],
                    latent_upscale=bool(values["latent_upscale"]),
                    postprocess=values["postprocess"],
                    alternate_decoder=bool(
                        values["use_int8_vae"]
                        or values["use_trt_vae"]
                        or values["use_lynnreal_vae"]
                        or values["lightvae_decoder"] not in {None, "none"}
                    ),
                    semantic_bridge=bool(values["semantic_bridge"]),
                )
            ]
        return (
            "\n\n".join(lines),
            preview,
            primary_update(count, accepted, saved=bool(run_id), edited_from=edited_from),
        )


def primary_label(count: int, accepted: int, *, saved: bool, edited_from: int | None = None) -> str:
    if not saved:
        return "Save sequence & generate"
    if edited_from is not None and edited_from <= accepted:
        # Continuing an edited request regenerates its first changed chunk.
        return f"Regenerate chunk {edited_from}"
    if accepted == 0:
        return "Generate chunk 1"
    if accepted < count:
        return f"Accept & generate chunk {accepted + 1}"
    return "Complete · add a chunk"


def primary_update(count: int, accepted: int, *, saved: bool, edited_from: int | None = None):
    # A complete, unedited sequence has nothing to continue until a chunk is added.
    return gr.update(
        value=primary_label(count, accepted, saved=saved, edited_from=edited_from),
        interactive=not saved or accepted < count or (edited_from or count + 1) <= accepted,
    )


def describe_run(run: dict[str, Any], request: ContinuumRequest, project) -> str:
    accepted = project.accepted_chunks if project else 0
    lines = [
        f"### {run['name']}",
        f"**{accepted}/{request.chunk_count} chunk(s) saved** · state {run['state']} · "
        f"{request.mode} · {request.width}×{request.height} · seed {request.seed} · "
        f"{request.steps} steps/{request.scheduler} · {request.chunk_seconds:g} s chunks · "
        f"{request.continuity} continuity",
    ]
    if project and project.review_unit and accepted < request.chunk_count:
        start, end = project.review_unit
        unit = f"chunk {start}" if start == end else f"chunks {start}–{end}"
        lines.append(f"Reviewing {unit}. Accept it by generating the next chunk, or retry it.")
    for warning in (run.get("report") or {}).get("warnings") or ():
        lines.append(f"⚠ {warning}")
    if run["state"] in {"failed", "interrupted"}:
        lines.append("The last operation did not finish; saved chunks are intact.")
    return "\n\n".join(lines)


__all__ = [
    "CLIP",
    "SEQUENCE",
    "VIDEO_MODES",
    "SEQUENCE_INPUT_NAMES",
    "DESCRIBE_INPUT_NAMES",
    "CONTINUITY_CHOICES",
    "ContinuumActions",
]
