"""Immutable Sequence (H3 Continuum) requests and review actions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from h3_app.continuum_policy import (
    CONTINUITY_CHOICES,
    MAX_REFERENCE_IMAGES,
    PROMPT_COMPILER_VERSION,
    REFERENCE_USE_ALL,
    REFERENCE_USE_PER_CHUNK,
    REVIEW_CONTINUE,
    REVIEW_REGENERATE_CURRENT,
    TAKE_AUTOMATIC,
    TAKE_CONTINUE,
    TAKE_USE,
    chunk_selector,
    compile_prompts,
)
from h3_app.errors import H3Error


@dataclass(frozen=True)
class ContinuumRequest:
    """Everything that determines sampling; saved verbatim per request version."""

    mode: str
    shared_prompt: str
    actions: tuple[str, ...]
    chunk_seconds: float
    continuity: str
    width: int
    height: int
    seed: int
    model_profile: str
    text_encoder: str
    steps: int
    scheduler: str
    first_image: str | None = None
    reference_images: tuple[str | None, ...] = (None,) * MAX_REFERENCE_IMAGES
    # None applies every image to every chunk; otherwise one chunk tuple per slot.
    reference_chunks: tuple[tuple[int, ...], ...] | None = None
    prompt_compiler_version: int = PROMPT_COMPILER_VERSION
    prompts: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self):
        if len(self.reference_images) != MAX_REFERENCE_IMAGES:
            raise H3Error("Sequence references use nine fixed slots.")
        if self.continuity not in CONTINUITY_CHOICES:
            raise H3Error(f"Unknown continuity preset: {self.continuity}")
        if int(self.seed) < 0:
            raise H3Error("Resolve the Sequence seed before saving the request.")
        if self.reference_chunks is not None and len(self.reference_chunks) != MAX_REFERENCE_IMAGES:
            raise H3Error("Per-chunk reference assignments need nine slots.")
        object.__setattr__(
            self, "prompts", compile_prompts(self.shared_prompt, self.actions, len(self.actions))
        )

    @property
    def chunk_count(self) -> int:
        return len(self.actions)

    @property
    def total_seconds(self) -> float:
        return self.chunk_count * float(self.chunk_seconds)

    @property
    def reference_use(self) -> str:
        return REFERENCE_USE_ALL if self.reference_chunks is None else REFERENCE_USE_PER_CHUNK

    def reference_selectors(self) -> tuple[str, ...]:
        if self.reference_chunks is None:
            return ("off",) * MAX_REFERENCE_IMAGES
        return tuple(
            chunk_selector(chunk for chunk in chunks if chunk <= self.chunk_count)
            for chunks in self.reference_chunks
        )

    def payload(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("prompts")
        return values

    @classmethod
    def from_payload(cls, values: Mapping[str, Any]) -> ContinuumRequest:
        data = dict(values)
        data.pop("prompts", None)
        data["actions"] = tuple(data["actions"])
        data["reference_images"] = tuple(data.get("reference_images") or (None,) * MAX_REFERENCE_IMAGES)
        chunks = data.get("reference_chunks")
        data["reference_chunks"] = (
            None if chunks is None else tuple(tuple(int(c) for c in row) for row in chunks)
        )
        return cls(**data)

    def digest(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def with_actions(self, actions: tuple[str, ...]) -> ContinuumRequest:
        return replace(self, actions=tuple(actions))


# App actions. Each one maps to a single upstream review operation, except
# "finish", which the orchestrator repeats as ordinary "next" operations.
NEXT = "next"
RETRY = "retry"
FINISH = "finish"
USE_TAKE = "use_take"
CONTINUE_FROM_TAKE = "continue_from_take"
REGENERATE_FROM = "regenerate_from"
REBUILD = "rebuild"
ACTIONS = (NEXT, RETRY, FINISH, USE_TAKE, CONTINUE_FROM_TAKE, REGENERATE_FROM, REBUILD)
SAMPLING_ACTIONS = frozenset({NEXT, RETRY, FINISH, CONTINUE_FROM_TAKE, REGENERATE_FROM})


@dataclass(frozen=True)
class ContinuumAction:
    kind: str
    take_group: int = 0
    take_revision_id: str = ""
    regenerate_from: int = 0

    def __post_init__(self):
        if self.kind not in ACTIONS:
            raise H3Error(f"Unknown Sequence action: {self.kind}")
        if self.kind in {USE_TAKE, CONTINUE_FROM_TAKE, REBUILD}:
            if self.take_group < 1 or not self.take_revision_id:
                raise H3Error("Choose a saved take first.")
            if not all(c in "0123456789abcdef" for c in self.take_revision_id):
                raise H3Error("The selected take id is invalid.")
        if self.kind == REGENERATE_FROM and self.regenerate_from < 1:
            raise H3Error("Choose the chunk to regenerate from.")

    def sampler_controls(self) -> dict[str, Any]:
        """Compile every one-shot upstream review field for this operation.

        Stale take selections and reroll intent are cleared on every action, so
        previewing a take never redirects Retry or Continue.
        """
        controls = {
            "review_action": REVIEW_CONTINUE,
            "reroll_from_chunk": "Auto",
            "reroll_nonce": 0,
            "take_action": TAKE_AUTOMATIC,
            "take_group": 0,
            "take_revision_id": "",
        }
        if self.kind == RETRY:
            controls["review_action"] = REVIEW_REGENERATE_CURRENT
        elif self.kind == REGENERATE_FROM:
            controls["reroll_from_chunk"] = f"Chunk {int(self.regenerate_from)}"
        elif self.kind in {USE_TAKE, REBUILD, CONTINUE_FROM_TAKE}:
            controls.update(
                take_action=TAKE_CONTINUE if self.kind == CONTINUE_FROM_TAKE else TAKE_USE,
                take_group=int(self.take_group),
                take_revision_id=str(self.take_revision_id),
            )
        return controls

    @property
    def samples(self) -> bool:
        return self.kind in SAMPLING_ACTIONS
