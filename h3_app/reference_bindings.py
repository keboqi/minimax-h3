"""Translate stable visible reference tags to dense engine ordinals.

The same map is used for prompt providers and generation. It is independent
of Gradio and preserves the exact user's prompt in the draft.
"""

from dataclasses import dataclass
import re
import uuid

from .errors import H3Error
from .policy import normalize_paths

TAG = re.compile(r"<(Picture|Video|Audio) ([1-9][0-9]*)>")
KINDS = (
    ("Picture", "ref_image", 9),
    ("Video", "ref_video", 3),
    ("Audio", "ref_audio", 3),
)


@dataclass(frozen=True)
class ReferenceBinding:
    asset_id: str
    kind: str
    slot: int
    path: str
    engine_ordinal: int
    unresolved: bool = False

    @property
    def tag(self):
        return f"<{self.kind} {self.slot}>"

    @property
    def engine_tag(self):
        return f"<{self.kind} {self.engine_ordinal}>"


@dataclass(frozen=True)
class ReferenceMap:
    version: int = 0
    bindings: tuple[ReferenceBinding, ...] = ()

    def update(self, values: dict, *, repair=False):
        previous = {binding.tag: binding for binding in self.bindings}
        bindings = []
        for kind, prefix, limit in KINDS:
            ordinal = 0
            for slot in range(1, limit + 1):
                paths = normalize_paths(values.get(f"{prefix}_{slot}"))
                if not paths:
                    continue
                if len(paths) != 1:
                    raise H3Error("Each reference card accepts one asset.")
                ordinal += 1
                old = previous.get(f"<{kind} {slot}>")
                same = old is not None and old.path == paths[0]
                bindings.append(
                    ReferenceBinding(
                        old.asset_id if same else uuid.uuid4().hex,
                        kind,
                        slot,
                        paths[0],
                        ordinal,
                        not repair and old is not None and (old.unresolved or not same),
                    )
                )
        # Remember removed bindings so replacing a tagged card still requires
        # an explicit repair, even after the upload was first cleared.
        active_tags = {b.tag for b in bindings}
        bindings.extend(
            ReferenceBinding(b.asset_id, b.kind, b.slot, b.path, 0, True)
            for b in self.bindings
            if b.tag not in active_tags
        )
        candidate = tuple(bindings)
        if candidate == self.bindings:
            return self
        return ReferenceMap(self.version + 1, candidate)

    def compile_prompt(self, prompt: str) -> str:
        mapping = {b.tag: b for b in self.bindings}

        def translate(match):
            binding = mapping.get(match[0])
            if binding is None or binding.unresolved or not binding.engine_ordinal:
                raise H3Error(
                    f"Repair the missing or replaced reference {match[0]} before continuing.",
                    code="reference_binding_missing",
                    field="reference_media",
                    recovery="Restore the missing asset or explicitly confirm the replacement.",
                )
            return binding.engine_tag

        return TAG.sub(translate, prompt)

    def restore_prompt(self, prompt: str) -> str:
        mapping = {
            b.engine_tag: b.tag
            for b in self.bindings
            if b.engine_ordinal and not b.unresolved
        }

        def translate(match):
            if match[0] not in mapping:
                raise H3Error(
                    f"Prompt writer returned an unknown reference {match[0]}."
                )
            return mapping[match[0]]

        return TAG.sub(translate, prompt)

    def dense_values(self, values: dict) -> dict:
        result = dict(values)
        for kind, prefix, limit in KINDS:
            for slot in range(1, limit + 1):
                result[f"{prefix}_{slot}"] = None
            for binding in self.bindings:
                if binding.kind == kind and binding.engine_ordinal:
                    result[f"{prefix}_{binding.engine_ordinal}"] = binding.path
        result["prompt"] = self.compile_prompt(values.get("prompt", ""))
        return result
