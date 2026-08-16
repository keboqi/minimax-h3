#!/usr/bin/env python3
"""Fail-closed compatibility patches for pinned third-party ComfyUI nodes."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


LARRY_TIMESTEP_PATCH_VERSION = 2


_LARRY_UNIQUE_T_ORIGINAL = """\
def _unique_t(timestep, shift_v, shift_a, has_vis_cond):
    sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
    t_v = 1.0 - sv
    t_a = 1.0 - _time_shift_sigma(sv, shift_v, shift_a)
    s = {t_v, t_a}
    if has_vis_cond:
        s.add(max(t_v, 0.999))
    return sorted(s)
"""

_LARRY_UNIQUE_T_PATCHED = """\
def _unique_t(timestep, shift_v, shift_a, payload):
    sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
    t_v = 1.0 - sv
    t_a = 1.0 - _time_shift_sigma(sv, shift_v, shift_a)
    layout = payload.get("layout")
    segments = getattr(layout, "segments", ()) or ()
    has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in segments)
    has_aud_cond = any(k == "ref_audio" for _, _, k in segments)
    if not segments:
        refs = payload.get("refs") or ()
        has_vis_cond = bool(payload.get("keyframes")) or any(
            ref.get("kind") in ("image", "video", "video_audio") for ref in refs)
        has_aud_cond = any(
            ref.get("kind") in ("audio", "video_audio")
            and int(ref.get("ref_audio_t") or 0) > 0
            for ref in refs
        )
    unique_t = {t_v, t_a}
    if has_vis_cond:
        unique_t.add(max(t_v, float(payload.get("visual_cond_noise_aug", 0.999))))
    if has_aud_cond:
        unique_t.add(max(t_a, float(payload.get("audio_cond_noise_aug", 1.0))))
    return sorted(unique_t)
"""

_LARRY_CALL_ORIGINAL = """\
        has_vc = bool(payload.get("keyframes") or payload.get("refs"))
        us = _unique_t(ts, shift_v, shift_a, has_vc)
"""

_LARRY_CALL_PATCHED = """\
        us = _unique_t(ts, shift_v, shift_a, payload)
"""

_LARRY_REPLACEMENTS = (
    (_LARRY_UNIQUE_T_ORIGINAL, _LARRY_UNIQUE_T_PATCHED),
    (_LARRY_CALL_ORIGINAL, _LARRY_CALL_PATCHED),
)

_LARRY_UPSTREAM_FIXED_MARKERS = (
    "def _unique_t(timestep, shift_v, shift_a, payload):",
    "adaln_t_table",
)


def _patch_larry_source(source: str) -> tuple[str, bool]:
    """Return validated patched source and whether a transformation occurred."""
    if all(marker in source for marker in _LARRY_UPSTREAM_FIXED_MARKERS):
        return source, False
    original_counts = [source.count(old) for old, _ in _LARRY_REPLACEMENTS]
    patched_counts = [source.count(new) for _, new in _LARRY_REPLACEMENTS]
    if original_counts == [0, 0] and patched_counts == [1, 1]:
        return source, False
    if original_counts != [1, 1] or patched_counts != [0, 0]:
        raise RuntimeError(
            "Larry Turbo compatibility patch does not match the pinned node "
            f"source (original={original_counts}, patched={patched_counts})"
        )

    for old, new in _LARRY_REPLACEMENTS:
        source = source.replace(old, new, 1)
    return source, True


def patch_larry_turbo_node(node_dir: Path) -> bool:
    """Mirror ComfyUI's dynamic AdaLN timestep rows in Larry's pruned path.

    Returns True when the file changed and False when it was already patched.
    Raises if the pinned source no longer matches, preventing a silent partial
    patch after an upstream revision change.
    """
    target = Path(node_dir) / "__init__.py"
    if not target.is_file():
        raise RuntimeError(f"Larry Turbo node entry point is missing: {target}")

    patched_source, changed = _patch_larry_source(
        target.read_text(encoding="utf-8")
    )
    if not changed:
        return False

    # Reject malformed transformations before touching the installed node. Use
    # an adjacent temporary so os.replace semantics remain atomic on one volume.
    compile(patched_source, str(target), "exec")
    temporary = target.with_name(target.name + ".h3-patch")
    try:
        temporary.write_text(patched_source, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"[h3-node-patch v{LARRY_TIMESTEP_PATCH_VERSION}] synchronized "
        f"Larry AdaLN timestep rows in {target}"
    )
    return True


def selftest() -> None:
    fixture = _LARRY_UNIQUE_T_ORIGINAL + "\ndef wrap():\n" + _LARRY_CALL_ORIGINAL
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "__init__.py"
        target.write_text(fixture, encoding="utf-8")
        assert patch_larry_turbo_node(Path(directory)) is True
        patched = target.read_text(encoding="utf-8")
        assert all(old not in patched for old, _ in _LARRY_REPLACEMENTS)
        assert all(new in patched for _, new in _LARRY_REPLACEMENTS)
        assert patch_larry_turbo_node(Path(directory)) is False
        assert not target.with_name(target.name + ".h3-patch").exists()

        try:
            _patch_larry_source("unexpected upstream source")
            raise AssertionError("unexpected Larry source was accepted")
        except RuntimeError as exc:
            assert "does not match the pinned node source" in str(exc)

        class Scalar(float):
            def __truediv__(self, other):
                return Scalar(super().__truediv__(other))

            def clamp(self, *, min):
                return Scalar(max(float(self), min))

        class Timestep:
            def __init__(self, value):
                self.value = Scalar(value)

            def flatten(self):
                return [self.value]

        class Layout:
            segments = ((0, 1, "ref_img"), (1, 2, "ref_audio"))

        namespace = {
            "_time_shift_sigma": lambda sigma, source, target: (
                target * (sigma / (source + sigma * (1.0 - source)))
                / (1.0 + (target - 1.0) * (sigma / (source + sigma * (1.0 - source))))
            )
        }
        exec(_LARRY_UNIQUE_T_PATCHED, namespace)
        unique_t = namespace["_unique_t"]
        payload = {"layout": Layout()}
        assert len(unique_t(Timestep(1000), 12.0, 3.0, payload)) == 3
        assert len(unique_t(Timestep(972.97), 12.0, 3.0, payload)) == 4
        assert len(unique_t(Timestep(1000), 12.0, 3.0, {})) == 1
        assert len(unique_t(Timestep(972.97), 12.0, 3.0, {})) == 2
    print("h3_node_patches selftest OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()


if __name__ == "__main__":
    main()
