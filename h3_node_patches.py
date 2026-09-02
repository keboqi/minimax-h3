#!/usr/bin/env python3
"""Fail-closed compatibility patches for pinned third-party ComfyUI nodes."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


LARRY_TIMESTEP_PATCH_VERSION = 2
TRT_VAE_PATCH_VERSION = 2


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

    patched_source, changed = _patch_larry_source(target.read_text(encoding="utf-8"))
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


_TRT_SINGLE_FRAME_ENCODE_ORIGINAL = """\
      if x.shape[2] == 1:
        moments = self.tiled_encode(self._normalize_pixels(x))[:, :, -1:, :, :]
      else:
        moments = self.encode_temporal(x)
"""

_TRT_SINGLE_FRAME_ENCODE_PATCHED = """\
      if x.shape[2] == 1:
        # The TensorRT encoder has a fixed 17-frame profile. Mirror the
        # temporal encoder's tail-padding behavior, then retain one latent.
        x_pad = x.repeat(1, 1, self.clip_length, 1, 1)
        moments = self.tiled_encode(self._normalize_pixels(x_pad))[:, :, -1:, :, :]
      else:
        moments = self.encode_temporal(x)
"""

_TRT_SINGLE_FRAME_DECODE_ORIGINAL = """\
      if z.shape[2] == 1:
        # 🌟 如果是单张图片 (T=1)，填充到 7 个 token 以满足 TRT 静态切片尺寸
        z_pad = z.repeat(1, 1, 7, 1, 1)
        return self._finalize_pixels(
            self.tiled_decode(z_pad)[:, :, -1:, :, :]
        )
      return self.decode_temporal(z)
"""

_TRT_SINGLE_FRAME_DECODE_PATCHED = """\
      if z.shape[2] == 1:
        # A lone token is out-of-distribution for the ViT decoder. Decode it
        # as the first token of a two-token clip, matching the reference VAE.
        z_pair = torch.cat([z, z], dim=2)
        return self.decode_temporal(z_pair)[:, :, :1]
      return self.decode_temporal(z)
"""

_TRT_TEMPORAL_RETURN_ORIGINAL = """\
    return torch.cat(dec_chunks, dim=2)

  def encode_temporal(self, x):
"""

_TRT_TEMPORAL_RETURN_PATCHED = """\
    dec = torch.cat(dec_chunks, dim=2)
    if pad_tokens > 0:
      # Remove pixel frames produced only by repeated tail tokens.
      intra_tail = self.clip_length % self.vae_ratio_t
      before_pad = z.shape[2] - pad_tokens
      pad_frames = sum(
          intra_tail
          if intra_tail and (before_pad + k) % self.tokens_chunk_size == 0
          else self.vae_ratio_t
          for k in range(pad_tokens)
      )
      if pad_frames > 0:
        dec = dec[:, :, :-pad_frames]
    return dec

  def encode_temporal(self, x):
"""

_TRT_FP32_NORMALIZATION_ORIGINAL = """\
    if hasattr(trt.BuilderFlag, "FP16"):
      config.set_flag(trt.BuilderFlag.FP16)

    workspace_size = (4 if is_decoder else 8) * (1024**3)
"""

_TRT_FP32_NORMALIZATION_PATCHED = """\
    if hasattr(trt.BuilderFlag, "FP16"):
      config.set_flag(trt.BuilderFlag.FP16)
    if is_decoder:
      # TensorRT warns that FP16 LayerNorm Reduce/Pow operations after
      # attention can overflow. Preserve FP32 for those operations while
      # retaining FP16 for the expensive matrix multiplications.
      constrained_layers = 0
      for layer_index in range(network.num_layers):
        layer = network.get_layer(layer_index)
        is_reduce = layer.type == trt.LayerType.REDUCE
        is_pow = (
            layer.type == trt.LayerType.ELEMENTWISE
            and getattr(layer, "op", None) == trt.ElementWiseOperation.POW
        )
        if not (is_reduce or is_pow):
          continue
        outputs = [layer.get_output(i) for i in range(layer.num_outputs)]
        floating_types = {trt.float16, trt.float32}
        if hasattr(trt, "bfloat16"):
          floating_types.add(trt.bfloat16)
        if not outputs or not all(output.dtype in floating_types for output in outputs):
          continue
        layer.precision = trt.float32
        for output_index in range(layer.num_outputs):
          layer.set_output_type(output_index, trt.float32)
        constrained_layers += 1
      constraint_flag = getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None)
      if constraint_flag is None:
        constraint_flag = getattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS", None)
      if constraint_flag is not None:
        config.set_flag(constraint_flag)
      logger.info(
          f"Forced {constrained_layers} decoder Reduce/Pow layers to FP32 precision"
      )

    workspace_size = (4 if is_decoder else 8) * (1024**3)
"""

_TRT_REPLACEMENTS = (
    (_TRT_SINGLE_FRAME_ENCODE_ORIGINAL, _TRT_SINGLE_FRAME_ENCODE_PATCHED),
    (_TRT_SINGLE_FRAME_DECODE_ORIGINAL, _TRT_SINGLE_FRAME_DECODE_PATCHED),
    (_TRT_TEMPORAL_RETURN_ORIGINAL, _TRT_TEMPORAL_RETURN_PATCHED),
    (_TRT_FP32_NORMALIZATION_ORIGINAL, _TRT_FP32_NORMALIZATION_PATCHED),
)


def _patch_trt_vae_source(source: str) -> tuple[str, bool]:
    """Synchronize TensorRT VAE behavior with the reference implementation."""
    changed = False
    states = []
    for original, patched in _TRT_REPLACEMENTS:
        original_count = source.count(original)
        patched_count = source.count(patched)
        states.append((original_count, patched_count))
        if original_count == 0 and patched_count == 1:
            continue
        if original_count != 1 or patched_count != 0:
            raise RuntimeError(
                "TensorRT VAE compatibility patch does not match the upstream "
                f"node source (states={states})"
            )
        source = source.replace(original, patched, 1)
        changed = True
    return source, changed


def patch_trt_vae_node(node_dir: Path) -> bool:
    """Synchronize upstream TensorRT encode, decode, and build behavior."""
    target = Path(node_dir) / "minimax_trt_node.py"
    if not target.is_file():
        raise RuntimeError(f"TensorRT VAE node entry point is missing: {target}")

    patched_source, changed = _patch_trt_vae_source(target.read_text(encoding="utf-8"))
    if not changed:
        return False
    compile(patched_source, str(target), "exec")
    temporary = target.with_name(target.name + ".h3-patch")
    try:
        temporary.write_text(patched_source, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"[h3-node-patch v{TRT_VAE_PATCH_VERSION}] synchronized TensorRT VAE "
        f"reference behavior and mixed precision in {target}"
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

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "minimax_trt_node.py"
        target.write_text(
            "class Fixture:\n"
            "  def encode(self, x):\n"
            + _TRT_SINGLE_FRAME_ENCODE_ORIGINAL
            + "  def decode(self, z):\n"
            + _TRT_SINGLE_FRAME_DECODE_ORIGINAL
            + "  def decode_temporal(self, z):\n"
            + _TRT_TEMPORAL_RETURN_ORIGINAL
            + "    pass\n\n"
            "def build():\n" + _TRT_FP32_NORMALIZATION_ORIGINAL,
            encoding="utf-8",
        )
        assert patch_trt_vae_node(Path(directory)) is True
        patched = target.read_text(encoding="utf-8")
        assert all(original not in patched for original, _ in _TRT_REPLACEMENTS)
        assert all(replacement in patched for _, replacement in _TRT_REPLACEMENTS)
        assert patch_trt_vae_node(Path(directory)) is False

        try:
            _patch_trt_vae_source("unexpected upstream source")
            raise AssertionError("unexpected TensorRT VAE source was accepted")
        except RuntimeError as exc:
            assert "does not match the upstream node source" in str(exc)

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
                target
                * (sigma / (source + sigma * (1.0 - source)))
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
