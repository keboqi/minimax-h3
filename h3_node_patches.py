#!/usr/bin/env python3
"""Fail-closed compatibility patches for pinned third-party ComfyUI nodes."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


LARRY_TIMESTEP_PATCH_VERSION = 2
QWEN_SPECTRUM_PATCH_VERSION = 1
TRT_VAE_PATCH_VERSION = 6
TRT_VAE_NODE_REPO = "https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT.git"
TRT_VAE_NODE_REF = "4360e00867eca86ab61b3899216c0ec281367b46"


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


# The upstream Qwen Spectrum node targets the older 20B MMDiT model. Qwen
# Image 2.1 is a separate 7B single-stream architecture: text projection is
# folded into txt_in, target tokens are normalized by LastLayer, and output is
# reshaped directly instead of through process_img. Keep this compatibility
# patch pinned and fail closed until upstream gains native Qwen 2.1 support.
_QWEN_SPECTRUM_INTROSPECTION_ORIGINAL = """\
def is_qwen_like_core(obj: Any) -> bool:
    if obj is None:
        return False
    if not all(hasattr(obj, field) for field in SUPPORTED_FORWARD_FIELDS):
        return False
    type_name = type(obj).__name__.lower()
    module_name = getattr(type(obj), "__module__", "").lower()
    if "qwen" in type_name or "qwen" in module_name:
        return True
    return hasattr(obj, "transformer_blocks") and hasattr(obj, "txt_in") and hasattr(obj, "img_in")
"""

_QWEN_SPECTRUM_INTROSPECTION_PATCHED = """\
QWEN_IMAGE21_FORWARD_FIELDS = {
    "img_in",
    "txt_in",
    "transformer_blocks",
    "norm_out",
    "proj_out",
    "time_text_embed",
    "build_sequence",
    "modulation",
}


def is_qwen_like_core(obj: Any) -> bool:
    if obj is None:
        return False
    is_legacy_qwen = all(
        hasattr(obj, field) for field in SUPPORTED_FORWARD_FIELDS
    )
    is_qwen_image21 = all(
        hasattr(obj, field) for field in QWEN_IMAGE21_FORWARD_FIELDS
    )
    if not (is_legacy_qwen or is_qwen_image21):
        return False
    type_name = type(obj).__name__.lower()
    module_name = getattr(type(obj), "__module__", "").lower()
    if "qwen" in type_name or "qwen" in module_name:
        return True
    return (
        hasattr(obj, "transformer_blocks")
        and hasattr(obj, "txt_in")
        and hasattr(obj, "img_in")
    )
"""

_QWEN_SPECTRUM_FORWARD_ORIGINAL = """\
def build_qwen_core_forward(
    core: Any,
    original_forward: Callable[..., Any],
) -> Callable[..., Any]:
    def spectrum_qwen_forward(
"""

_QWEN_SPECTRUM_FORWARD_PATCHED = """\
def _is_qwen_image21_core(core: Any) -> bool:
    return (
        hasattr(core, "build_sequence")
        and hasattr(core, "modulation")
        and not hasattr(core, "txt_norm")
    )


def _run_qwen_image21_forecast_forward(
    core: Any,
    state: QwenSpectrumState,
    runtime: QwenSpectrumRuntime,
    model_input: torch.Tensor,
    timestep: torch.Tensor,
) -> Any:
    if state.output_factory is None:
        raise RuntimeError("output factory missing before forecast")
    pred = state.forecaster.predict(runtime.current_time_coord)
    target_dtype = state.model_feature_dtype or pred.dtype
    pred = pred.to(device=model_input.device)
    pred = _sanitize_forecast_feature(pred, target_dtype)

    dtype = model_input.dtype
    t = ((timestep * 1000).to(dtype) / 1000).to(dtype)
    temb = core.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
    out_sample = core.proj_out(core.norm_out(pred, temb[:-1]))

    batch, _channels, height, width = model_input.shape
    out_sample = out_sample.transpose(1, 2).reshape(
        batch,
        int(core.out_channels),
        height,
        width,
    )
    state.record_forecast()
    log_debug(
        runtime.config.debug,
        (
            f"Spectrum Qwen 2.1 step={runtime.current_step_index + 1}/"
            f"{runtime.total_steps} branch={runtime.branch_key} "
            f"mode=forecast history={len(state.history_features)}"
        ),
    )
    return state.output_factory(out_sample, True)


def _build_qwen_image21_core_forward(
    core: Any,
    original_forward: Callable[..., Any],
) -> Callable[..., Any]:
    def spectrum_qwen_image21_forward(
        self: Any,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        ref_latents: Any = None,
        image_slots: Any = None,
        transformer_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        call_kwargs = {
            "timestep": timestep,
            "context": context,
            "ref_latents": ref_latents,
            "image_slots": image_slots,
            "transformer_options": transformer_options or {},
            **kwargs,
        }
        runtime: QwenSpectrumRuntime | None = getattr(
            self, "_spectrum_qwen_runtime", None
        )
        state: QwenSpectrumState | None = getattr(
            self, "_spectrum_qwen_state", None
        )
        if runtime is None or state is None:
            return original_forward(x, **call_kwargs)

        if runtime.decision_actual:
            return _run_actual_forward(
                self,
                state,
                runtime,
                original_forward,
                x,
                **call_kwargs,
            )

        try:
            return _run_qwen_image21_forecast_forward(
                self,
                state,
                runtime,
                model_input=x,
                timestep=timestep,
            )
        except Exception:
            return _run_actual_forward(
                self,
                state,
                runtime,
                original_forward,
                x,
                **call_kwargs,
            )

    return spectrum_qwen_image21_forward


def build_qwen_core_forward(
    core: Any,
    original_forward: Callable[..., Any],
) -> Callable[..., Any]:
    if _is_qwen_image21_core(core):
        return _build_qwen_image21_core_forward(core, original_forward)

    def spectrum_qwen_forward(
"""


def _patch_qwen_spectrum_sources(
    introspection_source: str,
    forward_source: str,
) -> tuple[str, str, bool]:
    states = {
        "model introspection": (
            introspection_source.count(_QWEN_SPECTRUM_INTROSPECTION_ORIGINAL),
            introspection_source.count(_QWEN_SPECTRUM_INTROSPECTION_PATCHED),
        ),
        "forward adapter": (
            forward_source.count(_QWEN_SPECTRUM_FORWARD_ORIGINAL),
            forward_source.count(_QWEN_SPECTRUM_FORWARD_PATCHED),
        ),
    }
    invalid = {
        name: state
        for name, state in states.items()
        if state not in ((1, 0), (0, 1))
    }
    if invalid:
        raise RuntimeError(
            "Qwen Spectrum 2.1 compatibility patch does not match the pinned "
            f"node source (invalid={invalid}, states={states})"
        )

    changed = False
    if states["model introspection"] == (1, 0):
        introspection_source = introspection_source.replace(
            _QWEN_SPECTRUM_INTROSPECTION_ORIGINAL,
            _QWEN_SPECTRUM_INTROSPECTION_PATCHED,
            1,
        )
        changed = True
    if states["forward adapter"] == (1, 0):
        forward_source = forward_source.replace(
            _QWEN_SPECTRUM_FORWARD_ORIGINAL,
            _QWEN_SPECTRUM_FORWARD_PATCHED,
            1,
        )
        changed = True
    return introspection_source, forward_source, changed


def patch_qwen_spectrum_node(node_dir: Path) -> bool:
    """Add the native Qwen Image 2.1 core and forecast-tail implementation."""
    node_dir = Path(node_dir)
    targets = {
        "model introspection": node_dir / "spectrum_qwen" / "model_introspection.py",
        "forward adapter": node_dir / "spectrum_qwen" / "forward_qwen.py",
    }
    missing = [str(path) for path in targets.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Qwen Spectrum node source is incomplete; missing: " + ", ".join(missing)
        )

    introspection_source = targets["model introspection"].read_text(encoding="utf-8")
    forward_source = targets["forward adapter"].read_text(encoding="utf-8")
    introspection_source, forward_source, changed = _patch_qwen_spectrum_sources(
        introspection_source,
        forward_source,
    )
    if not changed:
        return False

    compile(
        introspection_source,
        str(targets["model introspection"]),
        "exec",
    )
    compile(forward_source, str(targets["forward adapter"]), "exec")

    sources = {
        targets["model introspection"]: introspection_source,
        targets["forward adapter"]: forward_source,
    }
    temporaries = {
        target: target.with_name(target.name + ".h3-patch")
        for target in sources
    }
    try:
        for target, source in sources.items():
            temporaries[target].write_text(source, encoding="utf-8")
        for target in sources:
            temporaries[target].replace(target)
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)

    print(
        f"[h3-node-patch v{QWEN_SPECTRUM_PATCH_VERSION}] added native "
        f"Qwen Image 2.1 support in {node_dir}"
    )
    return True


# Upstream now owns single-frame encoding and optional encoder loading.
# Retain our established fixed-profile decoder workaround for single images.
_TRT_SINGLE_FRAME_DECODE_ORIGINAL = """\
    if z.shape[2] == 1:
      z_pad = z.repeat(1, 1, 7, 1, 1)
      return self._finalize_pixels(self.tiled_decode(z_pad)[:, :, -1:, :, :])
    return self.decode_temporal(z)
"""

_TRT_SINGLE_FRAME_DECODE_PATCHED = """\
    if z.shape[2] == 1:
      # Keep the first frame of a two-token clip for the fixed-profile engine.
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
      raise RuntimeError("Failed to parse ONNX:\\n" + "\\n".join(error_msgs))

    workspace_size = (4 if is_decoder else 8) * (1024**3)
"""


_TRT_FP32_NORMALIZATION_PATCHED = """\
      raise RuntimeError("Failed to parse ONNX:\\n" + "\\n".join(error_msgs))

    if is_decoder:
      # TensorRT 11 is strongly typed. Surround normalization Reduce/Pow
      # operations with explicit FP32 casts, then restore their output type.
      original_layers = [network.get_layer(i) for i in range(network.num_layers)]
      constrained_layers = 0
      for layer in original_layers:
        is_reduce = layer.type == trt.LayerType.REDUCE
        is_pow = (
            layer.type == trt.LayerType.ELEMENTWISE
            and getattr(layer, "op", None) == trt.ElementWiseOperation.POW
        )
        if not (is_reduce or is_pow):
          continue
        floating_types = {trt.float16, trt.float32}
        if hasattr(trt, "bfloat16"):
          floating_types.add(trt.bfloat16)
        outputs = [layer.get_output(i) for i in range(layer.num_outputs)]
        if not outputs or not all(output.dtype in floating_types for output in outputs):
          continue
        output_state = []
        for output_index, output in enumerate(outputs):
          if not output.name:
            output.name = f"{layer.name or 'layer'}_h3_output_{output_index}"
          consumers = []
          for consumer in original_layers:
            if consumer is layer:
              continue
            for input_index in range(consumer.num_inputs):
              candidate = consumer.get_input(input_index)
              if candidate is not None and candidate.name == output.name:
                consumers.append((consumer, input_index))
          output_state.append((output, output.dtype, consumers))
        for input_index in range(layer.num_inputs):
          input_tensor = layer.get_input(input_index)
          if input_tensor is None or input_tensor.dtype not in floating_types:
            continue
          if input_tensor.dtype != trt.float32:
            cast_in = network.add_cast(input_tensor, trt.float32)
            cast_in.name = f"{layer.name or 'layer'}_h3_fp32_input_{input_index}"
            layer.set_input(input_index, cast_in.get_output(0))
        for output_index, (output, original_dtype, consumers) in enumerate(output_state):
          if original_dtype == trt.float32:
            continue
          cast_out = network.add_cast(output, original_dtype)
          cast_out.name = f"{layer.name or 'layer'}_h3_restore_output_{output_index}"
          restored = cast_out.get_output(0)
          for consumer, input_index in consumers:
            consumer.set_input(input_index, restored)
        constrained_layers += 1
      logger.info(
          f"Wrapped {constrained_layers} decoder Reduce/Pow layers in FP32 casts"
      )

    workspace_size = (4 if is_decoder else 8) * (1024**3)
"""

_TRT_ONNX_IMPORT_ORIGINAL = """\
    try:
      model = onnx.load(onnx_path, load_external_data=False)
"""

_TRT_ONNX_IMPORT_PATCHED = """\
    try:
      import onnx
      model = onnx.load(onnx_path, load_external_data=False)
"""

_TRT_DESERIALIZE_FAILSAFE_ORIGINAL = """\
    self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
    if self.engine is None:
      raise RuntimeError(
          f"Failed to deserialize TensorRT engine:"
          f" '{os.path.basename(self.model_path)}'.\\n"
          f"Please re-compile the engine on this machine using the 'MiniMax-H3"
          " TRT VAE Compiler' node."
      )
"""

_TRT_DESERIALIZE_FAILSAFE_PATCHED = """\
    self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
    if self.engine is None:
      onnx_path = os.path.splitext(self.model_path)[0] + ".onnx"
      if os.path.isfile(onnx_path):
        logger.warning(
            f"Failed to deserialize TensorRT engine '{os.path.basename(self.model_path)}'. "
            "Re-compiling engine on this machine..."
        )
        is_decoder = "decoder" in os.path.basename(self.model_path).lower()
        try:
          mm.unload_all_models()
          mm.soft_empty_cache()
        except Exception:
          pass
        torch.cuda.empty_cache()
        MiniMaxH3TRTCompilerNode._build_engine(
            onnx_path,
            self.model_path,
            is_decoder=is_decoder,
        )
        self.engine_bytes = None
        self.load_to_ram()
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
      if self.engine is None:
        raise RuntimeError(
            f"Failed to deserialize TensorRT engine:"
            f" '{os.path.basename(self.model_path)}'.\\n"
            f"Please re-compile the engine on this machine using the 'MiniMax-H3"
            " TRT VAE Compiler' node."
        )
"""


_TRT_REPLACEMENTS = (
    (
        "ONNX quantization inspection",
        _TRT_ONNX_IMPORT_ORIGINAL,
        _TRT_ONNX_IMPORT_PATCHED,
    ),
    (
        "single-frame decode",
        _TRT_SINGLE_FRAME_DECODE_ORIGINAL,
        _TRT_SINGLE_FRAME_DECODE_PATCHED,
    ),
    (
        "temporal frame trimming",
        _TRT_TEMPORAL_RETURN_ORIGINAL,
        _TRT_TEMPORAL_RETURN_PATCHED,
    ),
    (
        "FP32 normalization",
        _TRT_FP32_NORMALIZATION_ORIGINAL,
        _TRT_FP32_NORMALIZATION_PATCHED,
    ),
    (
        "deserialization failsafe auto-recompile",
        _TRT_DESERIALIZE_FAILSAFE_ORIGINAL,
        _TRT_DESERIALIZE_FAILSAFE_PATCHED,
    ),
)


def _patch_trt_vae_source(source: str) -> tuple[str, bool]:
    """Synchronize TensorRT VAE behavior with the reference implementation."""
    changed = False
    states = {
        name: (source.count(original), source.count(patched))
        for name, original, patched in _TRT_REPLACEMENTS
    }
    invalid = {
        name: state
        for name, state in states.items()
        if state not in ((1, 0), (0, 1))
    }
    if invalid:
        raise RuntimeError(
            "TensorRT VAE compatibility patch does not match the upstream "
            f"node source (invalid={invalid}, states={states})"
        )

    for name, original, patched in _TRT_REPLACEMENTS:
        if states[name] == (1, 0):
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
    introspection, forward, changed = _patch_qwen_spectrum_sources(
        _QWEN_SPECTRUM_INTROSPECTION_ORIGINAL,
        _QWEN_SPECTRUM_FORWARD_ORIGINAL,
    )
    assert changed is True
    assert "QWEN_IMAGE21_FORWARD_FIELDS" in introspection
    assert "def _run_qwen_image21_forecast_forward" in forward
    assert "temb[:-1]" in forward
    assert ".reshape(" in forward

    introspection, forward, changed = _patch_qwen_spectrum_sources(
        introspection,
        forward,
    )
    assert changed is False

    try:
        _patch_qwen_spectrum_sources(
            "unexpected upstream source",
            "unexpected upstream source",
        )
        raise AssertionError("unexpected Qwen Spectrum source was accepted")
    except RuntimeError as exc:
        assert "does not match the pinned node source" in str(exc)
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
            "  def decode(self, z):\n"
            + _TRT_SINGLE_FRAME_DECODE_ORIGINAL
            + "  def decode_temporal(self, z):\n"
            + _TRT_TEMPORAL_RETURN_ORIGINAL
            + "    pass\n"
            "  def load_to_gpu(self):\n"
            + _TRT_DESERIALIZE_FAILSAFE_ORIGINAL
            + "\ndef build():\n"
            + "    if parse_failed:\n"
            + _TRT_FP32_NORMALIZATION_ORIGINAL
            + "\ndef inspect_quantization():\n"
            + _TRT_ONNX_IMPORT_ORIGINAL
            + "    except Exception:\n      pass\n",
            encoding="utf-8",
        )
        assert patch_trt_vae_node(Path(directory)) is True
        patched = target.read_text(encoding="utf-8")
        assert all(
            original not in patched for _, original, _ in _TRT_REPLACEMENTS
        )
        assert all(
            replacement in patched for _, _, replacement in _TRT_REPLACEMENTS
        )
        assert "network.add_cast" in patched
        assert "layer.precision" not in patched
        assert "layer.set_output_type" not in patched
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
