from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from dataclasses import dataclass
from fractions import Fraction

import av
import torch
import torch.nn.functional as F
import comfy.lora
import comfy.cli_args
import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sd
import comfy.utils
import comfy.weight_adapter
import folder_paths


LIGHTX_LORA_PATTERN = re.compile(
    r"^(diffusion_model\.(?:"
    r"(?:blocks\.\d+|token_refiner\.blocks\.\d+)\."
    r"(?:attn\.(?:qkv_proj|out_proj)|mlp\.fc[12])"
    r"|(?:blocks\.\d+|final_layer)\.adaln_proj\.linear"
    r"))\.(lora_[AB]\.weight|alpha)$"
)
LIGHTX_BACKBONE_BLOCKS = 50
LIGHTX_REFINER_BLOCKS = 2
LIGHTX_TARGETS_PER_BLOCK = 4
# v1.2 checkpoints add rank-16 AdaLN adapters; Turbo Fusion Full uses rank-32
# AdaLN with 320/960 QKV backbone ranks.
LIGHTX_ADALN_RANK = 16
LIGHTX_RANK_PROFILES = frozenset({
    (128, None),
    (128, 16),
    (320, 32),
})
LIGHTX_ADALN_MODULES = frozenset(
    f"diffusion_model.blocks.{block}.adaln_proj.linear"
    for block in range(LIGHTX_BACKBONE_BLOCKS)
) | {"diffusion_model.final_layer.adaln_proj.linear"}


def _logical_weight_shape(weight) -> tuple[int, ...]:
    """Return a quantized tensor's logical shape or a plain tensor's shape."""
    quant_params = getattr(weight, "_params", None)
    shape = getattr(quant_params, "orig_shape", None)
    if shape is None:
        shape = weight.shape
    return tuple(int(dimension) for dimension in shape)


class H3InplaceLoRAAdapter(comfy.weight_adapter.LoRAAdapter):
    """Apply a linear LoRA delta directly into the fresh base activation."""

    def __init__(self, loaded_keys, weights):
        super().__init__(loaded_keys, weights)
        up, down, alpha, mid, dora_scale, reshape = weights
        if getattr(self, "is_conv", False):
            raise RuntimeError("MiniMax H3 LightX2V bypass only supports linear layers")
        if mid is not None or dora_scale is not None or reshape is not None:
            raise RuntimeError("Unsupported non-linear LightX2V LoRA extension")

        rank = int(down.shape[0])
        self.intrinsic_scale = (
            float(alpha) / rank if alpha is not None else 1.0
        )

    def lora_input(self, x):
        return x

    def bypass_forward(self, original_forward, x, *args, **kwargs):
        up, down, _alpha, _mid, _dora_scale, _reshape = self.weights
        base_output = original_forward(x, *args, **kwargs)
        x = self.lora_input(x)
        if up.dtype != x.dtype or up.device != x.device:
            up = up.to(device=x.device, dtype=x.dtype)
        if down.dtype != x.dtype or down.device != x.device:
            down = down.to(device=x.device, dtype=x.dtype)
        scale = self.intrinsic_scale * float(getattr(self, "multiplier", 1.0))
        update = F.linear(F.linear(x, down), up)
        return base_output.add_(update, alpha=scale)


class H3CurveInplaceLoRAAdapter(H3InplaceLoRAAdapter):
    """Keep the base's curve input, but apply the LoRA to dense timestep rows."""

    def lora_input(self, x):
        rows = self.curve_context.get()
        if rows is None or rows.get("dense") is None:
            raise RuntimeError("LightX2V curve AdaLN ran without dense timestep rows")
        dense = rows["dense"]
        if dense.shape[0] != x.shape[0]:
            raise RuntimeError("LightX2V curve AdaLN timestep rows do not match")
        return dense.to(device=x.device, dtype=x.dtype)


def _lightx_curve_adaln(model, module, base_shape, down):
    backbone = getattr(model.model, "diffusion_model", None)
    return (
        module in LIGHTX_ADALN_MODULES
        and getattr(backbone, "use_adaln_curves", False)
        and len(base_shape) == 2
        and base_shape[1] == 8
        and int(down.shape[1]) == 2688
    )


def _lightx_curve_time_embedding(original, grid, context):
    """Use the model's exact timestep rows, including visual/audio reference rows."""
    def time_embedding(unique_t, device, dtype):
        state = context.get()
        if state is None:
            raise RuntimeError("LightX2V curve AdaLN ran outside the diffusion wrapper")
        table = grid.to(device=device, dtype=torch.float32)
        times = torch.tensor(unique_t, device=device, dtype=torch.float32)
        positions = times.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = positions.floor().long().clamp(max=table.shape[0] - 2)
        state["dense"] = torch.lerp(
            table[lower], table[lower + 1], (positions - lower).unsqueeze(1),
        )
        return original(unique_t, device, dtype)
    return time_embedding


def _lightx_curve_wrapper(context):
    def wrapper(executor, *args, **kwargs):
        token = context.set({})
        try:
            return executor(*args, **kwargs)
        finally:
            context.reset(token)
    return wrapper


def _lightx_curve_grid():
    # Setup already installs this pinned node and its dense SiLU timestep grid.
    from pathlib import Path
    for root in folder_paths.get_folder_paths("custom_nodes"):
        path = Path(root) / "ComfyUI-MiniMax-H3-Turbo" / "h3_silu_temb_grid.safetensors"
        if path.is_file():
            grid = comfy.utils.load_torch_file(str(path), safe_load=True)["silu_t_emb_grid"]
            if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] != 2688:
                raise ValueError("Invalid LightX2V dense timestep grid")
            return grid
    raise FileNotFoundError("LightX2V curve AdaLN needs the Larry Turbo timestep grid; re-run setup_h3.py")


def _lightx_key_map(model, lora):
    modules: dict[str, set[str]] = {}
    for key in lora:
        match = LIGHTX_LORA_PATTERN.fullmatch(key)
        if match is None:
            raise ValueError(f"Unexpected LightX2V LoRA key: {key}")
        modules.setdefault(match.group(1), set()).add(match.group(2))

    target_suffixes = (
        "attn.qkv_proj",
        "attn.out_proj",
        "mlp.fc1",
        "mlp.fc2",
    )
    expected_module_names = {
        f"diffusion_model.blocks.{block}.{suffix}"
        for block in range(LIGHTX_BACKBONE_BLOCKS)
        for suffix in target_suffixes
    } | {
        f"diffusion_model.token_refiner.blocks.{block}.{suffix}"
        for block in range(LIGHTX_REFINER_BLOCKS)
        for suffix in target_suffixes
    }
    expected_modules = (
        LIGHTX_BACKBONE_BLOCKS + LIGHTX_REFINER_BLOCKS
    ) * LIGHTX_TARGETS_PER_BLOCK
    # AdaLN adapters are all-or-none: older checkpoints have none, v1.2 and Fusion have all.
    has_adaln = bool(set(modules) & LIGHTX_ADALN_MODULES)
    if has_adaln:
        expected_module_names |= LIGHTX_ADALN_MODULES
        expected_modules += len(LIGHTX_ADALN_MODULES)
    if set(modules) != expected_module_names:
        raise ValueError(
            f"LightX2V LoRA has {len(modules)} modules; expected "
            f"the exact {expected_modules}-module H3 backbone/refiner layout"
        )
    expected_parts = {"lora_A.weight", "lora_B.weight", "alpha"}
    incomplete = sorted(
        module for module, parts in modules.items() if parts != expected_parts
    )
    if incomplete:
        raise ValueError(
            "LightX2V LoRA has incomplete module tensors: "
            + ", ".join(incomplete[:5])
        )

    probe_core = lora["diffusion_model.blocks.0.attn.out_proj.lora_A.weight"]
    core_rank = int(probe_core.shape[0]) if probe_core.ndim == 2 else -1
    if has_adaln:
        probe_adaln = lora["diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight"]
        adaln_rank = int(probe_adaln.shape[0]) if probe_adaln.ndim == 2 else -1
    else:
        adaln_rank = None
    if (core_rank, adaln_rank) not in LIGHTX_RANK_PROFILES:
        raise ValueError(
            f"Unexpected LightX2V rank profile: core={core_rank}, adaln={adaln_rank}"
        )

    for module in modules:
        down = lora[module + ".lora_A.weight"]
        up = lora[module + ".lora_B.weight"]
        alpha = lora[module + ".alpha"]
        if module in LIGHTX_ADALN_MODULES:
            expected_rank = adaln_rank
        elif module.endswith("attn.qkv_proj"):
            expected_rank = core_rank * 3
        else:
            expected_rank = core_rank
        if (
            down.ndim != 2
            or up.ndim != 2
            or int(down.shape[0]) != expected_rank
            or int(up.shape[1]) != expected_rank
            or alpha.numel() != 1
        ):
            raise ValueError(f"Unexpected LightX2V tensor layout for {module}")

    native_map = comfy.lora.model_lora_keys_unet(model.model, {})
    missing = sorted(module for module in modules if module not in native_map)
    if missing:
        raise ValueError(
            "LightX2V LoRA does not match this MiniMax H3 model: "
            + ", ".join(missing[:5])
        )

    for module in modules:
        down = lora[module + ".lora_A.weight"]
        up = lora[module + ".lora_B.weight"]
        base_weight = comfy.utils.get_attr(model.model, native_map[module])
        base_shape = _logical_weight_shape(base_weight)
        if (
            len(base_shape) != 2
            or (int(down.shape[1]) != int(base_shape[1])
                and not _lightx_curve_adaln(model, module, base_shape, down))
            or int(up.shape[0]) != int(base_shape[0])
        ):
            raise ValueError(
                f"LightX2V tensor shapes do not match {module}: "
                f"base={base_shape}, down={tuple(down.shape)}, up={tuple(up.shape)}"
            )
    return {module: native_map[module] for module in modules}


def _lightx_int8_fused_fc2_keys(
    model, key_map: dict[str, str]
) -> set[str]:
    """Return adapters whose ConvRot INT8 kernel bypasses fc2.forward.

    MiniMax H3's fused INT8 MLP path consumes fc2.weight directly, so a forward
    bypass hook on those modules never executes.  Route only those adapters
    through ComfyUI's transient weight-cast patch path; the LoRA delta is then
    applied after dequantization instead of being requantized into the base.
    """
    fused: set[str] = set()
    for module, key in key_map.items():
        if not module.endswith(".mlp.fc2"):
            continue
        # _lightx_key_map already proved this native key exists. Failing here
        # must abort adapter installation instead of silently losing an fc2.
        weight = comfy.utils.get_attr(model.model, key)
        if (
            getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(getattr(weight, "_params", None), "transposed", False)
        ):
            fused.add(key)
    return fused


class H3LightX2VBypassLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "model/patch/minimax"
    DESCRIPTION = (
        "Apply an official LightX2V MiniMax H3 LoRA in activation space, "
        "with fused ConvRot INT8 fc2 projections patched during weight cast."
    )

    def apply_lora(self, model, lora_name, strength):
        path = folder_paths.get_full_path("loras", lora_name)
        if path is None:
            raise FileNotFoundError(f"LightX2V LoRA is missing: {lora_name}")
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        key_map = _lightx_key_map(model, lora)
        loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
        if set(loaded) != set(key_map.values()):
            missing = sorted(set(key_map.values()) - set(loaded))
            raise ValueError(
                "LightX2V bypass failed to load every adapter: "
                + ", ".join(missing[:5])
            )

        patched = model.clone()
        curve_keys = {
            key for module, key in key_map.items()
            if _lightx_curve_adaln(
                model, module, _logical_weight_shape(comfy.utils.get_attr(model.model, key)),
                lora[module + ".lora_A.weight"],
            )
        }
        curve_context = None
        if curve_keys:
            grid = _lightx_curve_grid()
            curve_context = ContextVar(f"h3_lightx2v_curve_{id(patched)}", default=None)
            original = model.get_model_object("diffusion_model._time_embedding")
            patched.add_object_patch(
                "diffusion_model._time_embedding",
                _lightx_curve_time_embedding(original, grid, curve_context),
            )
            patched.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                "h3_lightx2v_curve", _lightx_curve_wrapper(curve_context),
            )
        fused_fc2_keys = _lightx_int8_fused_fc2_keys(model, key_map)
        fused_fc2 = {
            key: adapter for key, adapter in loaded.items() if key in fused_fc2_keys
        }
        bypass = {
            key: adapter for key, adapter in loaded.items() if key not in fused_fc2_keys
        }
        if fused_fc2:
            patched_fc2 = set(patched.add_patches(fused_fc2, float(strength)))
            if patched_fc2 != set(fused_fc2):
                missing_fc2 = sorted(set(fused_fc2) - patched_fc2)
                raise RuntimeError(
                    "LightX2V failed to patch fused INT8 fc2 adapters: "
                    + ", ".join(missing_fc2[:5])
                )

        manager = comfy.weight_adapter.BypassInjectionManager()
        for key, adapter in bypass.items():
            if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
                raise TypeError(f"Unsupported LightX2V adapter for {key}")
            if len(adapter.weights) != 6:
                raise ValueError(f"Unexpected LightX2V adapter layout for {key}")
            adapter_class = H3CurveInplaceLoRAAdapter if key in curve_keys else H3InplaceLoRAAdapter
            adapter = adapter_class(adapter.loaded_keys, adapter.weights)
            if key in curve_keys:
                adapter.curve_context = curve_context
            manager.add_adapter(key, adapter, strength=float(strength))

        injections = manager.create_injections(patched.model)
        if manager.get_hook_count() != len(bypass):
            raise RuntimeError(
                f"LightX2V bypass created {manager.get_hook_count()} hooks for "
                f"{len(bypass)} adapters"
            )
        if manager.get_hook_count() > 0:
            patched.set_injections("h3_lightx2v_bypass", injections)
        logging.info(
            "MiniMax H3 LightX2V runtime bypass: %d activation adapters, "
            "%d fused INT8 fc2 weight-cast adapters at strength %.3f",
            len(bypass),
            len(fused_fc2),
            float(strength),
        )
        return (patched,)


@dataclass(frozen=True)
class PresetConfig:
    threshold: float
    start_percent: float = 0.10
    end_percent: float = 0.95
    max_consecutive_hits: int = 2


PRESETS = {
    "Safe": PresetConfig(0.08),
    "Fast": PresetConfig(0.10),
    "Aggressive": PresetConfig(0.12),
}


@dataclass
class CacheContext:
    previous_first_residual: torch.Tensor | None = None
    remaining_blocks_residual: torch.Tensor | None = None
    first_block_output: torch.Tensor | None = None
    pending_first_residual: torch.Tensor | None = None
    use_cache: bool = False
    consecutive_hits: int = 0
    previous_sigma: float | None = None
    input_signature: tuple | None = None
    last_diff: float | None = None
    video_slice: tuple[int, int] | None = None
    latent_frames: int | None = None

    def clear_tensors(self):
        self.previous_first_residual = None
        self.remaining_blocks_residual = None
        self.first_block_output = None
        self.pending_first_residual = None
        self.use_cache = False
        self.consecutive_hits = 0
        self.previous_sigma = None
        self.input_signature = None
        self.last_diff = None
        self.video_slice = None
        self.latent_frames = None


class MiniMaxH3FirstBlockCache:
    def __init__(self, config, start_sigma, end_sigma, block_count, temporal_guard):
        self.config = config
        self.start_sigma = float(start_sigma)
        self.end_sigma = float(end_sigma)
        self.block_count = int(block_count)
        self.temporal_guard = bool(temporal_guard)
        self.contexts = {}
        self.current = None
        self.full_steps = 0
        self.cached_steps = 0
        self.diff_values = []
        self.temporal_diff_values = []
        self.cached_step_numbers = []

    def reset(self):
        for context in self.contexts.values():
            context.clear_tensors()
        self.contexts.clear()
        self.current = None
        self.full_steps = 0
        self.cached_steps = 0
        self.diff_values.clear()
        self.temporal_diff_values.clear()
        self.cached_step_numbers.clear()

    @staticmethod
    def _input_signature(x):
        tensors = x if isinstance(x, (tuple, list)) else (x,)
        return tuple(
            (tuple(t.shape), t.dtype, t.device)
            for t in tensors if torch.is_tensor(t)
        )

    @staticmethod
    def _video_layout(minimax_payload):
        if not minimax_payload:
            return None, None
        layout = minimax_payload.get("layout")
        if layout is None:
            return None, None
        video_slice = next(
            ((start, stop) for start, stop, kind in layout.segments if kind == "video"),
            None,
        )
        latent_frames = (
            layout.signature[1]
            if hasattr(layout, "signature") and len(layout.signature) > 1
            else None
        )
        return video_slice, latent_frames

    def begin_call(self, x, timestep, transformer_options, minimax_payload=None):
        sigma = float(timestep.flatten()[0].item()) / 1000.0
        uuids = transformer_options.get("uuids")
        key = tuple(str(value) for value in uuids) if uuids else ("default",)
        context = self.contexts.setdefault(key, CacheContext())
        signature = self._input_signature(x)
        restarted = (
            context.previous_sigma is not None
            and sigma > context.previous_sigma + 1e-7
        )
        if context.input_signature != signature or restarted:
            context.clear_tensors()
        context.input_signature = signature
        context.previous_sigma = sigma
        context.video_slice, context.latent_frames = self._video_layout(minimax_payload)
        context.first_block_output = None
        context.pending_first_residual = None
        context.use_cache = False
        self.current = context

    def end_call(self):
        self.current = None

    def _within_window(self, context):
        if context.previous_sigma is None:
            return False
        return self.end_sigma <= context.previous_sigma <= self.start_sigma

    @staticmethod
    def _temporal_diff(current, previous, context):
        if context.video_slice is None or not context.latent_frames:
            return None
        start, stop = context.video_slice
        current_video = current[start:stop]
        previous_video = previous[start:stop]
        if (
            current_video.shape != previous_video.shape
            or current_video.shape[0] % context.latent_frames
        ):
            return None
        rows_per_frame = current_video.shape[0] // context.latent_frames
        current_video = current_video.reshape(
            context.latent_frames, rows_per_frame, -1
        )
        previous_video = previous_video.reshape(
            context.latent_frames, rows_per_frame, -1
        )
        numerator = (current_video - previous_video).abs().mean(dim=(1, 2))
        denominator = previous_video.abs().mean(dim=(1, 2)).clamp(min=1e-8)
        return float((numerator / denominator).max().item())

    @torch.compiler.disable()
    def decide(self, first_residual, first_output):
        context = self.current
        if context is None:
            raise RuntimeError("MiniMax H3 FirstBlockCache called outside model execution")

        previous = context.previous_first_residual
        tail = context.remaining_blocks_residual
        can_compare = (
            previous is not None
            and tail is not None
            and previous.shape == first_residual.shape
            and tail.shape == first_output.shape
        )

        use_cache = False
        diff = None
        if can_compare and self._within_window(context):
            numerator = (first_residual - previous).abs().mean()
            denominator = previous.abs().mean().clamp(min=1e-8)
            diff = float((numerator / denominator).item())
            self.diff_values.append(diff)
            decision_diff = diff

            if self.temporal_guard:
                temporal_diff = self._temporal_diff(first_residual, previous, context)
                if temporal_diff is not None:
                    self.temporal_diff_values.append(temporal_diff)
                    decision_diff = max(decision_diff, temporal_diff)

            use_cache = (
                math.isfinite(decision_diff)
                and decision_diff <= self.config.threshold
                and context.consecutive_hits < self.config.max_consecutive_hits
            )

        context.last_diff = diff
        context.use_cache = use_cache
        if use_cache:
            self.cached_step_numbers.append(self.full_steps + self.cached_steps + 1)
            context.consecutive_hits += 1
            context.first_block_output = None
            context.pending_first_residual = None
        else:
            context.consecutive_hits = 0
            context.first_block_output = first_output.detach().clone()
            context.pending_first_residual = first_residual.detach()

    def finish_full_step(self, output):
        context = self.current
        if (
            context is None
            or context.first_block_output is None
            or context.pending_first_residual is None
        ):
            raise RuntimeError("MiniMax H3 FirstBlockCache full-step state is incomplete")
        context.remaining_blocks_residual = (
            output - context.first_block_output
        ).detach()
        context.previous_first_residual = context.pending_first_residual
        context.first_block_output = None
        context.pending_first_residual = None
        self.full_steps += 1

    def finish_cached_step(self, first_output):
        context = self.current
        if context is None or context.remaining_blocks_residual is None:
            raise RuntimeError("MiniMax H3 FirstBlockCache has no cached residual")
        self.cached_steps += 1
        return first_output + context.remaining_blocks_residual

    def summary(self):
        steps = self.full_steps + self.cached_steps
        if steps == 0:
            return "no model steps"

        executed_blocks = self.full_steps * self.block_count + self.cached_steps
        block_speedup = steps * self.block_count / max(executed_blocks, 1)

        finite_diffs = sorted(v for v in self.diff_values if math.isfinite(v))
        diff_details = ""
        if finite_diffs:
            middle = len(finite_diffs) // 2
            median = (
                finite_diffs[middle]
                if len(finite_diffs) % 2
                else (finite_diffs[middle - 1] + finite_diffs[middle]) / 2
            )
            diff_details = (
                f"; residual diff min/median/max "
                f"{finite_diffs[0]:.5f}/{median:.5f}/{finite_diffs[-1]:.5f}"
            )

        finite_temporal = sorted(
            v for v in self.temporal_diff_values if math.isfinite(v)
        )
        temporal_details = (
            f"; temporal guard max {finite_temporal[-1]:.5f}"
            if finite_temporal else ""
        )
        hit_details = (
            f"; cache steps {self.cached_step_numbers}"
            if self.cached_step_numbers else ""
        )

        return (
            f"cached {self.cached_steps}/{steps} steps; "
            f"estimated block-stack speedup {block_speedup:.2f}x"
            f"{diff_details}{temporal_details}{hit_details}"
        )


def make_block_patch(cache, index, last_index):
    def patch(args, extra):
        original_block = extra["original_block"]

        if index == 0:
            original_input = args["img"].detach().clone()
            output = original_block(args)["img"]
            cache.decide(output - original_input, output)
            return {"img": output}

        context = cache.current
        if context is None:
            raise RuntimeError("MiniMax H3 FirstBlockCache has no active context")

        if context.use_cache:
            if index == last_index:
                return {"img": cache.finish_cached_step(args["img"])}
            return {"img": args["img"]}

        output = original_block(args)["img"]
        if index == last_index:
            cache.finish_full_step(output)
        return {"img": output}

    return patch


def make_diffusion_wrapper(cache):
    def wrapper(executor, *args, **kwargs):
        transformer_options = (
            args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        )
        minimax_payload = (
            args[4] if len(args) > 4 else kwargs.get("minimax_payload")
        )
        cache.begin_call(args[0], args[1], transformer_options, minimax_payload)
        try:
            return executor(*args, **kwargs)
        finally:
            cache.end_call()
    return wrapper


def make_sample_wrapper(cache, label):
    def wrapper(executor, *args, **kwargs):
        cache.reset()
        logging.info("MiniMax H3 FBCache enabled: %s", label)
        try:
            return executor(*args, **kwargs)
        finally:
            logging.info("MiniMax H3 FBCache: %s", cache.summary())
            cache.reset()
    return wrapper


class H3FirstBlockCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (
                    ["Safe", "Fast", "Aggressive", "Custom"],
                    {"default": "Fast"},
                ),
                "residual_diff_threshold": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_consecutive_cache_hits": (
                    "INT",
                    {"default": 2, "min": 1, "max": 20, "step": 1},
                ),
                "temporal_guard": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch/minimax"
    DESCRIPTION = (
        "MiniMax H3 FirstBlockCache with calibrated presets, protected "
        "denoising window, refresh limit, and temporal frame guard."
    )

    def patch(
        self,
        model,
        preset,
        residual_diff_threshold,
        start_percent,
        end_percent,
        max_consecutive_cache_hits,
        temporal_guard,
    ):
        preset = str(preset)
        if preset == "Custom":
            if float(start_percent) >= float(end_percent):
                raise ValueError("FBCache start_percent must be smaller than end_percent")
            config = PresetConfig(
                float(residual_diff_threshold),
                float(start_percent),
                float(end_percent),
                max(1, int(max_consecutive_cache_hits)),
            )
        else:
            if preset not in PRESETS:
                raise ValueError(f"Unknown FBCache preset: {preset}")
            config = PRESETS[preset]

        diffusion_model = model.get_model_object("diffusion_model")
        if (
            diffusion_model.__class__.__name__ != "MiniMaxH3Model"
            or not hasattr(diffusion_model, "blocks")
        ):
            raise ValueError(
                "MiniMax H3 FirstBlockCache only supports native MiniMaxH3, "
                f"got {diffusion_model.__class__.__name__}"
            )

        block_count = len(diffusion_model.blocks)
        if block_count < 2:
            raise ValueError("MiniMax H3 FirstBlockCache needs at least two blocks")

        transformer_options = model.model_options.get("transformer_options", {})
        conflicts = (
            transformer_options.get("patches_replace", {}).get("dit", {})
        )
        for index in range(block_count):
            if ("double_block", index) in conflicts:
                raise ValueError(
                    "MiniMax H3 FirstBlockCache conflicts with another DiT block replacement"
                )

        model_sampling = model.get_model_object("model_sampling")
        start_sigma = float(model_sampling.percent_to_sigma(config.start_percent))
        end_sigma = float(model_sampling.percent_to_sigma(config.end_percent))

        patched = model.clone()
        cache = MiniMaxH3FirstBlockCache(
            config,
            start_sigma,
            end_sigma,
            block_count,
            bool(temporal_guard),
        )
        for index in range(block_count):
            patched.set_model_patch_replace(
                make_block_patch(cache, index, block_count - 1),
                "dit",
                "double_block",
                index,
            )

        key = f"minimax_h3_first_block_cache_{id(cache)}"
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            key,
            make_diffusion_wrapper(cache),
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            key,
            make_sample_wrapper(
                cache,
                (
                    f"{preset}: threshold={config.threshold:.2f}, "
                    f"window={config.start_percent:.2f}-{config.end_percent:.2f}, "
                    f"max={config.max_consecutive_hits}, "
                    f"temporal_guard={bool(temporal_guard)}"
                ),
            ),
        )
        return (patched,)


class H3SeparateAVLatent:
    """Expose MiniMax H3's video and audio tensors as ordinary LATENT values."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "separate"
    CATEGORY = "model/latent/minimax"

    def separate(self, latent):
        samples = latent.get("samples")
        if (
            samples is None
            or not getattr(samples, "is_nested", False)
            or len(samples.tensors) != 2
        ):
            raise ValueError("H3 Separate AV Latent expects a MiniMax H3 AV latent")
        video, audio = samples.tensors
        if video.ndim != 5 or video.shape[1] != 24:
            raise ValueError(
                "H3 video latent must have shape [B, 24, T, H, W]"
            )
        if audio.ndim != 4 or audio.shape[1] != 32:
            raise ValueError(
                "H3 audio latent must have shape [B, 32, 2, T]"
            )
        video_output = latent.copy()
        video_output["samples"] = video
        audio_output = latent.copy()
        audio_output["samples"] = audio
        return (video_output, audio_output)


class H3CombineAVLatent:
    """Rebuild MiniMax H3's nested AV latent after video-only processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "combine"
    CATEGORY = "model/latent/minimax"

    def combine(self, video_latent, audio_latent):
        video = video_latent.get("samples")
        audio = audio_latent.get("samples")
        if video is None or video.ndim != 5 or video.shape[1] != 24:
            raise ValueError(
                "H3 video latent must have shape [B, 24, T, H, W]"
            )
        if audio is None or audio.ndim != 4 or audio.shape[1] != 32:
            raise ValueError(
                "H3 audio latent must have shape [B, 32, 2, T]"
            )
        if video.shape[0] != audio.shape[0]:
            raise ValueError("H3 video and audio latent batch sizes must match")
        # The neural upscaler executes on its selected device and returns the
        # video latent there. The bypassed audio latent can remain on ComfyUI's
        # offload device (normally CPU). H3's sampler packs both nested tensors
        # with torch.cat, so reunify their device before the refinement pass.
        if audio.device != video.device:
            audio = audio.to(device=video.device, non_blocking=True)
        output = video_latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        return (output,)


class H3SingleFrameVAELoader:
    """Overlay the audited image-only decoder onto a complete official H3 VAE."""

    EXPECTED_DECODER_TENSORS = 585
    DECODER_PREFIXES = ("decoder.", "post_quant_conv.")
    DECODER_BLOCKS = 36
    ATTENTION_HEADS = 32
    ATTENTION_HEAD_DIM = 64

    @classmethod
    def INPUT_TYPES(cls):
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "base_vae_name": (vaes,),
                "decoder_name": (vaes,),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders/minimax"
    DESCRIPTION = (
        "Load the official MiniMax H3 VAE architecture and replace only its "
        "decoder with the experimental 500K single-frame checkpoint."
    )

    @staticmethod
    def _vae_path(name):
        path = folder_paths.get_full_path("vae", name)
        if path is None:
            raise FileNotFoundError(f"MiniMax H3 VAE file is missing: {name}")
        return path

    @classmethod
    def _convert_diffusers_decoder(cls, state):
        """Convert the published Diffusers decoder into ComfyUI's native layout."""

        def rename(source, target):
            if source not in state:
                raise ValueError(f"Single-frame decoder is missing {source}")
            if target in state:
                raise ValueError(f"Single-frame decoder already contains {target}")
            state[target] = state.pop(source)

        for suffix in ("weight", "bias"):
            rename(f"decoder.proj_in.{suffix}", f"decoder.x_embedder.{suffix}")

        inner_dim = cls.ATTENTION_HEADS * cls.ATTENTION_HEAD_DIM
        for index in range(cls.DECODER_BLOCKS):
            prefix = f"decoder.transformer_blocks.{index}"
            attention = f"{prefix}.attn"
            for suffix in ("weight", "bias"):
                projections = []
                for name in ("q", "k", "v"):
                    key = f"{attention}.to_{name}.{suffix}"
                    if key not in state:
                        raise ValueError(f"Single-frame decoder is missing {key}")
                    tensor = state.pop(key)
                    if tensor.shape[0] != inner_dim:
                        raise ValueError(
                            f"{key} has {tensor.shape[0]} rows; expected {inner_dim}"
                        )
                    projections.append(
                        tensor.reshape(
                            cls.ATTENTION_HEADS,
                            cls.ATTENTION_HEAD_DIM,
                            *tensor.shape[1:],
                        )
                    )
                # The native checkpoint stores q/k/v per head:
                # [head0 q k v, head1 q k v, ...].
                fused = torch.cat(projections, dim=1).reshape(
                    3 * inner_dim, *projections[0].shape[2:]
                )
                state[f"{attention}.to_qkv.{suffix}"] = fused.contiguous()

                rename(
                    f"{attention}.to_out.0.{suffix}",
                    f"{attention}.to_out.{suffix}",
                )

                first = f"{prefix}.ff.net.0.proj.{suffix}"
                if first not in state:
                    raise ValueError(f"Single-frame decoder is missing {first}")
                # Diffusers SwiGLU stores [up; gate]; ComfyUI expects [gate; up].
                up, gate = state.pop(first).chunk(2, dim=0)
                state[f"{prefix}.ff.w1.{suffix}"] = torch.cat(
                    (gate, up), dim=0
                ).contiguous()
                rename(
                    f"{prefix}.ff.net.2.{suffix}",
                    f"{prefix}.ff.w2.{suffix}",
                )
        return state

    def load_vae(self, base_vae_name, decoder_name):
        base_path = self._vae_path(base_vae_name)
        decoder_path = self._vae_path(decoder_name)
        base, metadata = comfy.utils.load_torch_file(
            base_path,
            safe_load=True,
            return_metadata=True,
        )
        decoder = comfy.utils.load_torch_file(decoder_path, safe_load=True)

        invalid = sorted(
            key for key in decoder if not key.startswith(self.DECODER_PREFIXES)
        )
        if invalid:
            raise ValueError(
                "Single-frame checkpoint contains non-decoder tensors: "
                + ", ".join(invalid[:5])
            )
        if len(decoder) != self.EXPECTED_DECODER_TENSORS:
            raise ValueError(
                f"Single-frame checkpoint contains {len(decoder)} tensors; "
                f"expected {self.EXPECTED_DECODER_TENSORS}."
            )

        decoder = self._convert_diffusers_decoder(decoder)
        expected = {
            key for key in base if key.startswith(self.DECODER_PREFIXES)
        }
        # mask_token is an unused all-zero training buffer which the published
        # Diffusers-format decoder intentionally omits. Retain it from the base.
        replaceable = expected - {"decoder.mask_token"}
        missing = sorted(replaceable - set(decoder))
        extra = sorted(set(decoder) - expected)
        if missing or extra:
            raise ValueError(
                "Single-frame decoder does not match the official H3 VAE keys; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        mismatched = sorted(
            key for key in replaceable
            if tuple(base[key].shape) != tuple(decoder[key].shape)
        )
        if mismatched:
            raise ValueError(
                "Single-frame decoder tensor shapes do not match the official H3 VAE: "
                + ", ".join(mismatched[:5])
            )

        base.update(decoder)
        # Core otherwise expands a single latent into a clip for the stock
        # decoder. This fine-tuned decoder consumes one latent directly and
        # selects its last real frame; retain the base architecture metadata.
        metadata = dict(metadata or {})
        metadata.update(h3_t1_direct="true", h3_t1_output_slice="3")
        logging.info(
            "Loaded MiniMax H3 single-frame decoder %s over %s",
            decoder_name,
            base_vae_name,
        )
        return (comfy.sd.VAE(sd=base, metadata=metadata),)


class H3VideoLatentSlicesToBatch:
    """Turn independent H3 temporal slices into one-frame batch elements."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": (
                    "INT",
                    {"default": 1, "min": 1, "max": 1, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "select"
    CATEGORY = "model/latent/minimax"

    def select(self, latent, frames):
        samples = latent.get("samples")
        if samples is not None and getattr(samples, "is_nested", False):
            if len(samples.tensors) != 2:
                raise ValueError("Expected a MiniMax H3 video/audio latent")
            samples = samples.tensors[0]
        if samples is None or samples.ndim != 5 or samples.shape[1] != 24:
            raise ValueError(
                "H3 single-frame decode expects video latents shaped [B, 24, T, H, W]"
            )

        count = int(frames)
        available = int(samples.shape[2])
        if count != 1:
            raise ValueError(
                "The 500K single-frame decoder supports exactly one image from "
                "the first H3 temporal latent slice."
            )
        if available < 1:
            raise ValueError("The sampled H3 latent contains no video slices")
        batch, channels, _time, height, width = samples.shape
        selected = samples[:, :, 0:1].reshape(
            batch,
            channels,
            1,
            height,
            width,
        )
        output = latent.copy()
        output["samples"] = selected
        return (output,)


# Upstream hardcodes small_input=True in both Qwen3-VL attention selectors.
# Install context-aware dispatchers once; only this CLIP encoding context can
# override that hint. Other models/threads retain their original dispatch.
_H3_ENCODER_SMALL_INPUT = ContextVar("h3_encoder_small_input", default=None)
_H3_ENCODER_ATTENTION_LOCK = threading.Lock()


def _h3_encoder_attention_selector(original):
    @wraps(original)
    def select(device, mask=False, small_input=False):
        override = _H3_ENCODER_SMALL_INPUT.get()
        return original(
            device, mask=mask,
            small_input=small_input if override is None else override,
        )

    select._h3_encoder_dispatch = True
    return select


@contextmanager
def _h3_encoder_attention(small_input):
    with _H3_ENCODER_ATTENTION_LOCK:
        for name in ("comfy.text_encoders.llama", "comfy.text_encoders.qwen35"):
            module = importlib.import_module(name)
            original = module.optimized_attention_for_device
            if not getattr(original, "_h3_encoder_dispatch", False):
                module.optimized_attention_for_device = _h3_encoder_attention_selector(original)
    token = _H3_ENCODER_SMALL_INPUT.set(bool(small_input))
    try:
        logging.info(
            "MiniMax H3 Qwen text/vision attention: %s",
            "small input (PyTorch/basic)" if small_input else "configured server backend",
        )
        yield
    finally:
        _H3_ENCODER_SMALL_INPUT.reset(token)


def _h3_encoder_input_signature(value):
    """Describe the actual token sequence and vision geometry without hashing pixels.

    The graph cache key already identifies source media (content-addressed when
    reuse is on). Resizing/cropping and video truncation happen inside the native
    node, so include the resulting token structure, tensor shapes and dtypes too.
    Unknown input types disable reuse rather than risk an incomplete signature.
    """
    if torch.is_tensor(value):
        return ("tensor", tuple(value.shape), str(value.dtype))
    if isinstance(value, dict):
        return ("dict", tuple(
            (key, _h3_encoder_input_signature(item))
            for key, item in sorted(value.items())
        ))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(
            _h3_encoder_input_signature(item) for item in value
        ))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported encoder cache input: {type(value).__name__}")


class _H3ConditioningReuseCache:
    """Bounded process-local cache of Qwen results, independent of VAE latents."""

    def __init__(self, max_entries: int = 2) -> None:
        self._entries: OrderedDict[tuple[str, str], object] = OrderedDict()
        self._fresh_since_policy: dict[str, bool] = {}
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._encoder_small_input = None
        self._attention_revision = 0

    def select_attention(self, small_input: bool) -> int:
        with self._lock:
            if self._encoder_small_input != small_input:
                self._entries.clear()
                self._fresh_since_policy.clear()
                self._encoder_small_input = small_input
                self._attention_revision += 1
                logging.info("MiniMax H3 invalidated conditioning for Qwen attention change")
            return self._attention_revision

    def encode(self, cache_key: str, encode, small_input: bool = True,
               *, reuse: bool = True, input_signature: str = ""):
        entry_key = (cache_key, input_signature)
        with self._lock:
            revision = self.select_attention(small_input)
            if reuse and entry_key in self._entries:
                conditioning = self._entries[entry_key]
                self._entries.move_to_end(entry_key)
                logging.info(
                    "MiniMax H3 Qwen cache hit [key=%s input=%s]; VAE conditioning is separate",
                    cache_key[:12], input_signature[:12],
                )
                return conditioning

        logging.info(
            "MiniMax H3 Qwen cache %s [key=%s input=%s]; encoding fresh",
            "miss" if reuse else "disabled", cache_key[:12], input_signature[:12],
        )
        conditioning = encode()
        with self._lock:
            # Do not publish a result from an older route if another execution
            # changed policy while this encode was in flight.
            if revision != self._attention_revision:
                return conditioning
            if not reuse:
                for key in list(self._entries):
                    if key[0] == cache_key:
                        del self._entries[key]
                self._fresh_since_policy.pop(cache_key, None)
                return conditioning
            self._fresh_since_policy[cache_key] = True
            self._entries[entry_key] = conditioning
            self._entries.move_to_end(entry_key)
            while len(self._entries) > self._max_entries:
                evicted_key, _value = self._entries.popitem(last=False)
                if not any(key[0] == evicted_key[0] for key in self._entries):
                    self._fresh_since_policy.pop(evicted_key[0], None)
        return conditioning

    def conditioning_was_reused(self, cache_key: str) -> bool:
        with self._lock:
            fresh = self._fresh_since_policy.pop(cache_key, False)
            return not fresh and any(key[0] == cache_key for key in self._entries)


_H3_CONDITIONING_REUSE_CACHE = _H3ConditioningReuseCache()


class _H3CachedCLIPProxy:
    def __init__(self, clip, cache_key: str, encoder_small_input: bool = True,
                 reuse_conditioning: bool = True) -> None:
        self._clip = clip
        self._cache_key = cache_key
        self._encoder_small_input = bool(encoder_small_input)
        self._reuse_conditioning = bool(reuse_conditioning)

    def __getattr__(self, name):
        return getattr(self._clip, name)

    def clone(self, *args, **kwargs):
        return type(self)(
            self._clip.clone(*args, **kwargs), self._cache_key,
            self._encoder_small_input, self._reuse_conditioning,
        )

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        def encode():
            with _h3_encoder_attention(self._encoder_small_input):
                return self._clip.encode_from_tokens_scheduled(tokens, *args, **kwargs)

        reuse = self._reuse_conditioning
        input_signature = ""
        if reuse:
            try:
                signature = _h3_encoder_input_signature((tokens, args, kwargs))
                input_signature = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()
            except TypeError as exc:
                logging.info("MiniMax H3 Qwen cache bypass: %s", exc)
                reuse = False
        return _H3_CONDITIONING_REUSE_CACHE.encode(
            self._cache_key, encode, self._encoder_small_input,
            reuse=reuse, input_signature=input_signature,
        )


class H3ConditioningCache:
    """Wrap H3 CLIP with scoped attention routing and optional encoding reuse."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "cache_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "encoder_small_input": ("BOOLEAN", {"default": True}),
                "reuse_conditioning": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "wrap"
    CATEGORY = "model/management/minimax"
    DESCRIPTION = (
        "Reuses matching Qwen token/vision inputs. VAE conditioning remains owned "
        "by the conditioning node. Disabling reuse forces fresh encoding."
    )

    @classmethod
    def IS_CHANGED(cls, clip, cache_key, encoder_small_input=True, reuse_conditioning=True):
        revision = _H3_CONDITIONING_REUSE_CACHE.select_attention(bool(encoder_small_input))
        # Force dependent native/T8 conditioning nodes to run even for text-only
        # requests, where there is no staged media filename to invalidate them.
        return revision if reuse_conditioning else float("nan")

    def wrap(self, clip, cache_key, encoder_small_input=True, reuse_conditioning=True):
        _H3_CONDITIONING_REUSE_CACHE.select_attention(bool(encoder_small_input))
        return (_H3CachedCLIPProxy(
            clip, str(cache_key), encoder_small_input, reuse_conditioning,
        ),)


def _offload_h3_models(label: str) -> None:
    before = len(comfy.model_management.loaded_models())
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    after = len(comfy.model_management.loaded_models())
    logging.info(
        "MiniMax H3 %s: resident Comfy models %d -> %d",
        label,
        before,
        after,
    )


class H3StageOffloadPolicy:
    """Disable BF16 stage offloads when conditioning was actually reused."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "additional_conditioning": ("CONDITIONING",),
                "additional_latent": ("LATENT",),
                "cache_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING", "LATENT", "CONDITIONING", "LATENT", "BOOLEAN"
    )
    RETURN_NAMES = (
        "conditioning",
        "latent",
        "additional_conditioning",
        "additional_latent",
        "stage_offload_enabled",
    )
    FUNCTION = "choose"
    CATEGORY = "model/management/minimax"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def choose(
        self,
        conditioning,
        latent,
        additional_conditioning,
        additional_latent,
        cache_key,
    ):
        reused = _H3_CONDITIONING_REUSE_CACHE.conditioning_was_reused(
            str(cache_key)
        )
        enabled = not reused
        if enabled:
            _offload_h3_models("encode-stage offload")
        else:
            logging.info(
                "MiniMax H3 conditioning reused; skipping stage offloads for this run"
            )
        return (
            conditioning,
            latent,
            additional_conditioning,
            additional_latent,
            enabled,
        )


class H3StageModelOffload:
    """Pass stage results through after forcing resident Comfy models off VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "additional_conditioning": ("CONDITIONING",),
                "additional_latent": ("LATENT",),
            },
            "optional": {"enabled": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "CONDITIONING", "LATENT")
    RETURN_NAMES = (
        "conditioning",
        "latent",
        "additional_conditioning",
        "additional_latent",
    )
    FUNCTION = "offload"
    CATEGORY = "model/management/minimax"
    DESCRIPTION = (
        "Force-unloads resident ComfyUI models at an H3 stage boundary while "
        "passing conditioning and latents through unchanged."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # The memory-management side effect must run even when upstream prompt
        # encoding is cached across otherwise-identical workflow submissions.
        return float("nan")

    def offload(
        self,
        conditioning,
        latent,
        additional_conditioning,
        additional_latent,
        enabled=True,
    ):
        if enabled:
            _offload_h3_models("stage offload")
        return conditioning, latent, additional_conditioning, additional_latent


_H3_REFINEMENT_COMPILER_LOCK = threading.RLock()


def _h3_refinement_video_volume(value) -> int:
    """Return T*H*W for an H3 video latent, or zero for unknown inputs."""
    tensors = getattr(value, "tensors", None)
    video = tensors[0] if isinstance(tensors, (list, tuple)) and tensors else value
    shape = getattr(video, "shape", ())
    if len(shape) < 5:
        return 0
    return math.prod(int(dimension) for dimension in shape[-3:])


def _make_h3_refinement_compiler_wrapper(min_video_volume: int):
    """Bypass AIMDO only for exceptionally large refinement model calls."""
    state = {"logged": False}

    def wrapper(executor, x, *args, **kwargs):
        volume = _h3_refinement_video_volume(x)
        if volume < min_video_volume:
            return executor(x, *args, **kwargs)

        with _H3_REFINEMENT_COMPILER_LOCK:
            previous = bool(comfy.cli_args.args.disable_comfy_compiler)
            comfy.cli_args.args.disable_comfy_compiler = True
            try:
                if not state["logged"]:
                    logging.info(
                        "MiniMax H3 refinement bypassing Comfy compiler for "
                        "large video latent volume %d (threshold %d)",
                        volume,
                        min_video_volume,
                    )
                    state["logged"] = True
                return executor(x, *args, **kwargs)
            finally:
                comfy.cli_args.args.disable_comfy_compiler = previous

    return wrapper


class H3RefinementCompilerGuard:
    """Disable malloc-graph capture only for exceptionally large refinement."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "min_video_volume": (
                    "INT",
                    {"default": 1_500_000, "min": 1, "max": 100_000_000},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/management/minimax"

    def patch(self, model, min_video_volume=1_500_000):
        patched = model.clone()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            f"h3_refinement_compiler_guard_{id(patched)}",
            _make_h3_refinement_compiler_wrapper(max(1, int(min_video_volume))),
        )
        return (patched,)


class H3SaveVideoNVENC:
    """Encode an in-memory VIDEO, preferring NVENC with a software fallback."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": (
                    "STRING",
                    {"default": "h3/generation"},
                ),
                "preset": (
                    ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                    {"default": "p4"},
                ),
                "constant_quality": (
                    "INT",
                    {"default": 23, "min": 0, "max": 51, "step": 1},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "video/minimax"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Saves an H.264 MP4 using NVENC when available, otherwise CPU libx264. "
        "The quality value controls NVENC CQ or software CRF."
    )

    @staticmethod
    def _probe_encoder(encoder, width, height, frame_rate, options):
        # Codec registration alone does not prove hardware support: H200 has
        # no NVENC, even when PyAV lists h264_nvenc. Open a real codec context.
        context = None
        try:
            context = av.CodecContext.create(encoder, "w")
            context.width = width
            context.height = height
            context.pix_fmt = "yuv420p"
            context.time_base = 1 / frame_rate
            context.framerate = frame_rate
            context.bit_rate = 0
            context.options = options.copy()
            context.open()
        finally:
            # Release the probe's hardware session before opening the output.
            del context

    @classmethod
    def _select_encoder(cls, width, height, frame_rate, preset, constant_quality):
        quality = str(int(constant_quality))
        options = {"preset": str(preset), "tune": "hq", "rc": "vbr", "cq": quality}
        try:
            cls._probe_encoder("h264_nvenc", width, height, frame_rate, options)
            return "h264_nvenc", options
        except Exception as nvenc_error:
            logging.warning(
                "[H3 video] NVENC unavailable; using CPU libx264: %s", nvenc_error
            )
            options = {"preset": "medium", "crf": quality}
            try:
                cls._probe_encoder("libx264", width, height, frame_rate, options)
            except Exception as software_error:
                raise RuntimeError(
                    "H3 video output requires a working h264_nvenc or libx264 encoder "
                    f"in PyAV. NVENC: {nvenc_error}. libx264: {software_error}"
                ) from software_error
            return "libx264", options

    def save(
        self,
        video,
        filename_prefix,
        preset,
        constant_quality,
        prompt=None,
        extra_pnginfo=None,
    ):
        components = video.get_components()
        images = components.images
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] < 3:
            raise ValueError(
                "H3 video output expects video images shaped [frames, H, W, RGB]"
            )

        height, width = int(images.shape[1]), int(images.shape[2])
        if width % 2 or height % 2:
            raise ValueError(
                f"H.264 requires even dimensions, got {width}x{height}"
            )
        full_output_folder, filename, counter, subfolder, _prefix = (
            folder_paths.get_save_image_path(
                str(filename_prefix),
                folder_paths.get_output_directory(),
                width,
                height,
            )
        )
        file = f"{filename}_{counter:05}_.mp4"
        path = os.path.join(full_output_folder, file)
        frame_rate = Fraction(
            round(float(components.frame_rate) * 1000), 1000
        )
        encoder, encoder_options = self._select_encoder(
            width, height, frame_rate, preset, constant_quality
        )

        try:
            with av.open(
                path,
                mode="w",
                format="mp4",
                options={"movflags": "use_metadata_tags+faststart"},
            ) as output:
                metadata = {}
                if extra_pnginfo:
                    metadata.update(extra_pnginfo)
                if prompt is not None:
                    metadata["prompt"] = prompt
                for key, value in metadata.items():
                    output.metadata[str(key)] = (
                        value if isinstance(value, str) else json.dumps(value)
                    )

                video_stream = output.add_stream(encoder, rate=frame_rate)
                video_stream.width = width
                video_stream.height = height
                video_stream.pix_fmt = "yuv420p"
                video_stream.bit_rate = 0
                video_stream.options = encoder_options

                audio = components.audio
                audio_stream = None
                audio_frame = None
                if audio:
                    sample_rate = int(audio["sample_rate"])
                    waveform = audio["waveform"][0]
                    sample_count = math.ceil(
                        sample_rate * int(images.shape[0]) / float(frame_rate)
                    )
                    waveform = waveform[:, :sample_count]
                    layout = {
                        1: "mono",
                        2: "stereo",
                        6: "5.1",
                    }.get(int(waveform.shape[0]), "stereo")
                    audio_stream = output.add_stream(
                        "aac", rate=sample_rate, layout=layout
                    )
                    audio_frame = av.AudioFrame.from_ndarray(
                        waveform.float().cpu().contiguous().numpy(),
                        format="fltp",
                        layout=layout,
                    )
                    audio_frame.sample_rate = sample_rate
                    audio_frame.pts = 0

                for image in images:
                    pixels = (
                        image[..., :3]
                        .mul(255.0)
                        .clamp(0.0, 255.0)
                        .to(device="cpu", dtype=torch.uint8)
                        .contiguous()
                        .numpy()
                    )
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    frame = frame.reformat(format="yuv420p")
                    for packet in video_stream.encode(frame):
                        output.mux(packet)
                for packet in video_stream.encode(None):
                    output.mux(packet)

                if audio_stream is not None and audio_frame is not None:
                    for packet in audio_stream.encode(audio_frame):
                        output.mux(packet)
                    for packet in audio_stream.encode(None):
                        output.mux(packet)
        except Exception:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
            raise

        return {
            "ui": {
                "videos": [{
                    "filename": file,
                    "subfolder": subfolder,
                    "type": "output",
                    "format": "video/mp4",
                }]
            }
        }


class H3SemanticBridge:
    """Standalone v1 FL2VA adapter; preserves cached native conditioning.

    Architecture and normalization follow speach1sdef178's Semantic Bridge v1:
    https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge
    Weights retain their upstream MiniMax H3 Community License terms.
    """

    FILENAME = "MiniMaxH3_SemanticBridge_v1.safetensors"
    SHA256 = "ac0dc8ac05f545ebdee12e2fcebe4515b049f9cfd9558eb4887a9bf3fd6d562e"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "alpha": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Semantic Bridge"
    DESCRIPTION = (
        "Experimental FL2VA-only prompt conditioning adapter. Do not use with Ref2VA."
    )

    @classmethod
    def _load_weights(cls):
        import hashlib
        from safetensors.torch import load_file

        path = os.path.join(folder_paths.models_dir, "semantic_bridge", cls.FILENAME)
        # Verify the pinned release before loading. No pickle or remote code execution.
        with open(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != cls.SHA256:
            raise RuntimeError(
                "Semantic Bridge v1 checksum mismatch; re-download the adapter."
            )
        weights = load_file(path, device="cpu")
        expected = {
            "fc1.weight": (512, 5120),
            "fc1.bias": (512,),
            "fc2.weight": (512, 512),
            "fc2.bias": (512,),
            "fc3.weight": (5120, 512),
            "fc3.bias": (5120,),
        }
        if set(weights) != set(expected):
            raise RuntimeError("Unexpected Semantic Bridge v1 tensors.")
        for name, shape in expected.items():
            if (
                tuple(weights[name].shape) != shape
                or not torch.isfinite(weights[name]).all()
            ):
                raise RuntimeError(f"Invalid Semantic Bridge tensor: {name}")
        return weights

    def apply(self, conditioning, alpha=0.10):
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or not 0 <= alpha <= 1
        ):
            raise ValueError(
                "Semantic Bridge strength must be finite and between 0 and 1."
            )
        if alpha == 0:
            return (conditioning,)
        for native, _metadata in conditioning:
            if (
                native.ndim != 3
                or native.shape[-1] != 5120
                or not native.is_floating_point()
            ):
                raise ValueError(
                    "Semantic Bridge expects floating FL2VA conditioning [B,T,5120]."
                )
        weights = self._load_weights()
        result = []
        # Keep weights local to this execution: no GPU cache survives stage offload.
        with torch.inference_mode():
            for native, metadata in conditioning:
                h = native.float()
                projected = h / torch.sqrt(h.square().mean(dim=-1, keepdim=True) + 1e-6)
                for layer in ("fc1", "fc2", "fc3"):
                    projected = F.linear(
                        projected,
                        weights[layer + ".weight"].to(
                            device=h.device, dtype=torch.float32
                        ),
                        weights[layer + ".bias"].to(
                            device=h.device, dtype=torch.float32
                        ),
                    )
                    if layer != "fc3":
                        projected = F.silu(projected)
                source_rms = torch.sqrt(
                    projected.square().mean(dim=-1, keepdim=True) + 1e-8
                )
                target_rms = torch.sqrt(h.square().mean(dim=-1, keepdim=True) + 1e-8)
                projected = projected * (target_rms / source_rms)
                hybrid = (h + float(alpha) * (projected - h)).to(dtype=native.dtype)
                if not torch.isfinite(hybrid).all():
                    raise RuntimeError(
                        "Semantic Bridge produced non-finite conditioning."
                    )
                result.append(
                    [
                        hybrid,
                        {
                            **metadata,
                            "semantic_bridge_adapter": self.FILENAME,
                            "semantic_bridge_sha256": self.SHA256,
                            "semantic_bridge_alpha": float(alpha),
                            "semantic_bridge_magnitude_match": "per_token",
                        },
                    ]
                )
        return (result,)


def _qwen21_target_tokens(latent_image):
    """Count native 16x Qwen 2.1 pixels, including Comfy's empty-latent resize.

    EmptyLatentImage uses an 8x grid and declares its downscale ratio. Edit
    conditioning already returns a native 16x grid. Qwen 2.1 packs one token
    per native latent pixel, unlike the 2x2 packing used by earlier Qwen models.
    """
    samples = latent_image["samples"]
    ratio = float(latent_image.get("downscale_ratio_spacial", 16)) / 16
    height = round(int(samples.shape[-2]) * ratio)
    width = round(int(samples.shape[-1]) * ratio)
    if height < 1 or width < 1:
        raise ValueError("Qwen Image 2.1 requires a nonempty spatial latent.")
    return height * width


class H3Flux3LayoutElements:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"layout": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("ARRAY",)
    FUNCTION = "parse"
    CATEGORY = "model/conditioning/flux"

    def parse(self, layout):
        elements = json.loads(layout)
        if not isinstance(elements, list):
            raise ValueError("FLUX layout must be a JSON list.")
        return (elements,)


class H3Flux3EditPrompt:
    """BFL's image-aware edit upsampler with ordered, independently sized references."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True}),
            "system_prompt": ("STRING", {"multiline": True}),
            "max_length": ("INT", {"default": 3584, "min": 1, "max": 32768}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "image0": ("IMAGE",),
        }, "optional": {f"image{i}": ("IMAGE",) for i in range(1, 10)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "text"

    def generate(self, clip, prompt, system_prompt, max_length, seed, image0, **references):
        # Native Qwen3.5 accepts a list of different-sized images. Each is a
        # distinct vision block, in the same order as ReferenceMultiLatent.
        images = [image0] + [references[f"image{i}"] for i in range(1, 10)
                             if references.get(f"image{i}") is not None]
        images = [image[:, :, :, :3] for image in images]
        tokens = clip.tokenize(prompt, images=images, skip_template=False, min_length=1,
                               thinking=False, system_prompt=system_prompt)
        generated_ids = clip.generate(tokens, do_sample=True, max_length=max_length,
                                      temperature=1.0, top_k=1, top_p=0.95, min_p=0.0,
                                      seed=seed, repetition_penalty=1.05,
                                      presence_penalty=0.0, mtp=False)
        text = clip.decode(generated_ids)
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        elif text.lstrip().startswith("<think>"):
            text = ""
        if not text.strip():
            raise ValueError("FLUX 3 edit upsampler returned no instruction. Retry or use the remote prompt rewrite.")
        return (text.strip(),)


class H3Qwen21PromptEnhance:
    """Native Qwen 2.1 PE models: explicit reasoning and independent image inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "prompt": ("STRING", {"multiline": True}),
            "system_prompt": ("STRING", {"multiline": True}),
            "max_length": ("INT", {"default": 4096, "min": 1, "max": 32768}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }, "optional": {f"image{i}": ("IMAGE",) for i in range(10)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "text"

    def generate(self, clip, prompt, system_prompt, max_length, seed, **references):
        images = [references[f"image{i}"][:, :, :, :3] for i in range(10)
                  if references.get(f"image{i}") is not None]
        tokens = clip.tokenize(prompt, images=images, skip_template=False, min_length=1,
                               thinking=True, system_prompt=system_prompt)
        ids = clip.generate(tokens, do_sample=True, max_length=max_length,
                            temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                            seed=seed, repetition_penalty=1.0,
                            presence_penalty=0.0 if images else 1.5, mtp=True)
        # The native Qwen3.5 thinking template opens <think> before generation.
        # Without a close marker, max_length ended during reasoning, not a prompt.
        _reasoning, closed, text = clip.decode(ids).partition("</think>")
        if not closed or not text.strip():
            raise ValueError("Qwen prompt enhancer did not finish its final prompt. Retry or turn native enhancement off.")
        return (text.strip(),)


class H3Qwen21Sigmas:
    """Official base flow schedule: dynamic resolution shift and 0.02 terminal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent_image": ("LATENT",),
            "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "calculate"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def calculate(self, latent_image, steps):
        count = int(steps)
        if not 1 <= count <= 100:
            raise ValueError("Qwen Image 2.1 requires between 1 and 100 steps.")
        tokens = _qwen21_target_tokens(latent_image)
        mu = 0.5 + 0.4 * (tokens - 256) / (8192 - 256)
        exponent = math.exp(mu)
        shifted = [
            exponent / (exponent + (1.0 / (1.0 - i / count) - 1.0))
            for i in range(count)
        ]
        # A single step starts at full noise; the upstream terminal stretch
        # is undefined for this degenerate schedule.
        if count > 1:
            scale = (1.0 - shifted[-1]) / (1.0 - 0.02)
            shifted = [1.0 - (1.0 - value) / scale for value in shifted]
        return (torch.tensor([*shifted, 0.0], dtype=torch.float32),)


class H3Qwen21LightningSigmas:
    """LightX2V four-step schedule with dynamic shift and no terminal stretch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent_image": ("LATENT",)}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "calculate"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def calculate(self, latent_image):
        tokens = _qwen21_target_tokens(latent_image)
        mu = 0.5 + 0.4 * (tokens - 256) / (8192 - 256)
        exponent = math.exp(mu)
        shifted = [
            exponent / (exponent + (1.0 / value - 1.0))
            for value in (1.0, 0.75, 0.5, 0.25)
        ]
        return (torch.tensor([*shifted, 0.0], dtype=torch.float32),)


class H3Qwen21TurboSigmas:
    """Viggle six-step / nine-step raw nodes with Qwen 2.1's dynamic shift.

    Comfy's Qwen model has a fixed shift at load time; the adapter needs the
    resolution-dependent shift used by its training pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent_image": ("LATENT",),
            "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "calculate"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def calculate(self, latent_image, steps):
        tokens = _qwen21_target_tokens(latent_image)
        mu = 0.5 + 0.4 * (tokens - 256) / (8192 - 256)
        count = int(steps)
        if count == 6:
            raw = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
        elif count == 9:
            raw = (1.0, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25, 1 / 6, 1 / 12)
        else:
            raise ValueError("Viggle Turbo requires 6 or 9 steps.")
        exponent = math.exp(mu)
        shifted = [
            exponent / (exponent + (1.0 / value - 1.0))
            for value in raw
        ]
        # Viggle's scheduler has shift_terminal=null; no terminal stretch.
        return (torch.tensor([*shifted, 0.0], dtype=torch.float32),)



def _h3_viggle_lora_fwd(x, factors):
    down, up = factors
    if down.device != x.device or down.dtype != x.dtype:
        down = down.to(device=x.device, dtype=x.dtype)
    if up.device != x.device or up.dtype != x.dtype:
        up = up.to(device=x.device, dtype=x.dtype)
    return F.linear(F.linear(x, down), up)


def _h3_viggle_mlp_hooks(mlp, gate, up, down):
    saved = {}

    def gate_up_hook(_module, inputs, output):
        saved["gate_up"] = output + torch.cat((
            _h3_viggle_lora_fwd(inputs[0], gate),
            _h3_viggle_lora_fwd(inputs[0], up),
        ), dim=-1)
        return saved["gate_up"]

    def mlp_hook(_module, inputs, output):
        gate_values, up_values = saved.pop("gate_up").chunk(2, dim=-1)
        return output + _h3_viggle_lora_fwd(
            F.silu(gate_values) * up_values, down
        )

    return [
        mlp.gate_up.register_forward_hook(gate_up_hook),
        mlp.register_forward_hook(mlp_hook),
    ]


def _h3_run_viggle_lora(lora, executor, *args, **kwargs):
    diffusion_model = executor.class_obj
    hooks = []
    try:
        for name, factors in lora.items():
            parent_name, _, leaf = name.rpartition(".")
            parent = diffusion_model.get_submodule(parent_name)
            if factors[0].device != args[0].device:
                factors[0] = factors[0].to(args[0].device)
                factors[1] = factors[1].to(args[0].device)
            if getattr(parent, "fused", False):
                if leaf == "out":
                    hooks.extend(_h3_viggle_mlp_hooks(
                        parent, lora[parent_name + ".gate_layer"],
                        lora[parent_name + ".proj"], factors,
                    ))
                continue

            def hook(_module, inputs, output, factors=factors):
                return output + _h3_viggle_lora_fwd(inputs[0], factors)

            hooks.append(diffusion_model.get_submodule(name).register_forward_hook(hook))
        return executor(*args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()


class H3Qwen21ViggleLora:
    """Apply the Viggle adapter at inference time without merging its weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, model, lora_name):
        state, metadata = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("loras", lora_name),
            return_metadata=True,
        )
        adapter_config = json.loads(
            (metadata or {}).get("lora_adapter_metadata", "{}")
        )
        scale = float(adapter_config.get("transformer.lora_alpha", 1)) / float(
            adapter_config.get("transformer.r", 1)
        )
        lora = {
            key.removeprefix("transformer.").removesuffix(".lora_A.weight"):
                [state[key], state[key.replace("lora_A", "lora_B")] * scale]
            for key in state if key.endswith(".lora_A.weight")
        }
        if not lora:
            raise ValueError("No Viggle-format Qwen Image 2.1 LoRA weights found.")
        patched = model.clone()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "h3_qwen21_viggle_lora",
            lambda executor, *args, **kwargs: _h3_run_viggle_lora(
                lora, executor, *args, **kwargs
            ),
        )
        return (patched,)


class H3LTXPadVideoFrames:
    """Pad a guide to 8n+1 without dropping source frames; retain its trim length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "padded_frames", "source_frames")
    FUNCTION = "pad"
    CATEGORY = "H3/LTX"

    def pad(self, images):
        if images.ndim != 4 or images.shape[0] < 1:
            raise ValueError("LTX upscale requires a nonempty IMAGE frame batch.")
        frames = int(images.shape[0])
        padded = ((frames - 1 + 7) // 8) * 8 + 1
        if padded != frames:
            images = torch.cat((images, images[-1:].expand(padded - frames, -1, -1, -1)))
        return images, padded, frames


class H3LTXHDRConditioning:
    """Load fixed HDR video context with a neutral, unused AV audio context."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_name": (folder_paths.get_filename_list("embeddings"),)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "H3/LTX"

    def load(self, file_name):
        path = folder_paths.get_full_path_or_raise("embeddings", file_name)
        state = comfy.utils.load_torch_file(path, safe_load=True)
        video = next((state[key] for key in ("video_context", "video_prompt_embeds") if key in state), None)
        if video is None:
            raise ValueError("HDR scene embeddings must contain video_context or video_prompt_embeds.")
        if video.ndim == 2:
            video = video.unsqueeze(0)
        if video.ndim != 3 or video.shape[-1] not in (4096, 6144):
            raise ValueError("HDR video context must have shape [B,T,4096] or [B,T,6144].")
        if video.shape[-1] == 4096:
            # Native Comfy LTX-AV splits video/audio context even when the
            # sampled latent is video-only. No audio tokens are generated.
            audio = next((state[key] for key in ("audio_context", "audio_prompt_embeds") if key in state), None)
            if audio is None:
                audio = video.new_zeros((*video.shape[:-1], 2048))
            elif audio.ndim == 2:
                audio = audio.unsqueeze(0)
            if audio.shape != (*video.shape[:-1], 2048):
                raise ValueError("HDR audio context must match the video's batch/token dimensions.")
            video = torch.cat((video, audio.to(video)), dim=-1)
        return ([[video, {}]],)

    @classmethod
    def IS_CHANGED(cls, file_name):
        path = folder_paths.get_full_path_or_raise("embeddings", file_name)
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()


class H3LatentUpscaleModelLoader:
    """Use Core's optimized H3 model while honoring the selected compute dtype."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("latent_upscale_models"),),
            "precision": (["bf16", "fp16", "fp32"],),
        }}

    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "model/loaders"

    def load_model(self, model_name, precision):
        # Imports stay local so updating an old backend produces a useful error.
        try:
            from comfy.ldm.minimax.latent_upscaler import MiniMaxH3LatentUpscaler
        except ImportError as exc:
            raise RuntimeError("Native H3 upscaling requires the updated ComfyUI pin; rerun setup or rebuild Modal.") from exc
        import comfy.model_patcher
        import comfy.ops
        import comfy.storage

        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if precision not in dtypes:
            raise ValueError("Unsupported H3 latent upscaler precision")
        path = folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        required = {"in_blocks.0.emb_layers.1.weight", "conv_in.weight", "in_blocks.1.dwconv.weight"}
        if not required <= sd.keys():
            raise ValueError("Expected a MiniMax H3 3D-conv latent upscaler checkpoint")
        model = MiniMaxH3LatentUpscaler(
            in_channels=sd["conv_in.weight"].shape[1],
            channels=sd["conv_in.weight"].shape[0],
            in_blocks=len({k.split(".")[1] for k in sd if k.startswith("in_blocks.") and ".in_layers." in k}),
            out_blocks=len({k.split(".")[1] for k in sd if k.startswith("out_blocks.") and ".in_layers." in k}),
            temporal_kernel=sd["in_blocks.1.dwconv.weight"].shape[2],
            dtype=dtypes[precision], operations=comfy.ops.disable_weight_init,
        )
        comfy.model_management.archive_model_dtypes(model)
        patcher = comfy.model_patcher.CoreModelPatcher(
            model, load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            fast_disk=comfy.storage.state_dict_fast_disk(sd),
        )
        model.load_state_dict(sd, assign=patcher.is_dynamic())
        return (patcher,)


class H3RegenerateEncodeVideo:
    """Pad to H3's temporal grid without dropping any source frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "vae": ("VAE",)}}

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("video_latent", "padded_frames", "source_frames")
    FUNCTION = "encode"
    CATEGORY = "H3/Regenerate"

    def encode(self, images, vae):
        count = images.shape[0]
        if count < 1:
            raise ValueError("Regenerate needs at least one video frame")
        padded = max(5, count)
        padded += (5 - padded) % 17
        if padded > count:
            images = torch.cat((images, images[-1:].expand(padded - count, -1, -1, -1)))
        samples = vae.encode_tiled(
            images[..., :3], tile_x=512, tile_y=512, overlap=64,
            tile_t=64, overlap_t=8,
        )
        return ({"samples": samples}, padded, count)


class H3RegeneratePrepareLatent:
    """Refine video against clean encoded silence; preserve delivery audio elsewhere."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "audio_vae": ("VAE",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "prepare"
    CATEGORY = "H3/Regenerate"

    def prepare(self, latent, audio_vae):
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
            raise ValueError("Regenerate expects a MiniMax H3 AV latent")
        video, audio = samples.tensors
        # H3 audio has 40 latent frames/sec; its VAE consumes [B, samples, stereo].
        if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
            raise ValueError("Regenerate expects H3 video [B,24,T,H,W] and audio [B,32,2,T]")
        silence = torch.zeros((1, audio.shape[-1] * 800, 2), dtype=torch.float32)
        encoded = audio_vae.encode(silence)
        if encoded.shape[-1] < audio.shape[-1]:
            encoded = torch.cat((encoded, encoded[..., -1:].expand(
                -1, -1, -1, audio.shape[-1] - encoded.shape[-1],
            )), dim=-1)
        encoded = encoded[..., :audio.shape[-1]].to(video)
        encoded = encoded.expand(video.shape[0], -1, -1, -1)
        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video, encoded))
        # Native H3 masks set the audio timestep to clean and prevent audio sampling.
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((
            torch.ones_like(video), torch.zeros_like(encoded),
        ))
        return (out,)


# Storyboard continuation. The tail selection and per-stream masks follow
# ComfyUI-MiniMaxH3-PrefixStream (MIT, knoic, reviewed at 775be90), which in turn
# credits Herrgott's H3 Infinite Continuation Suite. Native ComfyUI applies the
# nested (video, audio) noise_mask per token (Comfy-Org/ComfyUI#15375).
H3_FRAMES_PER_STEP = (1, 4, 4, 4, 4)
H3_VIDEO_FPS = 24
H3_AUDIO_HZ = 40
H3_CONTINUATION_FRAMES = (22, 39, 56)


def h3_latent_frame_boundaries(steps):
    """Exclusive pixel-frame boundaries of an H3 video latent with ``steps`` steps."""
    boundaries = [0]
    for step in range(int(steps)):
        boundaries.append(boundaries[-1] + H3_FRAMES_PER_STEP[step % 5])
    return boundaries


def h3_continuation_plan(source_steps, target_steps, context_frames):
    """Pick the latest phase-aligned source tail of at most ``context_frames`` frames.

    Contexts follow the H3 ``5 + 17k`` frame grid: ``2 + 5k`` latent steps that
    start on a single-frame step. The target must keep new frames after the head.
    """
    source = h3_latent_frame_boundaries(source_steps)
    target_frames = h3_latent_frame_boundaries(target_steps)[-1]
    cap = min(int(context_frames), source[-1], target_frames - 1)
    if cap < 5:
        raise ValueError(
            "Continuation needs at least 5 source frames and a target longer than its context."
        )
    frames = 5 + ((cap - 5) // 17) * 17
    context_steps = 2 + 5 * ((frames - 5) // 17)
    end_step = next(
        (end for end in range(int(source_steps), context_steps - 1, -1)
         if end % 5 == 2 and (end - context_steps) % 5 == 0),
        None,
    )
    if end_step is None:
        raise ValueError("The source latent has no phase-aligned continuation tail.")
    start_step = end_step - context_steps
    start_frame, end_frame = source[start_step], source[end_step]
    audio_steps = round((end_frame - start_frame) * H3_AUDIO_HZ / H3_VIDEO_FPS)
    audio_end = round(end_frame * H3_AUDIO_HZ / H3_VIDEO_FPS)
    return {
        "start_step": start_step,
        "end_step": end_step,
        "context_steps": context_steps,
        "frames": end_frame - start_frame,
        "audio_start": audio_end - audio_steps,
        "audio_end": audio_end,
        "audio_steps": audio_steps,
        "ignored_tail_frames": source[-1] - end_frame,
    }


def h3_av_streams(latent, label):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
        raise ValueError(f"{label} expects a MiniMax H3 AV latent")
    video, audio = samples.tensors
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1:3] != (32, 2):
        raise ValueError(f"{label} expects H3 video [B,24,T,H,W] and audio [B,32,2,T]")
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError(f"{label} supports batch size 1")
    return video, audio


def h3_apply_continuation(target_video, target_audio, source_video, source_audio, context_frames):
    """Copy the source AV tail into the target head; zero masks protect it."""
    if source_video.shape[1:2] + source_video.shape[3:] != target_video.shape[1:2] + target_video.shape[3:]:
        raise ValueError(
            "Continuation needs matching video latent sizes; keep the storyboard's size and VAE."
        )
    plan = h3_continuation_plan(source_video.shape[2], target_video.shape[2], context_frames)
    audio_end = min(plan["audio_end"], source_audio.shape[-1])
    audio_start = max(0, audio_end - plan["audio_steps"])
    audio_steps = audio_end - audio_start
    if not 0 < audio_steps < target_audio.shape[-1]:
        raise ValueError("The continued audio context must be shorter than the new clip.")
    video = target_video.clone()
    audio = target_audio.clone()
    steps = plan["context_steps"]
    video[:, :, :steps] = source_video[:, :, plan["start_step"]:plan["end_step"]].to(video)
    audio[..., :audio_steps] = source_audio[..., audio_start:audio_end].to(audio)
    video_mask = torch.ones_like(video, dtype=torch.float32)
    audio_mask = torch.ones_like(audio, dtype=torch.float32)
    video_mask[:, :, :steps] = 0.0
    audio_mask[..., :audio_steps] = 0.0
    plan.update(audio_start=audio_start, audio_end=audio_end, audio_steps=audio_steps)
    return video, audio, video_mask, audio_mask, plan


class H3ContinuationPrefix:
    """Start a new H3 clip from the tail of a previous one (storyboard flow)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "source": ("LATENT",),
                "context_frames": ("INT", {"default": 39, "min": 5, "max": 192, "step": 17}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "H3/Storyboard"

    def apply(self, latent, source, context_frames):
        target_video, target_audio = h3_av_streams(latent, "Continuation target")
        source_video, source_audio = h3_av_streams(source, "Continuation source")
        video, audio, video_mask, audio_mask, plan = h3_apply_continuation(
            target_video, target_audio, source_video, source_audio, context_frames
        )
        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        # Always replace: a stage-one sampler output keeps its low-resolution mask.
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        out["h3_prefix_frames"] = plan["frames"]
        logging.info(
            "[H3 storyboard] Continued %d frames / %d audio ticks from source steps %d:%d.",
            plan["frames"], plan["audio_steps"], plan["start_step"], plan["end_step"],
        )
        return (out,)


def h3_storyboard_latent_path(relative):
    """Resolve a saved storyboard latent strictly inside the ComfyUI output directory."""
    root = os.path.realpath(folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(root, str(relative)))
    if os.path.commonpath((root, path)) != root or not path.endswith(".safetensors"):
        raise ValueError("Storyboard latents must be .safetensors files inside the output directory")
    return path


class H3SaveAVLatent:
    """Save a MiniMax H3 AV latent so a later storyboard scene can continue from it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "h3/.storyboard/latent"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "H3/Storyboard"
    OUTPUT_NODE = True

    def save(self, latent, filename_prefix):
        from safetensors.torch import save_file

        video, audio = h3_av_streams(latent, "Save AV latent")
        full_output_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(
            str(filename_prefix), folder_paths.get_output_directory()
        )
        file = f"{filename}_{counter:05}_.safetensors"
        metadata = {"h3_prefix_frames": str(int(latent.get("h3_prefix_frames", 0)))}
        save_file(
            {"video": video.detach().float().cpu().contiguous(),
             "audio": audio.detach().float().cpu().contiguous()},
            os.path.join(full_output_folder, file),
            metadata=metadata,
        )
        return {"ui": {"h3_latents": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


class H3LoadAVLatent:
    """Load a latent saved by ``H3SaveAVLatent`` from the output directory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "H3/Storyboard"

    @classmethod
    def IS_CHANGED(cls, path):
        resolved = h3_storyboard_latent_path(path)
        return os.path.getmtime(resolved) if os.path.isfile(resolved) else float("nan")

    def load(self, path):
        from safetensors.torch import load_file

        tensors = load_file(h3_storyboard_latent_path(path))
        video, audio = tensors["video"], tensors["audio"]
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        h3_av_streams(latent, "Load AV latent")
        return (latent,)


H3_TAIL_FRAMES = 73  # 5 + 17 * 4: ends on a latent group, and covers every continuation overlap.


def h3_tail_latent_steps(frames):
    """Video latent steps a ``5 + 17k`` frame clip encodes to."""
    frames = int(frames)
    if frames < 5 or (frames - 5) % 17:
        raise ValueError(f"A continuation tail is 5 + 17k frames, not {frames}")
    return 2 + 5 * ((frames - 5) // 17)


def h3_tail_pixels(images, frames, width, height):
    """The clip's last ``frames`` frames on the ``width`` x ``height`` canvas."""
    if images.shape[0] < frames:
        raise ValueError(f"The clip has {images.shape[0]} frames; continuing needs its last {frames}")
    pixels = images[-frames:, :, :, :3]
    if tuple(pixels.shape[1:3]) != (height, width):
        pixels = comfy.utils.common_upscale(
            pixels.movedim(-1, 1), width, height, "lanczos", "center",
        ).movedim(1, -1)
    return pixels


def h3_tail_wave(audio, total_frames, frames, ticks):
    """The stereo 32 kHz soundtrack under the tail, padded to exactly ``ticks`` audio latents."""
    rate, frame_samples = 32000, 800  # H3 audio: 40 latent frames/sec at 32 kHz.
    length = ticks * frame_samples
    wave = torch.zeros(2, length)
    if audio is not None:
        source_rate = int(audio["sample_rate"])
        wave = audio["waveform"][0].float()
        wave = wave.repeat(2, 1) if wave.shape[0] == 1 else wave[:2]
        start = round((total_frames - frames) * source_rate / H3_VIDEO_FPS)
        wave = wave[:, start:start + round(frames * source_rate / H3_VIDEO_FPS)]
        if source_rate != rate:
            import comfy.audio

            wave = comfy.audio.resample(wave, source_rate, rate)
    # Exact latent frames here: VAE.encode would centre-crop and shift the sound.
    return F.pad(wave, (0, max(0, length - wave.shape[1])))[:, :length]


class H3EncodeAVTail:
    """Encode the end of a real clip as an H3 AV latent that a storyboard scene can continue."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 4096, "step": 32}),
                "frames": ("INT", {"default": H3_TAIL_FRAMES, "min": 22, "max": 192, "step": 17}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "H3/Storyboard"

    def encode(self, images, vae, audio_vae, width, height, frames, audio=None):
        steps = h3_tail_latent_steps(frames)
        video = vae.encode(h3_tail_pixels(images, int(frames), int(width), int(height)))
        if video.shape[2] != steps:
            raise ValueError(f"Encoding {frames} frames gave {video.shape[2]} latent steps, not {steps}")
        ticks = round(int(frames) * H3_AUDIO_HZ / H3_VIDEO_FPS)
        wave = h3_tail_wave(audio, images.shape[0], int(frames), ticks)
        encoded = audio_vae.encode(wave[None].movedim(1, -1))
        encoded = F.pad(encoded, (0, max(0, ticks - encoded.shape[-1])))[..., :ticks].to(video)
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, encoded))}
        h3_av_streams(latent, "Encode AV tail")
        return (latent,)


class H3SourceAudioLatent:
    """Hold a clip's own soundtrack clean, so H3 redraws video against the real speech."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"latent": ("LATENT",), "audio_vae": ("VAE",)},
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "hold"
    CATEGORY = "H3/Regenerate"

    def hold(self, latent, audio_vae, audio=None):
        import comfy.audio

        samples = latent["samples"]
        if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
            raise ValueError("Source audio expects a MiniMax H3 AV latent")
        video, template = samples.tensors
        rate, frame_samples = 32000, 800  # H3 audio: 40 latent frames/sec at 32 kHz.
        length = template.shape[-1] * frame_samples
        wave = torch.zeros(2, length)
        if audio is not None:
            wave = audio["waveform"][0].float()
            wave = wave.repeat(2, 1) if wave.shape[0] == 1 else wave[:2]
            if int(audio["sample_rate"]) != rate:
                wave = comfy.audio.resample(wave, int(audio["sample_rate"]), rate)
        # Exact latent frames here: VAE.encode would centre-crop and shift the speech.
        wave = F.pad(wave, (0, max(0, length - wave.shape[1])))[:, :length]
        encoded = audio_vae.encode(wave[None].movedim(1, -1))
        encoded = F.pad(encoded, (0, max(0, template.shape[-1] - encoded.shape[-1])))
        encoded = encoded[..., :template.shape[-1]].to(video)
        return ({
            **latent,
            "samples": comfy.nested_tensor.NestedTensor((video, encoded)),
            "noise_mask": comfy.nested_tensor.NestedTensor((
                torch.ones_like(video), torch.zeros_like(encoded),
            )),
        },)


class H3LipSyncMask:
    """Apply a tracked mouth mask to H3's video tokens while holding its audio clean."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "source_frames": ("IMAGE",), "mouth_mask": ("MASK",)}}

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("masked_latent", "composite_mask")
    FUNCTION = "apply"
    CATEGORY = "H3/Dubbing"

    def apply(self, latent, source_frames, mouth_mask):
        video, audio = h3_av_streams(latent, "Lip sync mask")
        height, width = video.shape[-2:]
        boundaries = h3_latent_frame_boundaries(video.shape[2])
        expected = (boundaries[-1], height * 16, width * 16)
        if tuple(source_frames.shape[:3]) != expected or tuple(mouth_mask.shape) != expected:
            raise ValueError(
                "Lip sync requires a matching crop and per-frame mask on H3's 17k+5 frame grid "
                f"(expected {expected}, got crop={tuple(source_frames.shape[:3])}, mask={tuple(mouth_mask.shape)})"
            )
        if height % 2 or width % 2:
            raise ValueError("Lip sync crop dimensions must be multiples of 32 pixels")

        # A video token covers a 2x2 latent patch. Sample any patch touched by the
        # mask in its pixel-frame span; the softer per-frame mask handles delivery.
        patches = F.max_pool2d((mouth_mask >= 0.5).float().unsqueeze(1), kernel_size=32).squeeze(1)
        editable = patches.flatten(1).any(dim=1)
        if not editable.any():
            raise ValueError("The lip-sync mouth mask has no editable pixels")
        grouped = torch.stack([patches[first:last].amax(dim=0)
                               for first, last in zip(boundaries, boundaries[1:])])
        grouped = grouped.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        mask = grouped.to(video).unsqueeze(0).unsqueeze(0).expand_as(video)
        # Prevent faint matte values on skipped frames from blending VAE reconstruction
        # into the original picture, even when a neighboring frame shares the token.
        composite_mask = torch.where(editable[:, None, None], mouth_mask, 0.0)
        masked = {**latent, "noise_mask": comfy.nested_tensor.NestedTensor((mask, torch.zeros_like(audio)))}
        return masked, composite_mask


_LOWER_FACE_LANDMARKS = (
    # Outer lips + philtrum + lower cheeks + chin/jawline (MediaPipe 468-point FaceMesh)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    164, 207, 427, 212, 432, 202, 422, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
)


def _mediapipe_mouth_polygon(
    frame_rgb_uint8,
    width,
    height,
    face_mesh,
    target_center=(0.5, 0.5),
    jaw_margin=0.06,
    *,
    pts_override=None,
):
    """Return a binary (H, W) float32 polygon mask over the real mouth/jaw landmarks, or None."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if pts_override is None:
        result = face_mesh.process(frame_rgb_uint8)
        faces = getattr(result, "multi_face_landmarks", None)
        if not faces:
            return None
        # Pick the face closest to the tracked face_rect center.
        tx, ty = target_center
        best = min(
            faces,
            key=lambda f: (f.landmark[1].x - tx) ** 2 + (f.landmark[1].y - ty) ** 2,
        )
        lm = {
            idx: (float(best.landmark[idx].x), float(best.landmark[idx].y))
            for idx in set(_LOWER_FACE_LANDMARKS) | {2, 13, 14, 33, 263, 152}
        }
    else:
        lm = pts_override

    p_nose = np.array([lm[2][0] * width, lm[2][1] * height], dtype=np.float32)
    p_chin = np.array([lm[152][0] * width, lm[152][1] * height], dtype=np.float32)
    p_mouth = np.array([
        0.5 * (lm[13][0] + lm[14][0]) * width,
        0.5 * (lm[13][1] + lm[14][1]) * height,
    ], dtype=np.float32)
    eye_vec = np.array([
        (lm[263][0] - lm[33][0]) * width,
        (lm[263][1] - lm[33][1]) * height,
    ], dtype=np.float32)
    eye_len = float(np.linalg.norm(eye_vec))
    if eye_len > 1e-3:
        u = eye_vec / eye_len
        n = np.array([-u[1], u[0]], dtype=np.float32)
    else:
        n = np.array([0.0, 1.0], dtype=np.float32)
    if float(np.dot(p_chin - p_nose, n)) < 0.0:
        n = -n

    nose_to_chin = max(4.0, float(np.dot(p_chin - p_nose, n)))
    jaw_pad = max(0.0, float(jaw_margin)) * nose_to_chin
    mouth_proj = float(np.dot(p_mouth - p_nose, n))
    pts_list = []
    for i in _LOWER_FACE_LANDMARKS:
        pt = np.array([lm[i][0] * width, lm[i][1] * height], dtype=np.float32)
        proj = float(np.dot(pt - p_nose, n))
        if proj > mouth_proj:
            pt = pt + jaw_pad * n
            proj = float(np.dot(pt - p_nose, n))
        if proj < 0.0:
            pt = pt - proj * n
        px = max(0.0, min(float(width - 1), float(pt[0])))
        py = max(0.0, min(float(height - 1), float(pt[1])))
        pts_list.append([px, py])
    pts = np.array(pts_list, dtype=np.float32)
    hull = cv2.convexHull(pts).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask


class H3LowerFaceMask:
    """Build per-frame lower-face denoise and composite masks from H3FaceTrackCrop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crops": ("IMAGE",),
                "transform": ("H3FACEXFORM",),
            },
            "optional": {
                "active_start": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "active_end": ("INT", {"default": -1, "min": -1, "max": 999999}),
                "mouth_top": ("FLOAT", {"default": 0.52, "min": 0.2, "max": 0.85, "step": 0.01}),
                "mouth_bottom": ("FLOAT", {"default": 1.02, "min": 0.6, "max": 1.35, "step": 0.01}),
                "width_scale": ("FLOAT", {"default": 0.82, "min": 0.4, "max": 1.3, "step": 0.02}),
                "denoise_expand": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 1.6, "step": 0.02}),
                "jaw_margin": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.35, "step": 0.02}),
                "separate_opacity": ("BOOLEAN", {"default": False}),
                "stitch_feather": ("INT", {"default": -1, "min": -1, "max": 64,
                    "tooltip": "Feather only the edge of the mouth patch, in source pixels. -1 uses the legacy mask."}),
                "edge_ramp": ("INT", {"default": 0, "min": 0, "max": 12,
                    "tooltip": "Cosine temporal fade frames at active lip-sync boundaries in single-composite mode."}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("denoise_mask", "composite_mask", "spatial_mask")
    FUNCTION = "build"
    CATEGORY = "H3/Dubbing"

    @staticmethod
    def _extract_smoothed_landmarks(
        crops, face_rects, frames, width, height, face_mesh, eligible_indices,
    ):
        import numpy as np

        lm_ids = tuple(dict.fromkeys((*_LOWER_FACE_LANDMARKS, 2, 13, 14, 33, 263, 152)))
        raw = {}
        for index in eligible_indices:
            if index < len(face_rects):
                fx, fy, fw, fh = (float(v) for v in face_rects[index])
            else:
                fx, fy, fw, fh = width * 0.25, height * 0.25, width * 0.5, height * 0.5
            if fw <= 1.0 or fh <= 1.0:
                continue
            try:
                frame_np = (crops[index, ..., :3].detach().cpu().clamp(0, 1).numpy() * 255.0).astype("uint8")
                result = face_mesh.process(frame_np)
                faces = getattr(result, "multi_face_landmarks", None)
                if not faces:
                    continue
                tx, ty = ((fx + 0.5 * fw) / width, (fy + 0.5 * fh) / height)
                best = min(faces, key=lambda f: (f.landmark[1].x - tx) ** 2 + (f.landmark[1].y - ty) ** 2)
                raw[index] = np.array(
                    [[float(best.landmark[idx].x), float(best.landmark[idx].y)] for idx in lm_ids],
                    dtype=np.float32,
                )
            except Exception:
                continue

        # Linearly interpolate short 1-3 frame MediaPipe misses across eligible frames.
        eligible_set = set(eligible_indices)
        interp = dict(raw)
        valid_sorted = sorted(raw)
        for a, b in zip(valid_sorted, valid_sorted[1:]):
            gap = b - a - 1
            if 1 <= gap <= 3 and all(k in eligible_set for k in range(a + 1, b)):
                for k in range(a + 1, b):
                    alpha = float(k - a) / float(b - a)
                    interp[k] = (1.0 - alpha) * raw[a] + alpha * raw[b]

        # 3-frame [0.25, 0.5, 0.25] temporal smoothing across contiguous detections.
        smoothed = {}
        for index in interp:
            acc = 0.5 * interp[index]
            w_sum = 0.5
            if (index - 1) in interp:
                acc = acc + 0.25 * interp[index - 1]
                w_sum += 0.25
            if (index + 1) in interp:
                acc = acc + 0.25 * interp[index + 1]
                w_sum += 0.25
            arr = acc / w_sum
            smoothed[index] = {
                lm_id: (float(arr[pos, 0]), float(arr[pos, 1]))
                for pos, lm_id in enumerate(lm_ids)
            }
        return smoothed

    def build(self, crops, transform, active_start=0, active_end=-1,
              mouth_top=0.52, mouth_bottom=1.02, width_scale=0.82, denoise_expand=1.15,
              jaw_margin=0.06, separate_opacity=False, stitch_feather=-1, edge_ramp=0):
        import math

        frames, height, width = int(crops.shape[0]), int(crops.shape[1]), int(crops.shape[2])
        if frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("H3LowerFaceMask requires a non-empty crop batch")
        if mouth_bottom <= mouth_top:
            raise ValueError("mouth_bottom must be greater than mouth_top")

        start = max(0, min(int(active_start), frames))
        end = frames if int(active_end) < 0 else max(start, min(int(active_end), frames))
        # The legacy mask fades over at most three active frames per edge.
        ramp = min(3, max(1, (end - start + 1) // 2))
        denoise_start = max(0, start - 2)
        denoise_end = min(frames, end + 2)

        single_composite = stitch_feather >= 0
        boxes = transform.get("boxes") or []
        face_rects = transform.get("face_rect") or []
        absent = transform.get("absent") or []
        detected = transform.get("detected") or []
        weights = transform.get("weights") or []

        device = crops.device
        yy = torch.arange(height, device=device, dtype=torch.float32)[:, None] + 0.5
        xx = torch.arange(width, device=device, dtype=torch.float32)[None, :] + 0.5
        denoise_mask = torch.zeros((frames, height, width), device=device, dtype=torch.float32)
        composite_mask = torch.zeros((frames, height, width), device=device, dtype=torch.float32)
        spatial_mask = torch.zeros_like(composite_mask) if separate_opacity else composite_mask

        active_vec = [
            not (index < len(absent) and absent[index])
            and (bool(detected[index]) if index < len(detected) else True)
            and (start <= index < end)
            for index in range(frames)
        ]
        eff_edge_ramp = (
            max(1, min(int(edge_ramp), (end - start + 1) // 2))
            if single_composite and int(edge_ramp) > 0 and start < end else 0
        )

        face_mesh = None
        cv2_mod = None
        try:
            import cv2 as _cv2
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=2, refine_landmarks=False,
                    min_detection_confidence=0.35, min_tracking_confidence=0.35,
                )
                cv2_mod = _cv2
        except Exception:
            face_mesh = None

        try:
            eligible_indices = [
                index for index in range(denoise_start, denoise_end)
                if not (index < len(absent) and absent[index])
                and (
                    (bool(detected[index]) if index < len(detected) else True)
                    or (float(weights[index]) if index < len(weights) else 1.0) > 0.0
                )
            ]
            has_mesh_process = face_mesh is not None and hasattr(face_mesh, "process")
            smoothed_lms = (
                self._extract_smoothed_landmarks(
                    crops, face_rects, frames, width, height, face_mesh, eligible_indices,
                )
                if has_mesh_process else {}
            )

            for index in range(frames):
                if index < denoise_start or index >= denoise_end:
                    continue
                if index < len(absent) and absent[index]:
                    continue
                is_detected = bool(detected[index]) if index < len(detected) else True
                weight = float(weights[index]) if index < len(weights) else (1.0 if is_detected else 0.0)
                if not is_detected and weight <= 0.0:
                    continue
                if single_composite:
                    if not active_vec[index]:
                        opacity = 0.0
                    elif eff_edge_ramp > 0:
                        d_left = eff_edge_ramp
                        for step in range(1, eff_edge_ramp):
                            if index - step >= 0 and not active_vec[index - step]:
                                d_left = step
                                break
                        d_right = eff_edge_ramp
                        for step in range(1, eff_edge_ramp):
                            if index + step < frames and not active_vec[index + step]:
                                d_right = step
                                break
                        ramp_t = min(1.0, min(d_left, d_right) / float(eff_edge_ramp))
                        opacity = 0.5 - 0.5 * math.cos(math.pi * ramp_t)
                    else:
                        opacity = 1.0
                    source_h = float(boxes[index][3]) if index < len(boxes) else height
                    feather_px = stitch_feather * height / max(1.0, source_h)
                else:
                    edge_weight = max(0, min(ramp, index - start + 1, end - index)) / ramp
                    opacity = max(0.0, min(1.0, weight)) * edge_weight
                if index < len(face_rects):
                    fx, fy, fw, fh = (float(v) for v in face_rects[index])
                else:
                    fx, fy, fw, fh = width * 0.25, height * 0.25, width * 0.5, height * 0.5
                if fw <= 1.0 or fh <= 1.0:
                    continue

                poly_mask = None
                if face_mesh is not None and (not has_mesh_process or index in smoothed_lms):
                    try:
                        frame_np = (crops[index, ..., :3].detach().cpu().clamp(0, 1).numpy() * 255.0).astype("uint8")
                        target_center = ((fx + 0.5 * fw) / width, (fy + 0.5 * fh) / height)
                        poly_mask = _mediapipe_mouth_polygon(
                            frame_np, width, height, face_mesh, target_center,
                            jaw_margin=jaw_margin, pts_override=smoothed_lms.get(index),
                        )
                    except Exception:
                        poly_mask = None

                if poly_mask is not None and cv2_mod is not None:
                    base_pad = max(3, int(round(0.04 * fh * max(1.0, float(denoise_expand)))))
                    if single_composite and feather_px > 0.0:
                        pad_px = max(base_pad, math.ceil(feather_px) + 2)
                    else:
                        pad_px = base_pad
                    k_denoise = cv2_mod.getStructuringElement(cv2_mod.MORPH_ELLIPSE, (pad_px * 2 + 1, pad_px * 2 + 1))
                    denoise_np = cv2_mod.dilate(poly_mask, k_denoise)
                    denoise_mask[index] = torch.from_numpy(denoise_np).to(device=device, dtype=torch.float32)
                    if opacity > 0.0:
                        comp_np = poly_mask
                        if single_composite:
                            # Put the transition outside the lip/jaw polygon and inside the denoise mask.
                            edge_width = min(feather_px, float(pad_px))
                            if edge_width > 0.0:
                                outside = (poly_mask == 0).astype("uint8")
                                distance = cv2_mod.distanceTransform(outside, cv2_mod.DIST_L2, cv2_mod.DIST_MASK_PRECISE)
                                comp_np = (1.0 - distance / edge_width).clip(0.0, 1.0) * denoise_np
                        else:
                            blur_px = max(3, int(round(0.03 * fh)) | 1)
                            comp_np = cv2_mod.GaussianBlur(poly_mask, (blur_px, blur_px), 0)
                        soft = torch.from_numpy(comp_np).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                else:
                    cx = fx + 0.5 * fw
                    cy = fy + 0.5 * (mouth_top + mouth_bottom) * fh
                    rx = max(1.0, 0.5 * fw * float(width_scale))
                    ry = max(1.0, 0.5 * fh * float(mouth_bottom - mouth_top))
                    d2_comp = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
                    effective_expand = float(denoise_expand)
                    if single_composite and feather_px > 0.0:
                        effective_expand = max(effective_expand, 1.0 + feather_px / max(1.0, min(rx, ry)) + 0.05)
                    denoise_mask[index] = (d2_comp <= effective_expand ** 2).float()
                    if opacity > 0.0:
                        if single_composite:
                            edge_width = min(feather_px / min(rx, ry), effective_expand - 1.0)
                            if edge_width > 0.0:
                                soft = ((1.0 + edge_width - d2_comp.sqrt()) / edge_width).clamp(0.0, 1.0)
                            else:
                                soft = (d2_comp <= 1.0).float()
                        else:
                            soft = ((1.0 - d2_comp) / 0.25).clamp(0.0, 1.0)
                if opacity > 0.0:
                    if separate_opacity:
                        spatial_mask[index] = soft
                    if separate_opacity and not single_composite:
                        # Blend the spatial taper in crop space, then fade the patch at stitch time.
                        composite_mask[index] = (soft > 1e-4) * opacity
                    else:
                        composite_mask[index] = soft * opacity
        finally:
            if face_mesh is not None:
                try:
                    face_mesh.close()
                except Exception:
                    pass

        if not (denoise_mask >= 0.5).any() or not (composite_mask > 0.0).any():
            raise ValueError("No detected face frames available in the active lip-sync interval")
        return denoise_mask, composite_mask, spatial_mask


class H3LipSyncPrepareCrop:
    """Neutralize source lip texture inside the denoise mask before VAE injection to prevent mouth ghosting."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crops": ("IMAGE",),
                "denoise_mask": ("MASK",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("prepared_crops",)
    FUNCTION = "prepare"
    CATEGORY = "H3/Dubbing"

    @staticmethod
    def _torch_blur_2d(tensor_nchw, kernel_size):
        k = max(3, int(kernel_size) | 1)
        pad = k // 2
        h, w = tensor_nchw.shape[-2:]
        mode = "reflect" if pad < min(h, w) else "replicate"
        x = tensor_nchw
        for _ in range(2):
            x = F.pad(x, (pad, pad, pad, pad), mode=mode)
            x = F.avg_pool2d(x, kernel_size=k, stride=1)
        return x

    def prepare(self, crops, denoise_mask, strength=0.65):
        if tuple(crops.shape[:3]) != tuple(denoise_mask.shape[:3]):
            raise ValueError("H3LipSyncPrepareCrop requires crops and denoise_mask with matching frame and spatial dimensions")
        s = max(0.0, min(1.0, float(strength)))
        if s <= 0.0:
            return (crops.clone(),)
        try:
            import cv2 as _cv2
        except ImportError:
            _cv2 = None

        frames = int(crops.shape[0])
        out = crops.clone()
        for index in range(frames):
            mask_t = denoise_mask[index]
            active_t = (mask_t >= 0.5)
            if not active_t.any():
                continue
            ys = torch.nonzero(active_t, as_tuple=False)[:, 0]
            mask_h = max(8, int(ys.max() - ys.min() + 1))
            erode_r = max(2, int(round(0.04 * mask_h)))
            k_erode = erode_r * 2 + 1
            k_feather = max(5, (max(3, int(round(0.08 * mask_h))) * 2 + 1))
            k_blur = max(15, int(round(0.35 * mask_h)) | 1)
            if _cv2 is not None:
                src_np = crops[index, ..., :3].detach().cpu().clamp(0.0, 1.0).numpy().astype("float32")
                bin_np = active_t.detach().cpu().numpy().astype("float32")
                kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k_erode, k_erode))
                eroded = _cv2.erode(bin_np, kernel)
                if eroded.max() <= 0.0:
                    eroded = bin_np
                feathered = _cv2.GaussianBlur(eroded, (k_feather, k_feather), 0)
                weight_np = (feathered * bin_np).clip(0.0, 1.0) * s
                smooth_np = _cv2.GaussianBlur(src_np, (k_blur, k_blur), 0, borderType=_cv2.BORDER_REFLECT_101)
                blended_np = src_np * (1.0 - weight_np[..., None]) + smooth_np * weight_np[..., None]
                out[index, ..., :3] = torch.from_numpy(blended_np).to(device=crops.device, dtype=crops.dtype)
            else:
                bin_t = active_t.float()[None, None]
                eroded_t = 1.0 - F.max_pool2d(1.0 - bin_t, kernel_size=k_erode, stride=1, padding=erode_r)
                if not (eroded_t > 0.0).any():
                    eroded_t = bin_t
                feathered_t = self._torch_blur_2d(eroded_t, k_feather).clamp(0.0, 1.0)
                weight_t = (feathered_t * bin_t).squeeze(0).squeeze(0) * s
                src_t = crops[index, ..., :3].float().permute(2, 0, 1).unsqueeze(0)
                smooth_t = self._torch_blur_2d(src_t, k_blur).squeeze(0).permute(1, 2, 0)
                blended_t = crops[index, ..., :3].float() * (1.0 - weight_t[..., None]) + smooth_t * weight_t[..., None]
                out[index, ..., :3] = blended_t.to(dtype=crops.dtype)
        return (out,)


class H3LipSyncSelectRefFrames:
    """Pick the sharpest resting-mouth and open-mouth source crops for H3 identity conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crops": ("IMAGE",),
                "transform": ("H3FACEXFORM",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("ref_neutral", "ref_open")
    FUNCTION = "select"
    CATEGORY = "H3/Dubbing"

    def select(self, crops, transform):
        import math

        frames, height, width = int(crops.shape[0]), int(crops.shape[1]), int(crops.shape[2])
        if frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("H3LipSyncSelectRefFrames requires a non-empty crop batch")

        source_frames = int(transform.get("source_frames") or frames)
        limit = max(1, min(frames, source_frames))
        absent = transform.get("absent") or []
        detected = transform.get("detected") or []
        face_rects = transform.get("face_rect") or []

        candidates = [
            i for i in range(limit)
            if (i >= len(absent) or not absent[i]) and (i >= len(detected) or bool(detected[i]))
        ]
        if not candidates:
            candidates = list(range(limit))

        face_mesh = None
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=2, refine_landmarks=False,
                    min_detection_confidence=0.35,
                )
        except Exception:
            face_mesh = None

        lap_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
            device=crops.device, dtype=torch.float32,
        )
        scored = []
        try:
            for idx in candidates:
                if idx < len(face_rects):
                    fx, fy, fw, fh = (float(v) for v in face_rects[idx])
                else:
                    fx, fy, fw, fh = width * 0.25, height * 0.25, width * 0.5, height * 0.5
                x0 = max(0, min(width - 2, int(round(fx))))
                y0 = max(0, min(height - 2, int(round(fy))))
                x1 = max(x0 + 2, min(width, int(round(fx + fw))))
                y1 = max(y0 + 2, min(height, int(round(fy + fh))))
                roi = crops[idx, y0:y1, x0:x1, :3].float()
                luma = 0.299 * roi[..., 0] + 0.587 * roi[..., 1] + 0.114 * roi[..., 2]
                if luma.shape[0] >= 3 and luma.shape[1] >= 3:
                    lap = F.conv2d(luma.unsqueeze(0).unsqueeze(0), lap_kernel)
                    sharpness = float(lap.var(unbiased=False).item())
                else:
                    sharpness = 0.0

                openness = None
                if face_mesh is not None:
                    try:
                        frame_np = (crops[idx, ..., :3].detach().cpu().clamp(0, 1).numpy() * 255.0).astype("uint8")
                        result = face_mesh.process(frame_np)
                        faces = getattr(result, "multi_face_landmarks", None)
                        if faces:
                            tx, ty = ((fx + 0.5 * fw) / width, (fy + 0.5 * fh) / height)
                            best = min(faces, key=lambda f: (f.landmark[1].x - tx) ** 2 + (f.landmark[1].y - ty) ** 2)
                            top, bot = best.landmark[13], best.landmark[14]
                            left, right = best.landmark[78], best.landmark[308]
                            mouth_w = math.hypot(right.x - left.x, right.y - left.y)
                            if mouth_w > 1e-4:
                                openness = math.hypot(bot.x - top.x, bot.y - top.y) / mouth_w
                    except Exception:
                        openness = None
                if openness is None:
                    rh, rw = luma.shape
                    my0 = max(0, min(rh - 2, int(round(0.60 * rh))))
                    my1 = max(my0 + 2, min(rh, int(round(0.85 * rh))))
                    mx0 = max(0, min(rw - 2, int(round(0.25 * rw))))
                    mx1 = max(mx0 + 2, min(rw, int(round(0.75 * rw))))
                    strip = luma[my0:my1, mx0:mx1]
                    v_grad = float((strip[1:, :] - strip[:-1, :]).abs().mean().item())
                    spread = float(strip.std(unbiased=False).item())
                    openness = v_grad + spread
                scored.append((idx, sharpness, float(openness)))
        finally:
            if face_mesh is not None:
                try:
                    face_mesh.close()
                except Exception:
                    pass

        scored.sort(key=lambda item: item[1], reverse=True)
        sharp_pool = scored[: max(1, (len(scored) + 1) // 2)]
        idx_neutral = min(sharp_pool, key=lambda item: item[2])[0]
        idx_open = max(sharp_pool, key=lambda item: item[2])[0]
        if idx_open == idx_neutral and len(sharp_pool) > 1:
            idx_open = max(
                (item for item in sharp_pool if item[0] != idx_neutral),
                key=lambda item: (abs(item[0] - idx_neutral), item[1]),
            )[0]
        return (
            crops[idx_neutral : idx_neutral + 1, ..., :3].clone(),
            crops[idx_open : idx_open + 1, ..., :3].clone(),
        )


class H3LipSyncBlend:
    """Skin-ring LAB color match, adaptive high-pass detail match, and Laplacian pyramid blend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_crops": ("IMAGE",),
                "generated_crops": ("IMAGE",),
                "composite_mask": ("MASK",),
            },
            "optional": {
                "color_match": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sharpen": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "pyramid_levels": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "match_only": ("BOOLEAN", {"default": False}),
                "temporal_smooth": ("INT", {"default": 1, "min": 1, "max": 15, "step": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_crops",)
    FUNCTION = "blend"
    CATEGORY = "H3/Dubbing"

    @staticmethod
    def _torch_gaussian_blur(img_nchw, sigma):
        radius = max(1, int(round(2.5 * float(sigma))))
        k = 2 * radius + 1
        coords = torch.arange(-radius, radius + 1, device=img_nchw.device, dtype=torch.float32)
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        ch = img_nchw.shape[1]
        kx = kernel_1d.view(1, 1, 1, k).expand(ch, 1, 1, k)
        ky = kernel_1d.view(1, 1, k, 1).expand(ch, 1, k, 1)
        h, w = img_nchw.shape[-2:]
        mode = "reflect" if radius < min(h, w) else "replicate"
        x = F.pad(img_nchw, (radius, radius, 0, 0), mode=mode)
        x = F.conv2d(x, kx, groups=ch)
        x = F.pad(x, (0, 0, radius, radius), mode=mode)
        return F.conv2d(x, ky, groups=ch)

    @staticmethod
    def _skin_ring_np(mask_np, cv2_mod):
        outer = mask_np > 0.05
        if not outer.any():
            return outer
        ys, xs = outer.nonzero()
        span = max(8, int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)))
        r_inner = max(2, int(round(0.06 * span)))
        core = (mask_np > 0.5).astype("uint8")
        k_in = cv2_mod.getStructuringElement(cv2_mod.MORPH_ELLIPSE, (2 * r_inner + 1, 2 * r_inner + 1))
        eroded_core = cv2_mod.erode(core, k_in) > 0
        ring = outer & (~eroded_core)
        if int(ring.sum()) < 32:
            r_out = max(4, int(round(0.08 * span)))
            k_out = cv2_mod.getStructuringElement(cv2_mod.MORPH_ELLIPSE, (2 * r_out + 1, 2 * r_out + 1))
            dilated = cv2_mod.dilate(outer.astype("uint8"), k_out) > 0
            ring = dilated & (~eroded_core)
        return ring if int(ring.sum()) >= 32 else outer

    @staticmethod
    def _skin_ring_torch(mask_t):
        outer = mask_t > 0.05
        if not outer.any():
            return outer
        nz = torch.nonzero(outer, as_tuple=False)
        span = max(8, int(max(nz[:, 0].max() - nz[:, 0].min() + 1, nz[:, 1].max() - nz[:, 1].min() + 1)))
        r_inner = max(2, int(round(0.06 * span)))
        core_t = (mask_t > 0.5).float()[None, None]
        eroded_core = (1.0 - F.max_pool2d(1.0 - core_t, kernel_size=2 * r_inner + 1, stride=1, padding=r_inner)) > 0.5
        eroded_core = eroded_core.squeeze(0).squeeze(0)
        ring = outer & (~eroded_core)
        if int(ring.sum()) < 32:
            r_out = max(4, int(round(0.08 * span)))
            dilated = F.max_pool2d(outer.float()[None, None], kernel_size=2 * r_out + 1, stride=1, padding=r_out) > 0.5
            ring = dilated.squeeze(0).squeeze(0) & (~eroded_core)
        return ring if int(ring.sum()) >= 32 else outer

    @staticmethod
    def _laplacian_blend_np(src, gen, mask, levels, cv2_mod):
        h, w = mask.shape
        if levels <= 1 or min(h, w) < 8:
            return gen * mask[..., None] + src * (1.0 - mask[..., None])
        g_src = [src]
        g_gen = [gen]
        g_mask = [mask]
        for _ in range(levels - 1):
            if min(g_mask[-1].shape[0], g_mask[-1].shape[1]) < 8:
                break
            g_src.append(cv2_mod.pyrDown(g_src[-1]))
            g_gen.append(cv2_mod.pyrDown(g_gen[-1]))
            g_mask.append(cv2_mod.pyrDown(g_mask[-1]))
        n = len(g_mask)
        if n <= 1:
            return gen * mask[..., None] + src * (1.0 - mask[..., None])
        l_out = []
        for idx in range(n - 1):
            th, tw = g_mask[idx].shape[:2]
            up_src = cv2_mod.pyrUp(g_src[idx + 1], dstsize=(tw, th))
            up_gen = cv2_mod.pyrUp(g_gen[idx + 1], dstsize=(tw, th))
            lap_src = g_src[idx] - up_src
            lap_gen = g_gen[idx] - up_gen
            m = g_mask[idx][..., None]
            l_out.append(lap_gen * m + lap_src * (1.0 - m))
        m_top = g_mask[-1][..., None]
        acc = g_gen[-1] * m_top + g_src[-1] * (1.0 - m_top)
        for idx in range(n - 2, -1, -1):
            th, tw = g_mask[idx].shape[:2]
            acc = cv2_mod.pyrUp(acc, dstsize=(tw, th)) + l_out[idx]
        return acc.clip(0.0, 1.0)

    def _laplacian_blend_torch(self, src_t, gen_t, mask_t, levels):
        h, w = mask_t.shape
        if levels <= 1 or min(h, w) < 8:
            return (gen_t * mask_t[..., None] + src_t * (1.0 - mask_t[..., None])).clamp(0.0, 1.0)
        g_src = [src_t.permute(2, 0, 1).unsqueeze(0)]
        g_gen = [gen_t.permute(2, 0, 1).unsqueeze(0)]
        g_mask = [mask_t.unsqueeze(0).unsqueeze(0)]
        for _ in range(levels - 1):
            cur_h, cur_w = g_mask[-1].shape[-2:]
            if min(cur_h, cur_w) < 8:
                break
            next_size = (max(2, (cur_h + 1) // 2), max(2, (cur_w + 1) // 2))
            g_src.append(F.interpolate(self._torch_gaussian_blur(g_src[-1], 1.0), size=next_size, mode="bilinear", align_corners=False))
            g_gen.append(F.interpolate(self._torch_gaussian_blur(g_gen[-1], 1.0), size=next_size, mode="bilinear", align_corners=False))
            g_mask.append(F.interpolate(self._torch_gaussian_blur(g_mask[-1], 1.0), size=next_size, mode="bilinear", align_corners=False))
        n = len(g_mask)
        if n <= 1:
            return (gen_t * mask_t[..., None] + src_t * (1.0 - mask_t[..., None])).clamp(0.0, 1.0)
        l_out = []
        for idx in range(n - 1):
            th, tw = g_mask[idx].shape[-2:]
            up_src = F.interpolate(g_src[idx + 1], size=(th, tw), mode="bilinear", align_corners=False)
            up_gen = F.interpolate(g_gen[idx + 1], size=(th, tw), mode="bilinear", align_corners=False)
            m = g_mask[idx]
            l_out.append((g_gen[idx] - up_gen) * m + (g_src[idx] - up_src) * (1.0 - m))
        acc = g_gen[-1] * g_mask[-1] + g_src[-1] * (1.0 - g_mask[-1])
        for idx in range(n - 2, -1, -1):
            th, tw = g_mask[idx].shape[-2:]
            acc = F.interpolate(acc, size=(th, tw), mode="bilinear", align_corners=False) + l_out[idx]
        return acc.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

    @staticmethod
    def _clamped_std_scale(s_std, g_std):
        return 1.0 if g_std < 1e-4 else max(0.75, min(1.33, s_std / g_std))

    @staticmethod
    def _adaptive_sharpen_gain(sh, src_energy, gen_energy):
        ratio = src_energy / max(gen_energy, 1e-4)
        return sh * max(0.25, min(1.5, ratio - 0.85))

    def _blend_frame_np(self, src_np, gen_np, mask_np, cm, sh, levels, cv2_mod, np_mod):
        ring = self._skin_ring_np(mask_np, cv2_mod)
        gen_cm = gen_np
        if cm > 0.0 and ring.any():
            src_lab = cv2_mod.cvtColor(src_np, cv2_mod.COLOR_RGB2LAB)
            gen_lab = cv2_mod.cvtColor(gen_np, cv2_mod.COLOR_RGB2LAB)
            matched_lab = gen_lab.copy()
            for ch in range(3):
                s_vals = src_lab[..., ch][ring]
                g_vals = gen_lab[..., ch][ring]
                scale = self._clamped_std_scale(float(s_vals.std()), float(g_vals.std()))
                matched_lab[..., ch] = (gen_lab[..., ch] - float(g_vals.mean())) * scale + float(s_vals.mean())
            matched_lab[..., 0] = matched_lab[..., 0].clip(0.0, 100.0)
            matched_lab[..., 1:] = matched_lab[..., 1:].clip(-127.0, 127.0)
            matched_rgb = cv2_mod.cvtColor(matched_lab, cv2_mod.COLOR_LAB2RGB).clip(0.0, 1.0)
            gen_cm = gen_np * (1.0 - cm) + matched_rgb * cm

        gen_sharp = gen_cm
        if sh > 0.0 and ring.any():
            src_hp = src_np - cv2_mod.GaussianBlur(src_np, (0, 0), 1.5, borderType=cv2_mod.BORDER_REFLECT_101)
            gen_hp = gen_cm - cv2_mod.GaussianBlur(gen_cm, (0, 0), 1.5, borderType=cv2_mod.BORDER_REFLECT_101)
            gain = self._adaptive_sharpen_gain(sh, float(src_hp[ring].std()), float(gen_hp[ring].std()))
            detail = gen_cm - cv2_mod.GaussianBlur(gen_cm, (0, 0), 1.2, borderType=cv2_mod.BORDER_REFLECT_101)
            gen_sharp = (gen_cm + gain * detail).clip(0.0, 1.0)

        blended = self._laplacian_blend_np(src_np, gen_sharp, mask_np, levels, cv2_mod)
        return np_mod.where(mask_np[..., None] > 1e-4, blended, src_np)

    def _blend_frame_torch(self, src_t, gen_t, mask_t, cm, sh, levels):
        ring = self._skin_ring_torch(mask_t)
        gen_cm = gen_t
        if cm > 0.0 and ring.any():
            matched = gen_t.clone()
            for ch in range(3):
                s_vals = src_t[..., ch][ring]
                g_vals = gen_t[..., ch][ring]
                scale = self._clamped_std_scale(
                    float(s_vals.std(unbiased=False)), float(g_vals.std(unbiased=False)),
                )
                matched[..., ch] = ((gen_t[..., ch] - g_vals.mean()) * scale + s_vals.mean()).clamp(0.0, 1.0)
            gen_cm = gen_t * (1.0 - cm) + matched * cm

        gen_sharp = gen_cm
        if sh > 0.0 and ring.any():
            src_b = self._torch_gaussian_blur(src_t.permute(2, 0, 1).unsqueeze(0), 1.5).squeeze(0).permute(1, 2, 0)
            gen_b = self._torch_gaussian_blur(gen_cm.permute(2, 0, 1).unsqueeze(0), 1.5).squeeze(0).permute(1, 2, 0)
            gain = self._adaptive_sharpen_gain(
                sh,
                float((src_t - src_b)[ring].std(unbiased=False)),
                float((gen_cm - gen_b)[ring].std(unbiased=False)),
            )
            d_blur = self._torch_gaussian_blur(gen_cm.permute(2, 0, 1).unsqueeze(0), 1.2).squeeze(0).permute(1, 2, 0)
            gen_sharp = (gen_cm + gain * (gen_cm - d_blur)).clamp(0.0, 1.0)

        blended = self._laplacian_blend_torch(src_t, gen_sharp, mask_t, levels)
        return torch.where(mask_t[..., None] > 1e-4, blended, src_t)

    @classmethod
    def _exterior_skin_ring_np(cls, mask_np, cv2_mod):
        core = mask_np > 0.5
        if not core.any():
            core = mask_np > 0.05
        if not core.any():
            return core
        ys, xs = core.nonzero()
        span = max(8, int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)))
        r_out = max(4, int(round(0.12 * span)))
        k_out = cv2_mod.getStructuringElement(cv2_mod.MORPH_ELLIPSE, (2 * r_out + 1, 2 * r_out + 1))
        dilated = cv2_mod.dilate(core.astype("uint8"), k_out) > 0
        ring = dilated & (~core)
        if int(ring.sum()) >= 32:
            return ring
        return cls._skin_ring_np(mask_np, cv2_mod)

    @classmethod
    def _exterior_skin_ring_torch(cls, mask_t):
        core = mask_t > 0.5
        if not core.any():
            core = mask_t > 0.05
        if not core.any():
            return core
        nz = torch.nonzero(core, as_tuple=False)
        span = max(8, int(max(nz[:, 0].max() - nz[:, 0].min() + 1, nz[:, 1].max() - nz[:, 1].min() + 1)))
        r_out = max(4, int(round(0.12 * span)))
        dilated = F.max_pool2d(core.float()[None, None], kernel_size=2 * r_out + 1, stride=1, padding=r_out) > 0.5
        ring = dilated.squeeze(0).squeeze(0) & (~core)
        if int(ring.sum()) >= 32:
            return ring
        return cls._skin_ring_torch(mask_t)

    @staticmethod
    def _smooth_stat_vectors(stats, window):
        import math

        k = max(1, int(window))
        if k <= 1:
            return stats
        radius = k // 2
        sigma = max(1.0, radius * 0.6)
        valid = sorted(stats)
        if len(valid) <= 1:
            return stats
        dim = len(stats[valid[0]])
        out = {}
        for idx in valid:
            acc = [0.0] * dim
            w_sum = 0.0
            for j in range(idx - radius, idx + radius + 1):
                item = stats.get(j)
                if item is None:
                    continue
                w = math.exp(-0.5 * ((j - idx) / sigma) ** 2)
                for d in range(dim):
                    acc[d] += w * item[d]
                w_sum += w
            out[idx] = [v / w_sum for v in acc] if w_sum > 0.0 else stats[idx]
        return out

    def _apply_match_np(self, src_np, gen_np, gen_lab, stat_vec, cm, sh, cv2_mod):
        s_mean = stat_vec[0:3]
        s_std = stat_vec[3:6]
        g_mean = stat_vec[6:9]
        g_std = stat_vec[9:12]
        src_energy, gen_energy = stat_vec[12], stat_vec[13]

        gen_cm = gen_np
        if cm > 0.0:
            matched_lab = gen_lab.copy()
            for ch in range(3):
                scale = self._clamped_std_scale(s_std[ch], g_std[ch])
                matched_lab[..., ch] = (gen_lab[..., ch] - g_mean[ch]) * scale + s_mean[ch]
            matched_lab[..., 0] = matched_lab[..., 0].clip(0.0, 100.0)
            matched_lab[..., 1:] = matched_lab[..., 1:].clip(-127.0, 127.0)
            matched_rgb = cv2_mod.cvtColor(matched_lab, cv2_mod.COLOR_LAB2RGB).clip(0.0, 1.0)
            gen_cm = gen_np * (1.0 - cm) + matched_rgb * cm

        if sh <= 0.0:
            return gen_cm
        gain = self._adaptive_sharpen_gain(sh, src_energy, gen_energy)
        detail = gen_cm - cv2_mod.GaussianBlur(gen_cm, (0, 0), 1.2, borderType=cv2_mod.BORDER_REFLECT_101)
        return (gen_cm + gain * detail).clip(0.0, 1.0)

    def _apply_match_torch(self, gen_t, stat_vec, cm, sh):
        s_mean = stat_vec[0:3]
        s_std = stat_vec[3:6]
        g_mean = stat_vec[6:9]
        g_std = stat_vec[9:12]
        src_energy, gen_energy = stat_vec[12], stat_vec[13]

        gen_cm = gen_t
        if cm > 0.0:
            matched = gen_t.clone()
            for ch in range(3):
                scale = self._clamped_std_scale(s_std[ch], g_std[ch])
                matched[..., ch] = ((gen_t[..., ch] - g_mean[ch]) * scale + s_mean[ch]).clamp(0.0, 1.0)
            gen_cm = gen_t * (1.0 - cm) + matched * cm

        if sh <= 0.0:
            return gen_cm
        gain = self._adaptive_sharpen_gain(sh, src_energy, gen_energy)
        d_blur = self._torch_gaussian_blur(gen_cm.permute(2, 0, 1).unsqueeze(0), 1.2).squeeze(0).permute(1, 2, 0)
        return (gen_cm + gain * (gen_cm - d_blur)).clamp(0.0, 1.0)

    def blend(self, source_crops, generated_crops, composite_mask,
              color_match=0.85, sharpen=0.35, pyramid_levels=3,
              match_only=False, temporal_smooth=1):
        if (tuple(source_crops.shape[:3]) != tuple(generated_crops.shape[:3])
                or tuple(source_crops.shape[:3]) != tuple(composite_mask.shape[:3])):
            raise ValueError("H3LipSyncBlend requires source_crops, generated_crops, and composite_mask with matching dimensions")
        cm = max(0.0, min(1.0, float(color_match)))
        sh = max(0.0, min(1.0, float(sharpen)))
        levels = max(1, min(5, int(pyramid_levels)))
        try:
            import cv2 as _cv2
            import numpy as _np
        except ImportError:
            _cv2 = None
            _np = None

        frames = int(source_crops.shape[0])
        if match_only:
            out = generated_crops.clone()
            if cm <= 0.0 and sh <= 0.0:
                return (out,)
            raw_stats = {}
            cached_frames = {}
            for index in range(frames):
                mask_t = composite_mask[index].float().clamp(0.0, 1.0)
                if float(mask_t.max()) <= 1e-4:
                    continue
                if _cv2 is not None and _np is not None:
                    src_np = source_crops[index, ..., :3].detach().cpu().clamp(0.0, 1.0).numpy().astype("float32")
                    gen_np = generated_crops[index, ..., :3].detach().cpu().clamp(0.0, 1.0).numpy().astype("float32")
                    mask_np = mask_t.detach().cpu().numpy().astype("float32")
                    ring = self._exterior_skin_ring_np(mask_np, _cv2)
                    if not ring.any():
                        continue
                    src_lab = _cv2.cvtColor(src_np, _cv2.COLOR_RGB2LAB)
                    gen_lab = _cv2.cvtColor(gen_np, _cv2.COLOR_RGB2LAB)
                    s_mean = [float(src_lab[..., ch][ring].mean()) for ch in range(3)]
                    s_std = [float(src_lab[..., ch][ring].std()) for ch in range(3)]
                    g_mean = [float(gen_lab[..., ch][ring].mean()) for ch in range(3)]
                    g_std = [float(gen_lab[..., ch][ring].std()) for ch in range(3)]
                    if sh > 0.0:
                        gen_cm_raw = gen_np
                        if cm > 0.0:
                            m_lab = gen_lab.copy()
                            for ch in range(3):
                                sc = self._clamped_std_scale(s_std[ch], g_std[ch])
                                m_lab[..., ch] = (gen_lab[..., ch] - g_mean[ch]) * sc + s_mean[ch]
                            m_lab[..., 0] = m_lab[..., 0].clip(0.0, 100.0)
                            m_lab[..., 1:] = m_lab[..., 1:].clip(-127.0, 127.0)
                            gen_cm_raw = gen_np * (1.0 - cm) + _cv2.cvtColor(m_lab, _cv2.COLOR_LAB2RGB).clip(0.0, 1.0) * cm
                        src_hp = src_np - _cv2.GaussianBlur(src_np, (0, 0), 1.5, borderType=_cv2.BORDER_REFLECT_101)
                        gen_hp = gen_cm_raw - _cv2.GaussianBlur(gen_cm_raw, (0, 0), 1.5, borderType=_cv2.BORDER_REFLECT_101)
                        src_e, gen_e = float(src_hp[ring].std()), float(gen_hp[ring].std())
                    else:
                        src_e, gen_e = 0.0, 0.0
                    raw_stats[index] = [*s_mean, *s_std, *g_mean, *g_std, src_e, gen_e]
                    cached_frames[index] = (src_np, gen_np, gen_lab)
                else:
                    src_t = source_crops[index, ..., :3].float().clamp(0.0, 1.0)
                    gen_t = generated_crops[index, ..., :3].float().clamp(0.0, 1.0)
                    ring = self._exterior_skin_ring_torch(mask_t)
                    if not ring.any():
                        continue
                    s_mean = [float(src_t[..., ch][ring].mean()) for ch in range(3)]
                    s_std = [float(src_t[..., ch][ring].std(unbiased=False)) for ch in range(3)]
                    g_mean = [float(gen_t[..., ch][ring].mean()) for ch in range(3)]
                    g_std = [float(gen_t[..., ch][ring].std(unbiased=False)) for ch in range(3)]
                    if sh > 0.0:
                        gen_cm_raw = gen_t
                        if cm > 0.0:
                            matched = gen_t.clone()
                            for ch in range(3):
                                sc = self._clamped_std_scale(s_std[ch], g_std[ch])
                                matched[..., ch] = ((gen_t[..., ch] - g_mean[ch]) * sc + s_mean[ch]).clamp(0.0, 1.0)
                            gen_cm_raw = gen_t * (1.0 - cm) + matched * cm
                        src_b = self._torch_gaussian_blur(src_t.permute(2, 0, 1).unsqueeze(0), 1.5).squeeze(0).permute(1, 2, 0)
                        gen_b = self._torch_gaussian_blur(gen_cm_raw.permute(2, 0, 1).unsqueeze(0), 1.5).squeeze(0).permute(1, 2, 0)
                        src_e = float((src_t - src_b)[ring].std(unbiased=False))
                        gen_e = float((gen_cm_raw - gen_b)[ring].std(unbiased=False))
                    else:
                        src_e, gen_e = 0.0, 0.0
                    raw_stats[index] = [*s_mean, *s_std, *g_mean, *g_std, src_e, gen_e]
                    cached_frames[index] = gen_t
            smooth_stats = self._smooth_stat_vectors(raw_stats, temporal_smooth)
            for index, stat_vec in smooth_stats.items():
                if _cv2 is not None and _np is not None:
                    src_np, gen_np, gen_lab = cached_frames[index]
                    matched = self._apply_match_np(src_np, gen_np, gen_lab, stat_vec, cm, sh, _cv2)
                    out[index, ..., :3] = torch.from_numpy(matched).to(device=generated_crops.device, dtype=generated_crops.dtype)
                else:
                    gen_t = cached_frames[index]
                    matched_t = self._apply_match_torch(gen_t, stat_vec, cm, sh)
                    out[index, ..., :3] = matched_t.to(dtype=generated_crops.dtype)
            return (out,)

        out = source_crops.clone()
        for index in range(frames):
            mask_t = composite_mask[index].float().clamp(0.0, 1.0)
            if float(mask_t.max()) <= 1e-4:
                continue
            if _cv2 is not None and _np is not None:
                src_np = source_crops[index, ..., :3].detach().cpu().clamp(0.0, 1.0).numpy().astype("float32")
                gen_np = generated_crops[index, ..., :3].detach().cpu().clamp(0.0, 1.0).numpy().astype("float32")
                mask_np = mask_t.detach().cpu().numpy().astype("float32")
                blended = self._blend_frame_np(src_np, gen_np, mask_np, cm, sh, levels, _cv2, _np)
                out[index, ..., :3] = torch.from_numpy(blended).to(device=source_crops.device, dtype=source_crops.dtype)
            else:
                src_t = source_crops[index, ..., :3].float().clamp(0.0, 1.0)
                gen_t = generated_crops[index, ..., :3].float().clamp(0.0, 1.0)
                blended = self._blend_frame_torch(src_t, gen_t, mask_t, cm, sh, levels)
                out[index, ..., :3] = blended.to(dtype=source_crops.dtype)
        return (out,)


class H3SaveDetectedMasks:
    """Save SAM detections independently, including a successful empty result."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",), "filename_prefix": ("STRING",)}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3/Effects"

    def save(self, masks, filename_prefix):
        if masks.shape[0] == 0:
            return {"ui": {"images": []}}

        from nodes import SaveImage

        images = masks.unsqueeze(-1).expand(-1, -1, -1, 3)
        return {"ui": SaveImage().save_images(images, filename_prefix)["ui"]}


_DEPTH_PRO2_LOCK = threading.Lock()
_DEPTH_PRO2_MODEL_CACHE: dict[str, object] = {}


def _encode_depthpro2_artifacts(depth_np, rgb_np, f_px, preset):
    """Pack metric depth into lossless 16-bit RG8+sky PNG and auto-derived scene anchors."""
    import numpy as np
    from PIL import Image

    depth_arr = np.asarray(depth_np, dtype=np.float32)
    height, width = int(depth_arr.shape[0]), int(depth_arr.shape[1])
    rgb_arr = np.asarray(rgb_np, dtype=np.uint8)

    sky_mask = (~np.isfinite(depth_arr)) | (depth_arr <= 0.0) | (depth_arr > 150.0)
    sky_depth = np.where((~np.isfinite(depth_arr)) | (depth_arr <= 0.0), 150.0, depth_arr)
    sky_float = np.clip((sky_depth - 60.0) / 90.0, 0.0, 1.0).astype(np.float32)
    sky_u8 = np.rint(sky_float * 255.0).astype(np.uint8)
    inv = np.where(sky_mask, 0.0, 1.0 / np.clip(depth_arr, 0.05, 150.0)).astype(np.float32)

    valid_mask = ~sky_mask
    valid_count = int(valid_mask.sum())
    if valid_count >= 16:
        valid_inv = inv[valid_mask]
        valid_z = np.clip(depth_arr[valid_mask], 0.1, 150.0)
        inv_min = 0.0
        inv_max = float(max(float(np.percentile(valid_inv, 99.5)), 1e-3))
        z_p10, z_p25, z_p50, z_p75, z_p90 = [
            float(v) for v in np.percentile(valid_z, (10.0, 25.0, 50.0, 75.0, 90.0))
        ]
    else:
        inv_min, inv_max = 0.0, 1.0
        z_p10, z_p25, z_p50, z_p75, z_p90 = 2.0, 4.0, 8.0, 16.0, 32.0
        valid_z = np.array([z_p50], dtype=np.float32)

    u = np.clip((inv - inv_min) / (inv_max - inv_min), 0.0, 1.0).astype(np.float32)
    q = np.rint(u * 65535.0).astype(np.uint16)
    r_ch = (q >> 8).astype(np.uint8)
    g_ch = (q & 0xFF).astype(np.uint8)
    png_image = Image.fromarray(np.stack((r_ch, g_ch, sky_u8), axis=-1), mode="RGB")

    # Subsample to a bounded analysis grid for deterministic, sub-millisecond anchor extraction.
    scale = min(1.0, 160.0 / max(width, height))
    aw = max(16, int(round(width * scale)))
    ah = max(16, int(round(height * scale)))
    ys = np.linspace(0, height - 1, ah).astype(np.intp)
    xs = np.linspace(0, width - 1, aw).astype(np.intp)
    u_s = u[np.ix_(ys, xs)]
    z_s = np.clip(depth_arr[np.ix_(ys, xs)], 0.1, 150.0)
    valid_s = valid_mask[np.ix_(ys, xs)]
    sky_s = sky_float[np.ix_(ys, xs)]
    rgb_s = np.asarray(rgb_arr[np.ix_(ys, xs)], dtype=np.float32) / 255.0
    gx = (np.arange(aw, dtype=np.float32) + 0.5) / aw
    gy = (np.arange(ah, dtype=np.float32) + 0.5) / ah
    uu, vv = np.meshgrid(gx, gy)

    su, sv, subject_z = 0.5, 0.46, z_p50
    if int(valid_s.sum()) >= 16:
        salience = np.exp(-(((uu - 0.5) ** 2) / 0.09 + ((vv - 0.46) ** 2) / 0.11)) * (0.5 + 0.5 * u_s)
        salience = np.where(valid_s, salience, 0.0)
        lo_u, hi_u = [float(v) for v in np.percentile(u_s[valid_s], (20.0, 92.0))]
        if hi_u - lo_u > 1e-4:
            hist, edges = np.histogram(
                u_s[valid_s], bins=32, range=(lo_u, hi_u), weights=salience[valid_s]
            )
            peak = int(np.argmax(hist))
            target_u = 0.5 * float(edges[peak] + edges[peak + 1])
            in_bin = valid_s & (u_s >= edges[peak]) & (u_s <= edges[peak + 1] + 1e-6)
            w_bin = np.where(in_bin, salience, 0.0)
            w_sum = float(w_bin.sum())
            if w_sum > 1e-6:
                su = float((uu * w_bin).sum() / w_sum)
                sv = float((vv * w_bin).sum() / w_sum)
            subject_inv = inv_min + target_u * (inv_max - inv_min)
        else:
            subject_inv = float(np.median(inv[valid_mask]))
        subject_z = float(min(150.0, max(0.1, 1.0 / max(subject_inv, 1e-4))))

    fg_frac = float((valid_z < subject_z * 0.68).mean()) if valid_count >= 16 else 0.0
    if z_p10 < subject_z * 0.62 and fg_frac >= 0.04:
        rack_start_z = float(z_p10)
    else:
        rack_start_z = float(min(150.0, max(z_p75, subject_z * 1.9)))

    luma = 0.299 * rgb_s[..., 0] + 0.587 * rgb_s[..., 1] + 0.114 * rgb_s[..., 2]
    warm = np.where(
        valid_s,
        np.clip(rgb_s[..., 0] - rgb_s[..., 2], 0.0, 1.0)
        * np.clip(rgb_s[..., 0] - 0.45 * rgb_s[..., 1], 0.0, 1.0)
        * (luma ** 2),
        0.0,
    )
    wu, wv, wz, warm_strength = 0.5, 0.62, float(min(150.0, subject_z * 0.9)), 0.45
    if int(valid_s.sum()) >= 16:
        warm_valid = warm[valid_s]
        thresh = float(np.percentile(warm_valid, 98.5))
        top_mask = valid_s & (warm >= max(thresh, 1e-6))
        if top_mask.any():
            mean_top_warm = float(warm[top_mask].mean())
            if mean_top_warm > 0.02:
                w_warm = np.where(top_mask, warm, 0.0)
                w_sum = float(w_warm.sum())
                if w_sum > 1e-6:
                    wu = float((uu * w_warm).sum() / w_sum)
                    wv = float((vv * w_warm).sum() / w_sum)
                    wz = float(np.median(z_s[top_mask]))
                    warm_strength = float(min(1.0, mean_top_warm * 8.0))

    upper = vv <= 0.55
    sky_w = np.where(upper, (0.7 * sky_s + 0.3 * (1.0 - u_s)) * (luma ** 2), 0.0)
    sky_w_sum = float(sky_w.sum())
    if sky_w_sum > max(0.05, 0.002 * sky_w.size):
        sky_u = float((uu * sky_w).sum() / sky_w_sum)
        sky_v = float((vv * sky_w).sum() / sky_w_sum)
    else:
        sky_u, sky_v = 0.26, 0.12

    log_span = math.log(max(z_p90, 0.2) / max(z_p10, 0.1))
    auto_strength = round(min(1.0, max(0.0, (log_span - 0.15) / 0.85)), 3)
    focal_px = float(f_px) if (f_px is not None and math.isfinite(float(f_px)) and float(f_px) > 1.0) else max(width, height) * 1.1

    meta = {
        "version": 1,
        "preset": str(preset),
        "width": width,
        "height": height,
        "f_px": round(focal_px, 2),
        "inv_min": round(inv_min, 6),
        "inv_max": round(inv_max, 6),
        "is_metric": True,
        "z_p10": round(z_p10, 3),
        "z_p25": round(z_p25, 3),
        "z_p50": round(z_p50, 3),
        "z_p75": round(z_p75, 3),
        "z_p90": round(z_p90, 3),
        "sky_fraction": round(float(sky_mask.mean()), 4),
        "subject_z": round(subject_z, 3),
        "subject_uv": [round(su, 4), round(sv, 4)],
        "rack_start_z": round(rack_start_z, 3),
        "warm_anchor": {"u": round(wu, 4), "v": round(wv, 4), "z": round(wz, 3), "strength": round(warm_strength, 3)},
        "sky_dir": {"u": round(sky_u, 4), "v": round(sky_v, 4)},
        "auto_strength": auto_strength,
    }
    return png_image, meta


class H3EstimateDepthPro2:
    """Estimate metric depth maps and scene anchors for Soundstage stills using Depth Pro 2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["depthpro2-s", "depthpro2-b", "depthpro2-l", "depthpro2-h"],),
                "model_name": ("STRING", {"default": "depthpro2_s.safetensors"}),
                "focal_model_name": ("STRING", {"default": "depthpro2_focal.safetensors"}),
                "filename_prefix": ("STRING", {"default": "h3/soundstage_depth/depth"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "estimate"
    OUTPUT_NODE = True
    CATEGORY = "H3/Effects"

    @staticmethod
    def _resolve_weight(name: str) -> str:
        base_dir = os.path.realpath(os.path.join(folder_paths.models_dir, "depth_pro_2"))
        clean = os.path.basename(str(name or "").strip())
        if not clean:
            raise ValueError("Depth Pro 2 checkpoint filename must not be empty.")
        resolved = os.path.realpath(os.path.join(base_dir, clean))
        if os.path.commonpath((base_dir, resolved)) != base_dir or not os.path.isfile(resolved):
            raise FileNotFoundError(f"Missing Depth Pro 2 checkpoint: {resolved}")
        return resolved

    @staticmethod
    def _load_model(preset: str, model_path: str, focal_path: str, precision: torch.dtype):
        from depth_pro_2.model import PRESETS, DepthPro2
        from safetensors.torch import load_file

        if preset not in PRESETS:
            raise ValueError(f"Unknown Depth Pro 2 preset: {preset}")
        key = f"{preset}|{model_path}|{focal_path}|{precision}"
        with _DEPTH_PRO2_LOCK:
            cached = _DEPTH_PRO2_MODEL_CACHE.get("entry")
            if cached is not None and cached[0] == key:
                return cached[1]
            _DEPTH_PRO2_MODEL_CACHE.clear()
            cfg = PRESETS[preset]
            model = DepthPro2(
                backbone_preset=cfg.backbone_preset,
                reassembled_blocks=cfg.reassembled_blocks,
                cross_frame_blocks=cfg.cross_frame_blocks,
                dims_reassemble=cfg.dims_reassemble,
                dim_decoder=cfg.dim_decoder,
                fov=True,
                precision=precision,
            )
            state_dict = dict(load_file(model_path))
            fov_state = load_file(focal_path)
            state_dict.update({f"fov.{k}": v for k, v in fov_state.items()})
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            _DEPTH_PRO2_MODEL_CACHE["entry"] = (key, model)
            return model

    def estimate(self, image, preset, model_name, focal_model_name, filename_prefix):
        from depth_pro_2.model import to_normalized_tensor

        model_path = self._resolve_weight(model_name)
        focal_path = self._resolve_weight(focal_model_name)
        device = comfy.model_management.get_torch_device()
        precision = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
        model = self._load_model(preset, model_path, focal_path, precision)

        results = []
        full_output_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(
            str(filename_prefix), folder_paths.get_output_directory(), int(image.shape[2]), int(image.shape[1])
        )
        os.makedirs(full_output_folder, exist_ok=True)
        with _DEPTH_PRO2_LOCK:
            try:
                if device.type != "cpu":
                    memory = comfy.model_management.module_size(model) + comfy.model_management.minimum_inference_memory()
                    comfy.model_management.free_memory(memory, device)
                model = model.to(device).eval()
                batch_uint8 = (
                    image[..., :3].detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
                )
                with torch.no_grad():
                    for index in range(batch_uint8.shape[0]):
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        rgb_np = batch_uint8[index]
                        tensor = to_normalized_tensor(rgb_np, device)
                        prediction = model.infer(tensor, f_px=None)
                        depth_np = prediction["depth"].detach().float().cpu().numpy()
                        focal_t = prediction.get("focallength_px")
                        f_px = (
                            float(focal_t.detach().float().cpu().item())
                            if focal_t is not None
                            else float(max(rgb_np.shape[0], rgb_np.shape[1]) * 1.1)
                        )
                        png_img, meta = _encode_depthpro2_artifacts(depth_np, rgb_np, f_px, preset)
                        out_name = f"{filename}_{counter + index:05}_.png"
                        png_img.save(os.path.join(full_output_folder, out_name), format="PNG", compress_level=6)
                        results.append({
                            "filename": out_name,
                            "subfolder": subfolder,
                            "type": "output",
                            "meta": meta,
                        })
            finally:
                try:
                    model.to("cpu")
                finally:
                    comfy.model_management.soft_empty_cache()
        return {"ui": {"depth_maps": results}}


NODE_CLASS_MAPPINGS = {
    "H3ContinuationPrefix": H3ContinuationPrefix,
    "H3SaveAVLatent": H3SaveAVLatent,
    "H3LoadAVLatent": H3LoadAVLatent,
    "H3EncodeAVTail": H3EncodeAVTail,
    "H3LatentUpscaleModelLoader": H3LatentUpscaleModelLoader,
    "H3RegenerateEncodeVideo": H3RegenerateEncodeVideo,
    "H3RegeneratePrepareLatent": H3RegeneratePrepareLatent,
    "H3SourceAudioLatent": H3SourceAudioLatent,
    "H3LipSyncMask": H3LipSyncMask,
    "H3LowerFaceMask": H3LowerFaceMask,
    "H3LipSyncPrepareCrop": H3LipSyncPrepareCrop,
    "H3LipSyncSelectRefFrames": H3LipSyncSelectRefFrames,
    "H3LipSyncBlend": H3LipSyncBlend,
    "H3LTXPadVideoFrames": H3LTXPadVideoFrames,
    "H3Flux3EditPrompt": H3Flux3EditPrompt,
    "H3Flux3LayoutElements": H3Flux3LayoutElements,
    "H3Qwen21PromptEnhance": H3Qwen21PromptEnhance,
    "H3Qwen21Sigmas": H3Qwen21Sigmas,
    "H3LTXHDRConditioning": H3LTXHDRConditioning,
    "H3Qwen21TurboSigmas": H3Qwen21TurboSigmas,
    "H3Qwen21LightningSigmas": H3Qwen21LightningSigmas,
    "H3Qwen21ViggleLora": H3Qwen21ViggleLora,
    "H3SemanticBridge": H3SemanticBridge,
    "H3FirstBlockCache": H3FirstBlockCache,
    "H3LightX2VBypassLoRA": H3LightX2VBypassLoRA,
    "H3SeparateAVLatent": H3SeparateAVLatent,
    "H3CombineAVLatent": H3CombineAVLatent,
    "H3SingleFrameVAELoader": H3SingleFrameVAELoader,
    "H3VideoLatentSlicesToBatch": H3VideoLatentSlicesToBatch,
    "H3ConditioningCache": H3ConditioningCache,
    "H3StageOffloadPolicy": H3StageOffloadPolicy,
    "H3StageModelOffload": H3StageModelOffload,
    "H3RefinementCompilerGuard": H3RefinementCompilerGuard,
    "H3SaveVideoNVENC": H3SaveVideoNVENC,
    "H3SaveDetectedMasks": H3SaveDetectedMasks,
    "H3EstimateDepthPro2": H3EstimateDepthPro2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuationPrefix": "MiniMax H3 Storyboard Continuation",
    "H3SaveAVLatent": "MiniMax H3 Save AV Latent",
    "H3LoadAVLatent": "MiniMax H3 Load AV Latent",
    "H3EncodeAVTail": "MiniMax H3 Encode AV Tail",
    "H3LTXPadVideoFrames": "LTX Pad Video Frames",
    "H3Flux3EditPrompt": "FLUX 3 Image Edit Prompt Upsampler",
    "H3Flux3LayoutElements": "FLUX 3 Layout Elements",
    "H3Qwen21PromptEnhance": "Qwen Image 2.1 Native Prompt Enhancer",
    "H3Qwen21Sigmas": "Qwen Image 2.1 Resolution-Aware Sigmas",
    "H3Qwen21TurboSigmas": "Qwen Image 2.1 Viggle Turbo Sigmas",
    "H3Qwen21LightningSigmas": "Qwen Image 2.1 LightX2V Lightning Sigmas",
    "H3Qwen21ViggleLora": "Qwen Image 2.1 Viggle Turbo LoRA (unmerged)",
    "H3SemanticBridge": "MiniMax H3 Semantic Bridge (experimental)",
    "H3FirstBlockCache": "MiniMax H3 FirstBlockCache",
    "H3LightX2VBypassLoRA": "MiniMax H3 LightX2V Bypass LoRA",
    "H3SeparateAVLatent": "MiniMax H3 Separate AV Latent",
    "H3CombineAVLatent": "MiniMax H3 Combine AV Latent",
    "H3SingleFrameVAELoader": "MiniMax H3 Single-Frame VAE Loader",
    "H3VideoLatentSlicesToBatch": "MiniMax H3 Video Slices to Image Batch",
    "H3ConditioningCache": "MiniMax H3 Conditioning Cache",
    "H3StageOffloadPolicy": "MiniMax H3 Stage Offload Policy",
    "H3StageModelOffload": "MiniMax H3 Stage Model Offload",
    "H3RefinementCompilerGuard": "MiniMax H3 Refinement Compiler Guard",
    "H3SaveVideoNVENC": "MiniMax H3 Save Video (Auto H.264)",
    "H3SaveDetectedMasks": "H3 Save Detected Masks",
    "H3EstimateDepthPro2": "Soundstage Estimate Depth Pro 2",
    "H3LipSyncMask": "MiniMax H3 Lip Sync Latent Mask",
    "H3LowerFaceMask": "MiniMax H3 Lower-Face Mouth Mask",
    "H3LipSyncPrepareCrop": "MiniMax H3 Lip Sync Prepare Crop",
    "H3LipSyncSelectRefFrames": "MiniMax H3 Lip Sync Reference Frames",
    "H3LipSyncBlend": "MiniMax H3 Lip Sync Blend",
}
