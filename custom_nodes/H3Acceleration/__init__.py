from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction

import av
import torch
import torch.nn.functional as F
import comfy.lora
import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sd
import comfy.utils
import comfy.weight_adapter
import folder_paths


LIGHTX_LORA_PATTERN = re.compile(
    r"^(diffusion_model\.(?:blocks\.\d+|token_refiner\.blocks\.\d+)\."
    r"(?:attn\.(?:qkv_proj|out_proj)|mlp\.fc[12]))\."
    r"(lora_[AB]\.weight|alpha)$"
)
LIGHTX_BACKBONE_BLOCKS = 50
LIGHTX_REFINER_BLOCKS = 2
LIGHTX_TARGETS_PER_BLOCK = 4


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

    def bypass_forward(self, original_forward, x, *args, **kwargs):
        up, down, _alpha, _mid, _dora_scale, _reshape = self.weights
        base_output = original_forward(x, *args, **kwargs)
        if up.dtype != x.dtype or up.device != x.device:
            up = up.to(device=x.device, dtype=x.dtype)
        if down.dtype != x.dtype or down.device != x.device:
            down = down.to(device=x.device, dtype=x.dtype)
        scale = self.intrinsic_scale * float(getattr(self, "multiplier", 1.0))
        update = F.linear(F.linear(x, down), up)
        return base_output.add_(update, alpha=scale)


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

    for module in modules:
        down = lora[module + ".lora_A.weight"]
        up = lora[module + ".lora_B.weight"]
        alpha = lora[module + ".alpha"]
        expected_rank = 384 if module.endswith("attn.qkv_proj") else 128
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
            or int(down.shape[1]) != int(base_shape[1])
            or int(up.shape[0]) != int(base_shape[0])
        ):
            raise ValueError(f"LightX2V tensor shapes do not match {module}")
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
            adapter = H3InplaceLoRAAdapter(adapter.loaded_keys, adapter.weights)
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


class _H3ConditioningReuseCache:
    """Small process-local cache keyed only by encoder conditioning inputs."""

    def __init__(self, max_entries: int = 2) -> None:
        self._entries: OrderedDict[str, object] = OrderedDict()
        self._fresh_since_policy: dict[str, bool] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def encode(self, cache_key: str, encode):
        with self._lock:
            if cache_key in self._entries:
                conditioning = self._entries[cache_key]
                self._entries.move_to_end(cache_key)
                logging.info("MiniMax H3 reused cached text/media conditioning")
                return conditioning

        conditioning = encode()
        with self._lock:
            self._entries[cache_key] = conditioning
            self._entries.move_to_end(cache_key)
            self._fresh_since_policy[cache_key] = True
            while len(self._entries) > self._max_entries:
                evicted_key, _value = self._entries.popitem(last=False)
                self._fresh_since_policy.pop(evicted_key, None)
        return conditioning

    def conditioning_was_reused(self, cache_key: str) -> bool:
        with self._lock:
            if cache_key not in self._entries:
                return False
            fresh = self._fresh_since_policy.pop(cache_key, False)
            return not fresh


_H3_CONDITIONING_REUSE_CACHE = _H3ConditioningReuseCache()


class _H3CachedCLIPProxy:
    def __init__(self, clip, cache_key: str) -> None:
        self._clip = clip
        self._cache_key = cache_key

    def __getattr__(self, name):
        return getattr(self._clip, name)

    def clone(self, *args, **kwargs):
        # Native H3 nodes clone CLIP before encoding. Preserve the proxy so
        # both latent-upscale stages use the same prompt/media cache entry.
        return type(self)(self._clip.clone(*args, **kwargs), self._cache_key)

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        return _H3_CONDITIONING_REUSE_CACHE.encode(
            self._cache_key,
            lambda: self._clip.encode_from_tokens_scheduled(
                tokens, *args, **kwargs
            ),
        )


class H3ConditioningCache:
    """Wrap H3 CLIP so unchanged prompt/media conditioning skips encoding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "cache_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "wrap"
    CATEGORY = "model/management/minimax"
    DESCRIPTION = (
        "Caches H3 text/media conditioning independently from resolution, "
        "duration, seed, sampler, and other generation-only settings."
    )

    def wrap(self, clip, cache_key):
        return (_H3CachedCLIPProxy(clip, str(cache_key)),)


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


class H3SaveVideoNVENC:
    """Encode an in-memory ComfyUI VIDEO with NVIDIA's H.264 hardware encoder."""

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
        "Saves an H.264 MP4 using the NVIDIA NVENC hardware encoder. "
        "Requires PyAV/FFmpeg with h264_nvenc and an NVIDIA driver."
    )

    @staticmethod
    def _require_encoder() -> None:
        try:
            av.Codec("h264_nvenc", "w")
        except Exception as exc:
            raise RuntimeError(
                "H3 NVENC output requires an FFmpeg/PyAV build with the "
                "h264_nvenc encoder. Verify that ffmpeg -encoders lists "
                "h264_nvenc and install an NVENC-enabled PyAV/FFmpeg build."
            ) from exc

    def save(
        self,
        video,
        filename_prefix,
        preset,
        constant_quality,
        prompt=None,
        extra_pnginfo=None,
    ):
        self._require_encoder()
        components = video.get_components()
        images = components.images
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] < 3:
            raise ValueError(
                "H3 NVENC output expects video images shaped [frames, H, W, RGB]"
            )

        height, width = int(images.shape[1]), int(images.shape[2])
        if width % 2 or height % 2:
            raise ValueError(
                f"H.264 NVENC requires even dimensions, got {width}x{height}"
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

                video_stream = output.add_stream("h264_nvenc", rate=frame_rate)
                video_stream.width = width
                video_stream.height = height
                video_stream.pix_fmt = "yuv420p"
                video_stream.bit_rate = 0
                video_stream.options = {
                    "preset": str(preset),
                    "tune": "hq",
                    "rc": "vbr",
                    "cq": str(int(constant_quality)),
                }

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


# Kijai's temporary FastVideo VSA implementation from comfy-kitchen PR #117.
from .fast_h3_vsa import H3FastVideoVSA  # noqa: E402


NODE_CLASS_MAPPINGS = {
    "H3FastVideoVSA": H3FastVideoVSA,
    "H3FirstBlockCache": H3FirstBlockCache,
    "H3LightX2VBypassLoRA": H3LightX2VBypassLoRA,
    "H3SeparateAVLatent": H3SeparateAVLatent,
    "H3CombineAVLatent": H3CombineAVLatent,
    "H3SingleFrameVAELoader": H3SingleFrameVAELoader,
    "H3VideoLatentSlicesToBatch": H3VideoLatentSlicesToBatch,
    "H3ConditioningCache": H3ConditioningCache,
    "H3StageOffloadPolicy": H3StageOffloadPolicy,
    "H3StageModelOffload": H3StageModelOffload,
    "H3SaveVideoNVENC": H3SaveVideoNVENC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FastVideoVSA": "MiniMax H3 FastVideo VSA (Kijai, experimental)",
    "H3FirstBlockCache": "MiniMax H3 FirstBlockCache",
    "H3LightX2VBypassLoRA": "MiniMax H3 LightX2V Bypass LoRA",
    "H3SeparateAVLatent": "MiniMax H3 Separate AV Latent",
    "H3CombineAVLatent": "MiniMax H3 Combine AV Latent",
    "H3SingleFrameVAELoader": "MiniMax H3 Single-Frame VAE Loader",
    "H3VideoLatentSlicesToBatch": "MiniMax H3 Video Slices to Image Batch",
    "H3ConditioningCache": "MiniMax H3 Conditioning Cache",
    "H3StageOffloadPolicy": "MiniMax H3 Stage Offload Policy",
    "H3StageModelOffload": "MiniMax H3 Stage Model Offload",
    "H3SaveVideoNVENC": "MiniMax H3 Save Video (NVENC)",
}
