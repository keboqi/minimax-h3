from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import comfy.lora
import comfy.patcher_extension
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

    model_state = model.model.state_dict()
    for module in modules:
        down = lora[module + ".lora_A.weight"]
        up = lora[module + ".lora_B.weight"]
        base_weight = model_state[native_map[module]]
        if (
            base_weight.ndim != 2
            or int(down.shape[1]) != int(base_weight.shape[1])
            or int(up.shape[0]) != int(base_weight.shape[0])
        ):
            raise ValueError(f"LightX2V tensor shapes do not match {module}")
    return {module: native_map[module] for module in modules}


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
        "avoiding BF16 base-weight duplication."
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
        manager = comfy.weight_adapter.BypassInjectionManager()
        for key, adapter in loaded.items():
            if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
                raise TypeError(f"Unsupported LightX2V adapter for {key}")
            if len(adapter.weights) != 6:
                raise ValueError(f"Unexpected LightX2V adapter layout for {key}")
            adapter = H3InplaceLoRAAdapter(adapter.loaded_keys, adapter.weights)
            manager.add_adapter(key, adapter, strength=float(strength))

        injections = manager.create_injections(patched.model)
        if manager.get_hook_count() != len(loaded):
            raise RuntimeError(
                f"LightX2V bypass created {manager.get_hook_count()} hooks for "
                f"{len(loaded)} adapters"
            )
        patched.set_injections("h3_lightx2v_bypass", injections)
        logging.info(
            "MiniMax H3 LightX2V runtime bypass: %d adapters at strength %.3f",
            len(loaded),
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


NODE_CLASS_MAPPINGS = {
    "H3FirstBlockCache": H3FirstBlockCache,
    "H3LightX2VBypassLoRA": H3LightX2VBypassLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FirstBlockCache": "MiniMax H3 FirstBlockCache",
    "H3LightX2VBypassLoRA": "MiniMax H3 LightX2V Bypass LoRA",
}
