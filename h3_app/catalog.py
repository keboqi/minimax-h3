"""Extracted catalog boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from h3_app.settings import PRESETS
from h3_models import (
    DEFAULT_H3_LATENT_UPSCALER_MODEL,
    DEFAULT_LTX25_MODEL,
    DEFAULT_MUSIC3_MODEL,
    DEFAULT_SEEDVR2_MODEL,
    PROFILE_LABELS,
)

COMFY_PROXY_PATH = "/comfyui"


UVICORN_WEBSOCKET_OPTIONS = {
    "ws": "wsproto",
    "ws_per_message_deflate": False,
}


AUTO_SOL_TOKEN_THRESHOLD = 8_192


MAX_REFERENCE_IMAGES = 9


MAX_REFERENCE_VIDEOS = 3


MAX_REFERENCE_AUDIOS = 3


STAGED_INPUT_HASH_CHUNK_BYTES = 8 * 1024 * 1024


REFERENCE_VIDEO_TRANSCODE_CACHE_VERSION = "video-v1"


GEMINI_PROMPT_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)


DEFAULT_GEMINI_PROMPT_MODEL = GEMINI_PROMPT_MODELS[0]


GEMINI_API_ROOT = "https://generativelanguage.googleapis.com"


LIGHTNING_PROMPT_MODEL = "openai/gpt-5.6-luna"


LIGHTNING_API_ROOT = "https://lightning.ai/api/v1/"


PROMPT_WRITER_BACKENDS = ("Local MiniMax-H3 8B", "Gemini", "Lightning AI")


DEFAULT_PROMPT_WRITER_BACKEND = PROMPT_WRITER_BACKENDS[1]


DEFAULT_FBCACHE_PRESET = "Fast"


DEFAULT_FBCACHE_THRESHOLD = 0.10


DEFAULT_FBCACHE_START = 0.10


DEFAULT_FBCACHE_END = 0.95


DEFAULT_FBCACHE_MAX_HITS = 2


DEFAULT_FBCACHE_TEMPORAL_GUARD = True


DEFAULT_ACCELERATOR = "Spectrum"


DEFAULT_SLA_PRESET = "Balanced"


SLA_PRESET_INPUTS = {
    "Fast": {
        "sparsity_ratio": 0.90,
        "block_size": "64",
        "min_seq_len": 8192,
        "dense_last_steps": 0,
        "protect_audio": True,
    },
    "Balanced": {
        "sparsity_ratio": 0.85,
        "block_size": "64",
        "min_seq_len": 8192,
        "dense_last_steps": 0,
        "protect_audio": True,
    },
    "Quality": {
        "sparsity_ratio": 0.85,
        "block_size": "64",
        "min_seq_len": 8192,
        "dense_last_steps": 1,
        "protect_audio": True,
    },
}


LIGHTX2V_4STEP_TURBO = "LightX2V / 4-step (FL2V 768p · Ref2V 544p)"


LIGHTX2V_8STEP_TURBO = "LightX2V v1.0 / 8-step 768p"


LARRY_TURBO = "Larry v4-600 EMA"


DEFAULT_TURBO = LIGHTX2V_4STEP_TURBO


RESULT_FORMATS = ("Video", "Image", "Audio")


DEFAULT_RESULT_FORMAT = RESULT_FORMATS[0]


MIN_VIDEO_BATCH_COUNT = 1


MAX_VIDEO_BATCH_COUNT = 4


DEFAULT_VIDEO_BATCH_COUNT = 1


OFFICIAL_IMAGE_VAE = "Official video VAE"


SINGLE_FRAME_IMAGE_VAE = "Single-frame 500K (experimental)"


IMAGE_VAE_CHOICES = (OFFICIAL_IMAGE_VAE, SINGLE_FRAME_IMAGE_VAE)


DEFAULT_IMAGE_VAE = OFFICIAL_IMAGE_VAE


DEFAULT_IMAGE_FRAMES = 5


MIN_IMAGE_FRAMES = 1


MAX_IMAGE_FRAMES = 20


LTX25_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


LTX25_DEFAULTS = {
    "model": DEFAULT_LTX25_MODEL,
    "mode": "Text to video",
    "duration": 5,
    "fps": 24,
    "width": 960,
    "height": 544,
    "seed": -1,
    "cfg": 1.0,
    "sampler": "euler_ancestral",
    "image_strength": 0.7,
    "middle_time": 2.5,
    "middle_strength": 0.7,
    "end_strength": 0.7,
}


MUSIC3_DEFAULTS = {
    "model": DEFAULT_MUSIC3_MODEL,
    "duration": 120,
    "seed": -1,
    "steps": 30,
    "cfg": 1.7,
    "ar_cfg": 1.7,
    "top_k": 50,
    "tiled_decode": True,
}


LTX25_WORKFLOWS = {
    "Text / image to video — single stage": {
        "id": "t2v-i2v-single-stage",
        "filename": "LTX-2.5_T2V_I2V_Single_Stage_Distilled.json",
        "description": "The official 8-step text-to-video and start-image workflow.",
        "inputs": "Prompt and optional start image.",
        "extra_models": (),
    },
    "Text / image to video — two stage": {
        "id": "t2v-i2v-two-stage",
        "filename": "LTX-2.5_T2V_I2V_Two_Stage_Distilled.json",
        "description": "Generates low resolution, then performs a 2x latent refinement pass.",
        "inputs": "Prompt and optional start image.",
        "extra_models": ("ltx25_spatial_upscaler",),
    },
    "Audio to video — two stage": {
        "id": "audio-to-video-two-stage",
        "filename": "LTX-2.5_A2V_Two_Stage_Distilled.json",
        "description": ("Generates video synchronized to a trimmed source soundtrack."),
        "inputs": "Audio, prompt, and optional first-frame image.",
        "extra_models": ("ltx25_spatial_upscaler",),
    },
    "Text to audio": {
        "id": "text-to-audio",
        "filename": "LTX-2.5_T2A_Single_Stage_Distilled.json",
        "description": "Generates standalone audio from a text description.",
        "inputs": "Prompt, duration, and frame rate; no image or video.",
        "extra_models": (),
        "audio_only": True,
    },
    "Ingredients / reference sheet — IC-LoRA": {
        "id": "iclora-ingredients",
        "filename": "LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json",
        "description": "Uses a reference sheet of characters, props, wardrobe, or locations.",
        "inputs": "Prompt and one reference-sheet image.",
        "extra_models": ("ltx25_iclora_ingredients",),
    },
    "Video to video / instant shave — IC-LoRA": {
        "id": "iclora-video-to-video",
        "filename": "LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json",
        "description": "Applies the official instant-shave edit while preserving timing and audio.",
        "inputs": "Source video, prompt, and optional start image.",
        "extra_models": ("ltx25_iclora_instant_shave",),
    },
    "Motion tracking — IC-LoRA": {
        "id": "iclora-motion-track",
        "filename": "LTX-2.5_ICLoRA_Motion_Track_Distilled.json",
        "description": "Draws sparse point tracks that control motion through the clip.",
        "inputs": "Prompt, start image, and tracks drawn in ComfyUI.",
        "extra_models": ("ltx25_iclora_motion_track",),
    },
    "Video inpaint — two stage IC-LoRA": {
        "id": "iclora-inpaint",
        "filename": "LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json",
        "description": "Regenerates masked areas and blends them into a source video.",
        "inputs": "Source video, matching black/white mask video, prompt, and optional start image.",
        "extra_models": (
            "ltx25_iclora_in_outpaint",
            "ltx25_spatial_upscaler",
        ),
    },
    "Video outpaint — two stage IC-LoRA": {
        "id": "iclora-outpaint",
        "filename": "LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json",
        "description": "Extends a source video beyond its original frame and refines at 2x.",
        "inputs": "Source video, target canvas/padding, prompt, and optional start image.",
        "extra_models": (
            "ltx25_iclora_in_outpaint",
            "ltx25_spatial_upscaler",
        ),
    },
    "Pose / depth / canny control — IC-LoRA": {
        "id": "iclora-union-control",
        "filename": "LTX-2.5_ICLoRA_Union_Control_Distilled.json",
        "description": "Controls generation from pose, depth, or canny guidance extracted from video.",
        "inputs": "Reference video, control type, prompt, and optional start image.",
        "extra_models": (
            "ltx25_iclora_union_control",
            "ltx25_spatial_upscaler",
        ),
    },
}


LTX25_WORKFLOW_COMMON_MODEL_KEYS = (
    "ltx25_distilled",
    "ltx25_text_encoder",
    "ltx25_video_vae",
    "ltx25_video_vae_full",
    "ltx25_audio_vae",
    "ltx25_text_enhancer",
)


MODEL_PROFILE_CHOICES = list(PROFILE_LABELS.values())


CORE_LORA_LOADER_NODE = "LoraLoaderModelOnly"


CORE_SAMPLER_NODE = "KSamplerSelect"


H3_SIGMA_SHIFT_NODE = "MiniMaxH3SigmaShift"


LARRY_TURBO_LORA_NODE = "MiniMaxH3TurboLoRA"


LARRY_TURBO_SAMPLER_NODE = "MiniMaxH3TurboSampler"


LIGHTX2V_BYPASS_LORA_NODE = "H3LightX2VBypassLoRA"


H3_LATENT_UPSCALER_NODE = "MinimaxH3LatentUpscaler3D"


H3_SEPARATE_AV_LATENT_NODE = "H3SeparateAVLatent"


H3_COMBINE_AV_LATENT_NODE = "H3CombineAVLatent"


H3_SPLIT_TEMPORAL_PARAMS_NODE = "MMH3TemporalSplitParamsV10"


H3_SPLIT_SPATIAL_PARAMS_NODE = "MMH3SpatialSplitParamsV10"


H3_SPLIT_UPSCALE_NODE = "MMH3SplitUpscale"


H3_SINGLE_FRAME_VAE_LOADER_NODE = "H3SingleFrameVAELoader"


H3_IMAGE_SLICES_NODE = "H3VideoLatentSlicesToBatch"


H3_STAGE_OFFLOAD_NODE = "H3StageModelOffload"


H3_SEMANTIC_BRIDGE_NODE = "H3SemanticBridge"


H3_CONDITIONING_CACHE_NODE = "H3ConditioningCache"


H3_STAGE_OFFLOAD_POLICY_NODE = "H3StageOffloadPolicy"


H3_NVENC_SAVE_NODE = "H3SaveVideoNVENC"


H3_LATENT_UPSCALE_SCALE = 2.0


H3_LATENT_UPSCALE_STANDARD = "Full-frame refinement"


H3_LATENT_UPSCALE_SPLIT = "MMH3 Split Upscale (experimental)"


H3_LATENT_UPSCALE_METHODS = (
    H3_LATENT_UPSCALE_STANDARD,
    H3_LATENT_UPSCALE_SPLIT,
)


SOL_ATTENTION_NODE = "MiniMaxH3MemoryEfficientSolAttentionPatch"


SAGE_ATTENTION_NODE = "PathchSageAttentionKJ"


SLA_ATTENTION_NODE = "H3SLAAttention"


FUSED_MODULATION_NODE = "MiniMaxH3FusedModulation"


CHUNK_FEED_FORWARD_NODE = "MiniMaxH3ChunkFeedForward"


SEEDVR2_UPSCALE = "SeedVR2 2x"


LTX25_UPSCALE = "LTX-2.5 IC-LoRA 2x"
LTX25_DECOMPRESSION = "LTX-2.5 IC-LoRA Decompression"
LTX25_DEBLUR = "LTX-2.5 IC-LoRA Deblur"
LTX25_POSTPROCESS_MODELS = {
    LTX25_UPSCALE: "ltx25_pixel_upscaler_x2",
    LTX25_DECOMPRESSION: "ltx25_decompression",
    LTX25_DEBLUR: "ltx25_deblur",
}
LTX25_RESTORATION_OPTIONS = {LTX25_DECOMPRESSION, LTX25_DEBLUR}


SWIFTVR_UPSCALE = "SwiftVR 2x"


COMFY_UPSCALE_OPTIONS = {SEEDVR2_UPSCALE, LTX25_UPSCALE}


COMFY_POSTPROCESS_OPTIONS = COMFY_UPSCALE_OPTIONS | LTX25_RESTORATION_OPTIONS
AI_POSTPROCESS_OPTIONS = COMFY_POSTPROCESS_OPTIONS | {SWIFTVR_UPSCALE}


POSTPROCESS_OPTIONS = [
    SEEDVR2_UPSCALE,
    LTX25_UPSCALE,
    LTX25_DECOMPRESSION,
    LTX25_DEBLUR,
    SWIFTVR_UPSCALE,
    "48 fps interpolation",
]


GENERATION_POSTPROCESS_OPTIONS = [
    "None",
    SEEDVR2_UPSCALE,
    LTX25_UPSCALE,
    SWIFTVR_UPSCALE,
]


UPSCALE_RESOLUTION_PRESETS = {
    "1280 × 1280": (1280, 1280),
    "1920 × 1920": (1920, 1920),
    "2560 × 2560": (2560, 2560),
    "3840 × 3840": (3840, 3840),
}


DEFAULT_UPSCALE_RESOLUTION = "1920 × 1920"


INPUT_IMAGE_UPSCALE_SLOTS = (
    "First frame",
    "Last frame",
    *(f"Picture {index}" for index in range(1, MAX_REFERENCE_IMAGES + 1)),
)


INPUT_IMAGE_FRAME_PRESETS = {
    "1280 × 1280": (1280, 1280),
    "1920 × 1920": (1920, 1920),
    "3840 × 3840": (3840, 3840),
    "Custom": None,
}


DEFAULT_INPUT_IMAGE_FRAME_PRESET = "1920 × 1920"


@dataclass(frozen=True)
class TurboSpec:
    steps: int
    strength: float
    lora_attr: str
    ref_lora_attr: str
    custom_nodes: bool = False


TURBO_SETTINGS = {
    LARRY_TURBO: TurboSpec(
        steps=6,
        strength=1.0,
        lora_attr="larry_turbo_lora",
        ref_lora_attr="larry_turbo_ref_lora",
        custom_nodes=True,
    ),
    LIGHTX2V_4STEP_TURBO: TurboSpec(
        steps=4,
        strength=1.0,
        lora_attr="turbo_lora",
        ref_lora_attr="turbo_ref_lora",
    ),
    LIGHTX2V_8STEP_TURBO: TurboSpec(
        steps=8,
        strength=1.0,
        lora_attr="turbo_8step_lora",
        ref_lora_attr="turbo_8step_ref_lora",
    ),
}


SAMPLING_PRESET_TEXT_ENCODERS = {
    "Singularity": "NVFP4 / AWQ",
    "Fast": "NVFP4 / AWQ",
    "Balanced": "INT8 ConvRot",
    "Quality": "BF16",
}


UI_DEFAULTS = {
    "mode": "Text to video",
    "result_format": DEFAULT_RESULT_FORMAT,
    "image_vae": DEFAULT_IMAGE_VAE,
    "image_frames": DEFAULT_IMAGE_FRAMES,
    "model_profile": "Singularity",
    "text_encoder": SAMPLING_PRESET_TEXT_ENCODERS["Fast"],
    "stage_model_offload": False,
    "encoder_small_input": True,
    "semantic_bridge": True,
    "semantic_bridge_alpha": 0.10,
    "reuse_unchanged_inputs": True,
    "use_int8_vae": False,
    "use_trt_vae": True,
    "generation_mode": "Turbo",
    "turbo_variant": DEFAULT_TURBO,
    "duration": 5,
    "width": 1376,
    "height": 768,
    "steps": 4,
    "scheduler": "simple",
    "seed": -1,
    "attention_mode": "SLA",
    "sla_preset": "Fast",
    "sol_tau": 1.2,
    "sol_thresh_type": "diag",
    "sol_exact_mode": "off",
    "sol_dense_steps": 1,
    "cache_mode": DEFAULT_ACCELERATOR,
    "fbcache_preset": DEFAULT_FBCACHE_PRESET,
    "fbcache_threshold": DEFAULT_FBCACHE_THRESHOLD,
    "fbcache_start": DEFAULT_FBCACHE_START,
    "fbcache_end": DEFAULT_FBCACHE_END,
    "fbcache_max_hits": DEFAULT_FBCACHE_MAX_HITS,
    "fbcache_temporal_guard": DEFAULT_FBCACHE_TEMPORAL_GUARD,
    "easycache_threshold": 0.10,
    "easycache_start": 0.15,
    "easycache_end": 0.85,
    "easycache_verbose": False,
    "ref_image_size": "match",
    "latent_upscale": True,
    "latent_upscaler_model": DEFAULT_H3_LATENT_UPSCALER_MODEL,
    "latent_upscale_refine_steps": 2,
    "latent_upscale_method": H3_LATENT_UPSCALE_STANDARD,
    "latent_split_tile_width": 512,
    "latent_split_tile_height": 512,
    "latent_split_overlap_ratio": 0.25,
    "latent_split_fade_ratio": 0.50,
    "latent_split_chunk_frames": 73,
    "latent_split_temporal_overlap_frames": 22,
    "latent_split_seam_denoise": 0.75,
    "latent_split_seam_polish": "off",
    "postprocess": "None",
    "seedvr2_model": DEFAULT_SEEDVR2_MODEL,
    "upscale_force_offload": False,
    "upscale_split_enabled": False,
    "upscale_split_seconds": 5.0,
}


SPECTRUM_DEFAULT_INPUTS = {
    "enabled": True,
    "blend_weight": 0.50,
    "degree": 1,
    "ridge_lambda": 0.10,
    "window_size": 2.0,
    "flex_window": 0.75,
    "warmup_steps": 1,
    "tail_actual_steps": 1,
    "max_history": 8,
    "debug": False,
    "history_storage": "system_ram",
    "bootstrap_first_forecast": True,
    "anchor_residual_feedback": False,
    "selective_rollback_correction": False,
    "offline_smoothing_replay": True,
    "audio_blend_weight": 0.0,
    # Spectrum separates the unbounded replay archive from the capped
    # causal history. Keep all replay anchors in host RAM unless a workflow
    # explicitly opts into the higher-VRAM path.
    "offline_archive_storage": "system_ram",
    # v0.2.7 keeps legacy behavior when model-aware forecasting is off. Make
    # that compatibility policy explicit instead of relying on the node default.
    "model_aware_mode": "off",
    "model_aware_risk_threshold": 0.65,
}


NATIVE_PIXEL_CAP = 768 * 1344


AUTO_RESOLUTION_MEGAPIXEL_PRESETS = {
    "1 MP": 1_000_000 - 1,
    "2 MP": 2_000_000 - 1,
    "4 MP": 4_000_000 - 1,
    "8 MP": 8_000_000 - 1,
}


DEFAULT_AUTO_RESOLUTION_MEGAPIXELS = "1 MP"


AUTO_RESOLUTION_PIXEL_CAP = AUTO_RESOLUTION_MEGAPIXEL_PRESETS[
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS
]


DRAFT_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1 · 768×768": (768, 768),
    "16:9 · 1376×768": (1376, 768),
    "9:16 · 768×1376": (768, 1376),
    "4:3 · 1024×768": (1024, 768),
    "3:4 · 768×1024": (768, 1024),
    "4:5 · 768×960": (768, 960),
    "5:4 · 960×768": (960, 768),
    "21:9 · 1792×768": (1792, 768),
    "9:21 · 768×1792": (768, 1792),
}


FAST_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1 · 1088×1088": (1088, 1088),
    "16:9 · 1920×1088": (1920, 1088),
    "9:16 · 1088×1920": (1088, 1920),
    "4:3 · 1440×1088": (1440, 1088),
    "3:4 · 1088×1440": (1088, 1440),
    "4:5 · 1088×1344": (1088, 1344),
    "5:4 · 1344×1088": (1344, 1088),
    "21:9 · 2528×1088": (2528, 1088),
    "9:21 · 1088×2528": (1088, 2528),
}


LARGE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1 · 1440×1440": (1440, 1440),
    "16:9 · 2560×1440": (2560, 1440),
    "9:16 · 1440×2560": (1440, 2560),
    "4:3 · 1920×1440": (1920, 1440),
    "3:4 · 1440×1920": (1440, 1920),
    "4:5 · 1440×1792": (1440, 1792),
    "5:4 · 1792×1440": (1792, 1440),
    "21:9 · 3360×1440": (3360, 1440),
    "9:21 · 1440×3360": (1440, 3360),
}


RESOLUTION_TIERS: dict[str, dict[str, tuple[int, int]]] = {
    "draft": DRAFT_RESOLUTIONS,
    "fast": FAST_RESOLUTIONS,
    "large": LARGE_RESOLUTIONS,
}


SAMPLING_PRESETS = {
    name: tuple(asdict(value).values())[:11] for name, value in PRESETS.items()
}


VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".mkv", ".gif"})


AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a"})


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
