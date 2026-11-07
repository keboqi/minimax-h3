"""Optional face pass: crop the largest face, re-render it with H3, stitch it back."""

from __future__ import annotations

from typing import Any

from h3_app.catalog import (
    H3_CONDITIONING_CACHE_NODE,
    H3_NVENC_SAVE_NODE,
    H3_REFINEMENT_COMPILER_GUARD_NODE,
    H3_SEMANTIC_BRIDGE_NODE,
    H3_SEPARATE_AV_LATENT_NODE,
    H3_SOURCE_AUDIO_LATENT_NODE,
)
from h3_app.graph import Graph
from h3_app.model_types import ModelConfig
from h3_app.refinement import FACE_REFINE_STEPS, resolve_face_refinement

REQUIRED_FACE_REFINE_NODES = frozenset({
    "H3FaceTrackCrop", "H3InjectVideoLatent", "H3PerFrameDenoise", "H3FaceStitch",
    H3_SOURCE_AUDIO_LATENT_NODE,
})
# The pack lists detectors from a folder only Impact Pack registers. Without it, it falls
# back to this bare name and looks in models/ultralytics/bbox.
FACE_DETECTOR = "face_yolov8m.pt"
# The tracker picks the real canvas, capped at this size; the stack needs one for attention policy.
CANVAS = 768


def build_h3_face_refine_graph(
    *, source_video: str, seed: int, models: ModelConfig, prompt: str, fps: float,
    strength: float, refinement: dict[str, Any], output_nonce: str = "",
) -> dict[str, Any]:
    from .h3 import add_h3_stage_offload, h3_conditioning_cache_key
    from .refinement import add_refinement_stack

    config = resolve_face_refinement(refinement, models)
    settings = config["settings"]
    graph = Graph()
    loaded = graph.add("LoadVideo", file=source_video)
    components = graph.add("GetVideoComponents", video=Graph.out(loaded))
    frames, audio = Graph.out(components), Graph.out(components, 1)
    # Identity tracking stays off: it needs insightface, and largest-face selection does not.
    track = graph.add(
        "H3FaceTrackCrop", images=frames, detector=FACE_DETECTOR, confidence=0.35,
        crop_factor=2.5, canvas_width=CANVAS, canvas_height=CANVAS,
        canvas_mode="auto_capped_768", smooth_window=21, size_smooth_window=51,
        smooth_method="gaussian", size_mode="per_frame", select="largest_face",
        identity_track=False, cut_detection="none",
    )
    crops, transform = Graph.out(track), Graph.out(track, 1)
    model, clip, encoder, decoder, audio_vae, schedule_model = add_refinement_stack(
        graph, models, config, CANVAS, CANVAS,
    )
    cache_key = h3_conditioning_cache_key(
        "face-refine", prompt, config["text_encoder"], [("source", source_video)],
        encoder_settings={"encoder_small_input": settings["encoder_small_input"],
                          "video_encoder": settings["video_encoder"]},
    )
    clip = Graph.out(graph.add(
        H3_CONDITIONING_CACHE_NODE, clip=clip, cache_key=cache_key,
        encoder_small_input=settings["encoder_small_input"],
        reuse_conditioning=settings["reuse_unchanged_inputs"],
    ))
    inputs = dict(clip=clip, vae=encoder, prompt=prompt, width=Graph.out(track, 4),
                  height=Graph.out(track, 5), length=Graph.out(track, 6))
    reference = settings["mode"] == "Reference media"
    if reference:
        inputs.update(audio_vae=audio_vae, ref_image_size=settings["ref_image_size"])
    conditioning = graph.add(
        "MiniMaxH3ReferenceToVideo" if reference else "MiniMaxH3ImageToVideo", **inputs,
    )
    conditioning_ref = Graph.out(conditioning)
    if settings["semantic_bridge"] and settings["semantic_bridge_alpha"]:
        conditioning_ref = Graph.out(graph.add(
            H3_SEMANTIC_BRIDGE_NODE, conditioning=conditioning_ref,
            alpha=settings["semantic_bridge_alpha"],
        ))
    injected = graph.add(
        "H3InjectVideoLatent", av_latent=Graph.out(conditioning, 1), images=crops, vae=encoder,
    )
    held = graph.add(H3_SOURCE_AUDIO_LATENT_NODE, latent=Graph.out(injected),
                     audio_vae=audio_vae, audio=audio)
    guarded = graph.add(H3_REFINEMENT_COMPILER_GUARD_NODE, model=model, min_video_volume=1_500_000)
    per_frame = graph.add(
        "H3PerFrameDenoise", model=Graph.out(guarded), av_latent=Graph.out(held),
        transform=transform, denoise_multiplier_small_face=1.0,
        denoise_multiplier_large_face=0.35, scale_mode="absolute_px", face_px_small=30.0,
        face_px_large=120.0, gamma=1.0, smooth_frames=9,
    )
    latent = Graph.out(per_frame)
    if settings["stage_model_offload"]:
        conditioning_ref, latent, _, _ = add_h3_stage_offload(graph, conditioning_ref, latent)
    guider = graph.add("BasicGuider", model=Graph.out(per_frame, 2), conditioning=conditioning_ref)
    # Strength is BasicScheduler's denoise. The pack warns that SplitSigmas stays at too
    # high a sigma after H3's shift and rewrites the frame.
    sigmas = graph.add("BasicScheduler", model=schedule_model, scheduler="simple",
                       steps=FACE_REFINE_STEPS, denoise=float(strength))
    sampled = graph.add(
        "SamplerCustomAdvanced", noise=Graph.out(graph.add("RandomNoise", noise_seed=int(seed))),
        guider=Graph.out(guider), sampler=Graph.out(graph.add("KSamplerSelect", sampler_name="er_sde")),
        sigmas=Graph.out(sigmas), latent_image=latent,
    )
    video_latent = graph.add(H3_SEPARATE_AV_LATENT_NODE, latent=Graph.out(sampled))
    decoded = graph.add("VAEDecode", samples=Graph.out(video_latent), vae=decoder)
    stitched = graph.add(
        "H3FaceStitch", base_images=frames, refined_crops=Graph.out(decoded), transform=transform,
        paste_region="face_only", mask_dilation=24, feather=24, colour_match=1.0, blend=1.0,
        undetected_frames="fade_out", feather_scales_with_crop=False,
    )
    video = graph.add("CreateVideo", images=Graph.out(stitched), audio=audio, fps=float(fps), bit_depth=8)
    graph.add(H3_NVENC_SAVE_NODE, video=Graph.out(video),
              filename_prefix=f"h3/face_refine_{output_nonce}", preset="p4", constant_quality=23)
    return graph.nodes
