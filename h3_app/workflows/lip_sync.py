"""Local H3 graphs for lip-sync rendering (explicit crop/mask and automatic face-tracked)."""

from h3_app.catalog import H3_REFINEMENT_COMPILER_GUARD_NODE, H3_SEPARATE_AV_LATENT_NODE, H3_SOURCE_AUDIO_LATENT_NODE
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.refinement import resolve_face_refinement

from .face_refine import FACE_DETECTOR
from .h3 import add_h3_stage_offload
from .refinement import add_partial_refinement_schedule, add_refinement_stack

FPS = 24
MIN_FRAMES = 5
FRAME_STRIDE = 17
MAX_FRAMES = 362
DEFAULT_CROP_SIZE = 640
DEFAULT_LIP_SYNC_CROP_FACTOR = 1.65
DEFAULT_AUTO_STRENGTH = 0.70
DEFAULT_AUTO_FEATHER = 6
DEFAULT_NEUTRALIZE_MOUTH = 0.65
DEFAULT_BLEND_COLOR_MATCH = 0.85
DEFAULT_BLEND_SHARPEN = 0.35
DEFAULT_PYRAMID_LEVELS = 3
LIP_SYNC_PROMPT = (
    "Preserve the source shot, framing, identity, lighting and expression. "
    "Synchronize the speaker's mouth and jaw to the supplied dialogue."
)


def snap_h3_frames(frames: int) -> int:
    """Round a positive frame count up to H3's 17k+5 grid."""
    if frames <= 0:
        raise H3Error("Lip-sync window must contain at least one frame.")
    if frames <= MIN_FRAMES:
        return MIN_FRAMES
    snapped = MIN_FRAMES + ((frames - MIN_FRAMES + FRAME_STRIDE - 1) // FRAME_STRIDE) * FRAME_STRIDE
    if snapped > MAX_FRAMES:
        raise H3Error(f"Lip-sync window exceeds H3's {MAX_FRAMES}-frame limit.")
    return snapped


def validate_lip_sync_window(width: int, height: int, frames: int, strength: float) -> None:
    """Validate the spatial and temporal grid before staging the shot."""
    if not (MIN_FRAMES <= frames <= MAX_FRAMES and (frames - MIN_FRAMES) % FRAME_STRIDE == 0):
        raise H3Error("Lip sync expects 17k+5 frames, up to 362, at 24 fps.")
    if min(width, height) < 32 or width % 32 or height % 32:
        raise H3Error("Lip-sync crop dimensions must be multiples of 32 pixels.")
    if not 0 < strength <= 1:
        raise H3Error("Lip-sync strength must be greater than 0 and at most 1.")


def build_lip_sync_graph(*, crop, mask, dialogue, width, height, frames, models, refinement,
                         output_prefix, seed=0, strength=0.4):
    """Render an H3-grid window. White in the grayscale mask means repaint."""
    validate_lip_sync_window(width, height, frames, strength)
    config = resolve_face_refinement(refinement, models)
    settings = config["settings"]
    graph = Graph()
    source = graph.add("GetVideoComponents", video=Graph.out(graph.add("LoadVideo", file=crop)))
    matte = graph.add("GetVideoComponents", video=Graph.out(graph.add("LoadVideo", file=mask)))
    mouth = graph.add("ImageToMask", image=Graph.out(matte), channel="red")
    audio = graph.add("LoadAudio", audio=dialogue)

    model, clip, encoder, decoder, audio_vae, schedule_model = add_refinement_stack(
        graph, models, config, width, height,
    )
    inputs = dict(clip=clip, vae=encoder, prompt=LIP_SYNC_PROMPT, width=width, height=height, length=frames)
    reference = settings["mode"] == "Reference media"
    if reference:
        inputs.update(audio_vae=audio_vae, ref_image_size=settings["ref_image_size"])
    conditioning = graph.add("MiniMaxH3ReferenceToVideo" if reference else "MiniMaxH3ImageToVideo", **inputs)
    injected = graph.add("H3InjectVideoLatent", av_latent=Graph.out(conditioning, 1),
                         images=Graph.out(source), vae=encoder)
    held = graph.add(H3_SOURCE_AUDIO_LATENT_NODE, latent=Graph.out(injected), audio=Graph.out(audio),
                     audio_vae=audio_vae)
    masked = graph.add("H3LipSyncMask", latent=Graph.out(held), source_frames=Graph.out(source),
                       mouth_mask=Graph.out(mouth))
    conditioning_ref, latent = Graph.out(conditioning), Graph.out(masked)
    if settings["stage_model_offload"]:
        conditioning_ref, latent, _, _ = add_h3_stage_offload(graph, conditioning_ref, latent)
    guarded = graph.add(H3_REFINEMENT_COMPILER_GUARD_NODE, model=model, min_video_volume=1_500_000)
    guider = graph.add("BasicGuider", model=Graph.out(guarded), conditioning=conditioning_ref)
    sampler, sigmas = add_partial_refinement_schedule(graph, schedule_model, config, strength)
    sampled = graph.add("SamplerCustomAdvanced", noise=Graph.out(graph.add("RandomNoise", noise_seed=int(seed))),
                        guider=Graph.out(guider), sampler=Graph.out(sampler), sigmas=Graph.out(sigmas),
                        latent_image=latent)
    video_latent = graph.add(H3_SEPARATE_AV_LATENT_NODE, latent=Graph.out(sampled))
    pixels = graph.add("VAEDecode", samples=Graph.out(video_latent), vae=decoder)
    composite = graph.add("ImageCompositeMasked", destination=Graph.out(source), source=Graph.out(pixels),
                          x=0, y=0, resize_source=False, mask=Graph.out(masked, 1))
    preview = graph.add("CreateVideo", images=Graph.out(composite), audio=Graph.out(audio), fps=FPS, bit_depth=8)
    graph.add("SaveVideo", video=Graph.out(preview), filename_prefix=output_prefix, format="auto", codec="auto")
    return graph.nodes


def build_auto_lip_sync_graph(
    *, video, dialogue, frames, models, refinement, output_prefix,
    crop_size=DEFAULT_CROP_SIZE, seed=0, strength=DEFAULT_AUTO_STRENGTH,
    active_start=0, active_end=-1, face_target=None, feather=DEFAULT_AUTO_FEATHER,
):
    """Track one face automatically, repaint its lower face to fitted audio, and stitch back."""
    validate_lip_sync_window(crop_size, crop_size, frames, strength)
    config = resolve_face_refinement(refinement, models)
    settings = config["settings"]
    graph = Graph()
    source = graph.add("GetVideoComponents", video=Graph.out(graph.add("LoadVideo", file=video)))
    audio = graph.add("LoadAudio", audio=dialogue)

    track_inputs = dict(
        images=Graph.out(source), detector=FACE_DETECTOR, confidence=0.35,
        crop_factor=DEFAULT_LIP_SYNC_CROP_FACTOR, canvas_width=crop_size, canvas_height=crop_size,
        canvas_mode="manual", smooth_window=21, size_smooth_window=51,
        smooth_method="gaussian", size_mode="per_frame",
        identity_track=False, cut_detection="none", absent_shots="off",
    )
    if face_target is not None:
        target_x, target_y, target_frame = face_target
        track_inputs.update(
            select="closest_to_xy",
            X=int(target_x),
            Y=int(target_y),
            frame_index=max(0, min(int(target_frame), frames - 1)),
        )
    else:
        track_inputs["select"] = "largest_face"
    track = graph.add("H3FaceTrackCrop", **track_inputs)
    crops, transform = Graph.out(track), Graph.out(track, 1)

    masks = graph.add(
        "H3LowerFaceMask", crops=crops, transform=transform,
        active_start=int(active_start), active_end=int(active_end), separate_opacity=True,
    )
    denoise_mask, composite_mask = Graph.out(masks), Graph.out(masks, 1)
    prepared = graph.add(
        "H3LipSyncPrepareCrop", crops=crops, denoise_mask=denoise_mask,
        strength=DEFAULT_NEUTRALIZE_MOUTH,
    )

    model, clip, encoder, decoder, audio_vae, schedule_model = add_refinement_stack(
        graph, models, config, crop_size, crop_size,
    )
    inputs = dict(clip=clip, vae=encoder, prompt=LIP_SYNC_PROMPT, width=crop_size, height=crop_size, length=frames)
    reference = settings["mode"] == "Reference media"
    if reference:
        inputs.update(audio_vae=audio_vae, ref_image_size=settings["ref_image_size"])
    conditioning = graph.add("MiniMaxH3ReferenceToVideo" if reference else "MiniMaxH3ImageToVideo", **inputs)
    injected = graph.add(
        "H3InjectVideoLatent", av_latent=Graph.out(conditioning, 1),
        images=Graph.out(prepared), vae=encoder,
    )
    held = graph.add(H3_SOURCE_AUDIO_LATENT_NODE, latent=Graph.out(injected), audio=Graph.out(audio),
                     audio_vae=audio_vae)
    masked = graph.add("H3LipSyncMask", latent=Graph.out(held), source_frames=crops, mouth_mask=denoise_mask)
    conditioning_ref, latent = Graph.out(conditioning), Graph.out(masked)
    if settings["stage_model_offload"]:
        conditioning_ref, latent, _, _ = add_h3_stage_offload(graph, conditioning_ref, latent)
    guarded = graph.add(H3_REFINEMENT_COMPILER_GUARD_NODE, model=model, min_video_volume=1_500_000)
    guider = graph.add("BasicGuider", model=Graph.out(guarded), conditioning=conditioning_ref)
    sampler, sigmas = add_partial_refinement_schedule(graph, schedule_model, config, strength)
    sampled = graph.add("SamplerCustomAdvanced", noise=Graph.out(graph.add("RandomNoise", noise_seed=int(seed))),
                        guider=Graph.out(guider), sampler=Graph.out(sampler), sigmas=Graph.out(sigmas),
                        latent_image=latent)
    video_latent = graph.add(H3_SEPARATE_AV_LATENT_NODE, latent=Graph.out(sampled))
    decoded = graph.add("VAEDecode", samples=Graph.out(video_latent), vae=decoder)
    blended = graph.add(
        "H3LipSyncBlend", source_crops=crops, generated_crops=Graph.out(decoded),
        composite_mask=Graph.out(masks, 2), color_match=DEFAULT_BLEND_COLOR_MATCH,
        sharpen=DEFAULT_BLEND_SHARPEN, pyramid_levels=DEFAULT_PYRAMID_LEVELS,
    )
    stitched = graph.add(
        "H3FaceStitch", base_images=Graph.out(source), refined_crops=Graph.out(blended), transform=transform,
        paste_region="face_ellipse", mask_dilation=0, feather=int(feather), colour_match=0.0, blend=1.0,
        undetected_frames="fade_out", masks=composite_mask, feather_scales_with_crop=False,
    )
    rendered = graph.add("CreateVideo", images=Graph.out(stitched), audio=Graph.out(audio), fps=FPS, bit_depth=8)
    graph.add("SaveVideo", video=Graph.out(rendered), filename_prefix=output_prefix, format="auto", codec="auto")
    return graph.nodes
