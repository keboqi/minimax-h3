"""Local H3 graph for a single aligned face crop and mouth-mask window."""

from h3_app.catalog import H3_REFINEMENT_COMPILER_GUARD_NODE, H3_SEPARATE_AV_LATENT_NODE, H3_SOURCE_AUDIO_LATENT_NODE
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.refinement import resolve_face_refinement

from .h3 import add_h3_stage_offload
from .refinement import add_partial_refinement_schedule, add_refinement_stack

FPS = 24
MAX_FRAMES = 362
LIP_SYNC_PROMPT = (
    "Preserve the source shot, framing, identity, lighting and expression. "
    "Synchronize the speaker's mouth and jaw to the supplied dialogue."
)


def validate_lip_sync_window(width: int, height: int, frames: int, strength: float) -> None:
    """Validate the spatial and temporal grid before staging the shot."""
    if not (5 <= frames <= MAX_FRAMES and (frames - 5) % 17 == 0):
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
