"""Adapt the existing H3 Image action to Pruna's image LoRA workflow."""

from dataclasses import dataclass

from h3_app.errors import H3Error
from h3_app.jobs import record_failure
from h3_app.policy import collect_reference_slots
from h3_app.progress import no_progress
from h3_app.workflows.ph3_image import PH3_IMAGE_KEYS, build_ph3_image_graph

from .checks import check_seed, whole_number
from .image_jobs import ImageEngine, PreparedImage, output_naming, run_image_job, size_note
from .results import GenerationUpdate


@dataclass(frozen=True)
class PH3ImageRequest:
    prompt: str
    model_name: str
    width: int
    height: int
    seed: int
    reference_images: tuple[str, ...] = ()


def validate_ph3_image(request):
    if not request.prompt.strip():
        raise H3Error("Write an image prompt.")
    check_seed(request.seed)
    whole_number(request.width, 256, 8192, "Width", 32)
    whole_number(request.height, 256, 8192, "Height", 32)
    if len(request.reference_images) > 8:
        raise H3Error("P-H3-Image supports at most eight images.")


def image_references(media, *, label="P-H3-Image"):
    if media.mode == "Text to video":
        return ()
    if media.mode == "First / last frame":
        images = collect_reference_slots(media.first_image, media.last_image)
    elif media.mode == "Reference media":
        if any(getattr(media, f"ref_{kind}_{i}") for kind in ("video", "audio") for i in range(1, 4)):
            raise H3Error(f"{label} editing accepts images; remove video and audio references.")
        images = collect_reference_slots(*(getattr(media, f"ref_image_{i}") for i in range(1, 10)))
    else:
        raise H3Error("Choose text generation or an image editing mode.")
    if not images:
        raise H3Error("Provide an image to edit.")
    return tuple(images)


def generate_ph3_image(request, services, runtime, *, progress=no_progress):
    try:
        references = image_references(request.media)
        plan = services.policy.resolve_request_settings(request.values())
        if plan.issues:
            raise H3Error(" ".join(plan.issues))
        output = plan.effective.output
        models = services.models.load_model_config()
        profile_key = models.profile_key(request.sampling.model_profile)
        profile = models.profiles[profile_key]
        image_request = PH3ImageRequest(request.media.prompt, profile.fl2va,
                                        output.width, output.height, output.seed, references)
        validate_ph3_image(image_request)

        def prepare(image_request, values, info, runtime):
            # Stage only after validation. The shared job checks Core support
            # before provisioning either the selected base or the image weights.
            values["reference_images"] = tuple(services.media.stage_file(path, "ph3_images")
                                               for path in image_request.reference_images)
            graph = build_ph3_image_graph(**values, **output_naming())
            yield from ()
            return PreparedImage(graph, PH3_IMAGE_KEYS, {
                "model_profile": profile.label, "steps": 8, "sampler": "euler",
                "cfg": 1.0, "reference_count": len(references),
                "adapter": "p-h3-image_lora_bf16.safetensors",
            }, size_note(values), lambda: services.models.ensure_profile_model(
                profile_key, profile, "Text to video"))

        engine = ImageEngine("P-H3-Image", "minimax", "P-H3-Image", validate_ph3_image, prepare)
        yield from run_image_job(engine, image_request, services, runtime, progress=progress)
    except Exception as exc:
        record_failure(exc)
        yield GenerationUpdate(None, f"Error: {exc}")
