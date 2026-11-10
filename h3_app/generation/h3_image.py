"""Official image jobs, independent of the video request's reference slots and settings."""

from dataclasses import dataclass

from h3_app.errors import H3Error
from h3_app.progress import no_progress
from h3_app.workflows.h3_image import H3_IMAGE_KEYS, build_h3_image_graph
from h3_app.h3_image import (H3_IMAGE_OFFICIAL, H3_IMAGE_MAX_REFERENCES,
                             h3_image_prompt_path, validate_h3_image_canvas)
from h3_app.prompt_writers.common import read_system_prompt

from .checks import check_seed
from .image_jobs import ImageEngine, PreparedImage, output_naming, run_image_job, size_note


@dataclass(frozen=True)
class H3ImageRequest:
    prompt: str
    width: int = 4096
    height: int = 4096
    seed: int = -1
    reference_images: tuple[str, ...] = ()
    expand_prompt: bool = True


def validate_h3_image(request):
    if not request.prompt.strip():
        raise H3Error("Write an image prompt or edit instruction.")
    check_seed(request.seed)
    validate_h3_image_canvas(request.width, request.height)
    if len(request.reference_images) > H3_IMAGE_MAX_REFERENCES:
        raise H3Error("MiniMax H3 Image supports at most sixteen images.")


def generate_h3_image(request, services, runtime, *, progress=no_progress):
    def prepare(request, values, info, runtime):
        values["reference_images"] = tuple(services.media.stage_file(path, "h3_images")
                                           for path in request.reference_images)
        system_prompt = None
        if request.expand_prompt:
            editing = bool(request.reference_images)
            target = "MiniMax H3 Image edit" if editing else "MiniMax H3 Image"
            path = getattr(runtime, "prompt_systems", {}).get(target, h3_image_prompt_path(editing))
            system_prompt = read_system_prompt(path, target)
        graph = build_h3_image_graph(**values, system_prompt=system_prompt, **output_naming())
        keys = H3_IMAGE_KEYS + (("h3_image_upsampler",) if request.expand_prompt else ())
        yield from ()
        return PreparedImage(graph, keys, {
            "backend": H3_IMAGE_OFFICIAL, "steps": 6, "sampler": "euler", "scheduler": "simple",
            "cfg": 1.0, "reference_count": len(request.reference_images),
        }, size_note(values))

    engine = ImageEngine(H3_IMAGE_OFFICIAL, "minimax", H3_IMAGE_OFFICIAL, validate_h3_image, prepare)
    yield from run_image_job(engine, request, services, runtime, progress=progress)


def generate_h3_image_from_h3(request, services, runtime, *, progress=no_progress):
    """Legacy Image result format defaults to the official backend too."""
    from h3_app.jobs import record_failure
    from .ph3_image import image_references
    from .results import GenerationUpdate

    try:
        references = image_references(request.media, label=H3_IMAGE_OFFICIAL)
        plan = services.policy.resolve_request_settings(request.values())
        if plan.issues:
            raise H3Error(" ".join(plan.issues))
        output = plan.effective.output
        image = H3ImageRequest(request.media.prompt, output.width, output.height, output.seed, references)
        yield from generate_h3_image(image, services, runtime, progress=progress)
    except Exception as exc:
        record_failure(exc)
        yield GenerationUpdate(None, f"Error: {exc}")
