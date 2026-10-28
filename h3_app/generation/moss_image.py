"""MOSS-Image text-to-image over the shared ComfyUI image job, with optional Spectrum-MOSS."""

from h3_app.errors import H3Error
from h3_app.progress import no_progress
from h3_app.workflows.flux3_layout import parse_flux3_layout
from h3_app.workflows.moss_image import (
    MOSS_SPECTRUM_NODE, build_moss_image_graph, moss_image_model_keys, moss_spectrum_enabled,
    validate_moss_edit, validate_moss_image_sampling,
)
from h3_app.workflows.moss_image_prompt import is_layout_prompt

from .checks import check_cfg, check_seed, whole_number
from .image_jobs import ImageEngine, PreparedImage, output_naming, run_image_job, size_note
from .results import GenerationUpdate

MOSS_IMAGE_FAMILY = "MOSS-Image"


def validate_moss_image(request):
    check_seed(request.seed)
    if not request.prompt.strip():
        raise H3Error("Write an image prompt.")
    whole_number(request.width, 256, 2048, "Width", 16)
    whole_number(request.height, 256, 2048, "Height", 16)
    whole_number(request.steps, 1, 100, "Steps")
    check_cfg(request.cfg, 1, 10, "MOSS-Image CFG must be between 1 and 10.")
    try:
        validate_moss_edit(request.reference_images, request.turbo_variant, request.layout, request.prompt)
        moss_image_model_keys(request.model_choice, request.text_encoder_choice, request.turbo_variant,
                              request.vae_choice)
        validate_moss_image_sampling(request.turbo_variant, request.steps, request.cfg,
                                     request.width, request.height)
        parse_flux3_layout(request.layout)
    except ValueError as exc:
        raise H3Error(str(exc)) from exc
    if request.layout.strip() and is_layout_prompt(request.prompt):
        raise H3Error("The prompt is already a MOSS-Image Layout with its own boxes. Clear the layout boxes.")


def _prepare(request, values, info, runtime):
    # Installs that predate the Spectrum-MOSS pin keep the original graph.
    wanted = not request.reference_images and moss_spectrum_enabled(request.accelerate, request.turbo_variant, request.steps)
    spectrum = wanted and MOSS_SPECTRUM_NODE in info
    if wanted and not spectrum:
        yield GenerationUpdate(None, "Spectrum-MOSS is not installed; using the original sampler. "
                                     "Rerun setup and restart ComfyUI to enable it.")
    graph = build_moss_image_graph(
        **{k: v for k, v in values.items() if k not in {"preset", "accelerate"}}, accelerate=spectrum,
        **output_naming(),
    )
    keys = moss_image_model_keys(request.model_choice, request.text_encoder_choice,
                                 request.turbo_variant, request.vae_choice)
    if request.reference_images:
        keys += ("moss_image_edit_lora",)
    return PreparedImage(graph, keys, {
        "layout_box_count": len(parse_flux3_layout(request.layout)),
        "reference_count": len(request.reference_images),
        "spectrum_applied": spectrum,
        "scheduler": "sgm_uniform" if spectrum else "simple",
    }, size_note(values))


ENGINE = ImageEngine(MOSS_IMAGE_FAMILY, "moss_image", "MOSS-Image", validate_moss_image, _prepare)


def generate_moss_image(request, services, runtime, *, progress=no_progress):
    return run_image_job(ENGINE, request, services, runtime, progress=progress)
