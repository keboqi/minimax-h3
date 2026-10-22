"""FLUX 3 Image Klein generation and editing over the shared ComfyUI image job."""

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX
from h3_app.errors import H3Error
from h3_app.progress import no_progress
from h3_app.workflows.flux3 import build_flux3_graph, flux3_model_keys
from h3_app.workflows.flux3_layout import parse_flux3_layout
from h3_models import FLUX3_BASE_MODEL_CHOICES

from .checks import check_cfg, check_seed, whole_number
from .image_jobs import ImageEngine, PreparedImage, aspect_edit_dimensions, output_naming, run_image_job, size_note
from .qwen import first_reference_dimensions

FLUX3_EDIT_SIZES = (QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX, QWEN_EDIT_SIZE_MANUAL)
FLUX3_MAX_PIXELS, FLUX3_MIN_SIDE, FLUX3_MAX_SIDE, FLUX3_GRID = 4_194_304, 256, 4096, 16


def flux3_edit_dimensions(source_width, source_height, *, maximize=False):
    return aspect_edit_dimensions(source_width, source_height, maximize=maximize, max_pixels=FLUX3_MAX_PIXELS,
                                  min_side=FLUX3_MIN_SIDE, max_side=FLUX3_MAX_SIDE, grid=FLUX3_GRID, label="FLUX 3")


def resolve_flux3_dimensions(request, *, input_dir=None):
    """Edits default to the first reference's size; manual size applies otherwise."""
    if not request.reference_images or request.edit_size == QWEN_EDIT_SIZE_MANUAL:
        return int(request.width), int(request.height)
    size = first_reference_dimensions(request.reference_images[0], input_dir)
    return flux3_edit_dimensions(*size, maximize=request.edit_size == QWEN_EDIT_SIZE_MAX)


def validate_flux3(request):
    check_seed(request.seed)
    if not request.prompt.strip():
        raise H3Error("Write an image prompt or edit request.")
    flux3_model_keys(request.model_choice, request.expand_prompt, request.upsampler_choice)
    whole_number(request.width, 256, 4096, "Width", 16)
    whole_number(request.height, 256, 4096, "Height", 16)
    whole_number(request.steps, 1, 100, "Steps")
    if request.cfg is not None:
        check_cfg(request.cfg, 1, 20, "FLUX CFG must be between 1 and 20.")
        if request.model_choice not in FLUX3_BASE_MODEL_CHOICES and request.cfg != 1:
            raise H3Error("Distilled FLUX 3 requires CFG 1.")
    parse_flux3_layout(request.layout)
    if request.edit_size not in FLUX3_EDIT_SIZES:
        raise H3Error("Choose a valid FLUX 3 edit output size.")
    if len(request.reference_images) > 10:
        raise H3Error("FLUX 3 supports at most ten reference images.")


def _prepare(request, values, info, runtime):
    values["width"], values["height"] = resolve_flux3_dimensions(request, input_dir=runtime.input_dir)
    # The shared image resolver only reads the managed staging directory.
    graph = build_flux3_graph(
        **{k: v for k, v in values.items() if k != "edit_size"}, **output_naming(),
        system_prompt=runtime.prompt_systems[
            "FLUX 3 Image Klein edit" if request.reference_images else "FLUX 3 Image Klein"
        ].read_text(encoding="utf-8").strip(),
    )
    return PreparedImage(
        graph, flux3_model_keys(request.model_choice, request.expand_prompt, request.upsampler_choice),
        {"reference_count": len(request.reference_images),
         "layout_box_count": len(parse_flux3_layout(request.layout))},
        size_note(values),
    )
    yield  # a generator, like engines that report notes while preparing


ENGINE = ImageEngine("FLUX 3 Image Klein", "flux3", "FLUX 3", validate_flux3, _prepare)


def generate_flux3(request, services, runtime, *, progress=no_progress):
    return run_image_job(ENGINE, request, services, runtime, progress=progress)
