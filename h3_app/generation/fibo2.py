"""Fibo 2.0 generation and editing over the shared ComfyUI image job."""

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL
from h3_app.errors import H3Error
from h3_app.progress import no_progress
from h3_app.workflows.fibo2 import (
    FIBO2_EDIT_SIZE_MAX, FIBO2_EDIT_SIZE_TRAINED, FIBO2_EDIT_SIZES, FIBO2_MAX_PIXELS,
    build_fibo2_graph, fibo2_model_keys, validate_fibo2_inputs,
)
from h3_models import MODEL_SPECS

from .checks import check_cfg, check_seed, whole_number
from .image_jobs import ImageEngine, PreparedImage, aspect_edit_dimensions, output_naming, run_image_job, size_note
from .qwen import first_reference_dimensions


def resolve_fibo2_dimensions(request, *, input_dir=None):
    """Edits default to the node's trained size (0 × 0); Match and Max keep the first image's aspect."""
    if not request.reference_images or request.edit_size == QWEN_EDIT_SIZE_MANUAL:
        return int(request.width), int(request.height)
    if request.edit_size == FIBO2_EDIT_SIZE_TRAINED:
        return 0, 0
    size = first_reference_dimensions(request.reference_images[0], input_dir)
    return aspect_edit_dimensions(*size, maximize=request.edit_size == FIBO2_EDIT_SIZE_MAX,
                                  max_pixels=FIBO2_MAX_PIXELS, min_side=256, max_side=2048, grid=16,
                                  label="Fibo 2.0")


def validate_fibo2(request):
    check_seed(request.seed)
    whole_number(request.width, 256, 2048, "Width", 16)
    whole_number(request.height, 256, 2048, "Height", 16)
    whole_number(request.steps, 1, 100, "Steps")
    check_cfg(request.cfg, 1, 20, "Fibo CFG must be between 1 and 20.")
    if request.edit_size not in FIBO2_EDIT_SIZES:
        raise H3Error("Choose a valid Fibo 2.0 edit output size.")
    try:
        validate_fibo2_inputs(request.model_choice, request.prompt, request.reference_images,
                              request.steps, request.cfg, request.precision)
    except ValueError as exc:
        raise H3Error(str(exc)) from exc


def _prepare(request, values, info, runtime):
    values["width"], values["height"] = resolve_fibo2_dimensions(request, input_dir=runtime.input_dir)
    graph = build_fibo2_graph(**{k: v for k, v in values.items() if k != "edit_size"}, **output_naming())
    keys = fibo2_model_keys(request.model_choice, request.precision, bool(request.reference_images))
    trained = values["width"] == 0
    return PreparedImage(graph, keys, {
        "reference_count": len(request.reference_images),
        "text_encoder": MODEL_SPECS[keys[1]].local_name,
    }, " · trained size" if trained else size_note(values))
    yield  # a generator, like engines that report notes while preparing


ENGINE = ImageEngine("Fibo 2.0", "fibo2", "Fibo 2.0", validate_fibo2, _prepare)


def generate_fibo2(request, services, runtime, *, progress=no_progress):
    return run_image_job(ENGINE, request, services, runtime, progress=progress)
