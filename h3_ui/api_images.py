"""CPU-only image workspace, sharing Studio's controller and page."""
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import gradio as gr

from h3_app.gpt_image import (DEFAULT_GPT_IMAGE_MODEL, GPT_IMAGE_MODELS, GPT_IMAGE_QUALITIES,
                              GPT_IMAGE_QUALITY, GPT_IMAGE_MAX_CONCURRENCY, GPT_IMAGE_RESOLUTION_PRESETS)
from h3_app.jobs import JOBS
from .image_controller import ImageGenerationController
from .image.page import ImagePageServices, ImagePageState, install_image_page, private_image_root
from .job_bindings import owned_generation
from .page_fields import FieldSpec


class ApiImageWorkspace(NamedTuple):
    controller: ImageGenerationController
    deps: SimpleNamespace
    pages: ImagePageState


def build_api_images(ui, runtime, *, library=None):
    specs = {
        "models": FieldSpec("choices", "Models", [DEFAULT_GPT_IMAGE_MODEL],
                            tuple((model, model) for model in GPT_IMAGE_MODELS)),
        "use_failback": FieldSpec("bool", "Failback", False),
        "quality": FieldSpec("choice", "Image quality", GPT_IMAGE_QUALITY,
                             tuple((quality, quality) for quality in GPT_IMAGE_QUALITIES)),
        "width": FieldSpec("choice", "Width", 1920, (("1920", 1920), ("1088", 1088))),
        "height": FieldSpec("choice", "Height", 1088, (("1088", 1088), ("1920", 1920))),
        "seed": FieldSpec("int", "Seed", -1),  # Input position for retained jobs.
        "batch_count": FieldSpec("int", "Concurrent images", 1, minimum=1,
                                 maximum=GPT_IMAGE_MAX_CONCURRENCY, step=1),
    }
    fields = {f"gpt_image__{name}": spec for name, spec in specs.items()}
    deps = SimpleNamespace(runtime_config=lambda: runtime,
                           resolve_result=lambda name: runtime.output_dir / Path(name).name)
    controller = ImageGenerationController.api_only(
        deps, {name: SimpleNamespace(value=spec.default) for name, spec in fields.items()})

    def generate(engine, mode, prompt, references, *values):
        if engine != "gpt_image":
            from h3_app.errors import H3Error
            raise H3Error("Choose GPT Image for API image tasks in Workshop.")
        yield from controller.generate(engine, mode, prompt, references, dict(zip(fields, values, strict=True)))

    JOBS.register_callback("image", owned_generation(
        generate, "image", input_names=("engine", "mode", "prompt", "references", *fields)))
    pages = ImagePageState(ImagePageServices(
        root=lambda: private_image_root(runtime, JOBS.store.root),
        fields={name: spec for name, spec in fields.items() if name != "gpt_image__seed"},
        job_fields=tuple(fields), presets={"gpt_image": (GPT_IMAGE_RESOLUTION_PRESETS,)},
        suggest=None, interrupt=interrupt_api_image, library=lambda: library, engines=("gpt_image",)))
    with gr.Tab("Image"):
        gr.HTML('<iframe src="/image/" title="Image" style="width:100%;height:90vh;border:0"></iframe>')
    ui.h3_routes = (*getattr(ui, "h3_routes", ()), partial(install_image_page, pages=pages, gpu=False))
    return ApiImageWorkspace(controller, deps, pages)


def interrupt_api_image(request, *, family="image", job_id=None):
    """Cancel owner-scoped API work without contacting ComfyUI."""
    from .job_admission import require_owner

    return JOBS.cancel(require_owner(request), family, lambda *a, **k: None,
                       lambda *a, **k: None, job_id=job_id)
