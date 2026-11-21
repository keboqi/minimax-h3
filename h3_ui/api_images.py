"""CPU-only image workspace, sharing Studio's controller and page."""
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import gradio as gr

from h3_app.gpt_image import (DEFAULT_GPT_IMAGE_MODEL, GPT_IMAGE_ENDPOINT_CHOICES, GPT_IMAGE_MODELS, GPT_IMAGE_QUALITIES,
                              GPT_IMAGE_QUALITY, GPT_IMAGE_MAX_CONCURRENCY, GPT_IMAGE_RESOLUTION_PRESETS,
                              gpt_image_credentials)
from h3_app.jobs import JOBS
from h3_app.hosted_image import API_IMAGE_ENGINES, DEFAULT_ASPECT_RATIO, HOSTED_IMAGE_PROVIDERS, hosted_aspect_ratios
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
        "endpoints": FieldSpec("choices", "Endpoints", list(gpt_image_credentials().default_endpoints),
                               GPT_IMAGE_ENDPOINT_CHOICES),
        "quality": FieldSpec("choice", "Image quality", GPT_IMAGE_QUALITY,
                             tuple((quality, quality) for quality in GPT_IMAGE_QUALITIES)),
        "width": FieldSpec("choice", "Width", 1920, (("1920", 1920), ("1088", 1088))),
        "height": FieldSpec("choice", "Height", 1088, (("1088", 1088), ("1920", 1920))),
        "seed": FieldSpec("int", "Seed", -1),  # Input position for retained jobs.
        "batch_count": FieldSpec("int", "Concurrent images", 1, minimum=1,
                                 maximum=GPT_IMAGE_MAX_CONCURRENCY, step=1),
    }
    fields = {f"gpt_image__{name}": spec for name, spec in specs.items()}
    for engine, provider in HOSTED_IMAGE_PROVIDERS.items():
        fields.update({
            f"{engine}__model_choice": FieldSpec("choice", "Model", provider.default_model,
                                                tuple((model, model) for model in provider.native_prices)),
            f"{engine}__resolution": FieldSpec("choice", "Output size", "1K",
                                               provider.resolution_choices(provider.default_model)),
            f"{engine}__batch_count": FieldSpec("int", "Concurrent images", 1, minimum=1,
                                               maximum=provider.requests_per_minute, step=1),
        })
    # Retained jobs use the order of these fields.
    fields.update({f"{engine}__aspect_ratio": FieldSpec(
        "choice", "Aspect ratio", DEFAULT_ASPECT_RATIO, tuple((ratio, ratio) for ratio in hosted_aspect_ratios(engine)),
    ) for engine in HOSTED_IMAGE_PROVIDERS})
    fields.update({f"{engine}__match_first_aspect": FieldSpec("bool", "Match first image", False)
                   for engine in API_IMAGE_ENGINES})
    deps = SimpleNamespace(runtime_config=lambda: runtime,
                           resolve_result=lambda name: runtime.output_dir / Path(name).name)
    controller = ImageGenerationController.api_only(
        deps, {name: SimpleNamespace(value=spec.default) for name, spec in fields.items()})

    def generate(engine, mode, prompt, references, *values):
        if engine not in API_IMAGE_ENGINES:
            from h3_app.errors import H3Error
            raise H3Error("Choose a hosted image engine in Workshop.")
        # Retained jobs may predate the hosted-engine fields appended to this callback.
        yield from controller.generate(engine, mode, prompt, references, dict(zip(fields, values)))

    JOBS.register_callback("image", owned_generation(
        generate, "image", input_names=("engine", "mode", "prompt", "references", *fields)))
    pages = ImagePageState(ImagePageServices(
        root=lambda: private_image_root(runtime, JOBS.store.root),
        fields={name: spec for name, spec in fields.items() if name != "gpt_image__seed"},
        job_fields=tuple(fields), presets={"gpt_image": (GPT_IMAGE_RESOLUTION_PRESETS,)},
        suggest=None, interrupt=interrupt_api_image, library=lambda: library, engines=API_IMAGE_ENGINES))
    with gr.Tab("Image"):
        gr.HTML('<iframe src="/image/" title="Image" style="width:100%;height:90vh;border:0"></iframe>')
    ui.h3_routes = (*getattr(ui, "h3_routes", ()), partial(install_image_page, pages=pages, gpu=False))
    return ApiImageWorkspace(controller, deps, pages)


def interrupt_api_image(request, *, family="image", job_id=None):
    """Cancel owner-scoped API work without contacting ComfyUI."""
    from .job_admission import require_owner

    return JOBS.cancel(require_owner(request), family, lambda *a, **k: None,
                       lambda *a, **k: None, job_id=job_id)
