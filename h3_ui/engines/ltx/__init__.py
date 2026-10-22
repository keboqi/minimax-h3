"""LTX 2.5 in Create: handlers, the official-workflow inventory and the feature record."""

import gradio as gr

from h3_app.prompt_writers import ltx as ltx_writer
from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, DEFAULT_LIGHTNING_PROMPT_MODEL, GEMINI_PROMPT_MODELS
from h3_app.catalog import LTX25_DEFAULTS, LTX25_WORKFLOWS
from h3_app.generation import ltx as ltx_generation
from h3_app.generation.requests import LtxRequest
from h3_models import LTX25_MODEL_CHOICES

from ...engine_readiness import bind_engine_readiness
from ...job_bindings import owned_interrupt
from ..feature import EngineDeps, EngineFeature, _picker_destinations
from .bindings import bind_ltx_view
from .view import LtxView, build_ltx_view


class LtxHandlers:
    """Generation, rewriting and model downloads; the parameter names are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps

    def generate_ltx25(
        self,
        mode: str,
        model_choice: str,
        prompt: str,
        negative_prompt: str,
        first_image: str | None,
        duration: float,
        fps: float,
        width: int,
        height: int,
        seed: int,
        cfg: float,
        sampler_name: str,
        image_strength: float,
        middle_image: str | None = None,
        middle_time: float = LTX25_DEFAULTS["middle_time"],
        middle_strength: float = LTX25_DEFAULTS["middle_strength"],
        end_image: str | None = None,
        end_strength: float = LTX25_DEFAULTS["end_strength"],
        reference_images: list[str] | None = None,
        progress=gr.Progress(track_tqdm=False),
    ):
        request = LtxRequest(
            mode=mode, model_choice=model_choice, prompt=prompt, negative_prompt=negative_prompt,
            first_image=first_image, duration=duration, fps=fps, width=width, height=height, seed=seed,
            cfg=cfg, sampler_name=sampler_name, image_strength=image_strength, middle_image=middle_image,
            middle_time=middle_time, middle_strength=middle_strength, end_image=end_image,
            end_strength=end_strength, reference_images=tuple(reference_images or ()),
        )
        yield from ltx_generation.generate_ltx25(
            request, self.deps.generation_services(), self.deps.runtime_config(), progress=progress
        )

    def enhance_ltx25_prompt(
        self,
        prompt: str,
        model: str,
        temporary_api_key: str,
        mode: str,
        start_image,
        middle_image,
        end_image,
        duration: float,
        width: int,
        height: int,
        backend: str = "Gemini",
        lightning_api_key: str = "",
        lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    ) -> tuple[str, str]:
        return ltx_writer.enhance_ltx25_prompt(
            prompt, model, temporary_api_key, mode, start_image, middle_image, end_image, duration, width,
            height, backend=backend, lightning_api_key=lightning_api_key, lightning_model=lightning_model,
            runtime=self.deps.runtime_config(),
        )

    def render_ltx25_workflow_details(self, workflow_label: str) -> str:
        return self.deps.models().render_ltx25_workflow_details(workflow_label)

    def render_ltx25_official_model_inventory(self) -> str:
        return self.deps.models().render_ltx25_official_model_inventory()

    def prepare_ltx25_official_workflow(self, workflow_label: str):
        yield from self.deps.models().prepare_ltx25_official_workflow(workflow_label)

    def prepare_all_ltx25_official_models(self):
        yield from self.deps.models().prepare_all_ltx25_official_models()


def build(root, deps: EngineDeps) -> LtxView:
    handlers = LtxHandlers(deps)
    return build_ltx_view(
        root,
        model_choices=LTX25_MODEL_CHOICES,
        defaults=LTX25_DEFAULTS,
        prompt_models=GEMINI_PROMPT_MODELS,
        default_prompt_model=DEFAULT_GEMINI_PROMPT_MODEL,
        workflows=tuple(LTX25_WORKFLOWS),
        initial_workflow_details=handlers.render_ltx25_workflow_details(next(iter(LTX25_WORKFLOWS))),
        model_inventory_text=handlers.render_ltx25_official_model_inventory(),
    )


def bind(view: LtxView, deps: EngineDeps) -> LtxHandlers:
    handlers = LtxHandlers(deps)
    event = bind_ltx_view(
        view,
        render_workflow=handlers.render_ltx25_workflow_details,
        prepare_workflow=handlers.prepare_ltx25_official_workflow,
        prepare_all_models=handlers.prepare_all_ltx25_official_models,
        render_inventory=handlers.render_ltx25_official_model_inventory,
        enhance_prompt=handlers.enhance_ltx25_prompt,
        generate=handlers.generate_ltx25,
    )
    bind_engine_readiness(view, "ltx")
    stopped = view.stop.click(owned_interrupt(deps.interrupt, "ltx"), outputs=view.status, queue=False, api_name=False)
    stopped.then(fn=None, cancels=[event], queue=False, api_name=False)
    return handlers


FEATURE = EngineFeature(
    "ltx", build, bind, settings_key="ltx25", settings=lambda view: dict(vars(view)),
    library_label="LTX", library_inputs=lambda view: tuple(c for c in vars(view).values() if hasattr(c, "h3_library_picker")),
    library_destinations=lambda view: _picker_destinations("LTX", vars(view).values()),
)
