"""Image draft policies and named adapters to existing generation handlers."""

from h3_app.contracts import GENERATION_COMPONENTS, GENERATION_FIELDS
from h3_app.errors import H3Error
from h3_app.generation.requests import H3Request
from h3_models import qwen_image21_reference_limit

IMAGE_MODELS = ("qwen", "h3", "flux3", "moss_image", "fibo2")


def reference_limit(engine, variant="Off"):
    if engine == "qwen":
        return qwen_image21_reference_limit(variant)
    return {"h3": 8, "flux3": 10, "moss_image": 0, "fibo2": 4}[engine]


def draft_issue(engine, mode, prompt, references, variant="Off", fibo_model="Dev", separate=False):
    if engine not in IMAGE_MODELS:
        return "Choose an image model."
    if mode not in {"Text to image", "Image edit"}:
        return "Choose Generate or Edit."
    editing = mode == "Image edit"
    if editing and engine == "moss_image":
        return "MOSS-Image supports generation only. Choose Generate or another model."
    if not editing and engine == "fibo2" and fibo_model == "Edit Turbo":
        return "Fibo Edit Turbo requires Edit mode. Choose Dev to generate from text."
    if not (prompt or "").strip():
        return "Write an image prompt or edit instruction."
    if editing:
        if not references:
            return "Add at least one image to edit."
        limit = reference_limit(engine, variant)
        if not (engine == "qwen" and separate) and len(references) > limit:
            return f"This model supports up to {limit} references per edit. Remove inputs or choose another model."
    return ""


def h3_image_values(settings, prompt, references, width, height, seed):
    """Override only image inputs in a fresh request; never mutate Video settings."""
    values = {**dict.fromkeys(GENERATION_FIELDS), **settings}
    for name in values:
        if name.startswith(("ref_image_", "ref_video_", "ref_audio_", "fl2va_audio_")):
            values[name] = None
    values.update(prompt=prompt, mode="Reference media" if references else "Text to video",
                  first_image=None, last_image=None, result_format="Image", width=width, height=height,
                  seed=seed, image_edit_size="Use width and height above", postprocess="None", latent_upscale=False)
    values.update({f"ref_image_{i}": path for i, path in enumerate(references, 1)})
    return values


# Handler parameter → option component. Common draft inputs are supplied separately.
GENERATION_OPTIONS = {
    "qwen": (
        ("model_choice", "model"), ("text_encoder_choice", "text_encoder"),
        ("negative_prompt", "negative_prompt"), ("width", "width"), ("height", "height"),
        ("reference_resolution", "reference_resolution"), ("edit_size", "edit_size"), ("seed", "seed"),
        ("steps", "steps"), ("cfg", "cfg"), ("sampler_name", "sampler"), ("scheduler", "scheduler"),
        ("cache_device", "cache_device"), ("cache_dtype", "cache_dtype"),
        ("attention_backend", "attention_backend"), ("accelerator", "accelerator"),
        ("turbo_variant", "turbo_variant"), ("output_resolution_1k", "output_resolution_1k"),
        ("output_resolution_2k", "output_resolution_2k"), ("batch_count", "batch_count"),
        ("batch_edit_inputs", "batch_edit_inputs"), ("prompt_enhancer", "prompt_enhancer"),
        ("lightning_strength", "lightning_strength"),
    ),
    "flux3": (
        ("model_choice", "model"), ("width", "width"), ("height", "height"),
        ("edit_size", "edit_size"), ("seed", "seed"), ("steps", "steps"),
        ("expand_prompt", "expand_prompt"), ("upsampler_choice", "upsampler"),
        ("cfg", "cfg"), ("layout", "layout"),
    ),
    "moss_image": (
        ("preset", "preset"), ("turbo_variant", "turbo_variant"), ("model_choice", "model"),
        ("text_encoder_choice", "text_encoder"), ("vae_choice", "vae"), ("width", "width"),
        ("height", "height"), ("seed", "seed"), ("steps", "steps"), ("cfg", "cfg"),
        ("layout", "layout"), ("accelerate", "accelerate"),
    ),
    "fibo2": (
        ("model_choice", "model"), ("precision", "precision"), ("width", "width"),
        ("height", "height"), ("edit_size", "edit_size"), ("seed", "seed"),
        ("steps", "steps"), ("cfg", "cfg"),
    ),
}
GENERATION_METHODS = {"qwen": "generate_qwen_image21", "flux3": "generate_flux3",
                      "moss_image": "generate_moss_image", "fibo2": "generate_fibo2"}


class ImageGenerationController:
    """Capture named option values and dispatch one immutable image draft.

    Field names are also the queue snapshot's names, so a saved job is readable
    without depending on the ordering of another model's controls.
    """

    def __init__(self, engine_views, handlers, h3, view, deps):
        self.handlers = handlers
        self.deps = deps
        self.fields = {
            f"{engine}__{parameter}": getattr(engine_views[engine], component)
            for engine, options in GENERATION_OPTIONS.items() for parameter, component in options
        }
        self.h3_fields = {
            field: getattr(h3, component)
            for field, component in zip(GENERATION_FIELDS, GENERATION_COMPONENTS, strict=True)
            if field not in {"prompt", "first_image", "last_image"}
            and not field.startswith(("ref_image_", "ref_video_", "ref_audio_", "fl2va_audio_"))
        }
        self.fields.update({f"h3__{name}": component for name, component in self.h3_fields.items()})
        self.fields.update(image__width=view.h3_width, image__height=view.h3_height, image__seed=view.h3_seed)

    def generate(self, engine, mode, prompt, references, captured):
        options = {name: captured[f"{engine}__{name}"] for name, _ in GENERATION_OPTIONS.get(engine, ())}
        issue = draft_issue(engine, mode, prompt, references, variant=options.get("turbo_variant", "Off"),
                            fibo_model=options.get("model_choice", "Dev"),
                            separate=options.get("batch_edit_inputs", False))
        if issue:
            raise H3Error(issue)
        active_references = list(references or []) if mode == "Image edit" else []
        if engine == "h3":
            from h3_app.generation.ph3_image import generate_ph3_image

            settings = {name: captured[f"h3__{name}"] for name in self.h3_fields}
            values = h3_image_values(settings, prompt, active_references, captured["image__width"],
                                     captured["image__height"], captured["image__seed"])
            return generate_ph3_image(H3Request.from_values(values), self.deps.generation_services(),
                                      self.deps.runtime_config())
        options["prompt"] = prompt
        if engine != "moss_image":
            options["reference_images"] = active_references
        if engine == "qwen":
            options["mode"] = mode
        return getattr(self.handlers[engine], GENERATION_METHODS[engine])(**options)


def suggest_prompt(handlers, engine, mode, prompt, references, options):
    """Use the selected model's recipe and actual output dimensions."""
    active_references = references if mode == "Image edit" else []
    writer = dict(backend=options["writer_backend"], openai_api_key=options["writer_openai_key"],
                  openai_model=options["writer_openai_model"], openai_base_url=options["writer_base_url"])
    if engine == "qwen":
        return handlers[engine].enhance_qwen_image21_prompt(
            prompt, options["writer_model"], options["writer_key"], mode, active_references,
            options["qwen_width"], options["qwen_height"], edit_size=options["qwen_edit_size"], **writer)
    if engine == "flux3":
        return handlers[engine].enhance_flux3_prompt(
            prompt, active_references, options["flux3_width"], options["flux3_height"],
            options["writer_model"], options["writer_key"], **writer)[:2]
    if engine == "moss_image" and mode == "Text to image":
        return handlers[engine].enhance_moss_image_prompt(
            prompt, options["moss_image_width"], options["moss_image_height"], options["moss_rewrite_mode"],
            options["writer_model"], options["writer_key"], **writer)[:2]
    raise H3Error("Prompt assistance is unavailable for this model and mode.")
