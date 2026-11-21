"""Image draft policies and named adapters to existing generation handlers."""

from math import log
from PIL import Image

from h3_app.catalog import H3_EDIT_SIZE_CAP, H3_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MANUAL
from h3_app.contracts import GENERATION_COMPONENTS, GENERATION_FIELDS
from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled, variant_indices, variant_seed
from h3_app.generation.results import GenerationUpdate
from h3_app.generation.requests import H3Request
from h3_models import qwen_image21_reference_limit
from h3_app.h3_image import H3_IMAGE_OFFICIAL, H3_IMAGE_PRUNA, H3_IMAGE_EDIT_SIZE_MAX, H3_IMAGE_MAX_REFERENCES, official_edit_dimensions
from h3_app.gpt_image import (
    DEFAULT_GPT_IMAGE_MODEL, GPT_IMAGE_MAX_REFERENCES, GPT_IMAGE_QUALITY, GPT_IMAGE_SIZES,
    normalize_gpt_image_endpoints, normalize_gpt_image_models, validate_gpt_image_settings,
)
from h3_app.hosted_image import (
    API_IMAGE_ENGINES, DEFAULT_ASPECT_RATIO, HOSTED_IMAGE_PROVIDERS,
    hosted_aspect_ratios, validate_hosted_image_settings,
)
from .h3.layout import start_frame_generation_resolution

IMAGE_MODELS = ("h3", "qwen", "flux3", "moss_image", "fibo2", *API_IMAGE_ENGINES)
GPT_IMAGE_ASPECT_SIZES = dict(zip(("16:9", "9:16"), GPT_IMAGE_SIZES, strict=True))


def matched_api_aspect(engine, mode, references, settings):
    """Match the first edit reference to the nearest supported aspect ratio."""
    if (engine not in API_IMAGE_ENGINES or mode != "Image edit" or not references
            or not settings.get(f"{engine}__match_first_aspect")):
        return None
    try:
        with Image.open(references[0]) as source:
            width, height = source.size
            if source.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions.")
        source_ratio = log(width / height)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise H3Error("Could not read the first reference image's dimensions.") from exc
    ratios = GPT_IMAGE_ASPECT_SIZES if engine == "gpt_image" else hosted_aspect_ratios(engine)

    def distance(ratio):
        ratio_width, ratio_height = map(int, ratio.split(":"))
        return abs(log(ratio_width / ratio_height) - source_ratio)

    return min(ratios, key=distance)


def image_mode(engine, intent):
    """Generation-only models leave the shared edit draft inactive."""
    return "Text to image" if engine == "moss_image" else intent


def reference_limit(engine, variant="Off", h3_backend=H3_IMAGE_OFFICIAL):
    if engine == "qwen":
        return qwen_image21_reference_limit(variant)
    if engine in HOSTED_IMAGE_PROVIDERS:
        return HOSTED_IMAGE_PROVIDERS[engine].max_references
    return {"h3": 8 if h3_backend == H3_IMAGE_PRUNA else H3_IMAGE_MAX_REFERENCES, "flux3": 10, "moss_image": 0, "fibo2": 4, "gpt_image": GPT_IMAGE_MAX_REFERENCES}[engine]


def reference_tag(engine, index, count, separate=False):
    """How a prompt names the reference at a 1-based position, or None where it has no name.

    A lone Qwen input is <image1> too, so its thumbnail can still add a name to the prompt;
    separate editing gives each run a different single input, so no one name fits them all.
    """
    if engine == "qwen":
        return None if separate and count > 1 else f"<image{index}>"
    return {"h3": f"<Picture {index}>", "flux3": f"<ref_image_{index - 1}>",
            "fibo2": f"<image_{index}>", "gpt_image": f"<image_{index}>"}.get(engine)


def draft_issue(engine, mode, prompt, references, variant="Off", fibo_model="Dev", separate=False, h3_backend=H3_IMAGE_OFFICIAL,
                *, gpt_models=(DEFAULT_GPT_IMAGE_MODEL,), gpt_use_failback=False, gpt_quality=GPT_IMAGE_QUALITY,
                gpt_width=1920, gpt_height=1088, gpt_batch_count=1, gpt_endpoints=None, hosted_options=None):
    if engine not in IMAGE_MODELS:
        return "Choose an image model."
    if mode not in {"Text to image", "Image edit"}:
        return "Choose Generate or Edit."
    editing = mode == "Image edit"
    if editing and engine == "moss_image":
        return "MOSS-Image supports generation only. Choose Generate or another model."
    if not editing and engine == "fibo2" and fibo_model == "Edit Turbo":
        return "Fibo Edit Turbo requires Edit mode. Choose Dev to generate from text."
    if engine == "gpt_image":
        try:
            validate_gpt_image_settings(
                models=gpt_models,
                use_failback=bool(gpt_use_failback),
                endpoints=gpt_endpoints,
                quality=gpt_quality,
                width=gpt_width,
                height=gpt_height,
                batch_count=gpt_batch_count,
            )
        except H3Error as exc:
            return str(exc)
    if engine in HOSTED_IMAGE_PROVIDERS:
        settings = hosted_options or {}
        try:
            validate_hosted_image_settings(
                engine, *hosted_image_options(engine, settings),
                aspect_ratio=settings.get(f"{engine}__aspect_ratio", DEFAULT_ASPECT_RATIO),
            )
        except H3Error as exc:
            return str(exc)
    if not (prompt or "").strip():
        return "Write an image prompt or edit instruction."
    if editing:
        if not references:
            return "Add at least one image to edit."
        limit = reference_limit(engine, variant, h3_backend)
        if not (engine == "qwen" and separate) and len(references) > limit:
            return f"This model supports up to {limit} references per edit. Remove inputs or choose another model."
        try:
            matched_api_aspect(engine, mode, references, hosted_options or {})
        except H3Error as exc:
            return str(exc)
    return ""


def hosted_image_options(engine, fields):
    provider = HOSTED_IMAGE_PROVIDERS[engine]
    return (fields.get(f"{engine}__model_choice", provider.default_model),
            fields.get(f"{engine}__resolution", "1K"),
            fields.get(f"{engine}__batch_count", 1))


def h3_image_size(edit_size, references, width, height, backend=H3_IMAGE_OFFICIAL):
    """Resolve manual dimensions or the first reference ratio within the backend pixel cap."""
    if not references or edit_size == QWEN_EDIT_SIZE_MANUAL:
        return width, height
    if backend == H3_IMAGE_OFFICIAL and edit_size in (H3_IMAGE_EDIT_SIZE_MAX, H3_EDIT_SIZE_MATCH):
        from PIL import Image, ImageOps
        with Image.open(references[0]) as image:
            width, height = ImageOps.exif_transpose(image).size
        return official_edit_dimensions(width, height, maximize=edit_size == H3_IMAGE_EDIT_SIZE_MAX)
    return start_frame_generation_resolution(
        references[:1], alignment=32, auto_megapixels="8 MP",
        edit_size=H3_EDIT_SIZE_MATCH if edit_size == H3_EDIT_SIZE_MATCH else H3_EDIT_SIZE_CAP,
    )


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

    @classmethod
    def api_only(cls, deps, fields):
        return cls({}, {}, None, None, deps, fields=fields)

    def __init__(self, engine_views, handlers, h3, view, deps, *, fields=None):
        self.handlers = handlers
        self.deps = deps
        if fields is not None:
            self.fields = fields
            self.h3_fields = {}
            return
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
        self.fields.update(image__width=view.h3_width, image__height=view.h3_height, image__seed=view.h3_seed,
                           image__edit_size=view.h3_edit_size)
        self.fields.update({f"{'image' if engine == 'h3' else engine}__batch_count": component
                            for engine, component in getattr(view, "batch_counts", {}).items() if engine != "qwen"})
        for name in ("backend", "expand_prompt"):
            component = getattr(view, f"h3_{name}", None)
            if component is not None:
                self.fields[f"image__{name}"] = component
        # Endpoints keep the previous failback checkbox's positional slot in retained jobs.
        for name in ("width", "height", "seed", "models", "endpoints", "quality"):
            component = getattr(view, f"gpt_image_{name}", None)
            if component is not None:
                self.fields[f"gpt_image__{name}"] = component
        self.fields.update(getattr(view, "hosted_fields", {}))

    def generate(self, engine, mode, prompt, references, captured):
        # Qwen and GPT Image already handle concurrency / variations in their native jobs.
        if engine != "qwen" and engine not in API_IMAGE_ENGINES:
            prefix = "image" if engine == "h3" else engine
            count = captured.get(f"{prefix}__batch_count", 1)
            if isinstance(count, bool) or not isinstance(count, (int, float)) or count not in (1, 2, 3, 4):
                raise H3Error("Variations must be between 1 and 4.")
            if count > 1:
                return self._variations(engine, mode, prompt, references, captured, prefix, int(count))
        return self._generate_one(engine, mode, prompt, references, captured)

    def _variations(self, engine, mode, prompt, references, captured, prefix, count):
        import random

        outputs = []
        base_seed = captured[f"{prefix}__seed"]
        for index in variant_indices(count):
            check_cancelled()
            seed = variant_seed(index, lambda: random.randrange(2**63 - 1) if base_seed < 0 else int(base_seed) + index)
            values = {**captured, f"{prefix}__seed": seed}
            current = []
            for result, status in self._generate_one(engine, mode, prompt, references, values):
                if result is not None:
                    current = [result] if isinstance(result, str) else list(result)
                yield GenerationUpdate(outputs + current, f"Image {index + 1}/{count} · {status}")
            check_cancelled()
            if not current:
                return  # Preserve completed images and the failing variation's error status.
            outputs.extend(current)

    def _generate_one(self, engine, mode, prompt, references, captured):
        options = {name: captured[f"{engine}__{name}"] for name, _ in GENERATION_OPTIONS.get(engine, ())}
        backend = captured.get("image__backend", H3_IMAGE_OFFICIAL)
        issue = draft_issue(
            engine, mode, prompt, references, variant=options.get("turbo_variant", "Off"),
            fibo_model=options.get("model_choice", "Dev"),
            separate=options.get("batch_edit_inputs", False), h3_backend=backend,
            gpt_models=captured.get("gpt_image__models", (DEFAULT_GPT_IMAGE_MODEL,)),
            gpt_use_failback=bool(captured.get("gpt_image__use_failback", False)),
            gpt_endpoints=captured.get("gpt_image__endpoints"),
            gpt_quality=captured.get("gpt_image__quality", GPT_IMAGE_QUALITY),
            gpt_width=captured.get("gpt_image__width", 1920),
            gpt_height=captured.get("gpt_image__height", 1088),
            gpt_batch_count=captured.get("gpt_image__batch_count", 1),
            hosted_options=captured,
        )
        if issue:
            raise H3Error(issue)
        active_references = list(references or []) if mode == "Image edit" else []
        matched = matched_api_aspect(engine, mode, active_references, captured)
        if engine in HOSTED_IMAGE_PROVIDERS:
            from h3_app.generation.hosted_image import generate_hosted_image

            return generate_hosted_image(engine, *hosted_image_options(engine, captured),
                                         prompt, active_references, self.deps.runtime_config(),
                                         aspect_ratio=matched or captured.get(f"{engine}__aspect_ratio", DEFAULT_ASPECT_RATIO))
        if engine == "gpt_image":
            from h3_app.generation.gpt_image import GptImageRequest, generate_gpt_image

            endpoints = captured.get("gpt_image__endpoints")
            width, height = GPT_IMAGE_ASPECT_SIZES[matched] if matched else (
                captured.get("gpt_image__width", 1920), captured.get("gpt_image__height", 1088),
            )
            request = GptImageRequest(
                prompt=prompt,
                models=normalize_gpt_image_models(captured.get("gpt_image__models", (DEFAULT_GPT_IMAGE_MODEL,))),
                use_failback=bool(captured.get("gpt_image__use_failback", False)),
                endpoints=normalize_gpt_image_endpoints(endpoints) if endpoints is not None else None,
                quality=str(captured.get("gpt_image__quality", GPT_IMAGE_QUALITY)),
                width=int(width),
                height=int(height),
                batch_count=int(captured.get("gpt_image__batch_count", 1)),
                reference_images=tuple(active_references),
            )
            return generate_gpt_image(request, self.deps.runtime_config())
        if engine == "h3":
            from h3_app.generation.ph3_image import generate_ph3_image

            if backend not in (H3_IMAGE_OFFICIAL, H3_IMAGE_PRUNA):
                raise H3Error("Choose an H3 image backend.")
            width, height = h3_image_size(captured["image__edit_size"], active_references,
                                          captured["image__width"], captured["image__height"], backend)
            if backend == H3_IMAGE_OFFICIAL:
                from h3_app.generation.h3_image import H3ImageRequest, generate_h3_image

                return generate_h3_image(H3ImageRequest(prompt, width, height, captured["image__seed"],
                                          tuple(active_references), captured.get("image__expand_prompt", True)),
                                         self.deps.generation_services(), self.deps.runtime_config())
            settings = {name: captured[f"h3__{name}"] for name in self.h3_fields}
            values = h3_image_values(settings, prompt, active_references, width, height, captured["image__seed"])
            return generate_ph3_image(H3Request.from_values(values), self.deps.generation_services(),
                                      self.deps.runtime_config())
        options["prompt"] = prompt
        if engine != "moss_image":
            options["reference_images"] = active_references
        if engine == "qwen":
            options["mode"] = mode
        return getattr(self.handlers[engine], GENERATION_METHODS[engine])(**options)


def suggest_prompt(handlers, engine, mode, prompt, references, options, *, runtime=None):
    """Use the selected model's recipe and actual output dimensions."""
    active_references = references if mode == "Image edit" else []
    writer = dict(backend=options["writer_backend"], openai_api_key=options["writer_openai_key"],
                  openai_model=options["writer_openai_model"], openai_base_url=options["writer_base_url"])
    if engine == "h3":
        from h3_app.prompt_writers.h3_image import enhance_h3_image_prompt

        width, height = h3_image_size(options.get("h3_edit_size", QWEN_EDIT_SIZE_MANUAL), active_references,
                                      options["h3_width"], options["h3_height"],
                                      options.get("h3_backend", H3_IMAGE_OFFICIAL))
        return enhance_h3_image_prompt(prompt, options["writer_model"], options["writer_key"],
                                        active_references, width, height, runtime=runtime, **writer)
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
