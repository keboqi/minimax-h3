"""MiniMax's official still-image writing recipe over the shared remote transports."""

from math import gcd
from pathlib import Path

from PIL import Image, ImageOps

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL
from h3_app.errors import H3Error
from h3_app.h3_image import (
    H3_IMAGE_MAX_REFERENCES, h3_image_prompt_path, h3_image_canvas_description, validate_h3_image_canvas,
)
from . import common


def enhance_h3_image_prompt(prompt, model, temporary_api_key, reference_images, width, height,
                            backend="Gemini", openai_api_key="", openai_model=DEFAULT_OPENAI_PROMPT_MODEL,
                            openai_base_url=None, *, runtime=None):
    """Keep picture bindings and original geometry when describing a reference edit."""
    try:
        references = reference_images or []
        if isinstance(references, (str, Path, dict)):
            references = [references]
        if len(references) > H3_IMAGE_MAX_REFERENCES:
            raise H3Error("MiniMax H3 Image supports at most sixteen reference images.")
        validate_h3_image_canvas(width, height)
        width, height = int(width), int(height)
        metadata, media = [], []
        for index, reference in enumerate(references, 1):
            path = common.uploaded_media_path(reference)
            if path is None:
                raise H3Error("A reference image is missing.")
            with Image.open(path) as image:
                w, h = ImageOps.exif_transpose(image).size
            factor = gcd(w, h)
            label = f"<Picture {index}> (original display aspect ratio {w // factor}:{h // factor})"
            media.append((label, path))
            metadata.append(label)
        target = "MiniMax H3 Image edit" if references else "MiniMax H3 Image"
        system_path = getattr(runtime, "prompt_systems", {}).get(target, h3_image_prompt_path(bool(references)))
        return common.write_prompt(
            prompt=prompt, model=model, temporary_api_key=temporary_api_key,
            target=target, system_path=system_path, media_values=tuple(media), context="",
            backend=backend, openai_api_key=openai_api_key, openai_model=openai_model,
            openai_base_url=openai_base_url,
            request_text=(h3_image_canvas_description(width, height) + "\n"
                          + "\n".join(metadata) + f"\nUser request:\n{prompt}"),
        )
    except (H3Error, OSError, ValueError) as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}"
