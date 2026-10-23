"""MOSS-Image prompt writer following OpenMOSS's prompt-enhancer skill."""

from __future__ import annotations

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL
from h3_app.config import RuntimeConfig
from . import common


def enhance_moss_image_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    width: int,
    height: int,
    rewrite_mode: str = "Auto",
    backend: str = "Gemini",
    openai_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
    openai_base_url: str | None = None,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """Expand a MOSS-Image request with OpenMOSS's prompt-enhancer recipe.

    The writer returns a caption or pixel Layout draft, which is validated and
    compiled locally exactly as the official skill's compile_prompt.py does.
    """
    from h3_app.workflows.moss_image_prompt import MOSS_IMAGE_REWRITE_MODES, compile_writer_draft

    request = str(prompt or "").strip()
    if not request:
        return str(prompt or ""), "Prompt enhancement failed: write an image request first."
    if rewrite_mode not in MOSS_IMAGE_REWRITE_MODES:
        return str(prompt or ""), "Prompt enhancement failed: choose Auto, Caption or Layout."
    reply, status = common.write_prompt(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target="MOSS-Image",
        system_path=runtime.prompt_systems["MOSS-Image"],
        backend=backend,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        media_values=(),
        context="",
        request_text=(
            f"TARGET CANVAS: {int(width)}x{int(height)}\n"
            f"DELIVERABLE: {rewrite_mode}\nUSER REQUEST:\n{request}"
        ),
    )
    if status.startswith("Prompt enhancement failed:"):
        return reply, status
    try:
        compiled, layout = compile_writer_draft(reply, width, height, rewrite_mode)
    except ValueError as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}. Try again or choose Caption."
    kind = f"Layout for {int(width)}x{int(height)}" if layout else "caption"
    return compiled, f"{status.removesuffix(' using 0 image(s).')} as a {kind}."
