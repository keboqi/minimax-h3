"""Canvas controls delegate dimension decisions to existing engine policy."""

import json

import gradio as gr


def bind_image_canvas_preview(output, first, result_format, conditioning):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from PIL import Image
    from h3_app.image_canvas import transform_canvas

    def preview(path, result, mode, canvas_mode, width, height):
        if (
            result != "Image"
            or mode != "First / last frame"
            or not path
            or canvas_mode == "Input-derived"
        ):
            return (
                gr.update(value=None, visible=False),
                "Input-derived is the default. Fit/pad or fill/crop applies only to a conditioned Image request.",
            )
        try:
            with TemporaryDirectory(prefix="h3-canvas-preview-") as directory:
                derived = transform_canvas(
                    path, Path(directory) / "preview.png", canvas_mode, width, height
                )
                with Image.open(derived) as image:
                    display = image.copy()
            return (
                gr.update(value=display, visible=True),
                f"{canvas_mode} · {int(width)}×{int(height)}. The original upload is preserved.",
            )
        except (ValueError, OSError) as exc:
            return gr.update(value=None, visible=False), str(exc)

    gr.on(
        triggers=[
            first.change,
            result_format.change,
            conditioning.change,
            *(c.input for c in output.canvas_controls),
        ],
        fn=preview,
        inputs=[first, result_format, conditioning, *output.canvas_controls],
        outputs=[output.canvas_preview, output.canvas_status],
        queue=False,
        api_name=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )


def bind_canvas_controls(components, resolve_dimensions, controller, presets):
    if components.aspect_ratio is None:
        return
    # This catalog comes from engine-owned constants, never formatted status text.
    choices = {
        tier: {name.split("·", 1)[0].strip(): name for name in source}
        for tier, source in presets.items()
    }

    def choose(ratio, tier, latent_upscale, result_format, mode, first):
        if result_format == "Image" and mode == "First / last frame" and first:
            return (
                gr.skip(),
                gr.skip(),
                "Input-derived image canvas; the first frame determines aligned dimensions.",
            )
        if tier == "custom" or ratio == "Custom":
            return gr.skip(), gr.skip(), "Use Width and Height in Fine-tune → Size, seed & canvas."
        name = choices.get(tier, {}).get(ratio)
        if name is None:
            return (
                gr.skip(),
                gr.skip(),
                "This aspect ratio is unavailable at the selected size. Use exact dimensions.",
            )
        # Policy owns alignment, supported bounds and latent-upscale behavior.
        return resolve_dimensions(name, tier, latent_upscale, result_format)

    def choose_settings(ratio, tier, latent_upscale, result_format, mode, first):
        width, height, info = choose(
            ratio, tier, latent_upscale, result_format, mode, first
        )
        overrides = {
            name: value
            for name, value in (("width", width), ("height", height))
            if not (isinstance(value, dict) and value.get("__type__") == "update")
        }
        return overrides, (info,)

    # One response resolves the preset dimensions and the settings they feed.
    controller.bind_refresh(
        [components.aspect_ratio.input, components.size_tier.input],
        choose_settings,
        inputs=[
            components.aspect_ratio,
            components.size_tier,
            components.latent_upscale,
            components.result_format,
            components.mode,
            components.first,
        ],
        extra_outputs=[components.resolution_info],
    )

    # Labelling the current dimensions needs only the preset catalog, so it
    # runs in the browser instead of costing a request per dimension change.
    catalog = [
        [name.split("·", 1)[0].strip(), tier, *map(int, dimensions)]
        for tier, source in presets.items()
        for name, dimensions in source.items()
    ]
    sync = f"""(width, height, resultFormat, mode, first) => {{
        const presets = {json.dumps(catalog)};
        const locked = resultFormat === "Image" && mode === "First / last frame" && !!first;
        const interactive = !locked && resultFormat !== "Audio";
        const match = presets.find(([, , w, h]) => w === width && h === height);
        return [
            {{__type__: "update", value: match ? match[0] : "Custom", interactive}},
            {{__type__: "update", value: match ? match[1] : "custom", interactive}},
        ];
    }}"""
    gr.on(
        triggers=[
            c.change
            for c in (
                components.width,
                components.height,
                components.result_format,
                components.mode,
                components.first,
            )
        ],
        fn=None,
        js=sync,
        inputs=[
            components.width,
            components.height,
            components.result_format,
            components.mode,
            components.first,
        ],
        outputs=[components.aspect_ratio, components.size_tier],
        queue=False,
        api_name=False,
        show_progress="hidden",
    )
