"""Canvas controls delegate dimension decisions to existing engine policy."""

import gradio as gr


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
        if tier == "custom":
            return gr.skip(), gr.skip(), "Use Width and Height in Exact dimensions."
        name = choices.get(tier, {}).get(ratio)
        if name is None:
            return (
                gr.skip(),
                gr.skip(),
                "This aspect ratio is unavailable at the selected size. Use exact dimensions.",
            )
        # Policy owns alignment, supported bounds and latent-upscale behavior.
        return resolve_dimensions(name, tier, latent_upscale, result_format)

    for component in (components.aspect_ratio, components.size_tier):
        event = component.input(
            choose,
            inputs=[
                components.aspect_ratio,
                components.size_tier,
                components.latent_upscale,
                components.result_format,
                components.mode,
                components.first,
            ],
            outputs=[components.width, components.height, components.resolution_info],
            queue=False,
            api_name=False,
            show_progress="hidden",
        )
        event.then(
            controller.refresh,
            inputs=[controller.memory, *controller.inputs],
            outputs=controller.outputs,
            queue=False,
            api_name=False,
            show_progress="hidden",
        )

    def sync(width, height, result_format, mode, first):
        locked = (
            result_format == "Image" and mode == "First / last frame" and bool(first)
        )
        for tier, source in presets.items():
            for name, dimensions in source.items():
                if tuple(dimensions) == (width, height):
                    return (
                        gr.update(
                            value=name.split("·", 1)[0].strip(),
                            interactive=not locked and result_format != "Audio",
                        ),
                        gr.update(
                            value=tier,
                            interactive=not locked and result_format != "Audio",
                        ),
                    )
        return (
            gr.update(interactive=not locked and result_format != "Audio"),
            gr.update(
                value="custom", interactive=not locked and result_format != "Audio"
            ),
        )

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
        fn=sync,
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
