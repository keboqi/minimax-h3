"""Compose-time prerequisites; inference remains owned by the existing queue."""

import gradio as gr


def bind_engine_readiness(view, family):
    if family == "ltx":
        inputs = [view.prompt, view.mode, view.image, view.reference_images]

        def missing(prompt, mode, image, references):
            if not prompt.strip():
                return "Write a video prompt to start."
            if mode == "Image to video" and not image:
                return "Add a start keyframe to generate this video."
            if mode == "Reference images" and not references:
                return "Add reference images to generate this video."
            return ""

    elif family == "qwen":
        inputs = [view.prompt, view.mode, view.reference_images]

        def missing(prompt, mode, references):
            if not prompt.strip():
                return "Write an image prompt or edit instruction to start."
            if mode == "Image edit" and not references:
                return "Add an image to edit."
            return ""

    else:
        inputs = [view.caption if family == "music" else view.style, view.lyrics]

        def missing(description, lyrics):
            return (
                ""
                if description.strip() or lyrics.strip()
                else "Write a style or lyrics to start."
            )

    def readiness(*values):
        message = missing(*values)
        return (
            f'<p class="h3-readiness" role="{"alert" if message else "status"}">{message or "Ready to generate."}</p>',
            gr.update(interactive=not message),
        )

    gr.on(
        triggers=[component.change for component in inputs],
        fn=readiness,
        inputs=inputs,
        outputs=[view.run.h3_readiness, view.run],
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_name=False,
    )
    if family == "yue2":
        view.mode.change(
            lambda mode: gr.update(interactive=mode != "off"),
            inputs=view.mode,
            outputs=view.abc,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
