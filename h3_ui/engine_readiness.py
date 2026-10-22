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

    elif family == "music":
        inputs = [view.caption, view.lyrics]

        # Music 3 rejects a request without a caption; lyrics alone need a description first.
        def missing(caption, lyrics):
            if caption.strip():
                return ""
            if lyrics.strip():
                return "Add a music description, or use Write / improve song to draft one from your lyrics."
            return "Describe your song or add lyrics to start."

    else:
        inputs = [view.style, view.lyrics]

        # YuE2 rejects a request without both, so Generate waits for both.
        def missing(style, lyrics):
            if not style.strip() and not lyrics.strip():
                return "Describe your song and add lyrics to start."
            if not lyrics.strip():
                return "Add lyrics or section tags, or use Write / improve song to draft them."
            if not style.strip():
                return "Add a style prompt, or use Write / improve song to suggest one."
            return ""

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
