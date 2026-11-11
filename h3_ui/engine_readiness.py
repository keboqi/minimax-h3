"""Compose-time prerequisites; inference remains owned by the existing queue."""

import gradio as gr


def song_issue(family: str, description: str, lyrics: str) -> str:
    """What a song still needs before Generate, or "" when it may start; the Song page asks too."""
    description, lyrics = str(description or "").strip(), str(lyrics or "").strip()
    if family == "music":
        # Music 3 rejects a request without a caption; lyrics alone need a description first.
        if description:
            return ""
        if lyrics:
            return "Add a music description in the brief, or press Write the song to draft one from your lyrics."
        return "Describe your song in the brief, or add lyrics, to start."
    # YuE2 rejects a request without both, so Generate waits for both.
    if not description and not lyrics:
        return "Describe your song in the brief and add lyrics to start."
    if not lyrics:
        return "Add lyrics or section tags to the draft, or press Write the song to draft them."
    if not description:
        return "Add a style in the brief, or press Write the song to suggest one."
    return ""


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
            if not (prompt or "").strip():
                return "Write an image prompt or edit instruction to start."
            if mode == "Image edit" and not references:
                return "Add an image to edit."
            return ""

    elif family == "music":
        inputs = [view.caption, view.lyrics]

        def missing(caption, lyrics):
            return song_issue("music", caption, lyrics)

    else:
        inputs = [view.style, view.lyrics]

        def missing(style, lyrics):
            return song_issue("yue2", style, lyrics)

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
