"""Shared building blocks for the guided Song views (Music 3 and YuE2)."""

from __future__ import annotations

from html import escape

import gradio as gr

from h3_app import song_brief


def song_intro(eyebrow: str, title: str, detail: str, badge: str) -> gr.HTML:
    return gr.HTML(
        f'<div class="h3-song-intro"><div><p class="h3-eyebrow">{eyebrow}</p>'
        f"<h2>{escape(title)}</h2><p>{detail}</p></div>"
        f'<span class="h3-song-badge">{escape(badge)}</span></div>'
    )


def panel_heading(step: str, title: str, detail: str, *, classes: str = "") -> gr.HTML:
    names = " ".join(("h3-panel-heading", classes)).strip()
    return gr.HTML(
        f'<div class="{names}"><span class="h3-step" aria-hidden="true">'
        f"{escape(step)}</span><div><h3>{escape(title)}</h3><p>{escape(detail)}</p></div></div>"
    )


def duration_presets(slider: gr.Slider, presets) -> None:
    """One-click maximum-length buttons; each sets the slider in the browser."""
    with gr.Row(elem_classes=["h3-song-duration-presets"]):
        for seconds, label in presets:
            gr.Button(label, size="sm", min_width=50).click(
                None, outputs=slider, js=f"() => {int(seconds)}", queue=False, api_name=False,
            )


def song_choices() -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    """Three plain questions that stand in for a producer's brief."""
    with gr.Row(elem_classes=["h3-song-choices"]):
        vocals = gr.Dropdown(
            list(song_brief.VOCAL_CHOICES), value=song_brief.AUTO, label="Who sings",
            info="Auto picks a voice that suits the idea.", min_width=150,
        )
        language = gr.Dropdown(
            list(song_brief.LANGUAGE_CHOICES), value=song_brief.AUTO, label="Lyric language",
            info="Auto follows your idea's language.", allow_custom_value=True, min_width=150,
        )
        use = gr.Dropdown(
            list(song_brief.USE_CHOICES), value=song_brief.AUTO, label="Made for",
            info="Shapes structure and ending.", min_width=150,
        )
    return vocals, language, use


# Listener words; the writer prompts translate each into a specific musical change.
QUICK_FIXES = (
    "Stronger chorus", "More personal", "Easier to sing along", "The voice gets lost",
    "More bounce, same pace", "Get to the good part sooner", "Calmer",
)


def change_request() -> gr.Textbox:
    notes = gr.Textbox(
        "", label="What should change? (optional)", lines=1, max_lines=3,
        placeholder="Say what you hear, for example: the chorus feels flat, too busy, too slow",
        info="Used when you press Write again on an existing draft. Only what you mention changes.",
    )
    with gr.Row(elem_classes=["h3-song-quick-fixes"]):
        for label in QUICK_FIXES:
            gr.Button(label, size="sm", min_width=0).click(
                None, inputs=notes, outputs=notes, queue=False, api_name=False,
                js=f"(notes) => notes && notes.trim() ? notes.trim() + ', ' + {label.lower()!r} : {label!r}",
            )
    return notes


def lyrics_policy() -> gr.Radio:
    return gr.Radio(
        list(song_brief.LYRICS_POLICIES), value=song_brief.KEEP_LYRICS,
        label="Your lyrics when you press Write",
        info="Keep sends your words exactly, though a take can still mis-sing a word. Polish fixes "
             "structure, long lines and rhymes but keeps your meaning; you review every change.",
        elem_classes=["h3-song-lyrics-policy"],
    )


def song_check_panel(engine: str, max_seconds: float) -> gr.HTML:
    return gr.HTML(
        song_brief.render_check(engine=engine, description="", lyrics="", max_seconds=max_seconds),
        label="Song check", elem_classes=["h3-song-check-panel"],
    )


def bind_song_check(view, engine: str, description: gr.Textbox) -> None:
    """Re-check the draft in the browser session on every edit; no GPU, no network."""
    inputs = [description, view.lyrics, view.duration, view.vocals, view.use]

    def check(text, lyrics, duration, vocals, use):
        return song_brief.render_check(
            engine=engine, description=text or "", lyrics=lyrics or "",
            max_seconds=float(duration or 1), vocals=vocals, use=use,
        )

    gr.on(
        triggers=[component.change for component in inputs],
        fn=check, inputs=inputs, outputs=view.song_check,
        queue=False, api_name=False, show_progress="hidden",
    )


def bind_write_and_generate(
    button: gr.Button,
    enhance_prompt,
    *,
    inputs: list,
    drafts: list,
    status: gr.Textbox,
    review,
    generate_id: str,
) -> None:
    """One click: write, apply without review (Undo stays available), then generate.

    The take starts by pressing the visible Generate button, so it takes the
    same admission, queue, Interrupt and readiness path as a manual click.
    """
    from .job_bindings import PROMPT_QUEUE

    draft_indices = [inputs.index(component) for component in drafts]

    def write(*values):
        result = enhance_prompt(*values)
        message = str(result[-1])
        if message.startswith("Prompt enhancement failed"):
            raise gr.Error(message)
        originals = tuple(values[i] for i in draft_indices)
        suggestions = tuple(result[:-1])
        if suggestions == originals:
            # Nothing to undo; keep any earlier Undo as it was.
            return (*suggestions, message + " The writer kept your draft as it is.",
                    gr.skip(), None, gr.skip(), gr.update(visible=False))
        return (
            *suggestions,
            message + " Applied without review.",
            originals,
            None,
            gr.update(visible=True),
            gr.update(visible=False),
        )

    button.click(
        write,
        inputs=inputs,
        outputs=[*drafts, status, review.previous, review.proposal, review.undo, review.panel],
        show_progress="minimal",
        api_name=False,
        **PROMPT_QUEUE,
    ).success(
        None,
        inputs=status,
        outputs=status,
        api_name=False,
        queue=False,
        # Readiness re-enables Generate once the new draft lands; wait briefly for
        # it, and say so when it stays disabled (a take is still running).
        js=(
            "async (status) => { for (let i = 0; i < 50; i++) { "
            f"const b = document.getElementById({generate_id!r}); "
            "if (b && !b.disabled) { b.click(); return status + ' Starting a take.'; } "
            "await new Promise((r) => setTimeout(r, 100)); } "
            "return status + ' No take started: Generate is not available yet. "
            "Press Generate when the current take finishes.'; }"
        ),
    )
