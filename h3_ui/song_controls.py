"""Song-specific building blocks for the guided Music 3 and YuE2 views."""

from __future__ import annotations

from html import escape

import gradio as gr

from h3_app import song_brief

# The four steps every Song view walks through, in order.
IDEA, DRAFT, TAKE, LISTEN = range(4)


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
        "", label="What should change? (optional)", lines=2, max_lines=4,
        placeholder="Say what you hear, for example: the chorus feels flat, too busy, too slow",
        info="Revise the draft sends this to the writer, and so does any later Write until you "
             "clear it. Only what you mention changes.",
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


def render_recap(*, engine: str, sound_label: str, description: str, lyrics: str,
                 max_seconds: float, vocals: str = song_brief.AUTO,
                 language: str = song_brief.AUTO, use: str = song_brief.AUTO) -> str:
    """What the next take will use: the sound, the voice, and the song check."""
    sound = " ".join(description.split())
    if len(sound) > 220:
        sound = sound[:219] + "…"
    if vocals == song_brief.INSTRUMENTAL:
        voice = "Instrumental"
    else:
        voice = " · ".join((
            "Voice chosen from your idea" if vocals == song_brief.AUTO else vocals,
            "language from your idea" if language in (song_brief.AUTO, "", None) else str(language),
        ))
    if use not in (song_brief.AUTO, "", None):
        voice += f" · {use}"
    rows = (
        (sound_label, escape(sound) if sound else
         '<em>Not described yet. Add it on the Idea step, or press Write the song.</em>'),
        ("Voice", escape(voice)),
    )
    check = song_brief.render_check(
        engine=engine, description=description, lyrics=lyrics,
        max_seconds=max_seconds, vocals=vocals, use=use,
    )
    return (
        '<div class="h3-song-recap"><dl>'
        + "".join(f"<div><dt>{escape(name)}</dt><dd>{value}</dd></div>" for name, value in rows)
        + f"</dl>{check}</div>"
    )


def bind_song_check(view, engine: str, description: gr.Textbox, sound_label: str) -> None:
    """Re-check the draft in the browser session on every edit; no GPU, no network."""
    inputs = [description, view.lyrics, view.duration, view.vocals, view.use, view.language]

    def check(text, lyrics, duration, vocals, use, language):
        values = dict(
            engine=engine, description=text or "", lyrics=lyrics or "",
            max_seconds=float(duration or 1), vocals=vocals, use=use,
        )
        return (
            song_brief.render_check(**values),
            render_recap(sound_label=sound_label, language=language, **values),
        )

    gr.on(
        triggers=[component.change for component in inputs],
        fn=check, inputs=inputs, outputs=[view.song_check, view.recap],
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
