"""Song-specific building blocks for the Music 3 and YuE2 studio views."""

from __future__ import annotations

import gradio as gr

from h3_app import song_brief

from .step_cards import card_heading

# The studio's three columns, side by side on a wide screen and stacked in this order on a phone.
BRIEF, DRAFT, TAKES = "brief", "draft", "takes"


def studio_column(engine_id: str, part: str, *, scale: int, min_width: int, title: str,
                  detail: str = "") -> gr.Column:
    """One studio column, opened with its heading; enter it with ``with``."""
    column = gr.Column(scale=scale, min_width=min_width, elem_id=f"h3-{engine_id}-{part}",
                       elem_classes=["h3-song-col", f"h3-song-{part}"])
    with column:
        card_heading(title, detail)
    return column


def reveal(trigger: gr.Button, elem_id: str) -> None:
    """Bring a column into view on press; on a wide screen it is already there."""
    trigger.click(
        None, queue=False, api_name=False,
        js=f"() => document.getElementById({elem_id!r})?.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }})",
    )


def press(label: str, elem_id: str, *, variant: str = "secondary") -> gr.Button:
    """A second button for an action that lives elsewhere: clicking it presses ``elem_id``."""
    button = gr.Button(label, variant=variant)
    button.click(None, js=f"() => document.getElementById({elem_id!r})?.click()", queue=False, api_name=False)
    return button


# Section tags the insert buttons add, in the spelling both writers use.
SECTION_INSERTS = ("Verse", "Pre-Chorus", "Chorus", "Bridge", "Instrumental")


def section_inserts(lyrics: gr.Textbox) -> None:
    """Add a section tag at the end of the lyrics, or repeat the last chorus; browser only."""
    end = r"(lyrics) => { const text = (lyrics || '').replace(/\s+$/, ''); return (text ? text + '\n\n' : '') + "
    with gr.Row(elem_classes=["h3-song-sections"]):
        for tag in SECTION_INSERTS:
            gr.Button(f"+ {tag}", size="sm", min_width=0).click(
                None, inputs=lyrics, outputs=lyrics, queue=False, api_name=False,
                js=end + f"'[{tag}]\\n'; }}",
            )
        # The last [Chorus] block, up to the next tag; without one, a new empty chorus.
        gr.Button("Repeat chorus", size="sm", min_width=0).click(
            None, inputs=lyrics, outputs=lyrics, queue=False, api_name=False,
            js=end + r"((text.match(/\[chorus[^\]]*\][^\[]*/gi) || []).pop() || '[Chorus]\n').trim() + '\n'; }",
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


def bind_song_check(
    view, engine: str, description: gr.Textbox,
    *, score: gr.Textbox | None = None, mode: gr.Dropdown | None = None,
    score_words: gr.State | None = None,
) -> None:
    """Re-check the draft in the browser session on every edit; no GPU, no network.

    With ``score_words`` the check also compares the lyrics with the words a
    reused score was sung with.
    """
    inputs = [description, view.lyrics, view.duration, view.vocals, view.use]
    if score is not None and mode is not None and score_words is not None:
        inputs += [score, mode, score_words]

    def check(text, lyrics, duration, vocals, use, *reused):
        return song_brief.render_check(
            engine=engine, description=text or "", lyrics=lyrics or "",
            max_seconds=float(duration or 1), vocals=vocals, use=use,
            score_lyrics=reused_score_lyrics(*reused) if reused else "",
        )

    gr.on(
        triggers=[component.change for component in inputs],
        fn=check, inputs=inputs, outputs=view.song_check,
        queue=False, api_name=False, show_progress="hidden",
    )


def reused_score_lyrics(score: str, mode: str, words) -> str:
    """The words the score in the field was sung with, while it is a reused take's score.

    An edited or pasted score, or No score, leaves nothing to compare.
    """
    if not words or mode == "off":
        return ""
    if str(score or "").strip() != str(words.get("score") or "").strip():
        return ""
    return str(words.get("lyrics") or "")


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
            "if (b && !b.disabled && !b.classList.contains('h3-busy')) { b.click(); return status + ' Starting a take.'; } "
            "await new Promise((r) => setTimeout(r, 100)); } "
            "return status + ' No take started: Generate is not available yet. "
            "Press Generate when the current take finishes.'; }"
        ),
    )
