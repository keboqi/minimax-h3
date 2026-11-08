"""Soundstage slot contracts: how many speaker and sound slots a project has, and their API names.

The writer and examples answer with one value per slot, shaped by ``_slot_visibility``.
"""

import gradio as gr


DRAMA3_VOICE_SLOTS = 16


DRAMA3_VISIBLE_SLOTS = 1


DRAMA3_SLOT_INPUT_NAMES = tuple(
    f"speaker_{speaker}_{field}"
    for speaker in range(DRAMA3_VOICE_SLOTS)
    for field in ("voice", "audio", "transcript")
)


DRAMA3_CAST_INPUT_NAMES = tuple(
    f"speaker_{speaker}_{field}"
    for speaker in range(DRAMA3_VOICE_SLOTS)
    for field in ("name", "description", "preview_text")
)


UPLOAD_VOICE = "Upload a clip"


DRAMA3_SOUND_SLOTS = 16


DRAMA3_VISIBLE_SOUND_SLOTS = 1


UPLOAD_SOUND = "Upload a clip"


DRAMA3_SOUND_SLOT_INPUT_NAMES = tuple(
    f"sound_{index}_{field}"
    for index in range(DRAMA3_SOUND_SLOTS)
    for field in ("preset", "audio", "name", "description", "duration")
)


DRAMA3_SAMPLING_PRESETS = {
    "Strict": (0.7, 0.7, 30),
    "Balance": (1.0, 0.9, 0),
    "Expressive": (2.0, 1.0, 30),
}


def _clamp_count(count, total):
    try:
        return max(1, min(total, int(count or 1)))
    except (TypeError, ValueError):
        return 1


def _slot_visibility(count, previous=None, total=DRAMA3_VOICE_SLOTS):
    """Show ``count`` slots; disable Add at the limit and Remove at one slot.

    With the ``previous`` count only slots whose visibility changes are sent:
    each visibility update of an unrendered slot makes Gradio re-walk its layout.
    """
    count = _clamp_count(count, total)
    if previous is not None:
        previous = _clamp_count(previous, total)
        if previous == count:
            return (gr.skip(),) * (total + 3)
    groups = [gr.skip() if previous is not None and (index < count) == (index < previous)
              else gr.update(visible=index < count) for index in range(total)]
    return (count, *groups, gr.update(interactive=count < total), gr.update(interactive=count > 1))


def _sound_slot_visibility(count, previous=None):
    """Show ``count`` sound slots; disable Add at the limit and Remove at one slot."""
    return _slot_visibility(count, previous, DRAMA3_SOUND_SLOTS)

