"""Rich result views shared by Soundstage, Dub & Translate and Media.

A Soundstage result shows its lines and sounds on a timeline, subtitles that follow the
player and the stage, drawn from its settings snapshot. A dub plays in its own result
player from the tracks saved beside its outputs. The generate flows show them as results
arrive; Media shows them again for the selected file and hides them for other media.
Free of the generation stack, so the GPU-free Library Console uses them too.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from h3_app.drama3.results import DRAMA3_FAMILY, result_settings
from h3_app.dubbing.results import DUB_FAMILY, Player, load_player
from h3_app.provenance import read_snapshot
from h3_ui.drama3.result_view import DRAMA3_SUBTITLES_JS, subtitles_html
from h3_ui.dubbing.player import ELEM_CLASS as PLAYER_CLASS, PLAYER_JS, player_html

ELEM_CLASS = "h3-result-view"


def soundstage_panel() -> gr.HTML:
    """Hidden until a result with line timings is shown; the page loads its script."""
    return gr.HTML("", visible=False, elem_classes=["h3-drama3-subtitle-panel", ELEM_CLASS])


def dub_panel() -> gr.HTML:
    return gr.HTML("", visible=False, js_on_load=PLAYER_JS, elem_classes=[PLAYER_CLASS, ELEM_CLASS])


def load_result_view_scripts(blocks: gr.Blocks) -> None:
    """Soundstage views follow their player, so the script must register media before any loads."""
    blocks.load(None, js=DRAMA3_SUBTITLES_JS, api_name=False)


def panel_update(value: str):
    return gr.update(value=value, visible=bool(value))


def _media_path(path) -> Path | None:
    """A file path, from Gradio's cached-file dict or a plain value."""
    if isinstance(path, dict):
        path = path.get("path") or path.get("value")
    return Path(str(path)) if path else None


def soundstage_html(path, settings: dict | None = None) -> str:
    """A Soundstage result's view, or "" when it has no line timings."""
    path = _media_path(path)
    if path is None:
        return ""
    return subtitles_html(result_settings(path) if settings is None else settings, path)


def dub_html(player: Player | None, *, scope: str) -> str:
    """A dub's result player; ``scope`` keeps its controls apart from the same dub shown elsewhere."""
    # Each dub's player tracks sit in a folder named after its outputs.
    return player_html(player, key=f"{scope}-{player.original.parent.name}") if player is not None else ""


def result_view_values(selected) -> tuple[str, str]:
    """(Soundstage view, dub player) HTML for the media selected in Media; "" where it has none."""
    path = _media_path(selected)
    snapshot = read_snapshot(path) if path else None
    family = (snapshot or {}).get("family")
    if family == DRAMA3_FAMILY:
        try:
            return soundstage_html(path, snapshot.get("settings") or {}), ""
        except (TypeError, ValueError, KeyError):
            return "", ""
    if family == DUB_FAMILY:
        return "", dub_html(load_player(path), scope="media")
    return "", ""


def result_view_updates(selected):
    """Updates for (Soundstage view, dub player, video preview, audio preview) in Media.

    A dub's player has its own picture and voices, so it replaces the plain preview; selecting
    other media shows that preview again. A Soundstage view follows the audio preview.
    """
    soundstage, dub = result_view_values(selected)
    preview = gr.update(visible=False) if dub else gr.skip()
    return panel_update(soundstage), panel_update(dub), preview, preview
