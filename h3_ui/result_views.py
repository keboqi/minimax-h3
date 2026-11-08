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
from h3_ui.drama3.stage_video import ELEM_CLASS as STAGE_VIDEO_CLASS
from h3_ui.drama3.stage_video import ILLUSTRATED_FAMILY, STAGE_VIDEO_JS, latest_export, stage_video_html
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


def stage_panel() -> gr.HTML:
    """An illustrated video with its mix's subtitles over it; hidden until one is shown."""
    return gr.HTML("", visible=False, js_on_load=STAGE_VIDEO_JS, elem_classes=[STAGE_VIDEO_CLASS, ELEM_CLASS])


def stage_mode_switch() -> gr.Radio:
    """Audio or Video for a Soundstage mix that has an illustrated video; hidden otherwise."""
    return gr.Radio(["Audio", "Video"], value="Audio", show_label=False, visible=False,
                    elem_classes=["h3-drama3-result-mode"])


def result_view_values(selected) -> tuple[str, str, str, bool]:
    """(Soundstage view, dub player, illustrated video, offers a switch) for the media selected in Media.

    A Soundstage mix shows its newest illustrated video behind an Audio / Video switch; a selected
    illustrated video plays with its source mix's subtitles, which sit beside it.
    """
    path = _media_path(selected)
    snapshot = read_snapshot(path) if path else None
    family = (snapshot or {}).get("family")
    try:
        if family == DRAMA3_FAMILY:
            latest = latest_export(path)
            return (soundstage_html(path, snapshot.get("settings") or {}), "",
                    stage_video_html(latest), latest is not None)
        if family == ILLUSTRATED_FAMILY:
            mix = path.with_name(Path(str((snapshot.get("settings") or {}).get("source") or "")).name)
            subtitles = soundstage_html(mix) if mix.name and mix.is_file() else ""
            return subtitles, "", stage_video_html(path), False
    except (OSError, TypeError, ValueError, KeyError):
        return "", "", "", False
    if family == DUB_FAMILY:
        return "", dub_html(load_player(path), scope="media"), "", False
    return "", "", "", False


def result_view_updates(selected):
    """Updates for (Soundstage view, dub player, video preview, audio preview, illustrated video, switch).

    A dub's player has its own picture and voices, so it replaces the plain preview; an illustrated
    video replaces the plain video preview with its captioned one. A mix with an illustrated video
    starts on Audio, with the switch offered. Selecting other media shows the plain preview again.
    """
    soundstage, dub, stage, switch = result_view_values(selected)
    illustrated = bool(stage) and not switch
    video_preview = gr.update(visible=False) if dub or illustrated else gr.skip()
    audio_preview = gr.update(visible=False) if dub else gr.skip()
    return (panel_update(soundstage), panel_update(dub), video_preview, audio_preview,
            gr.update(value=stage, visible=illustrated), gr.update(visible=switch, value="Audio"))


def switch_stage(mode):
    """Show the selected mix's audio preview or its illustrated video."""
    return gr.update(visible=mode != "Video"), gr.update(visible=mode == "Video")
