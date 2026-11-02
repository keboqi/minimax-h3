"""HTML for the scene list and the full-video timeline."""

from __future__ import annotations

from html import escape

from h3_app import storyboard as sb

# Clicks report which control was pressed and for which scene.
CLICK_JS = """
const press = (control) => {
  if (!control) return;
  trigger('click', {action: control.dataset.sbAction, scene_id: control.dataset.sceneId || ''});
};
element.addEventListener('click', event => press(event.target.closest('[data-sb-action]')));
"""

MODE_NAMES = {sb.TEXT: "text only", sb.FRAMES: "picture frames", sb.REFERENCES: "references"}


def seconds_label(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") + " s"


def clock(value: float) -> str:
    total = int(round(value))
    return f"{total // 60}:{total % 60:02d}"


def scene_summary(board: dict, index: int) -> str:
    scene = board["scenes"][index]
    start = "Opening" if index == 0 else f"Continues scene {index}" if sb.flows_from_previous(board, index) \
        else "Fresh start"
    mode = sb.scene_media(scene)["mode"]
    name = "last frame" if mode == sb.FRAMES and sb.flows_from_previous(board, index) else MODE_NAMES[mode]
    return f"{start} · {name}"


def scene_status(board: dict, index: int, state: str) -> tuple[str, str]:
    """The card's status label and CSS state; drafts that can't run yet say so."""
    if state == sb.DRAFT:
        own = [issue for issue in sb.scene_issues(board, index) if not issue.startswith("Generate scene")]
        if own:
            return "Needs input", "needs"
    return sb.STATE_NAMES[state], state


def scene_list_html(board: dict | None, selected: str | None) -> str:
    if board is None:
        return ""
    states = sb.scene_states(board)
    items = []
    for index, (scene, state) in enumerate(zip(board["scenes"], states)):
        if index:
            flow = sb.flows_from_previous(board, index)
            label = "↓ continues into" if flow else "✂ cut to a fresh start"
            items.append(f'<div class="h3-sb-join h3-sb-join-{"flow" if flow else "cut"}">{label}</div>')
        label, css = scene_status(board, index, state)
        items.append(
            f'<button type="button" class="h3-sb-scene" data-sb-action="select" data-scene-id="{escape(scene["id"])}" '
            f'aria-pressed="{str(scene["id"] == selected).lower()}">'
            f'<span class="h3-sb-scene-top"><span class="h3-sb-title">{index + 1} · '
            f'{escape(scene["title"] or "Untitled scene")}</span>'
            f'<span class="h3-sb-meta">{seconds_label(scene["seconds"])}</span></span>'
            f'<span class="h3-sb-meta">{escape(scene_summary(board, index))}</span>'
            f'<span class="h3-sb-state h3-sb-{css}">{label}</span></button>'
        )
    items.append('<button type="button" class="h3-sb-add" data-sb-action="add">+ Add a scene</button>')
    done = states.count(sb.DONE)
    return (f'<p class="h3-sb-list-meta">{done} of {len(states)} done</p>'
            '<div class="h3-sb-list">' + "".join(items) + "</div>")


def timeline_html(board: dict | None, selected: str | None) -> str:
    if board is None:
        return ""
    states = sb.scene_states(board)
    segments = []
    for index, (scene, state) in enumerate(zip(board["scenes"], states)):
        added = max(sb.new_seconds(board, index), 0.5)
        classes = ["h3-sb-segment", f"h3-sb-{state}"]
        if index and not sb.flows_from_previous(board, index):
            classes.append("h3-sb-segment-cut")
        if scene["id"] == selected:
            classes.append("h3-sb-segment-selected")
        segments.append(
            f'<button type="button" class="{" ".join(classes)}" style="flex: {added:.2f} 1 0" '
            f'data-sb-action="select" data-scene-id="{escape(scene["id"])}" '
            f'aria-label="Scene {index + 1}, {seconds_label(added)}, {sb.STATE_NAMES[state]}">{index + 1}</button>'
        )
    return (
        '<div class="h3-sb-timeline">' + "".join(segments) + "</div>"
        f'<p class="h3-sb-timeline-meta"><span>0:00</span>'
        f'<span>{states.count(sb.DONE)} of {len(states)} scenes done</span>'
        f'<span>{clock(sb.total_seconds(board))}</span></p>'
        '<p class="h3-sb-legend"><span class="h3-sb-key h3-sb-done"></span>Done'
        '<span class="h3-sb-key h3-sb-outdated"></span>Out of date'
        '<span class="h3-sb-key h3-sb-draft"></span>Not generated'
        '<span class="h3-sb-key h3-sb-key-cut"></span>Fresh start (cut)</p>'
    )


def cast_html(board: dict | None) -> str:
    if board is None or not board["cast"]:
        return ('<p class="h3-sb-empty">No cast yet. Add a picture of each person so they look the same in '
                'every scene that uses Reference media.</p>')
    return '<ul class="h3-sb-cast">' + "".join(
        f'<li>{escape(member["name"])}</li>' for member in board["cast"]
    ) + "</ul>"
