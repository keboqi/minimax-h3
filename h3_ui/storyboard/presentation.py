"""HTML for the storyboard strip and the full-video timeline."""

from __future__ import annotations

from html import escape

from h3_app import storyboard

# Clicks inside the strip report which control was pressed and for which scene.
STRIP_JS = """
const press = (control) => {
  if (!control) return;
  trigger('click', {action: control.dataset.sbAction, scene_id: control.dataset.sceneId || ''});
};
element.addEventListener('click', event => press(event.target.closest('[data-sb-action]')));
"""

STATE_LABELS = {storyboard.DONE: "Done", storyboard.OUTDATED: "Outdated", storyboard.DRAFT: "Draft"}


def seconds_label(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") + " s"


def clock(value: float) -> str:
    total = int(round(value))
    return f"{total // 60}:{total % 60:02d}"


def strip_html(board: dict | None, selected: str | None) -> str:
    if board is None:
        return '<p class="h3-sb-empty">Create a storyboard to start adding scenes.</p>'
    states = storyboard.scene_states(board)
    items = []
    for index, (scene, state) in enumerate(zip(board["scenes"], states)):
        if index:
            flow = scene["link"] == storyboard.FLOW
            label = "Flow" if flow else "Cut"
            items.append(
                f'<button type="button" class="h3-sb-link h3-sb-link-{scene["link"]}" data-sb-action="link" '
                f'data-scene-id="{escape(scene["id"])}" aria-label="Scene {index + 1} {label.lower()}s from '
                f'scene {index}. Switch to {"cut" if flow else "flow"}.">{label}</button>'
            )
        takes = len(scene["takes"])
        in_use = scene["takes"].index(scene["in_use"]) + 1 if scene["in_use"] in scene["takes"] else 0
        thumb = f"Take {in_use} of {takes}" if in_use else "Not generated"
        added = storyboard.new_seconds(board, index)
        length = seconds_label(scene["seconds"])
        if added != scene["seconds"]:
            length += f" · {seconds_label(added)} new"
        items.append(
            f'<button type="button" class="h3-sb-card" data-sb-action="select" data-scene-id="{escape(scene["id"])}" '
            f'aria-pressed="{str(scene["id"] == selected).lower()}">'
            f'<span class="h3-sb-thumb{" h3-sb-thumb-empty" if not in_use else ""}">{escape(thumb)}</span>'
            f'<span class="h3-sb-meta">Scene {index + 1} · {escape(length)}</span>'
            f'<span class="h3-sb-title">{escape(scene["title"] or "Untitled scene")}</span>'
            f'<span class="h3-sb-state h3-sb-{state}">{STATE_LABELS[state]}</span></button>'
        )
    items.append('<button type="button" class="h3-sb-add" data-sb-action="add">+ Add scene</button>')
    return '<div class="h3-sb-strip">' + "".join(items) + "</div>"


def timeline_html(board: dict | None, selected: str | None) -> str:
    if board is None:
        return ""
    states = storyboard.scene_states(board)
    segments = []
    for index, (scene, state) in enumerate(zip(board["scenes"], states)):
        added = max(storyboard.new_seconds(board, index), 0.5)
        classes = ["h3-sb-segment", f"h3-sb-{state}"]
        if index and scene["link"] == storyboard.CUT:
            classes.append("h3-sb-segment-cut")
        if scene["id"] == selected:
            classes.append("h3-sb-segment-selected")
        segments.append(
            f'<button type="button" class="{" ".join(classes)}" style="flex: {added:.2f} 1 0" '
            f'data-sb-action="select" data-scene-id="{escape(scene["id"])}" '
            f'aria-label="Scene {index + 1}, {seconds_label(added)}, {STATE_LABELS[state]}">{index + 1}</button>'
        )
    done = states.count(storyboard.DONE)
    return (
        '<div class="h3-sb-timeline">' + "".join(segments) + "</div>"
        f'<p class="h3-sb-timeline-meta"><span>0:00</span><span>{len(states)} scenes · {done} done</span>'
        f'<span>{clock(storyboard.total_seconds(board))}</span></p>'
    )


def cast_html(board: dict | None) -> str:
    if board is None or not board["cast"]:
        return '<p class="h3-sb-empty">No cast yet. Add reference pictures so people look the same across cuts.</p>'
    return '<ul class="h3-sb-cast">' + "".join(
        f'<li>{escape(member["name"])}</li>' for member in board["cast"]
    ) + "</ul>"
