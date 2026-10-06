"""Lazy media-library pickers for image and multi-image inputs."""

from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from .media_previews import browser_gallery_items


@dataclass(frozen=True)
class ImageLibraryPicker:
    panel: gr.Accordion
    query: gr.Textbox
    refresh: gr.Button
    grid: gr.Gallery
    paths: gr.State
    shown: gr.State
    more: gr.Button
    status: gr.Markdown


def _build_input(component_type, **kwargs):
    visible = kwargs.get("visible", True)
    with gr.Column(visible=visible, elem_classes=["h3-image-library-input"]) as container:
        component = component_type(**kwargs)
        with gr.Accordion("Choose from media library", open=False) as panel:
            query = gr.Textbox(
                value="", label="Search library images", placeholder="Filename, tags or settings"
            )
            refresh = gr.Button("Search / refresh images", size="sm")
            grid = gr.Gallery(
                value=[], label="Library images", columns=3, height=240,
                object_fit="cover", allow_preview=False, interactive=False,
            )
            paths = gr.State([])
            shown = gr.State(24)
            more = gr.Button("Show more images", size="sm", interactive=False)
            status = gr.Markdown("Select a thumbnail to use its original image.")
    component.h3_library_container = container
    component.h3_library_picker = ImageLibraryPicker(
        panel, query, refresh, grid, paths, shown, more, status
    )
    return component


def build_image_input(**kwargs):
    return _build_input(gr.Image, **kwargs)


def build_image_file_input(**kwargs):
    return _build_input(gr.File, **kwargs)


def select_library_image(paths, current, event, *, validate_path, multiple=False):
    """Resolve a server-side thumbnail index to a live managed original."""
    index = event.index
    if isinstance(index, (tuple, list)):
        index = index[0] if index else None
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(paths):
        raise gr.Error("Refresh the library and select an image thumbnail.")
    return library_image_value(
        paths[index], current, validate_path=validate_path, multiple=multiple
    )


def library_image_value(source, current, *, validate_path, multiple=False):
    """Use a managed original for either direction of library selection."""
    validate_path(source, "Image")
    if not Path(source).is_file():
        raise gr.Error("This image is unavailable. Refresh the library and choose another.")
    if multiple:
        values = list(current or [])
        if source not in values:
            values.append(source)
        return values
    return source


def build_media_image_destinations(view, destinations, *, validate_path):
    """Send the selected library image to a named generator input."""
    slots = {
        name: list(value) if isinstance(value, (tuple, list)) else [value]
        for name, value in destinations.items()
    }
    targets = list(dict.fromkeys(component for group in slots.values() for component in group))
    reference_targets = [
        component for component in targets if hasattr(component, "h3_reference_slot")
    ]
    shown = reference_targets[0].h3_reference_shown if reference_targets else None
    add_reference = reference_targets[0].h3_reference_add if reference_targets else None
    with view.inspector:
        with gr.Group(visible=False) as group:
            with gr.Row():
                destination = gr.Dropdown(
                    choices=list(destinations), label="Set as", value=None,
                    interactive=False,
                    info="Single images replace; references fill the first empty slot or append to the list.",
                )
                apply = gr.Button("Set image", interactive=False, scale=0)
            status = gr.Markdown()

    def availability(mode, selected):
        active = mode == "Image" and bool(selected)
        return (
            gr.update(visible=mode == "Image"),
            gr.update(interactive=active), gr.update(interactive=active), "",
        )

    view.selected.h3_image_destination_sync = (
        availability, [view.mode, view.selected], [group, destination, apply, status],
    )
    outputs = [*targets, *(c.h3_library_container for c in targets), status]
    if shown is not None:
        outputs.extend([shown, add_reference])

    def assign(mode, selected, name, *values):
        if mode != "Image" or not selected:
            raise gr.Error("Select an image in the Media library first.")
        if name not in destinations:
            raise gr.Error("Choose an image input from Set as.")
        group = slots[name]
        if len(group) > 1:
            component = next((c for c in group if not values[targets.index(c)]), None)
            if component is None:
                raise gr.Error(f"{name} is full. Clear a reference slot in Create first.")
        else:
            component = group[0]
        current = values[targets.index(component)]
        multiple = isinstance(component, gr.File) and component.file_count == "multiple"
        value = library_image_value(
            selected, current, validate_path=validate_path, multiple=multiple
        )
        updates = {
            component: gr.update(value=value, visible=True),
            component.h3_library_container: gr.update(visible=True),
            status: (
                f"Added image to {name} · {component.label}. Open Create to use it."
                if len(group) > 1 else
                f"{'Added image to' if multiple else 'Set image as'} {name}. Open Create to use it."
            ),
        }
        if component in reference_targets:
            count = max(int(values[-1]), component.h3_reference_slot)
            updates[shown] = count
            updates[add_reference] = gr.update(interactive=count < len(reference_targets))
            for reference in reference_targets[:count]:
                updates.setdefault(reference, gr.update(visible=True))
                updates[reference.h3_library_container] = gr.update(visible=True)
        return updates

    apply.click(
        assign,
        inputs=[view.mode, view.selected, destination, *targets]
        + ([shown] if shown is not None else []),
        outputs=outputs, queue=True, concurrency_id="h3-settings",
        concurrency_limit=1, api_name=False, show_progress="hidden",
    )


def bind_image_library(component, *, index, refresh_page, validate_path):
    picker = component.h3_library_picker
    outputs = [picker.grid, picker.paths, picker.shown, picker.more, picker.status]

    def browse(query, limit=24, request: gr.Request = None):
        page = refresh_page("Image", limit, paths=index.inventory("Image", query or "", limit=limit))
        return (
            browser_gallery_items(page.items, request, version=index.versions.get("Image")), list(page.paths), limit,
            gr.update(interactive=page.next_cursor is not None),
            f"{page.total} matching images. Select a thumbnail to "
            + ("append it to the inputs." if isinstance(component, gr.File) else "use it."),
        )

    gr.on(
        triggers=[picker.panel.expand, picker.refresh.click, picker.query.submit],
        fn=browse, inputs=picker.query, outputs=outputs,
        queue=False, api_name=False, show_progress="hidden",
        trigger_mode="always_last",
    )
    def more(query, shown, request: gr.Request):
        return browse(query, shown + 24, request=request)

    picker.more.click(
        more,
        inputs=[picker.query, picker.shown], outputs=outputs,
        queue=False, api_name=False, show_progress="hidden",
    )

    def choose(paths, current, evt: gr.SelectData):
        return select_library_image(
            paths, current, evt, validate_path=validate_path,
            multiple=isinstance(component, gr.File) and component.file_count == "multiple",
        )

    picker.grid.select(
        choose, inputs=[picker.paths, component], outputs=component,
        queue=False, api_name=False, show_progress="hidden",
    )
