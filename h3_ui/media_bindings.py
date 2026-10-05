"""Reusable Gradio event-binding helpers."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
import gradio as gr
from h3_app.catalog import (
    LTX25_CQ_ENHANCER,
    LTX25_CQ_IMAGE_ENHANCER,
    LTX25_DEBLUR,
    LTX25_RESTORATION_OPTIONS,
    LTX25_SAME_RESOLUTION_OPTIONS,
    LTX25_SDR_TO_HDR,
)
from .job_bindings import bind_gpu_action, owned_generation, owned_interrupt
from .media_view import GalleryView


def bind_gallery_view(
    view: GalleryView,
    *,
    tab: gr.Tab,
    selected_ltx_model: gr.Dropdown,
    ai_options: set[str],
    seedvr_option: str,
    ltx_option: str,
    refresh: Callable[..., Any],
    select: Callable[..., Any],
    import_video: Callable[..., Any],
    postprocess: Callable[..., Any],
    interrupt: Callable[..., Any],
    delete: Callable[..., Any],
    empty: Callable[..., Any],
    list_paths: Callable[..., Any] | None = None,
) -> None:
    page_size = 48

    filters = view.filters or gr.State({"query": "", "favorite": False})
    index = getattr(view.search, "h3_asset_index", None)
    page_outputs = [view.grid, view.paths, view.status, view.shown, view.show_more]

    def refresh_page(mode: str, filter_values, limit: int = page_size):
        matching = (
            index.inventory(
                mode,
                filter_values.get("query", ""),
                filter_values.get("favorite", False),
            )
            if index
            else None
        )
        page = (
            refresh(mode, limit, paths=matching)
            if matching is not None
            else refresh(mode, limit)
        )
        return (
            list(page.items),
            list(page.paths),
            f"{page.total} matching assets · {page.status}",
            min(max(0, int(limit)), page.total),
            gr.update(interactive=page.next_cursor is not None),
        )

    def sync_more(mode: str, filter_values):
        return refresh_page(mode, filter_values)

    def validate_selection(mode, paths, selected):
        if selected and selected in paths:
            return (gr.skip(),) * 5
        return (
            gr.update(value=None, visible=mode == "Video"),
            gr.update(value=None, visible=mode == "Image"),
            gr.update(value=None, visible=mode == "Audio"),
            "",
            None,
        )

    def synchronize(event, *, refreshed=False):
        if refreshed:
            event = event.then(
                validate_selection,
                inputs=[view.mode, view.paths, view.selected],
                outputs=[
                    view.player,
                    view.image,
                    view.audio,
                    view.download,
                    view.selected,
                ],
                queue=False,
                show_progress="hidden",
                api_name=False,
            )
        callbacks = getattr(view.selected, "h3_sync", None)
        if callbacks:
            fn, inputs, outputs = callbacks
            event.then(
                fn,
                inputs=inputs,
                outputs=outputs,
                queue=False,
                show_progress="hidden",
                api_name=False,
            )
        image_destination_sync = getattr(view.selected, "h3_image_destination_sync", None)
        if image_destination_sync:
            fn, inputs, outputs = image_destination_sync
            event.then(
                fn, inputs=inputs, outputs=outputs, queue=False,
                show_progress="hidden", api_name=False,
            )
        event.then(
            lambda mode, selected: (
                gr.update(interactive=bool(selected) and mode != "Audio"),
                gr.update(interactive=bool(selected)),
            ),
            inputs=[view.mode, view.selected],
            outputs=[view.post_run, view.delete],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        return event

    if view.search is not None:

        def apply_filters(mode, text, favorite, current):
            current.update(query=text.strip(), favorite=bool(favorite))
            return refresh_page(mode, current)

        filtered = gr.on(
            triggers=[view.search.click, view.query.submit, view.favorite_only.input],
            fn=apply_filters,
            inputs=[view.mode, view.query, view.favorite_only, filters],
            outputs=page_outputs,
            queue=False,
            show_progress="hidden",
            api_name=False,
            trigger_mode="always_last",
        )

        synchronize(filtered, refreshed=True)

    ltx_options = {ltx_option} | LTX25_SAME_RESOLUTION_OPTIONS
    video_postprocess_options = [
        choice[1] if isinstance(choice, (tuple, list)) else choice
        for choice in view.postprocess.choices
    ]
    mode_changed = view.mode.change(
        lambda value: (
            gr.update(visible=value == "Video", value=None),
            gr.update(visible=value == "Image", value=None),
            gr.update(visible=value == "Audio", value=None),
            None,
            "",
            gr.update(visible=value == "Video" or True),
            gr.update(value=False),
            gr.update(
                choices=(
                    [seedvr_option, LTX25_CQ_IMAGE_ENHANCER] if value == "Image" else video_postprocess_options
                ),
                value=seedvr_option,
            ),
            gr.update(
                value=(
                    "Enhance selected image"
                    if value == "Image"
                    else "Enhance selected media"
                )
            ),
            gr.update(visible=value != "Audio"),
        ),
        inputs=view.mode,
        outputs=[
            view.player,
            view.image,
            view.audio,
            view.selected,
            view.download,
            view.manage,
            view.confirm_delete,
            view.postprocess,
            view.post_run,
            view.enhance,
        ],
        queue=False,
        show_progress="hidden",
    )
    refreshed = mode_changed.then(
        refresh_page,
        inputs=[view.mode, filters],
        outputs=page_outputs,
        queue=False,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    view.postprocess.change(
        lambda value: (
            gr.update(visible=value in ai_options or value == LTX25_CQ_IMAGE_ENHANCER),
            gr.update(visible=value == seedvr_option),
            gr.update(
                visible=value in ltx_options
                and value not in {LTX25_CQ_ENHANCER, LTX25_SDR_TO_HDR},
                info=(
                    "Describe the source scene; focus restoration instructions are added automatically. Preserves source resolution. Uses the finishing model selected here."
                    if value == LTX25_DEBLUR
                    else (
                        "Describe the source scene; compression artifact removal instructions are added automatically. Preserves source resolution. Uses the finishing model selected here."
                        if value in LTX25_RESTORATION_OPTIONS
                        else "Optional but recommended. Uses the finishing model selected here."
                    )
                ),
            ),
            gr.update(visible=value in ltx_options),
            gr.update(visible=value in ltx_options),
            gr.update(
                visible=(value in ai_options or value == LTX25_CQ_IMAGE_ENHANCER)
                and value not in LTX25_SAME_RESOLUTION_OPTIONS
            ),
            gr.update(visible=value in ltx_options),
        ),
        inputs=view.postprocess,
        outputs=[
            view.ai_settings,
            view.seedvr2_model,
            view.ltx25_prompt,
            view.split_upscale,
            view.split_seconds,
            view.upscale_resolution,
            selected_ltx_model,
        ],
        queue=False,
        show_progress="hidden",
    )
    opened = tab.select(
        lambda value: (
            gr.update(value=None, visible=value == "Video"),
            gr.update(value=None, visible=value == "Image"),
            gr.update(value=None, visible=value == "Audio"),
            "",
            None,
            False,
        ),
        inputs=view.mode,
        outputs=[
            view.player,
            view.image,
            view.audio,
            view.download,
            view.selected,
            view.confirm_delete,
        ],
        queue=False,
        show_progress="hidden",
    )
    refreshed = opened.then(
        refresh_page,
        inputs=[view.mode, filters],
        outputs=page_outputs,
        queue=False,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    refreshed = view.refresh.click(
        refresh_page,
        inputs=[view.mode, filters],
        outputs=page_outputs,
        queue=False,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    refreshed = view.show_more.click(
        lambda mode, current, shown: refresh_page(mode, current, shown + page_size),
        inputs=[view.mode, filters, view.shown],
        outputs=page_outputs,
        queue=False,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    selected = view.grid.select(
        select,
        inputs=[view.mode, view.paths],
        outputs=[view.player, view.image, view.audio, view.download, view.selected],
        queue=False,
        show_progress="hidden",
    )
    synchronize(selected)
    mutation_outputs = [
        view.grid,
        view.paths,
        view.status,
        view.player,
        view.image,
        view.audio,
        view.download,
        view.selected,
        view.confirm_delete,
    ]
    imported = view.import_video.click(
        import_video,
        inputs=[view.mode, view.upload_video],
        outputs=mutation_outputs,
        queue=False,
        show_progress="minimal",
        api_name=False,
    )
    refreshed = imported.then(
        sync_more,
        inputs=[view.mode, filters],
        outputs=page_outputs,
        queue=False,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    post_event = bind_gpu_action(
        view.post_run.click,
        owned_generation(postprocess, "gallery"),
        inputs=[
            view.mode,
            view.selected,
            view.postprocess,
            view.post_seed,
            view.seedvr2_model,
            selected_ltx_model,
            view.ltx25_prompt,
            view.force_offload,
            view.split_upscale,
            view.split_seconds,
            view.upscale_resolution,
        ],
        outputs=mutation_outputs + [view.post_status],
        show_progress="minimal",
        api_name=False,
    )
    refreshed = post_event.then(
        sync_more,
        inputs=[view.mode, filters],
        outputs=page_outputs,
        queue=False,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    stopped = view.post_stop.click(
        owned_interrupt(interrupt, "gallery"),
        outputs=view.post_status,
        api_name=False,
        queue=False,
    )
    stopped.then(fn=None, cancels=[post_event], queue=False, api_name=False)
    from .media_actions import bind_safe_deletion

    deleted = bind_safe_deletion(
        view,
        list_paths=list_paths,
        delete=delete,
        empty=empty,
        mutation_outputs=mutation_outputs,
        sync_more=sync_more,
        refresh_inputs=[view.mode, filters],
        refresh_outputs=page_outputs,
    )
    synchronize(deleted, refreshed=True)
