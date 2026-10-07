"""Reusable Gradio event-binding helpers."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
from time import sleep
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
from .media_previews import browser_preview_updates, browser_gallery_items


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
    page_queue = index is None
    page_outputs = [view.grid, view.paths, view.status, view.shown, view.show_more]
    refresh_events = []

    def page_values(page, limit, request, version):
        return (
            browser_gallery_items(page.items, request, version=version),
            list(page.paths),
            f"{page.total} matching assets · {page.status}" + (
                " · Use Scan historical media to add older or externally created outputs."
                if page.total == 0 else ""
            ),
            min(max(0, int(limit)), page.total),
            gr.update(interactive=page.next_cursor is not None),
        )

    def refresh_page(mode: str, filter_values, limit: int = page_size, request: gr.Request = None):
        # The catalog already contains ready previews. Return its page in the
        # ordinary HTTP response instead of opening a queue stream for one yield.
        matching = index.inventory(
            mode,
            filter_values.get("query", ""),
            filter_values.get("favorite", False),
            limit=limit,
        )
        page = refresh(mode, limit, paths=matching)
        return page_values(page, limit, request, index.versions.get(mode))

    def progressive_page(mode: str, filter_values, limit: int = page_size, request: gr.Request = None):
        # Legacy callers without a catalog can still render while decoding.
        previous = None
        while True:
            page = refresh(mode, limit, preview_timeout=0.1, previous=previous)
            if previous is None or page.items != previous.items or page.status != previous.status:
                yield page_values(page, limit, request, None)
            if not page.preparing:
                break
            previous = page
            sleep(0.25)

    def sync_more(mode: str, filter_values, request: gr.Request):
        return refresh_page(mode, filter_values, request=request)

    if page_queue:
        refresh_page = progressive_page

        def sync_more(mode: str, filter_values, request: gr.Request):
            yield from refresh_page(mode, filter_values, request=request)

    scan_event = getattr(view.search, "h3_scan_event", None)
    if scan_event is not None:
        scan_event.success(
            sync_more, inputs=[view.mode, filters], outputs=page_outputs,
            queue=page_queue, concurrency_limit=None, api_name=False,
        )

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
                postprocess=False,
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

    ltx_options = {ltx_option} | LTX25_SAME_RESOLUTION_OPTIONS
    video_postprocess_options = [
        choice[1] if isinstance(choice, (tuple, list)) else choice
        for choice in view.postprocess.choices
    ]
    def mode_values(value):
        return (
            gr.update(visible=value == "Video", value=None),
            gr.update(visible=value == "Image", value=None),
            gr.update(visible=value == "Audio", value=None),
            None,
            "",
            gr.update(visible=True),
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
        )

    def browse_controls(mode, target):
        controls = [gr.skip() for _ in range(10)]
        if target is tab:
            controls[:5] = [
                gr.update(value=None, visible=mode == "Video"),
                gr.update(value=None, visible=mode == "Image"),
                gr.update(value=None, visible=mode == "Audio"),
                None,
                "",
            ]
            controls[6] = gr.update(value=False)
        return controls

    def browse_parameters(mode, current, shown, evt, text, favorite):
        target = evt.target
        mode_changed = mode != current.get("_mode", mode) or (
            "_mode" not in current and target is view.mode
        )
        filters_changed = False
        if view.search is not None:
            query, starred = text.strip(), bool(favorite)
            filters_changed = query != current.get("query", "") or starred != current.get("favorite", False)
            current.update(query=query, favorite=starred)
        current["_mode"] = mode
        # Gradio replays a deferred dependency with its original event target,
        # but its inputs contain the latest controls. Use those inputs to reset
        # the mode and page even when the original target was Search/Show more.
        controls = mode_values(mode) if mode_changed else browse_controls(mode, target)
        limit = (
            shown + page_size
            if target is view.show_more and not mode_changed and not filters_changed
            else page_size
        )
        return controls, limit

    def browse_library(mode, current, shown, request: gr.Request, evt: gr.EventData, text="", favorite=False):
        controls, limit = browse_parameters(mode, current, shown, evt, text, favorite)
        return (*controls, *refresh_page(mode, current, limit, request=request))

    if page_queue:
        def browse_library(mode, current, shown, request: gr.Request, evt: gr.EventData, text="", favorite=False):
            controls, limit = browse_parameters(mode, current, shown, evt, text, favorite)
            if evt.target is view.mode:
                yield (*controls, [], [], "Loading library…", 0, gr.update(interactive=False))
                controls = [gr.skip() for _ in range(10)]
            for page in refresh_page(mode, current, limit, request=request):
                yield (*controls, *page)
                controls = [gr.skip() for _ in range(10)]

    browse_triggers = [view.mode.change, tab.select, view.refresh.click, view.show_more.click]
    browse_inputs = [view.mode, filters, view.shown]
    if view.search is not None:
        browse_triggers.extend([view.search.click, view.query.submit, view.favorite_only.input])
        browse_inputs.extend([view.query, view.favorite_only])
    # One dependency serializes all user browsing actions, so a late search or
    # pagination response cannot replace the final media-type switch.
    refreshed = gr.on(
        triggers=browse_triggers,
        fn=browse_library,
        inputs=browse_inputs,
        outputs=[
            view.player, view.image, view.audio, view.selected, view.download,
            view.manage, view.confirm_delete, view.postprocess, view.post_run,
            view.enhance, *page_outputs,
        ],
        queue=page_queue,
        concurrency_limit=None,
        show_progress="minimal",
        trigger_mode="always_last",
        api_name=False,
    )
    synchronize(refreshed, refreshed=True)
    refresh_events.append(refreshed)
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
    def select_gallery_media(mode: str, paths: list[str], request: gr.Request, evt: gr.SelectData):
        return browser_preview_updates(select(mode, paths, request, evt))

    selected = view.grid.select(
        select_gallery_media,
        inputs=[view.mode, view.paths],
        outputs=[view.player, view.image, view.audio, view.download, view.selected],
        queue=False,
        postprocess=False,
        show_progress="minimal",
        trigger_mode="always_last",
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
        queue=page_queue,
        concurrency_limit=None,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    refresh_events.append(refreshed)
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
        queue=page_queue,
        concurrency_limit=None,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    refresh_events.append(refreshed)
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
        refresh_queue=page_queue,
    )
    synchronize(deleted, refreshed=True)
    refresh_events.append(deleted)
    if page_queue:
        # Only legacy progressive pages have in-flight decoding to cancel.
        view.mode.input(fn=None, cancels=refresh_events, queue=False, api_name=False)
