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
    LTX25_TILED_UPSCALE,
    H3_REFINEMENT,
    H3_REGENERATE_MODELS,
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
    postprocess: Callable[..., Any] | None,
    interrupt: Callable[..., Any] | None,
    delete: Callable[..., Any],
    empty: Callable[..., Any],
    list_paths: Callable[..., Any] | None = None,
    download_actions: Any = None,
    h3_refinement_inputs=None,
) -> None:
    page_size = 48

    filters = view.filters
    index = view.search.h3_asset_index
    page_outputs = [view.grid, view.paths, view.status, view.shown, view.show_more]

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
            engine=filter_values.get("engine", ""),
            sort=filter_values.get("sort", "newest"),
        )
        page = refresh(mode, limit, paths=matching)
        return page_values(page, limit, request, index.versions.get(mode))

    def sync_more(mode: str, filter_values, request: gr.Request):
        return refresh_page(mode, filter_values, request=request)

    view.search.h3_scan_event.success(
        sync_more, inputs=[view.mode, filters], outputs=page_outputs,
        queue=False, concurrency_limit=None, api_name=False,
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

    def summary_updates(mode, filter_values):
        # Counts per type beside the status; the Engine list follows the type. Writing
        # the counts into the Media type choices would re-trigger its change event.
        counts, engines = index.summary()
        engine = filter_values.get("engine", "")
        options = [("All engines", ""), *((name, name) for name in engines.get(mode, []))]
        return (
            " · ".join(f"{kind} {counts.get(kind, 0)}" for kind in ("Video", "Image", "Audio")),
            gr.update(choices=options, value=engine if engine in dict(options).values() else ""),
        )

    def synchronize(event, *, refreshed=False):
        if refreshed:
            event.then(
                summary_updates,
                inputs=[view.mode, filters],
                outputs=[view.counts, view.engine],
                queue=False,
                show_progress="hidden",
                api_name=False,
            )
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

    ltx_options = {ltx_option, LTX25_TILED_UPSCALE} | LTX25_SAME_RESOLUTION_OPTIONS
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
                    [seedvr_option, LTX25_CQ_IMAGE_ENHANCER]
                    if value == "Image" else video_postprocess_options
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
            gr.update(visible=postprocess is not None and value != "Audio"),
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

    def browse_parameters(mode, current, shown, evt, text, favorite, engine="", sort="newest"):
        target = evt.target
        mode_changed = mode != current.get("_mode", mode) or (
            "_mode" not in current and target is view.mode
        )
        query, starred = text.strip(), bool(favorite)
        # Engines differ per media type, so a type switch shows every engine again.
        engine, sort = ("" if mode_changed else engine or ""), sort or "newest"
        filters_changed = (query, starred, engine, sort) != (
            current.get("query", ""), current.get("favorite", False),
            current.get("engine", ""), current.get("sort", "newest"),
        )
        current.update(query=query, favorite=starred, engine=engine, sort=sort)
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

    def browse_library(mode, current, shown, request: gr.Request, evt: gr.EventData, text="", favorite=False,
                       engine="", sort="newest"):
        controls, limit = browse_parameters(mode, current, shown, evt, text, favorite, engine, sort)
        return (*controls, *refresh_page(mode, current, limit, request=request))

    browse_triggers = [
        view.mode.change, tab.select, view.refresh.click, view.show_more.click,
        view.search.click, view.query.submit, view.favorite_only.input,
        # Live search: typing or deleting filters as you go; always_last drops stale keystrokes.
        # change, not input: Gradio fires input only for typed characters. No event writes
        # to the search box, so its change event cannot loop back here.
        view.query.change, view.engine.input, view.sort.input,
    ]
    browse_inputs = [view.mode, filters, view.shown, view.query, view.favorite_only, view.engine, view.sort]
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
        queue=False,
        concurrency_limit=None,
        show_progress="minimal",
        trigger_mode="always_last",
        api_name=False,
    )
    synchronize(refreshed, refreshed=True)
    view.postprocess.change(
        lambda value: (
            gr.update(visible=(value in ai_options or value == LTX25_CQ_IMAGE_ENHANCER) and value != H3_REFINEMENT),
            gr.update(visible=value == seedvr_option),
            gr.update(
                visible=value in ltx_options | set(H3_REGENERATE_MODELS)
                and value not in {LTX25_CQ_ENHANCER, LTX25_SDR_TO_HDR},
                info=(
                    "Uses the current Minimax H3 native upscale settings: method, upscaler precision, "
                    "refinement steps and LoRA. Preserves source timing and audio."
                    if value == H3_REFINEMENT else
                    "Experimental H3 refiner + LightX2V, three Euler steps. Describe the source scene. "
                    "Preserves frames, fps and audio; best tested on short 24 fps drafts. Uses the default H3 model profile."
                    if value in H3_REGENERATE_MODELS else
                    "Describe texture, lighting and sharpness only; named subjects can repeat across tiles. "
                    "Uses spatial tiles and temporal streaming."
                    if value == LTX25_TILED_UPSCALE else
                    "Describe the source scene; focus restoration instructions are added automatically. Preserves source resolution. Uses the finishing model selected here."
                    if value == LTX25_DEBLUR
                    else (
                        "Describe the source scene; compression artifact removal instructions are added automatically. Preserves source resolution. Uses the finishing model selected here."
                        if value in LTX25_RESTORATION_OPTIONS
                        else "Optional but recommended. Uses the finishing model selected here."
                    )
                ),
                label="Scene prompt",
            ),
            gr.update(
                visible=(value in ai_options or value == LTX25_CQ_IMAGE_ENHANCER)
                and value not in LTX25_SAME_RESOLUTION_OPTIONS
            ),
            gr.update(visible=value in ltx_options),
        ),
        inputs=[view.postprocess],
        outputs=[
            view.ai_settings,
            view.seedvr2_model,
            view.ltx25_prompt,
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
    # On a phone the inspector stacks under the grid; bring it into view after a pick.
    view.grid.select(
        None,
        js="""() => {
            if (!window.matchMedia('(max-width: 767px)').matches) return;
            const smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            document.querySelector('.h3-gallery-inspector')?.scrollIntoView({behavior: smooth ? 'smooth' : 'auto', block: 'start'});
        }""",
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
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
        concurrency_limit=None,
        show_progress="hidden",
    )
    synchronize(refreshed, refreshed=True)
    if view.downloads is not None and download_actions is not None:
        from .dubbing.bindings import bind_media_downloads

        def show_download(event):
            # A finished download is a video: switch to Video and show it at the top of the page.
            switched = event.success(
                lambda path: gr.update(value="Video") if path else gr.skip(),
                inputs=view.downloads.path, outputs=view.mode,
                queue=False, show_progress="hidden", api_name=False,
            )
            synchronize(switched.then(
                sync_more, inputs=[view.mode, filters], outputs=page_outputs,
                queue=False, concurrency_limit=None, show_progress="hidden",
            ), refreshed=True)

        bind_media_downloads(view.downloads, on_downloaded=show_download, actions=download_actions)
    if postprocess is not None:
        bind_postprocess(
            view, selected_ltx_model, postprocess, interrupt,
            mutation_outputs, sync_more, synchronize, h3_refinement_inputs,
        )
    from .media_actions import bind_safe_deletion

    def refresh_after_deletion(mode: str, filter_values, request: gr.Request):
        values = sync_more(mode, filter_values, request)
        return (*values[:2], gr.skip(), *values[3:])

    deleted = bind_safe_deletion(
        view,
        list_paths=list_paths,
        delete=delete,
        empty=empty,
        mutation_outputs=mutation_outputs,
        sync_more=refresh_after_deletion,
        refresh_inputs=[view.mode, filters],
        refresh_outputs=page_outputs,
    )
    synchronize(deleted, refreshed=True)


def bind_postprocess(
    view, selected_ltx_model, postprocess, interrupt,
    mutation_outputs, sync_more, synchronize, h3_refinement_inputs=None,
):
    """Bind the GPU enhancement actions; GPU-free deployments omit them."""
    page_outputs = [view.grid, view.paths, view.status, view.shown, view.show_more]
    callback = postprocess
    input_names = None
    if h3_refinement_inputs:
        import inspect
        from h3_app.refinement import H3_REFINEMENT_INPUTS, H3_REFINEMENT_FIELDS
        input_names = [name for name, param in inspect.signature(postprocess).parameters.items()
                       if name not in {"request", "progress"} and param.kind != inspect.Parameter.KEYWORD_ONLY]
        input_names += [f"h3_{name}" for name in H3_REFINEMENT_INPUTS]

        def callback(*values, request: gr.Request):
            count = len(H3_REFINEMENT_INPUTS)
            yield from postprocess(
                *values[:-count], request=request,
                h3_refinement=dict(zip(H3_REFINEMENT_FIELDS, values[-count:])),
            )

    post_event = bind_gpu_action(
        view.post_run.click,
        owned_generation(callback, "gallery", input_names=input_names),
        inputs=[
            view.mode,
            view.selected,
            view.postprocess,
            view.post_seed,
            view.seedvr2_model,
            selected_ltx_model,
            view.ltx25_prompt,
            view.force_offload,
            view.upscale_resolution,
            *(h3_refinement_inputs or []),
        ],
        outputs=mutation_outputs + [view.post_status],
        show_progress="minimal",
        api_name=False,
    )
    refreshed = post_event.then(
        sync_more,
        inputs=[view.mode, view.filters],
        outputs=page_outputs,
        queue=False,
        concurrency_limit=None,
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
