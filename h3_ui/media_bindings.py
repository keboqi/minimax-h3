"""Reusable Gradio event-binding helpers."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
import gradio as gr
from h3_app.catalog import (
    LTX25_CQ_ENHANCER,
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

    def refresh_page(mode: str, limit: int = page_size):
        page = refresh(mode, limit)
        return (
            list(page.items),
            list(page.paths),
            page.status,
            min(max(0, int(limit)), page.total),
            gr.update(interactive=page.next_cursor is not None),
        )

    def sync_more(mode: str):
        page = refresh(mode, page_size)
        return (
            min(page_size, page.total),
            gr.update(interactive=page.next_cursor is not None),
        )

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
                    [seedvr_option] if value == "Image" else video_postprocess_options
                ),
                value=seedvr_option,
            ),
            gr.update(
                value=(
                    "Upscale selected image"
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
    mode_changed.then(
        refresh_page,
        inputs=view.mode,
        outputs=[view.grid, view.paths, view.status, view.shown, view.show_more],
        queue=False,
        show_progress="hidden",
    )
    view.postprocess.change(
        lambda value: (
            gr.update(visible=value in ai_options),
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
                visible=value in ai_options
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
    opened.then(
        refresh_page,
        inputs=view.mode,
        outputs=[view.grid, view.paths, view.status, view.shown, view.show_more],
        queue=False,
        show_progress="hidden",
    )
    view.refresh.click(
        refresh_page,
        inputs=view.mode,
        outputs=[view.grid, view.paths, view.status, view.shown, view.show_more],
        queue=False,
        show_progress="hidden",
    )
    view.show_more.click(
        lambda mode, shown: refresh_page(mode, shown + page_size),
        inputs=[view.mode, view.shown],
        outputs=[view.grid, view.paths, view.status, view.shown, view.show_more],
        queue=False,
        show_progress="hidden",
    )
    view.grid.select(
        select,
        inputs=[view.mode, view.paths],
        outputs=[view.player, view.image, view.audio, view.download, view.selected],
        queue=False,
        show_progress="hidden",
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
    imported.then(
        sync_more,
        inputs=view.mode,
        outputs=[view.shown, view.show_more],
        queue=False,
        show_progress="hidden",
    )
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
    post_event.then(
        sync_more,
        inputs=view.mode,
        outputs=[view.shown, view.show_more],
        queue=False,
        show_progress="hidden",
    )
    stopped = view.post_stop.click(
        owned_interrupt(interrupt, "gallery"),
        outputs=view.post_status,
        api_name=False,
        queue=False,
    )
    stopped.then(fn=None, cancels=[post_event], queue=False, api_name=False)
    from .media_actions import bind_safe_deletion

    bind_safe_deletion(
        view,
        list_paths=list_paths,
        delete=delete,
        empty=empty,
        mutation_outputs=mutation_outputs,
        sync_more=sync_more,
    )
