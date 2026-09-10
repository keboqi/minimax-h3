"""Bind gallery actions."""

from __future__ import annotations

from ..contracts import AppComponents, AppServices


def bind_gallery(components: AppComponents, services: AppServices) -> None:
    services.bind_gallery_view(
        components.gallery_components,
        tab=components.gallery_tab,
        selected_ltx_model=components.ltx25_model,
        ai_options=services.AI_POSTPROCESS_OPTIONS,
        seedvr_option=services.SEEDVR2_UPSCALE,
        ltx_option=services.LTX25_UPSCALE,
        refresh=services.refresh_gallery,
        select=services.select_gallery_video,
        import_video=services.import_gallery_video,
        postprocess=services.postprocess_selected_gallery_video,
        interrupt=services.interrupt,
        delete=services.delete_selected_gallery_video,
        empty=services.empty_generated_gallery,
    )
