"""Bind gallery actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from h3_app.catalog import AI_POSTPROCESS_OPTIONS, LTX25_UPSCALE, SEEDVR2_UPSCALE

from ..contracts import AppComponents
from ..media_bindings import bind_gallery_view

if TYPE_CHECKING:
    from ..contracts import BootstrapServices


def bind_gallery(components: AppComponents, services: BootstrapServices) -> None:
    bind_gallery_view(
        components.gallery_components,
        tab=components.gallery_tab,
        selected_ltx_model=components.gallery_components.ltx25_model,
        ai_options=AI_POSTPROCESS_OPTIONS,
        seedvr_option=SEEDVR2_UPSCALE,
        ltx_option=LTX25_UPSCALE,
        refresh=services.refresh_media_page,
        select=services.select_gallery_media,
        import_video=services.import_gallery_media,
        postprocess=services.postprocess_selected_gallery_media,
        interrupt=services.interrupt,
        delete=services.delete_selected_gallery_media,
        empty=services.empty_generated_media_gallery,
        list_paths=services.list_media_paths,
        download_actions=services.dubbing_actions,
    )
