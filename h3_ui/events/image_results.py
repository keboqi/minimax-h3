"""Bind image results actions."""

from __future__ import annotations

from ..contracts import AppComponents, AppServices


def bind_image_results(components: AppComponents, services: AppServices) -> None:
    components.image_select_all.click(
        services.select_all_image_frames,
        inputs=components.image_frame_paths,
        outputs=components.image_selection,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    components.image_clear_selection.click(
        lambda: [],
        outputs=components.image_selection,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    components.image_save_selected.click(
        services.save_selected_image_frames,
        inputs=[components.image_frame_paths, components.image_selection],
        outputs=[components.image_saved_files, components.image_save_status],
        queue=False,
        show_progress="minimal",
        api_name="save_h3_image_frames",
    )
