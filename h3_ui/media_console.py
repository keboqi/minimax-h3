"""GPU-free Media library for the storage console.

Browsing, search, import, tags, lineage, comparison and deletion reuse the main
application's controller and bindings. Enhancement needs ComfyUI on the GPU
endpoint, so it is left out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import gradio as gr

from h3_app import gallery_store
from h3_app.catalog import (
    AI_POSTPROCESS_OPTIONS,
    AUDIO_EXTENSIONS,
    DEFAULT_UPSCALE_RESOLUTION,
    IMAGE_EXTENSIONS,
    LTX25_UPSCALE,
    POSTPROCESS_OPTIONS,
    SEEDVR2_UPSCALE,
    UPSCALE_RESOLUTION_PRESETS,
    VIDEO_EXTENSIONS,
)
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.media_catalog import catalog_inventory, configure_media_catalog
from h3_app.provenance import copy_media, render_snapshot, snapshot_path

from .library_tools import build_library_tools
from .media_bindings import bind_gallery_view
from .media_controller import MediaController
from .media_view import GalleryView, build_gallery_view

PAGE_SIZE = 48


def media_runtime(data: Path, script_dir: Path) -> RuntimeConfig:
    """The GPU endpoint's media layout on the shared volume, without ComfyUI."""
    data = Path(data)
    runtime = RuntimeConfig.from_environment(
        Path(script_dir),
        {"COMFY_DIR": str(Path(script_dir) / "ComfyUI"),
         "GRADIO_OUTPUT_DIR": str(data / "output")},
    )
    # The GPU endpoint links ComfyUI's input/output folders to these directories.
    return replace(runtime, input_root=data / "input", output_root=data / "output")


def media_controller(runtime: RuntimeConfig) -> tuple[MediaController, SimpleNamespace]:
    services = SimpleNamespace(
        AUDIO_EXTENSIONS=AUDIO_EXTENSIONS,
        IMAGE_EXTENSIONS=IMAGE_EXTENSIONS,
        VIDEO_EXTENSIONS=VIDEO_EXTENSIONS,
        GALLERY_THUMBNAILS_DIR=runtime.gallery_thumbnails_dir,
        OUTPUTS_DIR=runtime.outputs_dir,
        H3Error=H3Error,
        Path=Path,
        _runtime_config=lambda: runtime,
        copy_media=copy_media,
        gallery_store=gallery_store,
        gr=gr,
        snapshot_path=snapshot_path,
        time=time,
        uuid=uuid,
    )
    controller = MediaController(services)
    # The controller reaches its own operations through its services, like the
    # main application's module-level adapters.
    for name in vars(MediaController):
        if not name.startswith("_"):
            setattr(services, name, getattr(controller, name))

    def refresh_media_page(mode="Video", limit=PAGE_SIZE, *, paths=None):
        if paths is None:
            paths = catalog_inventory(mode, limit)
        return controller.refresh_media_page(mode, limit, paths=paths)

    services.refresh_media_page = refresh_media_page
    services.refresh_gallery_page = lambda limit=PAGE_SIZE: refresh_media_page("Video", limit)
    return controller, services


def build_media_library(
    tab: gr.Tab, runtime: RuntimeConfig, store, *, reload: Callable[[], str] | None = None,
) -> GalleryView:
    """Build and bind the library inside ``tab`` against an existing workspace store."""
    from h3_app.jobs import JOBS

    # Deletion checks the recorded jobs, so media used by unfinished work stays protected.
    JOBS.configure(store)
    configure_media_catalog(store, lambda: runtime)
    controller, services = media_controller(runtime)
    with tab:
        if reload is not None:
            with gr.Row(equal_height=True):
                sync = gr.Button("Reload shared volume", scale=0, min_width=200)
                sync_status = gr.Markdown(
                    "Reload to see outputs the GPU endpoint saved after this page opened."
                )
            sync.click(reload, outputs=sync_status, concurrency_id="model-volume",
                       concurrency_limit=1, api_name=False)
        library = gr.Column()
        maintenance = gr.Column()
    view = build_gallery_view(
        library,
        postprocess_options=POSTPROCESS_OPTIONS,
        resolution_choices=tuple(UPSCALE_RESOLUTION_PRESETS),
        default_resolution=DEFAULT_UPSCALE_RESOLUTION,
        seedvr2_choices=(),
        default_seedvr2=None,
        enhance=False,
    )
    with view.inspector:
        settings_used = gr.HTML("Select an output to inspect its settings.")
    view.selected.h3_snapshot = (render_snapshot, settings_used)
    build_library_tools(
        library,
        controller.list_media_paths,
        view=view,
        system_root=maintenance,
        validate_path=controller.gallery_media_download_path,
        preview_media=services.refresh_media_page,
    )
    bind_gallery_view(
        view,
        tab=tab,
        selected_ltx_model=view.ltx25_model,
        ai_options=AI_POSTPROCESS_OPTIONS,
        seedvr_option=SEEDVR2_UPSCALE,
        ltx_option=LTX25_UPSCALE,
        refresh=services.refresh_media_page,
        select=controller.select_gallery_media,
        import_video=controller.import_gallery_media,
        postprocess=None,
        interrupt=None,
        delete=controller.delete_selected_gallery_media,
        empty=controller.empty_generated_media_gallery,
        list_paths=controller.list_media_paths,
    )
    return view
