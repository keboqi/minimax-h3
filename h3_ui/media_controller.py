"""Media controller with explicit runtime dependencies."""

from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Any, Callable
import uuid

import re
from h3_app.policy import normalize_paths
from typing import Iterable
from h3_app.catalog import INPUT_IMAGE_UPSCALE_SLOTS, MAX_REFERENCE_IMAGES
import gradio as gr
import websocket
from h3_app import gallery_store
from h3_app.catalog import (
    AUDIO_EXTENSIONS,
    COMFY_POSTPROCESS_OPTIONS,
    IMAGE_EXTENSIONS,
    LTX25_CQ_IMAGE_ENHANCER,
    LTX25_POSTPROCESS_MODELS,
    LTX25_SAME_RESOLUTION_OPTIONS,
    POSTPROCESS_OPTIONS,
    SEEDVR2_UPSCALE,
    UPSCALE_RESOLUTION_PRESETS,
    VIDEO_EXTENSIONS,
)
from h3_app.errors import H3Error
from h3_app.media_types import UpscaleClipBatch
from h3_app.jobs import record_failure
from h3_app.provenance import copy_media, score_path, snapshot_path
from h3_app.status import progress_status
from h3_app.workflows.upscale import (
    required_cq_image_enhance_nodes,
    required_seedvr2_image_upscale_nodes,
    required_upscale_nodes,
)
from dataclasses import dataclass


from h3_app.gallery_store import AssetInventory, AssetPage
from .media_previews import preview_results
from .presentation import error_markdown

# How deletion messages name one item and a count of items of each type.
MEDIA_NOUNS = {"Video": ("a video", "video"), "Image": ("an image", "image"), "Audio": ("an audio clip", "audio clip")}
GalleryMediaMutationResult = tuple[
    list[tuple[str, str]], list[str], str, Any, Any, Any, Any, str | None, bool
]
GalleryMediaPostprocessResult = tuple[
    list[tuple[str, str]], list[str], str, Any, Any, Any, Any, str | None, bool, str
]


@dataclass(frozen=True)
class MediaServices:
    """Runtime, configuration and effects the Media library and its finishing jobs use."""

    _runtime_config: Callable[[], Any]
    catalog_inventory: Callable[[str, int], Any]
    load_model_config: Callable[[], Any]
    ensure_ltx25_upscale_models: Callable[..., bool]
    ensure_seedvr2_upscale_models: Callable[..., bool]
    ensure_cq_image_enhance_models: Callable[[], bool]
    build_seedvr2_image_upscale_graph: Callable[..., dict]
    build_cq_image_enhance_graph: Callable[..., dict]
    build_upscale_graph: Callable[..., dict]
    object_info: Callable[[], dict]
    submit_prompt: Callable[..., str]
    poll_comfy_progress: Callable[..., Any]
    stream_comfy_progress: Callable[..., Any]
    wait_for_history: Callable[..., dict]
    resolve_output: Callable[..., Path]
    resolve_seedvr2_input_upscale_outputs: Callable[..., Any]
    unload_comfy_models: Callable[[], None]
    stage_file: Callable[..., str]
    input_image_upscale_dimensions: Callable[..., Any]
    upscale_target_dimensions: Callable[..., Any]
    postprocess_video: Callable[..., Path]
    probe_video_metadata: Callable[..., Any]
    prepare_upscale_clip_batch: Callable[..., Any]
    concat_upscaled_clips: Callable[..., Path]
    cleanup_upscale_clip_batch: Callable[..., None]
    write_snapshot: Callable[..., Any]


class MediaController:
    @staticmethod
    def _retained_input(value):
        from h3_app.jobs import CURRENT_JOB

        job = CURRENT_JOB.get()
        path = Path(value).resolve()
        if job and job.family == "gallery" and job.has_gpu and job.input_lease:
            root = Path(job.input_lease).resolve()
            if path.is_file() and path.is_relative_to(root):
                return path
        return None

    def __init__(self, services: MediaServices):
        self.services = services

    def video_download_path(self, video: str | Path) -> str:
        return gallery_store.video_download_path(
            video, runtime=self.services._runtime_config()
        )

    def managed_video_path(
        self, video: str | Path, *, require_file: bool = True
    ) -> Path:
        retained = self._retained_input(video)
        if retained is not None:
            return retained
        return gallery_store.managed_video_path(
            video, require_file=require_file, runtime=self.services._runtime_config()
        )

    def absolute_video_url(
        self, video: str | Path, request: gr.Request, *, download: bool = False
    ) -> str:
        relative_url = self.video_download_path(video)
        base_url = str(request.request.base_url).rstrip("/")
        url = f"{base_url}{relative_url}"
        return f"{url}?download=1" if download else url

    def absolute_video_download_url(
        self, video: str | Path, request: gr.Request
    ) -> str:
        return self.absolute_video_url(video, request, download=True)

    def gallery_video_paths(self, *, limit: int | None = 200) -> list[Path]:
        return gallery_store.gallery_video_paths(
            limit=limit, runtime=self.services._runtime_config()
        )

    def gallery_thumbnail(self, video: Path) -> Path | None:
        return gallery_store.gallery_thumbnail(
            video, runtime=self.services._runtime_config()
        )

    def gallery_thumbnail_path(self, video: str | Path) -> Path:
        return gallery_store.gallery_thumbnail_path(
            video, runtime=self.services._runtime_config()
        )

    def gallery_resolution_text(self, video: Path) -> str:
        return gallery_store.gallery_resolution_text(
            video, runtime=self.services._runtime_config()
        )

    def generated_video_family(self, video: str | Path) -> str:
        return gallery_store.generated_video_family(
            video, runtime=self.services._runtime_config()
        )

    def forget_gallery_metadata(self, video: str | Path | None = None) -> None:
        from h3_app.media_catalog import get_media_catalog
        catalog = get_media_catalog()
        if video is not None and catalog is not None:
            catalog.store.forget_asset(video)
        return gallery_store.forget_gallery_metadata(video)

    def managed_gallery_image_path(
        self, image: str | Path, *, require_file: bool = True
    ) -> Path:
        retained = self._retained_input(image)
        if retained is not None:
            return retained
        return gallery_store.managed_image_path(
            image, require_file=require_file, runtime=self.services._runtime_config()
        )

    def gallery_image_paths(self, *, limit: int | None = 200) -> list[Path]:
        return gallery_store.gallery_image_paths(
            limit=limit, runtime=self.services._runtime_config()
        )

    def gallery_image_resolution_text(self, image: Path) -> str:
        return gallery_store.gallery_image_resolution_text(image)

    def managed_gallery_audio_path(
        self, audio: str | Path, *, require_file: bool = True
    ) -> Path:
        retained = self._retained_input(audio)
        if retained is not None:
            return retained
        return gallery_store.managed_audio_path(
            audio, require_file=require_file, runtime=self.services._runtime_config()
        )

    def gallery_audio_paths(self, *, limit: int | None = 200) -> list[Path]:
        return gallery_store.gallery_audio_paths(
            limit=limit, runtime=self.services._runtime_config()
        )

    def gallery_media_mode(self, mode: str) -> str:
        return str(mode) if str(mode) in {"Video", "Image", "Audio"} else "Video"

    def gallery_media_download_path(self, media: str | Path, mode: str) -> str:
        media_mode = self.gallery_media_mode(mode)
        if media_mode == "Image":
            return gallery_store.image_download_path(
                media, runtime=self.services._runtime_config()
            )
        if media_mode == "Audio":
            return gallery_store.audio_download_path(
                media, runtime=self.services._runtime_config()
            )
        return self.video_download_path(media)

    def absolute_gallery_media_download_url(
        self, media: str | Path, mode: str, request: gr.Request
    ) -> str:
        relative_url = self.gallery_media_download_path(media, mode)
        base_url = str(request.request.base_url).rstrip("/")
        return f"{base_url}{relative_url}?download=1"

    def import_gallery_media(self, mode: str, uploaded_media: str | None):
        media_mode = self.gallery_media_mode(mode)
        if not uploaded_media:
            return self.gallery_media_mutation_result(
                media_mode,
                f"Choose a local {media_mode.lower()} first.",
                clear_selection=False,
            )
        source = Path(uploaded_media).expanduser().resolve()
        extensions = {
            "Video": VIDEO_EXTENSIONS,
            "Image": IMAGE_EXTENSIONS,
            "Audio": AUDIO_EXTENSIONS,
        }[media_mode]
        if not source.is_file() or source.suffix.lower() not in extensions:
            return self.gallery_media_mutation_result(
                media_mode,
                f"The selected file is not a supported {media_mode.lower()}.",
                clear_selection=False,
            )
        destination_dir = self.services._runtime_config().outputs_dir / "imports"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = (
            destination_dir
            / f"import_{int(time.time())}_{uuid.uuid4().hex[:8]}{source.suffix.lower()}"
        )
        copy_media(source, destination)
        return self.gallery_media_mutation_result(
            media_mode,
            f"Imported `{source.name}`",
            selected_media=str(destination),
            clear_selection=False,
        )

    def refresh_gallery_page(self, limit: int = 48, *, paths=None) -> AssetPage:
        if isinstance(paths, AssetInventory) and paths.kind != "Video":
            raise ValueError("Gallery inventory does not match the media type.")
        if isinstance(paths, AssetInventory) and paths.kind == "Video":
            videos = list(paths.paths)
        else:
            videos = self.gallery_video_paths(limit=None)
        if paths is not None and not isinstance(paths, AssetInventory):
            selected = {Path(path).resolve() for path in paths}
            videos = [path for path in videos if path.resolve() in selected]
        shown_videos = videos[: max(0, limit)]
        items: list[tuple[str, str]] = []
        selectable_paths: list[str] = []
        failed = 0
        posters = preview_results(
            shown_videos, self.gallery_thumbnail, kind="Video",
            cache_root=self.services._runtime_config().gallery_thumbnails_dir,
        )
        for video, (thumbnail, _pending) in zip(shown_videos, posters):
            if thumbnail is None:
                failed += 1
                thumbnail = gallery_store.gallery_placeholder(
                    video, kind="Video", runtime=self.services._runtime_config()
                )
            if thumbnail is None:
                continue
            try:
                stat = video.stat()
            except OSError:
                continue
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
            )
            size_mb = stat.st_size / (1024 * 1024)
            caption = f"{self.generated_video_family(video)} · {video.name} · {timestamp} · {size_mb:.1f} MB"
            items.append((str(thumbnail), caption))
            selectable_paths.append(str(video))
        return AssetPage.from_scan(
            items, selectable_paths, len(videos), limit, "videos", failed
        )

    def list_media_paths(self, mode):
        return {
            "Video": gallery_store.gallery_video_paths,
            "Image": gallery_store.gallery_image_paths,
            "Audio": gallery_store.gallery_audio_paths,
        }[self.gallery_media_mode(mode)](
            limit=None, runtime=self.services._runtime_config(), strict=True,
        )

    def refresh_media_page(
        self, mode: str = "Video", limit: int = 48, *, paths=None
    ) -> AssetPage:
        """Refresh the active gallery from the media catalog, scanning only when it has no inventory."""
        if paths is None:
            paths = self.services.catalog_inventory(mode, limit)
        if isinstance(paths, AssetInventory) and paths.total is not None:
            if paths.kind != self.gallery_media_mode(mode):
                raise ValueError("Gallery inventory does not match the media type.")
            items, selected = [], []
            failed = 0
            root = self.services._runtime_config().gallery_thumbnails_dir
            for row in paths.records[:max(0, limit)]:
                info = row["metadata"]["_media"]
                name = info.get("thumbnail")
                if name and Path(name).name == name:
                    items.append((str(root / name), info["caption"]))
                    selected.append(row["path"])
                    failed += int(not info.get("preview_available", True))
            kind = {"Video": "videos", "Image": "images", "Audio": "audio files"}[paths.kind]
            return AssetPage.from_scan(items, selected, paths.total, limit, kind, failed)
        media_mode = self.gallery_media_mode(mode)
        if isinstance(paths, AssetInventory) and paths.kind != media_mode:
            raise ValueError("Gallery inventory does not match the media type.")
        if media_mode == "Video":
            return self.refresh_gallery_page(limit, paths=paths)
        if media_mode == "Audio":
            all_audio_files = (
                list(paths.paths)
                if isinstance(paths, AssetInventory) and paths.kind == "Audio"
                else self.gallery_audio_paths(limit=None)
            )
            if paths is not None and not isinstance(paths, AssetInventory):
                selected = {Path(path).resolve() for path in paths}
                all_audio_files = [
                    path for path in all_audio_files if path.resolve() in selected
                ]
            audio_files = all_audio_files[: max(0, limit)]
            items: list[tuple[str, str]] = []
            selectable_paths: list[str] = []
            failed = 0
            for audio in audio_files:
                thumbnail = gallery_store.gallery_audio_thumbnail(
                    audio, runtime=self.services._runtime_config()
                )
                if thumbnail is None:
                    failed += 1
                    continue
                try:
                    stat = audio.stat()
                except OSError:
                    continue
                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
                )
                size_mb = stat.st_size / (1024 * 1024)
                family = gallery_store.generated_audio_family(
                    audio, runtime=self.services._runtime_config()
                )
                caption = f"{family} · {audio.name} · {timestamp} · {size_mb:.1f} MB"
                items.append((str(thumbnail), caption))
                selectable_paths.append(str(audio))
            return AssetPage.from_scan(
                items,
                selectable_paths,
                len(all_audio_files),
                limit,
                "audio files",
                failed,
            )
        all_images = (
            list(paths.paths)
            if isinstance(paths, AssetInventory) and paths.kind == "Image"
            else self.gallery_image_paths(limit=None)
        )
        if paths is not None and not isinstance(paths, AssetInventory):
            selected = {Path(path).resolve() for path in paths}
            all_images = [path for path in all_images if path.resolve() in selected]
        images = all_images[: max(0, limit)]
        items: list[tuple[str, str]] = []
        selectable_paths: list[str] = []
        failed = 0
        runtime = self.services._runtime_config()
        posters = preview_results(
            images,
            lambda image: gallery_store.gallery_image_thumbnail(image, runtime=runtime),
            kind="Image", cache_root=runtime.gallery_thumbnails_dir,
        )
        for image, (thumbnail, _pending) in zip(images, posters):
            if thumbnail is None:
                failed += 1
                thumbnail = gallery_store.gallery_placeholder(
                    image, kind="Image", runtime=self.services._runtime_config()
                )
            if thumbnail is None:
                continue
            try:
                stat = image.stat()
            except OSError:
                continue
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
            )
            size_mb = stat.st_size / (1024 * 1024)
            caption = f"{gallery_store.generated_image_family(image, runtime=self.services._runtime_config())} · {image.name} · {timestamp} · {size_mb:.1f} MB"
            items.append((str(thumbnail), caption))
            selectable_paths.append(str(image))
        return AssetPage.from_scan(
            items, selectable_paths, len(all_images), limit, "images", failed
        )

    def refresh_media_gallery(self, mode: str = "Video", limit: int = 48):
        """Preserve the public tuple contract while the UI consumes typed pages."""
        return self.refresh_media_page(mode, limit).as_ui_values()

    def gallery_preview_updates(
        self,
        mode: str,
        *,
        video: str | None = None,
        image: str | None = None,
        audio: str | None = None,
    ) -> tuple[Any, Any, Any]:
        """Always update all preview visibility flags with the selected media."""
        media_mode = self.gallery_media_mode(mode)
        return (
            gr.update(value=video, visible=media_mode == "Video"),
            gr.update(value=image, visible=media_mode == "Image"),
            gr.update(value=audio, visible=media_mode == "Audio"),
        )

    def select_gallery_media(
        self, mode: str, paths: list[str], request: gr.Request, evt: gr.SelectData
    ) -> tuple[Any, Any, Any, str, str | None]:
        index = evt.index
        if isinstance(index, (tuple, list)):
            index = index[0]
        try:
            media = paths[int(index)]
        except (IndexError, TypeError, ValueError):
            return (*self.gallery_preview_updates(mode), "", None)
        media_mode = self.gallery_media_mode(mode)
        if media_mode == "Image":
            resolved = self.managed_gallery_image_path(media, require_file=False)
            source_info = resolved.stat()
            download_url = self.absolute_gallery_media_download_url(
                media, media_mode, request
            )
            preview_url = download_url.removesuffix("?download=1") + f"?v={source_info.st_mtime_ns}-{source_info.st_size}"
            return (
                *self.gallery_preview_updates(mode, image=preview_url),
                f"[Download original image]({download_url})",
                media,
            )
        if media_mode == "Audio":
            resolved = self.managed_gallery_audio_path(media)
            download_url = self.absolute_gallery_media_download_url(
                resolved, media_mode, request
            )
            return (
                *self.gallery_preview_updates(mode, audio=download_url.removesuffix("?download=1")),
                f"[Download audio]({download_url})",
                media,
            )
        video = self.managed_video_path(media)
        preview_url = self.absolute_video_url(video, request)
        download_url = self.absolute_video_download_url(video, request)
        return (
            *self.gallery_preview_updates(mode, video=preview_url),
            f"[Download video]({download_url})",
            media,
        )

    def gallery_media_mutation_result(
        self,
        mode: str,
        message: str,
        *,
        selected_media: str | None = None,
        clear_selection: bool,
    ) -> GalleryMediaMutationResult:
        items, paths, detail = self.refresh_media_gallery(mode)
        video = None if clear_selection else gr.skip()
        image = None if clear_selection else gr.skip()
        audio = None if clear_selection else gr.skip()
        download = "" if clear_selection else gr.skip()
        selected = None if clear_selection else selected_media
        return (
            items,
            paths,
            f"{message} · {detail}",
            video,
            image,
            audio,
            download,
            selected,
            False,
        )

    def gallery_media_progress_result(
        self, message: str
    ) -> GalleryMediaPostprocessResult:
        return (
            gr.skip(),
            gr.skip(),
            message,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            message,
        )

    def gallery_media_processed_result(
        self, mode: str, result: Path, option: str, elapsed: float, request: gr.Request
    ) -> GalleryMediaPostprocessResult:
        items, paths, detail = self.refresh_media_gallery(mode)
        download_url = self.absolute_gallery_media_download_url(
            result, mode, request
        )
        if str(mode) == "Image":
            resolution = self.gallery_image_resolution_text(result)
            video, image, audio, noun = (None, str(result), None, "image")
        else:
            resolution = self.gallery_resolution_text(result)
            video, image, audio, noun = (str(result), None, None, "video")
        return (
            items,
            paths,
            f"Completed {option} in {elapsed:.1f}s · {detail}",
            *self.gallery_preview_updates(
                mode, video=video, image=image, audio=audio
            ),
            f"**Resolution:** {resolution} · [Download processed {noun}]({download_url})",
            str(result),
            False,
            f"Completed {option} in {elapsed:.1f}s",
        )

    def _postprocess_selected_video(
        self,
        selected_video: str | None,
        option: str,
        seed: int,
        seedvr2_model: str,
        ltx25_model: str,
        ltx25_prompt: str,
        force_offload: bool,
        split_upscale: bool,
        split_seconds: float,
        upscale_resolution: str,
        request: gr.Request,
        progress=gr.Progress(track_tqdm=False),
    ):
        """Create a new post-processed output from one selected gallery video."""
        started = time.monotonic()
        ws: websocket.WebSocket | None = None
        clip_batch: UpscaleClipBatch | None = None
        clip_outputs: list[Path] = []
        try:
            if not selected_video:
                raise H3Error("Select a gallery video first.")
            if option not in POSTPROCESS_OPTIONS:
                raise H3Error("Choose a post-processing method.")
            source = self.managed_video_path(selected_video)
            actual_seed = (
                random.randrange(0, 2**63 - 1)
                if int(seed) < 0
                else int(seed)
            )
            progress(0, desc=f"Preparing {option}")
            yield self.gallery_media_progress_result(
                f"Preparing `{source.name}` for {option}"
            )
            if option not in COMFY_POSTPROCESS_OPTIONS:
                result = self.services.postprocess_video(source, option)
                progress(1, desc="Complete")
                yield self.gallery_media_processed_result(
                    "Video", result, option, time.monotonic() - started, request
                )
                return
            models = self.services.load_model_config()
            available = set(self.services.object_info())
            missing = required_upscale_nodes(option) - available
            if missing:
                raise H3Error(
                    f"{option} is unavailable. Missing ComfyUI nodes: "
                    + ", ".join(sorted(missing))
                )
            if option == SEEDVR2_UPSCALE:
                model_status = f"SeedVR2 {seedvr2_model}"
                yield self.gallery_media_progress_result(
                    f"Checking {model_status} models"
                )
                downloaded = self.services.ensure_seedvr2_upscale_models(
                    models, seedvr2_model
                )
                stage_bucket = "seedvr2_upscale"
            else:
                model_status = f"{option} with {ltx25_model}"
                yield self.gallery_media_progress_result(
                    f"Checking {model_status} models"
                )
                downloaded = self.services.ensure_ltx25_upscale_models(
                    ltx25_model, option=option
                )
                stage_bucket = (
                    LTX25_POSTPROCESS_MODELS[option]
                    if option in LTX25_SAME_RESOLUTION_OPTIONS
                    else "ltx25_upscale"
                )
            if downloaded:
                yield self.gallery_media_progress_result(
                    f"{option} models downloaded"
                )
            metadata = self.services.probe_video_metadata(source)
            target_width, target_height = (
                (metadata.width, metadata.height)
                if option in LTX25_SAME_RESOLUTION_OPTIONS
                else self.services.upscale_target_dimensions(
                    metadata.width, metadata.height, upscale_resolution
                )
            )
            use_split = option in LTX25_POSTPROCESS_MODELS and bool(
                split_upscale
            )
            clip_batch = self.services.prepare_upscale_clip_batch(
                source,
                category=stage_bucket,
                split_enabled=use_split,
                split_seconds=split_seconds,
                metadata=metadata,
            )
            clip_count = len(clip_batch.sources)
            if use_split:
                yield self.gallery_media_progress_result(
                    f"Split `{source.name}` into {clip_count} LTX-safe clips (target {float(split_seconds):g}s each)"
                )
            if force_offload:
                yield self.gallery_media_progress_result(
                    f"Unloading resident models before {option}"
                )
                self.services.unload_comfy_models()
            if not use_split:
                graph, configured_steps = self.services.build_upscale_graph(
                    option=option,
                    source_video=clip_batch.sources[0],
                    seed=actual_seed,
                    models=models,
                    seedvr2_model=seedvr2_model,
                    ltx25_model=ltx25_model,
                    prompt=ltx25_prompt,
                    width=metadata.width,
                    height=metadata.height,
                    target_width=target_width,
                    target_height=target_height,
                    fps=metadata.fps,
                )
            client_id = str(uuid.uuid4())
            ws = None
            if use_split:
                for clip_index, staged_source in enumerate(clip_batch.sources):
                    clip_seed = (actual_seed + clip_index) % (2**63 - 1)
                    graph, configured_steps = self.services.build_upscale_graph(
                        option=option,
                        source_video=staged_source,
                        seed=clip_seed,
                        models=models,
                        seedvr2_model=seedvr2_model,
                        ltx25_model=ltx25_model,
                        prompt=ltx25_prompt,
                        width=metadata.width,
                        height=metadata.height,
                        target_width=target_width,
                        target_height=target_height,
                        fps=metadata.fps,
                    )
                    queued_at = time.time()
                    prompt_id = self.services.submit_prompt(graph, client_id)
                    clip_label = f"Clip {clip_index + 1}/{clip_count}"
                    yield self.gallery_media_progress_result(
                        progress_status(
                            f"{option} queued",
                            started=started,
                            detail=f"{clip_label} 路 job `{prompt_id}` 路 seed {clip_seed}",
                        )
                    )
                    updates = (
                        self.services.stream_comfy_progress(
                            ws, prompt_id, graph, started
                        )
                        if ws is not None
                        else self.services.poll_comfy_progress(prompt_id, graph)
                    )
                    for (
                        stage,
                        completed_nodes,
                        total_nodes,
                        step,
                        step_total,
                    ) in updates:
                        if stage == "Generating video and audio":
                            stage = f"Processing with {option}"
                        if step is not None and step_total:
                            progress(
                                (
                                    clip_index * step_total + step,
                                    clip_count * step_total,
                                ),
                                desc=f"{clip_label}: {stage}",
                            )
                        elif total_nodes:
                            progress(
                                (
                                    clip_index * total_nodes + completed_nodes,
                                    clip_count * total_nodes,
                                ),
                                desc=f"{clip_label}: {stage}",
                            )
                        yield self.gallery_media_progress_result(
                            progress_status(
                                f"{clip_label}: {stage}",
                                started=started,
                                completed_nodes=completed_nodes,
                                total_nodes=total_nodes,
                                step=step,
                                step_total=step_total,
                                configured_steps=(
                                    configured_steps if step is not None else None
                                ),
                                detail=f"Post-process job `{prompt_id}`",
                            )
                        )
                    clip_outputs.append(
                        self.services.resolve_output(
                            self.services.wait_for_history(prompt_id), queued_at
                        )
                    )
                yield self.gallery_media_progress_result(
                    f"Concatenating {clip_count} processed clips and restoring source audio"
                )
                result = self.services.concat_upscaled_clips(
                    source,
                    clip_outputs,
                    option=option,
                    duration=metadata.duration,
                    frame_count=metadata.frame_count,
                )
                progress(1, desc="Complete")
                yield self.gallery_media_processed_result(
                    "Video", result, option, time.monotonic() - started, request
                )
                return
            queued_at = time.time()
            prompt_id = self.services.submit_prompt(graph, client_id)
            yield self.gallery_media_progress_result(
                progress_status(
                    f"{option} queued",
                    started=started,
                    detail=f"Job `{prompt_id}` · seed {actual_seed}",
                )
            )
            updates = (
                self.services.stream_comfy_progress(ws, prompt_id, graph, started)
                if ws is not None
                else self.services.poll_comfy_progress(prompt_id, graph)
            )
            for stage, completed_nodes, total_nodes, step, step_total in updates:
                if stage == "Generating video and audio":
                    stage = f"Processing with {option}"
                if step is not None and step_total:
                    progress((step, step_total), desc=stage)
                elif total_nodes:
                    progress((completed_nodes, total_nodes), desc=stage)
                yield self.gallery_media_progress_result(
                    progress_status(
                        stage,
                        started=started,
                        completed_nodes=completed_nodes,
                        total_nodes=total_nodes,
                        step=step,
                        step_total=step_total,
                        configured_steps=configured_steps if step is not None else None,
                        detail=f"Post-process job `{prompt_id}`",
                    )
                )
            result = self.services.resolve_output(
                self.services.wait_for_history(prompt_id), queued_at
            )
            progress(1, desc="Complete")
            yield self.gallery_media_processed_result(
                "Video", result, option, time.monotonic() - started, request
            )
        except Exception as exc:
            record_failure(exc)
            yield self.gallery_media_progress_result(
                error_markdown("Post-processing failed", exc)
            )
        finally:
            self.services.cleanup_upscale_clip_batch(
                clip_batch,
                clip_outputs if clip_batch and clip_batch.temporary_inputs else (),
            )
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

    def postprocess_selected_gallery_image(
        self,
        selected_image: str | None,
        option: str,
        seed: int,
        seedvr2_model: str,
        force_offload: bool,
        upscale_resolution: str,
        request: gr.Request,
        progress=gr.Progress(track_tqdm=False),
    ):
        """Enhance one selected gallery still and retain its original."""
        started = time.monotonic()
        try:
            if not selected_image:
                raise H3Error("Select a gallery image first.")
            cq_image = option == LTX25_CQ_IMAGE_ENHANCER
            family = "CQ image enhancement" if cq_image else "SeedVR2 image upscale"
            if option not in {SEEDVR2_UPSCALE, LTX25_CQ_IMAGE_ENHANCER}:
                raise H3Error(
                    f"Unsupported image enhancement method: {option}"
                )
            source = self.managed_gallery_image_path(selected_image)
            try:
                frame_width, frame_height = UPSCALE_RESOLUTION_PRESETS[
                    str(upscale_resolution)
                ]
            except KeyError as exc:
                raise H3Error(
                    f"Unknown upscale resolution preset: {upscale_resolution}"
                ) from exc
            source_width, source_height, target_width, target_height, scale_by = (
                self.services.input_image_upscale_dimensions(
                    source, frame_width, frame_height
                )
            )
            if cq_image:
                scale_by = min(frame_width / source_width, frame_height / source_height)
                target_width = max(1, min(frame_width, round(source_width * scale_by)))
                target_height = max(1, min(frame_height, round(source_height * scale_by)))
            if scale_by <= 1.0 and not cq_image:
                raise H3Error(
                    f"`{source.name}` is already {source_width}×{source_height}; choose a larger target resolution."
                )
            actual_seed = (
                random.randrange(0, 2**63 - 1)
                if int(seed) < 0
                else int(seed)
            )
            yield self.gallery_media_progress_result(
                f"Preparing `{source.name}` for {family}"
            )
            available = set(self.services.object_info())
            required = (
                required_cq_image_enhance_nodes()
                if cq_image else required_seedvr2_image_upscale_nodes()
            )
            missing = required - available
            if missing:
                raise H3Error(
                    f"{family} requires current ComfyUI nodes: "
                    + ", ".join(sorted(missing))
                )
            models = None if cq_image else self.services.load_model_config()
            downloaded = (
                self.services.ensure_cq_image_enhance_models()
                if cq_image else self.services.ensure_seedvr2_upscale_models(models, seedvr2_model)
            )
            if downloaded:
                yield self.gallery_media_progress_result(
                    f"{family} models downloaded"
                )
            if force_offload:
                yield self.gallery_media_progress_result(
                    f"Unloading resident models before {family}"
                )
                self.services.unload_comfy_models()
            staged = self.services.stage_file(
                str(source), "gallery_image_upscale", reuse=True
            )
            output_token = uuid.uuid4().hex
            if cq_image:
                output_token = f"cq_image_{output_token}"
            if cq_image:
                graph = self.services.build_cq_image_enhance_graph(
                    source_image=staged, seed=actual_seed,
                    target_width=target_width, target_height=target_height,
                    output_token=output_token,
                )
            else:
                graph = self.services.build_seedvr2_image_upscale_graph(
                    source_images=[("gallery", staged, scale_by)],
                    seed=actual_seed, models=models, model_choice=seedvr2_model,
                    output_token=output_token,
                )
            queued_at = time.time()
            prompt_id = self.services.submit_prompt(
                graph, str(uuid.uuid4())
            )
            yield self.gallery_media_progress_result(
                f"{family} queued · job `{prompt_id}` · seed {actual_seed}"
            )
            for (
                stage,
                completed,
                total,
                step,
                step_total,
            ) in self.services.poll_comfy_progress(prompt_id, graph):
                if step is not None and step_total:
                    progress((step, step_total), desc=stage)
                elif total:
                    progress((completed, total), desc=stage)
                yield self.gallery_media_progress_result(
                    progress_status(
                        stage,
                        started=started,
                        completed_nodes=completed,
                        total_nodes=total,
                        step=step,
                        step_total=step_total,
                        configured_steps=(8 if cq_image else 1) if step is not None else None,
                        detail=f"{family} job `{prompt_id}`",
                    )
                )
            history = self.services.wait_for_history(prompt_id)
            result = self.services.resolve_seedvr2_input_upscale_outputs(
                history, queued_at, output_token, ["gallery"]
            )["gallery"]
            self.services.write_snapshot(
                result,
                {
                    "job_id": prompt_id,
                    "family": family,
                    "settings": {
                        "source": source.name,
                        "source_resolution": f"{source_width}×{source_height}",
                        "target_resolution": f"{target_width}×{target_height}",
                        "model": "LTX-2.5 Dev INT8 + CQ image LoRA" if cq_image else seedvr2_model,
                        "method": option,
                        "seed": actual_seed,
                    },
                },
            )
            progress(1, desc="Complete")
            yield self.gallery_media_processed_result(
                "Image",
                result,
                option,
                time.monotonic() - started,
                request,
            )
        except Exception as exc:
            record_failure(exc)
            yield self.gallery_media_progress_result(
                error_markdown("Image enhancement failed", exc)
            )

    def postprocess_selected_gallery_media(
        self,
        mode: str,
        selected_media: str | None,
        option: str,
        seed: int,
        seedvr2_model: str,
        ltx25_model: str,
        ltx25_prompt: str,
        force_offload: bool,
        split_upscale: bool,
        split_seconds: float,
        upscale_resolution: str,
        request: gr.Request,
        progress=gr.Progress(track_tqdm=False),
    ):
        """Dispatch gallery enhancement according to the active media library."""
        media_mode = self.gallery_media_mode(mode)
        if media_mode == "Audio":
            yield self.gallery_media_progress_result(
                "Audio gallery outputs are available for playback and download."
            )
            return
        if media_mode == "Image":
            yield from self.postprocess_selected_gallery_image(
                selected_media,
                option,
                seed,
                seedvr2_model,
                force_offload,
                upscale_resolution,
                request,
                progress,
            )
            return
        yield from self._postprocess_selected_video(
            selected_media,
            option,
            seed,
            seedvr2_model,
            ltx25_model,
            ltx25_prompt,
            force_offload,
            split_upscale,
            split_seconds,
            upscale_resolution,
            request,
            progress,
        )

    def _delete_generated_media(self, selected_media, mode):
        """Delete only managed outputs and retire their catalog entries immediately."""
        store = gallery_store
        runtime = self.services._runtime_config()
        media = {
            "Video": store.managed_video_path,
            "Image": store.managed_image_path,
            "Audio": store.managed_audio_path,
        }[mode](selected_media, runtime=runtime)
        thumbnail = {
            "Video": store.gallery_thumbnail_path,
            "Image": store.gallery_image_thumbnail_path,
            "Audio": store.gallery_audio_thumbnail_path,
        }[mode](media, runtime=runtime)
        placeholder = store.gallery_placeholder_path(media, kind=mode, runtime=runtime)
        media.unlink()
        # Even if sidecar cleanup fails, browsing must stop offering this file.
        self.forget_gallery_metadata(media)
        snapshot_path(media).unlink(missing_ok=True)
        score_path(media).unlink(missing_ok=True)
        thumbnail.unlink(missing_ok=True)
        placeholder.unlink(missing_ok=True)
        return media.name

    def delete_selected_gallery_media(
        self, mode: str, selected_media: str | None, confirmed: bool
    ) -> GalleryMediaMutationResult:
        media_mode = self.gallery_media_mode(mode)
        if not confirmed:
            return self.gallery_media_mutation_result(
                media_mode,
                "Confirm permanent deletion first.",
                selected_media=selected_media,
                clear_selection=False,
            )
        if not selected_media:
            return self.gallery_media_mutation_result(
                media_mode, f"Select {MEDIA_NOUNS[media_mode][0]} to delete.", clear_selection=True,
            )
        try:
            name = self._delete_generated_media(selected_media, media_mode)
            message = f"Deleted `{name}`."
        except (H3Error, OSError) as exc:
            message = f"Delete failed: {exc}"
        return self.gallery_media_mutation_result(
            media_mode, message, clear_selection=True,
        )

    def empty_generated_media_gallery(
        self, mode: str, selected_media: str | None, confirmed: bool
    ) -> GalleryMediaMutationResult:
        media_mode = self.gallery_media_mode(mode)
        if not confirmed:
            return self.gallery_media_mutation_result(
                media_mode,
                "Confirm permanent deletion first.",
                selected_media=selected_media,
                clear_selection=False,
            )
        deleted = failed = 0
        for candidate in self.list_media_paths(media_mode):
            try:
                self._delete_generated_media(candidate, media_mode)
                deleted += 1
            except (H3Error, OSError):
                failed += 1
        noun = MEDIA_NOUNS[media_mode][1]
        message = f"Deleted {deleted} generated {noun}{'s' if deleted != 1 else ''}."
        if failed:
            message += f" {failed} file{'s' if failed != 1 else ''} could not be deleted."
        return self.gallery_media_mutation_result(
            media_mode, message, clear_selection=True,
        )

    def upscale_selected_input_images(
        self,
        selected_slots: Iterable[str],
        model_choice: str,
        seed: int,
        force_offload: bool,
        frame_width: int,
        frame_height: int,
        first_image: Any,
        last_image: Any,
        ref_image_1: Any,
        ref_image_2: Any,
        ref_image_3: Any,
        ref_image_4: Any,
        ref_image_5: Any,
        ref_image_6: Any,
        ref_image_7: Any,
        ref_image_8: Any,
        ref_image_9: Any,
        progress=gr.Progress(track_tqdm=False),
    ):
        """Upscale selected H3 stills to fit a frame and replace their UI values."""
        slot_labels = list(INPUT_IMAGE_UPSCALE_SLOTS)
        slot_keys = [
            "first",
            "last",
            *(f"picture_{index}" for index in range(1, MAX_REFERENCE_IMAGES + 1)),
        ]
        image_values = (
            first_image,
            last_image,
            ref_image_1,
            ref_image_2,
            ref_image_3,
            ref_image_4,
            ref_image_5,
            ref_image_6,
            ref_image_7,
            ref_image_8,
            ref_image_9,
        )
        selected = list(dict.fromkeys((str(value) for value in selected_slots or [])))
        if not selected:
            raise gr.Error("Select at least one start, end, or reference image to upscale.")
        unknown = sorted(set(selected) - set(slot_labels))
        if unknown:
            raise gr.Error("Unknown input image selection: " + ", ".join(unknown))
        values_by_label = dict(zip(slot_labels, image_values))
        keys_by_label = dict(zip(slot_labels, slot_keys))
        staged: list[tuple[str, str, float]] = []
        unchanged_results: dict[str, Path] = {}
        dimension_notes: list[str] = []
        for label in selected:
            paths = normalize_paths(values_by_label[label])
            if not paths:
                raise gr.Error(f"Upload {label} before selecting it for upscaling.")
            source_path = Path(paths[0]).resolve()
            try:
                source_width, source_height, dest_width, dest_height, scale_by = (
                    self.services.input_image_upscale_dimensions(
                        str(source_path), frame_width, frame_height
                    )
                )
            except Exception as exc:
                raise gr.Error(f"Could not prepare {label}: {exc}") from exc
            slot_key = keys_by_label[label]
            if scale_by > 1.0:
                staged.append(
                    (
                        slot_key,
                        self.services.stage_file(str(source_path), "input_image_upscale", reuse=True),
                        scale_by,
                    )
                )
                dimension_notes.append(
                    f"{label}: {source_width}×{source_height} → {dest_width}×{dest_height}"
                )
            else:
                unchanged_results[slot_key] = source_path
                dimension_notes.append(f"{label}: {source_width}×{source_height} unchanged")
        actual_seed = random.randrange(0, 2**63 - 1) if int(seed) < 0 else int(seed)
        generated_keys = {slot_key for slot_key, _path, _scale in staged}
        results = dict(unchanged_results)
        try:
            if staged:
                progress(0, desc="Checking SeedVR2 input upscaler")
                available = set(self.services.object_info())
                missing = required_seedvr2_image_upscale_nodes() - available
                if missing:
                    raise H3Error(
                        "SeedVR2 input upscaling requires current ComfyUI nodes: "
                        + ", ".join(sorted(missing))
                    )
                models = self.services.load_model_config()
                self.services.ensure_seedvr2_upscale_models(models, model_choice)
                if force_offload:
                    progress(0, desc="Unloading resident models")
                    self.services.unload_comfy_models()
                output_token = uuid.uuid4().hex
                graph = self.services.build_seedvr2_image_upscale_graph(
                    source_images=staged,
                    seed=actual_seed,
                    models=models,
                    model_choice=model_choice,
                    output_token=output_token,
                )
                client_id = str(uuid.uuid4())
                queued_at = time.time()
                prompt_id = self.services.submit_prompt(graph, client_id)
                for stage, completed, total, step, step_total in self.services.poll_comfy_progress(
                    prompt_id, graph
                ):
                    if step is not None and step_total:
                        progress((step, step_total), desc=f"Upscaling inputs: {stage}")
                    elif total:
                        progress((completed, total), desc=f"Upscaling inputs: {stage}")
                history = self.services.wait_for_history(prompt_id)
                results.update(
                    self.services.resolve_seedvr2_input_upscale_outputs(
                        history, queued_at, output_token, generated_keys
                    )
                )
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(f"Input image upscaling failed: {exc}") from exc
        component_updates: list[Any] = []
        downloads: list[str] = []
        for slot_key in slot_keys:
            if slot_key in generated_keys:
                path = str(results[slot_key])
                component_updates.append(gr.update(value=path))
            else:
                component_updates.append(gr.update())
            if slot_key in results:
                downloads.append(str(results[slot_key]))
        progress(1, desc="Input images ready")
        summary = (
            f"Fit selected images into a {int(frame_width)}×{int(frame_height)} frame: {len(staged)} upscaled with SeedVR2 {model_choice}, {len(unchanged_results)} already large enough. Aspect ratios were preserved and no image was downscaled.\n\n"
            + "  \n".join(dimension_notes)
        )
        return (*component_updates, downloads, summary)

    def save_selected_image_frames(
        self,
        frame_paths: Iterable[Any], selected_labels: Iterable[str]
    ) -> tuple[list[str], str]:
        paths = [Path(path).resolve() for path in normalize_paths(frame_paths)]
        labels = list(selected_labels or [])
        if not paths:
            raise gr.Error("Generate image frames before saving a selection.")
        if not labels:
            raise gr.Error("Select at least one image frame to save.")
        selected_indices: list[int] = []
        for label in labels:
            match = re.fullmatch("Frame\\s+(\\d+)", str(label).strip())
            if not match:
                raise gr.Error(f"Invalid frame selection: {label}")
            index = int(match.group(1)) - 1
            if index < 0 or index >= len(paths):
                raise gr.Error(f"Frame selection is out of range: {label}")
            if index not in selected_indices:
                selected_indices.append(index)
        staging_root = (self.services._runtime_config().output_dir / "h3" / "image_staging").resolve()
        for index in selected_indices:
            source = paths[index]
            if not source.is_relative_to(staging_root) or not source.is_file():
                raise gr.Error(
                    "A selected frame is outside the H3 image staging directory."
                )
        destination = (
            self.services._runtime_config().output_dir
            / "h3"
            / "images"
            / f"selection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        )
        destination.mkdir(parents=True, exist_ok=False)
        saved: list[str] = []
        for index in selected_indices:
            target = destination / f"frame_{index + 1:03d}.png"
            copy_media(paths[index], target)
            saved.append(str(target))
        return (saved, f"Saved {len(saved)} selected frame(s) to `{destination}`.")
