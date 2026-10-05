"""Media controller with explicit runtime dependencies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any
import gradio as gr
import websocket
from h3_app.media_types import UpscaleClipBatch
from h3_app.catalog import LTX25_CQ_IMAGE_ENHANCER
from dataclasses import dataclass


from h3_app.gallery_store import AssetInventory, AssetPage

_THUMBNAIL_WORKERS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="h3-posters")

GalleryMutationResult = tuple[
    list[tuple[str, str]], list[str], str, Any, Any, str | None, bool
]
GalleryPostprocessResult = tuple[
    list[tuple[str, str]], list[str], str, Any, Any, str | None, bool, str
]
GalleryMediaMutationResult = tuple[
    list[tuple[str, str]], list[str], str, Any, Any, Any, Any, str | None, bool
]
GalleryMediaPostprocessResult = tuple[
    list[tuple[str, str]], list[str], str, Any, Any, Any, Any, str | None, bool, str
]


@dataclass(frozen=True)
class MediaServices:
    AUDIO_EXTENSIONS: Any
    COMFY_POSTPROCESS_OPTIONS: Any
    GALLERY_THUMBNAILS_DIR: Any
    H3Error: Any
    IMAGE_EXTENSIONS: Any
    LTX25_POSTPROCESS_MODELS: Any
    LTX25_SAME_RESOLUTION_OPTIONS: Any
    OUTPUTS_DIR: Any
    POSTPROCESS_OPTIONS: Any
    Path: Any
    SEEDVR2_UPSCALE: Any
    SWIFTVR_UPSCALE: Any
    UPSCALE_RESOLUTION_PRESETS: Any
    VIDEO_EXTENSIONS: Any
    _runtime_config: Any
    build_seedvr2_image_upscale_graph: Any
    build_cq_image_enhance_graph: Any
    build_upscale_graph: Any
    cleanup_upscale_clip_batch: Any
    concat_upscaled_clips: Any
    copy_media: Any
    ensure_ltx25_upscale_models: Any
    ensure_seedvr2_upscale_models: Any
    ensure_cq_image_enhance_models: Any
    gallery_store: Any
    gr: Any
    input_image_upscale_dimensions: Any
    load_model_config: Any
    object_info: Any
    poll_comfy_progress: Any
    postprocess_swiftvr_video: Any
    postprocess_video: Any
    prepare_upscale_clip_batch: Any
    probe_video_metadata: Any
    progress_status: Any
    random: Any
    required_seedvr2_image_upscale_nodes: Any
    required_cq_image_enhance_nodes: Any
    required_upscale_nodes: Any
    resolve_output: Any
    resolve_seedvr2_input_upscale_outputs: Any
    snapshot_path: Any
    stage_file: Any
    stream_comfy_progress: Any
    submit_prompt: Any
    time: Any
    unload_comfy_models: Any
    upscale_target_dimensions: Any
    uuid: Any
    wait_for_history: Any
    write_snapshot: Any
    absolute_gallery_media_download_url: Any
    absolute_video_download_url: Any
    absolute_video_url: Any
    delete_selected_gallery_video: Any
    empty_generated_gallery: Any
    forget_gallery_metadata: Any
    gallery_audio_paths: Any
    gallery_image_paths: Any
    gallery_image_resolution_text: Any
    gallery_media_download_path: Any
    gallery_media_mode: Any
    gallery_media_mutation_result: Any
    gallery_media_processed_result: Any
    gallery_media_progress_result: Any
    gallery_mutation_result: Any
    gallery_preview_updates: Any
    gallery_processed_result: Any
    gallery_progress_result: Any
    gallery_resolution_text: Any
    gallery_thumbnail: Any
    gallery_thumbnail_path: Any
    gallery_video_paths: Any
    generated_video_family: Any
    managed_gallery_audio_path: Any
    managed_gallery_image_path: Any
    managed_video_path: Any
    postprocess_selected_gallery_image: Any
    postprocess_selected_gallery_video: Any
    refresh_gallery: Any
    refresh_gallery_page: Any
    refresh_media_gallery: Any
    refresh_media_page: Any
    select_gallery_video: Any
    video_download_path: Any


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
        return self.services.gallery_store.video_download_path(
            video, runtime=self.services._runtime_config()
        )

    def managed_video_path(
        self, video: str | Path, *, require_file: bool = True
    ) -> Path:
        retained = self._retained_input(video)
        if retained is not None:
            return retained
        return self.services.gallery_store.managed_video_path(
            video, require_file=require_file, runtime=self.services._runtime_config()
        )

    def absolute_video_url(
        self, video: str | Path, request: gr.Request, *, download: bool = False
    ) -> str:
        relative_url = self.services.video_download_path(video)
        base_url = str(request.request.base_url).rstrip("/")
        url = f"{base_url}{relative_url}"
        return f"{url}?download=1" if download else url

    def absolute_video_download_url(
        self, video: str | Path, request: gr.Request
    ) -> str:
        return self.services.absolute_video_url(video, request, download=True)

    def gallery_video_paths(self, *, limit: int | None = 200) -> list[Path]:
        return self.services.gallery_store.gallery_video_paths(
            limit=limit, runtime=self.services._runtime_config()
        )

    def gallery_thumbnail(self, video: Path) -> Path | None:
        return self.services.gallery_store.gallery_thumbnail(
            video, runtime=self.services._runtime_config()
        )

    def gallery_thumbnail_path(self, video: str | Path) -> Path:
        return self.services.gallery_store.gallery_thumbnail_path(
            video, runtime=self.services._runtime_config()
        )

    def gallery_video_resolution(self, video: Path) -> tuple[int, int] | None:
        return self.services.gallery_store.gallery_video_resolution(
            video, runtime=self.services._runtime_config()
        )

    def gallery_resolution_text(self, video: Path) -> str:
        return self.services.gallery_store.gallery_resolution_text(
            video, runtime=self.services._runtime_config()
        )

    def generated_video_family(self, video: str | Path) -> str:
        return self.services.gallery_store.generated_video_family(
            video, runtime=self.services._runtime_config()
        )

    def forget_gallery_metadata(self, video: str | Path | None = None) -> None:
        return self.services.gallery_store.forget_gallery_metadata(video)

    def managed_gallery_image_path(
        self, image: str | Path, *, require_file: bool = True
    ) -> Path:
        retained = self._retained_input(image)
        if retained is not None:
            return retained
        return self.services.gallery_store.managed_image_path(
            image, require_file=require_file, runtime=self.services._runtime_config()
        )

    def gallery_image_paths(self, *, limit: int | None = 200) -> list[Path]:
        return self.services.gallery_store.gallery_image_paths(
            limit=limit, runtime=self.services._runtime_config()
        )

    def gallery_image_resolution_text(self, image: Path) -> str:
        return self.services.gallery_store.gallery_image_resolution_text(image)

    def managed_gallery_audio_path(
        self, audio: str | Path, *, require_file: bool = True
    ) -> Path:
        retained = self._retained_input(audio)
        if retained is not None:
            return retained
        return self.services.gallery_store.managed_audio_path(
            audio, require_file=require_file, runtime=self.services._runtime_config()
        )

    def gallery_audio_paths(self, *, limit: int | None = 200) -> list[Path]:
        return self.services.gallery_store.gallery_audio_paths(
            limit=limit, runtime=self.services._runtime_config()
        )

    def gallery_media_mode(self, mode: str) -> str:
        return str(mode) if str(mode) in {"Video", "Image", "Audio"} else "Video"

    def gallery_media_download_path(self, media: str | Path, mode: str) -> str:
        media_mode = self.services.gallery_media_mode(mode)
        if media_mode == "Image":
            return self.services.gallery_store.image_download_path(
                media, runtime=self.services._runtime_config()
            )
        if media_mode == "Audio":
            return self.services.gallery_store.audio_download_path(
                media, runtime=self.services._runtime_config()
            )
        return self.services.video_download_path(media)

    def absolute_gallery_media_download_url(
        self, media: str | Path, mode: str, request: gr.Request
    ) -> str:
        relative_url = self.services.gallery_media_download_path(media, mode)
        base_url = str(request.request.base_url).rstrip("/")
        return f"{base_url}{relative_url}?download=1"

    def import_gallery_media(self, mode: str, uploaded_media: str | None):
        media_mode = self.services.gallery_media_mode(mode)
        if not uploaded_media:
            return self.services.gallery_media_mutation_result(
                media_mode,
                f"Choose a local {media_mode.lower()} first.",
                clear_selection=False,
            )
        source = self.services.Path(uploaded_media).expanduser().resolve()
        extensions = {
            "Video": self.services.VIDEO_EXTENSIONS,
            "Image": self.services.IMAGE_EXTENSIONS,
            "Audio": self.services.AUDIO_EXTENSIONS,
        }[media_mode]
        if not source.is_file() or source.suffix.lower() not in extensions:
            return self.services.gallery_media_mutation_result(
                media_mode,
                f"The selected file is not a supported {media_mode.lower()}.",
                clear_selection=False,
            )
        destination_dir = self.services.OUTPUTS_DIR / "imports"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = (
            destination_dir
            / f"import_{int(self.services.time.time())}_{self.services.uuid.uuid4().hex[:8]}{source.suffix.lower()}"
        )
        self.services.copy_media(source, destination)
        return self.services.gallery_media_mutation_result(
            media_mode,
            f"Imported `{source.name}`",
            selected_media=str(destination),
            clear_selection=False,
        )

    def import_gallery_video(self, uploaded_video: str | None) -> GalleryMutationResult:
        if not uploaded_video:
            return self.services.gallery_mutation_result(
                "Choose a local video first.", clear_selection=False
            )
        source = self.services.Path(uploaded_video).expanduser().resolve()
        if (
            not source.is_file()
            or source.suffix.lower() not in self.services.VIDEO_EXTENSIONS
        ):
            return self.services.gallery_mutation_result(
                "The selected file is not a supported video.", clear_selection=False
            )
        destination_dir = self.services.OUTPUTS_DIR / "imports"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = (
            destination_dir
            / f"import_{int(self.services.time.time())}_{self.services.uuid.uuid4().hex[:8]}{source.suffix.lower()}"
        )
        self.services.copy_media(source, destination)
        return self.services.gallery_mutation_result(
            f"Imported `{source.name}`",
            selected_video=str(destination),
            clear_selection=False,
        )

    def refresh_gallery_page(self, limit: int = 48, *, paths=None) -> AssetPage:
        if isinstance(paths, AssetInventory) and paths.kind != "Video":
            raise ValueError("Gallery inventory does not match the media type.")
        if isinstance(paths, AssetInventory) and paths.kind == "Video":
            videos = list(paths.paths)
        else:
            videos = self.services.gallery_video_paths(limit=None)
        if paths is not None and not isinstance(paths, AssetInventory):
            selected = {self.services.Path(path).resolve() for path in paths}
            videos = [path for path in videos if path.resolve() in selected]
        shown_videos = videos[: max(0, limit)]
        items: list[tuple[str, str]] = []
        selectable_paths: list[str] = []
        failed = 0
        # One shared bound across browser sessions. Copy job context so finishing
        # callbacks retain cancellation checks inside the media subprocesses.
        posters = [
            _THUMBNAIL_WORKERS.submit(
                copy_context().run, self.services.gallery_thumbnail, video
            )
            for video in shown_videos
        ]
        for video, poster in zip(shown_videos, posters):
            thumbnail = poster.result()
            if thumbnail is None:
                failed += 1
                thumbnail = self.services.gallery_store.gallery_placeholder(
                    video, kind="Video", runtime=self.services._runtime_config()
                )
            if thumbnail is None:
                continue
            try:
                stat = video.stat()
            except OSError:
                continue
            timestamp = self.services.time.strftime(
                "%Y-%m-%d %H:%M", self.services.time.localtime(stat.st_mtime)
            )
            size_mb = stat.st_size / (1024 * 1024)
            caption = f"{self.services.generated_video_family(video)} · {video.name} · {timestamp} · {size_mb:.1f} MB"
            items.append((str(thumbnail), caption))
            selectable_paths.append(str(video))
        return AssetPage.from_scan(
            items, selectable_paths, len(videos), limit, "videos", failed
        )

    def refresh_gallery(self, limit: int = 48):
        """Compatibility adapter for existing gallery callers."""
        return self.services.refresh_gallery_page(limit).as_ui_values()

    def select_gallery_video(
        self, paths: list[str], request: gr.Request, evt: gr.SelectData
    ) -> tuple[str | None, str, str | None]:
        index = evt.index
        if isinstance(index, (tuple, list)):
            index = index[0]
        try:
            video = paths[int(index)]
        except (IndexError, TypeError, ValueError):
            return (None, "", None)
        download_url = self.services.absolute_video_download_url(video, request)
        resolution = self.services.gallery_resolution_text(
            self.services.managed_video_path(video)
        )
        return (
            video,
            f"**Resolution:** {resolution} · [Download video]({download_url})",
            video,
        )

    def list_media_paths(self, mode):
        return {
            "Video": self.services.gallery_video_paths,
            "Image": self.services.gallery_image_paths,
            "Audio": self.services.gallery_audio_paths,
        }[self.services.gallery_media_mode(mode)](limit=None)

    def refresh_media_page(
        self, mode: str = "Video", limit: int = 48, *, paths=None
    ) -> AssetPage:
        """Refresh the active gallery, defaulting to the existing video library."""
        media_mode = self.services.gallery_media_mode(mode)
        if isinstance(paths, AssetInventory) and paths.kind != media_mode:
            raise ValueError("Gallery inventory does not match the media type.")
        if media_mode == "Video":
            return (
                self.refresh_gallery_page(limit, paths=paths)
                if paths is not None
                else self.services.refresh_gallery_page(limit)
            )
        if media_mode == "Audio":
            all_audio_files = (
                list(paths.paths)
                if isinstance(paths, AssetInventory) and paths.kind == "Audio"
                else self.services.gallery_audio_paths(limit=None)
            )
            if paths is not None and not isinstance(paths, AssetInventory):
                selected = {self.services.Path(path).resolve() for path in paths}
                all_audio_files = [
                    path for path in all_audio_files if path.resolve() in selected
                ]
            audio_files = all_audio_files[: max(0, limit)]
            items: list[tuple[str, str]] = []
            selectable_paths: list[str] = []
            failed = 0
            for audio in audio_files:
                thumbnail = self.services.gallery_store.gallery_audio_thumbnail(
                    audio, runtime=self.services._runtime_config()
                )
                if thumbnail is None:
                    failed += 1
                    continue
                try:
                    stat = audio.stat()
                except OSError:
                    continue
                timestamp = self.services.time.strftime(
                    "%Y-%m-%d %H:%M", self.services.time.localtime(stat.st_mtime)
                )
                size_mb = stat.st_size / (1024 * 1024)
                family = self.services.gallery_store.generated_audio_family(
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
            else self.services.gallery_image_paths(limit=None)
        )
        if paths is not None and not isinstance(paths, AssetInventory):
            selected = {self.services.Path(path).resolve() for path in paths}
            all_images = [path for path in all_images if path.resolve() in selected]
        images = all_images[: max(0, limit)]
        items: list[tuple[str, str]] = []
        selectable_paths: list[str] = []
        failed = 0
        for image in images:
            thumbnail = self.services.gallery_store.gallery_image_thumbnail(
                image, runtime=self.services._runtime_config()
            )
            if thumbnail is None:
                failed += 1
                thumbnail = self.services.gallery_store.gallery_placeholder(
                    image, kind="Image", runtime=self.services._runtime_config()
                )
            if thumbnail is None:
                continue
            try:
                stat = image.stat()
            except OSError:
                continue
            timestamp = self.services.time.strftime(
                "%Y-%m-%d %H:%M", self.services.time.localtime(stat.st_mtime)
            )
            size_mb = stat.st_size / (1024 * 1024)
            caption = f"{self.services.gallery_store.generated_image_family(image, runtime=self.services._runtime_config())} · {image.name} · {timestamp} · {size_mb:.1f} MB"
            items.append((str(thumbnail), caption))
            selectable_paths.append(str(image))
        return AssetPage.from_scan(
            items, selectable_paths, len(all_images), limit, "images", failed
        )

    def refresh_media_gallery(self, mode: str = "Video", limit: int = 48):
        """Preserve the public tuple contract while the UI consumes typed pages."""
        return self.services.refresh_media_page(mode, limit).as_ui_values()

    def gallery_preview_updates(
        self,
        mode: str,
        *,
        video: str | None = None,
        image: str | None = None,
        audio: str | None = None,
    ) -> tuple[Any, Any, Any]:
        """Always update all preview visibility flags with the selected media."""
        media_mode = self.services.gallery_media_mode(mode)
        return (
            self.services.gr.update(value=video, visible=media_mode == "Video"),
            self.services.gr.update(value=image, visible=media_mode == "Image"),
            self.services.gr.update(value=audio, visible=media_mode == "Audio"),
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
            return (*self.services.gallery_preview_updates(mode), "", None)
        media_mode = self.services.gallery_media_mode(mode)
        if media_mode == "Image":
            resolved = self.services.managed_gallery_image_path(media)
            resolution = self.services.gallery_image_resolution_text(resolved)
            download_url = self.services.absolute_gallery_media_download_url(
                media, media_mode, request
            )
            return (
                *self.services.gallery_preview_updates(mode, image=media),
                f"**Resolution:** {resolution} · [Download image]({download_url})",
                media,
            )
        if media_mode == "Audio":
            resolved = self.services.managed_gallery_audio_path(media)
            download_url = self.services.absolute_gallery_media_download_url(
                resolved, media_mode, request
            )
            return (
                *self.services.gallery_preview_updates(mode, audio=media),
                f"[Download audio]({download_url})",
                media,
            )
        video, download, selected = self.services.select_gallery_video(
            paths, request, evt
        )
        return (
            *self.services.gallery_preview_updates(mode, video=video),
            download,
            selected,
        )

    def gallery_media_mutation_result(
        self,
        mode: str,
        message: str,
        *,
        selected_media: str | None = None,
        clear_selection: bool,
    ) -> GalleryMediaMutationResult:
        items, paths, detail = self.services.refresh_media_gallery(mode)
        video = None if clear_selection else self.services.gr.skip()
        image = None if clear_selection else self.services.gr.skip()
        audio = None if clear_selection else self.services.gr.skip()
        download = "" if clear_selection else self.services.gr.skip()
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
            self.services.gr.skip(),
            self.services.gr.skip(),
            message,
            self.services.gr.skip(),
            self.services.gr.skip(),
            self.services.gr.skip(),
            self.services.gr.skip(),
            self.services.gr.skip(),
            self.services.gr.skip(),
            message,
        )

    def gallery_media_processed_result(
        self, mode: str, result: Path, option: str, elapsed: float, request: gr.Request
    ) -> GalleryMediaPostprocessResult:
        items, paths, detail = self.services.refresh_media_gallery(mode)
        download_url = self.services.absolute_gallery_media_download_url(
            result, mode, request
        )
        if str(mode) == "Image":
            resolution = self.services.gallery_image_resolution_text(result)
            video, image, audio, noun = (None, str(result), None, "image")
        else:
            resolution = self.services.gallery_resolution_text(result)
            video, image, audio, noun = (str(result), None, None, "video")
        return (
            items,
            paths,
            f"Completed {option} in {elapsed:.1f}s · {detail}",
            *self.services.gallery_preview_updates(
                mode, video=video, image=image, audio=audio
            ),
            f"**Resolution:** {resolution} · [Download processed {noun}]({download_url})",
            str(result),
            False,
            f"Completed {option} in {elapsed:.1f}s",
        )

    def gallery_mutation_result(
        self, message: str, *, selected_video: str | None = None, clear_selection: bool
    ) -> GalleryMutationResult:
        items, paths, detail = self.services.refresh_gallery()
        player = None if clear_selection else self.services.gr.skip()
        download = "" if clear_selection else self.services.gr.skip()
        selected = None if clear_selection else selected_video
        return (
            items,
            paths,
            f"{message} · {detail}",
            player,
            download,
            selected,
            False,
        )

    def gallery_progress_result(self, message: str) -> GalleryPostprocessResult:
        return (
            self.services.gr.skip(),
            self.services.gr.skip(),
            message,
            self.services.gr.skip(),
            self.services.gr.skip(),
            self.services.gr.skip(),
            self.services.gr.skip(),
            message,
        )

    def gallery_processed_result(
        self, result: Path, option: str, elapsed: float, request: gr.Request
    ) -> GalleryPostprocessResult:
        items, paths, detail = self.services.refresh_gallery()
        download_url = self.services.absolute_video_download_url(result, request)
        resolution = self.services.gallery_resolution_text(result)
        return (
            items,
            paths,
            f"Completed {option} in {elapsed:.1f}s · {detail}",
            str(result),
            f"**Resolution:** {resolution} · [Download processed video]({download_url})",
            str(result),
            False,
            f"Completed {option} in {elapsed:.1f}s",
        )

    def postprocess_selected_gallery_video(
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
        started = self.services.time.monotonic()
        ws: websocket.WebSocket | None = None
        clip_batch: UpscaleClipBatch | None = None
        clip_outputs: list[Path] = []
        try:
            if not selected_video:
                raise self.services.H3Error("Select a gallery video first.")
            if option not in self.services.POSTPROCESS_OPTIONS:
                raise self.services.H3Error("Choose a post-processing method.")
            source = self.services.managed_video_path(selected_video)
            actual_seed = (
                self.services.random.randrange(0, 2**63 - 1)
                if int(seed) < 0
                else int(seed)
            )
            progress(0, desc=f"Preparing {option}")
            yield self.services.gallery_progress_result(
                f"Preparing `{source.name}` for {option}"
            )
            if option == self.services.SWIFTVR_UPSCALE:
                progress(0, desc="Checking SwiftVR runtime and checkpoint")
                yield self.services.gallery_progress_result(
                    "Checking SwiftVR runtime and downloading its checkpoint on first use."
                )
                metadata = self.services.probe_video_metadata(source)
                target_width, target_height = self.services.upscale_target_dimensions(
                    metadata.width, metadata.height, upscale_resolution
                )
                result = self.services.postprocess_swiftvr_video(
                    source,
                    fps=metadata.fps,
                    target_width=target_width,
                    target_height=target_height,
                )
                progress(1, desc="Complete")
                yield self.services.gallery_processed_result(
                    result, option, self.services.time.monotonic() - started, request
                )
                return
            if option not in self.services.COMFY_POSTPROCESS_OPTIONS:
                result = self.services.postprocess_video(source, option)
                progress(1, desc="Complete")
                yield self.services.gallery_processed_result(
                    result, option, self.services.time.monotonic() - started, request
                )
                return
            models = self.services.load_model_config()
            available = set(self.services.object_info())
            missing = self.services.required_upscale_nodes(option) - available
            if missing:
                raise self.services.H3Error(
                    f"{option} is unavailable. Missing ComfyUI nodes: "
                    + ", ".join(sorted(missing))
                )
            if option == self.services.SEEDVR2_UPSCALE:
                model_status = f"SeedVR2 {seedvr2_model}"
                yield self.services.gallery_progress_result(
                    f"Checking {model_status} models"
                )
                downloaded = self.services.ensure_seedvr2_upscale_models(
                    models, seedvr2_model
                )
                stage_bucket = "seedvr2_upscale"
            else:
                model_status = f"{option} with {ltx25_model}"
                yield self.services.gallery_progress_result(
                    f"Checking {model_status} models"
                )
                downloaded = self.services.ensure_ltx25_upscale_models(
                    ltx25_model, option=option
                )
                stage_bucket = (
                    self.services.LTX25_POSTPROCESS_MODELS[option]
                    if option in self.services.LTX25_SAME_RESOLUTION_OPTIONS
                    else "ltx25_upscale"
                )
            if downloaded:
                yield self.services.gallery_progress_result(
                    f"{option} models downloaded"
                )
            metadata = self.services.probe_video_metadata(source)
            target_width, target_height = (
                (metadata.width, metadata.height)
                if option in self.services.LTX25_SAME_RESOLUTION_OPTIONS
                else self.services.upscale_target_dimensions(
                    metadata.width, metadata.height, upscale_resolution
                )
            )
            use_split = option in self.services.LTX25_POSTPROCESS_MODELS and bool(
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
                yield self.services.gallery_progress_result(
                    f"Split `{source.name}` into {clip_count} LTX-safe clips (target {float(split_seconds):g}s each)"
                )
            if force_offload:
                yield self.services.gallery_progress_result(
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
            client_id = str(self.services.uuid.uuid4())
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
                    queued_at = self.services.time.time()
                    prompt_id = self.services.submit_prompt(graph, client_id)
                    clip_label = f"Clip {clip_index + 1}/{clip_count}"
                    yield self.services.gallery_progress_result(
                        self.services.progress_status(
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
                        yield self.services.gallery_progress_result(
                            self.services.progress_status(
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
                yield self.services.gallery_progress_result(
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
                yield self.services.gallery_processed_result(
                    result, option, self.services.time.monotonic() - started, request
                )
                return
            queued_at = self.services.time.time()
            prompt_id = self.services.submit_prompt(graph, client_id)
            yield self.services.gallery_progress_result(
                self.services.progress_status(
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
                yield self.services.gallery_progress_result(
                    self.services.progress_status(
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
            yield self.services.gallery_processed_result(
                result, option, self.services.time.monotonic() - started, request
            )
        except Exception as exc:
            yield self.services.gallery_progress_result(
                f"Post-processing failed: {exc}"
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
        """Enhance one selected gallery still with SeedVR2 or CQ."""
        started = self.services.time.monotonic()
        try:
            if not selected_image:
                raise self.services.H3Error("Select a gallery image first.")
            cq_image = option == LTX25_CQ_IMAGE_ENHANCER
            family = "CQ image enhancement" if cq_image else "SeedVR2 image upscale"
            if option not in {self.services.SEEDVR2_UPSCALE, LTX25_CQ_IMAGE_ENHANCER}:
                raise self.services.H3Error(
                    f"Unsupported image enhancement method: {option}"
                )
            source = self.services.managed_gallery_image_path(selected_image)
            try:
                frame_width, frame_height = self.services.UPSCALE_RESOLUTION_PRESETS[
                    str(upscale_resolution)
                ]
            except KeyError as exc:
                raise self.services.H3Error(
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
                raise self.services.H3Error(
                    f"`{source.name}` is already {source_width}×{source_height}; choose a larger target resolution."
                )
            actual_seed = (
                self.services.random.randrange(0, 2**63 - 1)
                if int(seed) < 0
                else int(seed)
            )
            yield self.services.gallery_media_progress_result(
                f"Preparing `{source.name}` for {family}"
            )
            available = set(self.services.object_info())
            required = (
                self.services.required_cq_image_enhance_nodes()
                if cq_image else self.services.required_seedvr2_image_upscale_nodes()
            )
            missing = required - available
            if missing:
                raise self.services.H3Error(
                    f"{family} requires current ComfyUI nodes: "
                    + ", ".join(sorted(missing))
                )
            models = None if cq_image else self.services.load_model_config()
            downloaded = (
                self.services.ensure_cq_image_enhance_models()
                if cq_image else self.services.ensure_seedvr2_upscale_models(models, seedvr2_model)
            )
            if downloaded:
                yield self.services.gallery_media_progress_result(
                    f"{family} models downloaded"
                )
            if force_offload:
                yield self.services.gallery_media_progress_result(
                    f"Unloading resident models before {family}"
                )
                self.services.unload_comfy_models()
            staged = self.services.stage_file(
                str(source), "gallery_image_upscale", reuse=True
            )
            output_token = self.services.uuid.uuid4().hex
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
            queued_at = self.services.time.time()
            prompt_id = self.services.submit_prompt(
                graph, str(self.services.uuid.uuid4())
            )
            yield self.services.gallery_media_progress_result(
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
                yield self.services.gallery_media_progress_result(
                    self.services.progress_status(
                        stage,
                        started=started,
                        completed_nodes=completed,
                        total_nodes=total,
                        step=step,
                        step_total=step_total,
                        configured_steps=(8 if cq_image else 1) if step is not None else None,
                        detail=f"Image upscale job `{prompt_id}`",
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
            yield self.services.gallery_media_processed_result(
                "Image",
                result,
                option,
                self.services.time.monotonic() - started,
                request,
            )
        except Exception as exc:
            yield self.services.gallery_media_progress_result(
                f"Image enhancement failed: {exc}"
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
        media_mode = self.services.gallery_media_mode(mode)
        if media_mode == "Audio":
            yield self.services.gallery_media_progress_result(
                "Audio gallery outputs are available for playback and download."
            )
            return
        if media_mode == "Image":
            yield from self.services.postprocess_selected_gallery_image(
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
        for update in self.services.postprocess_selected_gallery_video(
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
        ):
            yield (*update[:4], None, None, *update[4:])

    def delete_selected_gallery_video(
        self, selected_video: str | None, confirmed: bool
    ) -> GalleryMutationResult:
        if not confirmed:
            return self.services.gallery_mutation_result(
                "Confirm permanent deletion first.",
                selected_video=selected_video,
                clear_selection=False,
            )
        if not selected_video:
            return self.services.gallery_mutation_result(
                "Select a video to delete.", clear_selection=True
            )
        try:
            video = self.services.managed_video_path(selected_video)
            thumbnail = self.services.gallery_thumbnail_path(video)
            name = video.name
            video.unlink()
            self.services.snapshot_path(video).unlink(missing_ok=True)
            thumbnail.unlink(missing_ok=True)
            self.services.forget_gallery_metadata(video)
            return self.services.gallery_mutation_result(
                f"Deleted `{name}`.", clear_selection=True
            )
        except (self.services.H3Error, OSError) as exc:
            return self.services.gallery_mutation_result(
                f"Delete failed: {exc}", clear_selection=True
            )

    def delete_selected_gallery_media(
        self, mode: str, selected_media: str | None, confirmed: bool
    ) -> GalleryMediaMutationResult:
        media_mode = self.services.gallery_media_mode(mode)
        if media_mode != "Video":
            return self.services.gallery_media_mutation_result(
                media_mode,
                f"{media_mode} deletion is not enabled in this gallery.",
                selected_media=selected_media,
                clear_selection=False,
            )
        result = self.services.delete_selected_gallery_video(selected_media, confirmed)
        return (*result[:4], None, None, *result[4:])

    def empty_generated_gallery(
        self, selected_video: str | None, confirmed: bool
    ) -> GalleryMutationResult:
        if not confirmed:
            return self.services.gallery_mutation_result(
                "Confirm permanent deletion first.",
                selected_video=selected_video,
                clear_selection=False,
            )
        deleted = 0
        failed = 0
        for candidate in self.services.gallery_video_paths(limit=None):
            try:
                video = self.services.managed_video_path(candidate)
                self.services.gallery_thumbnail_path(video).unlink(missing_ok=True)
                video.unlink()
                self.services.snapshot_path(video).unlink(missing_ok=True)
                deleted += 1
            except (self.services.H3Error, OSError):
                failed += 1
        if self.services.GALLERY_THUMBNAILS_DIR.is_dir():
            for thumbnail in self.services.GALLERY_THUMBNAILS_DIR.iterdir():
                if thumbnail.is_file() and thumbnail.suffix.lower() in {".jpg", ".tmp"}:
                    try:
                        thumbnail.unlink()
                    except OSError:
                        failed += 1
        self.services.forget_gallery_metadata()
        result = f"Deleted {deleted} generated video{('s' if deleted != 1 else '')}."
        if failed:
            result += (
                f" {failed} file{('s' if failed != 1 else '')} could not be deleted."
            )
        return self.services.gallery_mutation_result(result, clear_selection=True)

    def empty_generated_media_gallery(
        self, mode: str, selected_media: str | None, confirmed: bool
    ) -> GalleryMediaMutationResult:
        media_mode = self.services.gallery_media_mode(mode)
        if media_mode != "Video":
            return self.services.gallery_media_mutation_result(
                media_mode,
                f"{media_mode} library deletion is not enabled.",
                selected_media=selected_media,
                clear_selection=False,
            )
        result = self.services.empty_generated_gallery(selected_media, confirmed)
        return (*result[:4], None, None, *result[4:])
