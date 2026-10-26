"""H3 Storage Console: CPU-only models and media for the shared Modal volume.

No application runtime, ComfyUI, torch, or GPU dependencies are imported here.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from threading import RLock
from functools import partial

from h3_models import (
    DEFAULT_LTX25_MODEL,
    DEFAULT_MUSIC3_MODEL,
    DEFAULT_QWEN_IMAGE21_MODEL,
    DEFAULT_QWEN_IMAGE21_TEXT_ENCODER,
    DEFAULT_YUE2_MODEL,
    LTX25_ICLORA_MODEL_KEYS,
    LTX25_MODEL_CHOICES,
    LTX25_SHARED_MODEL_KEYS,
    MODEL_SPECS,
    MUSIC3_MODEL_CHOICES,
    MUSIC3_SHARED_MODEL_KEYS,
    PRELOAD_MODEL_KEYS,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
    DEFAULT_FIBO2_PRECISION,
    DEFAULT_MOSS_IMAGE_MODEL,
    FIBO2_MODEL_CHOICES,
    FIBO2_TEXT_ENCODERS,
    DEFAULT_MOSS_IMAGE_TEXT_ENCODER,
    DEFAULT_MOSS_IMAGE_VAE,
    MOSS_IMAGE_MODEL_CHOICES,
    MOSS_IMAGE_TEXT_ENCODER_CHOICES,
    MOSS_IMAGE_TURBO_MODES,
    MOSS_IMAGE_VAE_CHOICES,
    YUE2_MODEL_CHOICES,
    build_model_config,
    model_file_matches_manifest,
    model_manifest_key,
    model_manifest_lock,
    read_json,
    sync_models,
    write_json_atomic,
)


AUDIO_TOOLS_FAMILY = "Qwen VoiceDesign / Stable Audio"


@dataclass(frozen=True)
class SnapshotSpec:
    """A whole Hugging Face repository that its runtime downloads on first use."""

    family: str
    repo_id: str
    description: str
    # Each callable takes the models root, so the manager and runtime share one layout.
    path: Callable[[Path], Path]
    ready: Callable[[Path], bool]
    download: Callable[[Path], None]


def _audio_tools_path(model: str, root: Path) -> Path:
    from h3_app import audio_tools_worker

    if model == "voice_design":
        return audio_tools_worker.voice_design_cache(root)
    return audio_tools_worker.stable_audio_dir(root)


def _audio_tools_ready(model: str, root: Path) -> bool:
    from h3_app import audio_tools_worker

    return audio_tools_worker.weights_ready(model, root)


def _audio_tools_download(model: str, root: Path) -> None:
    from h3_app import audio_tools_worker

    # Stable Audio is gated; huggingface_hub reads HF_TOKEN from the Modal Secret.
    audio_tools_worker.download(model, str(root))


def _drama3_settings(root: Path, checkpoint: str = "fp8"):
    from h3_app.drama3.serving.server import ServerSettings

    # Match the GPU endpoint's H3_DRAMA3_SGLANG_MODELS and H3_DRAMA3_SGLANG_DATA.
    return ServerSettings(venv=Path(), data_dir=root.parent / "drama3-serving",
                          model_root=root / "drama3_serving", checkpoint=checkpoint)


def _drama3_path(checkpoint: str, root: Path) -> Path:
    return _drama3_settings(root, checkpoint).model_dir


def _drama3_ready(checkpoint: str, root: Path) -> bool:
    return (_drama3_path(checkpoint, root) / ".h3-complete").is_file()


def _drama3_download(checkpoint: str, root: Path) -> None:
    from h3_app.drama3.serving.server import ensure_model

    ensure_model(_drama3_settings(root, checkpoint), token=os.environ.get("HF_TOKEN"),
                 log=lambda message: print(f"[storage-console] {message}", flush=True))


def _drama3_spec(checkpoint: str, description: str) -> SnapshotSpec:
    from h3_app.drama3.serving.server import CHECKPOINTS

    pin = CHECKPOINTS[checkpoint]
    return SnapshotSpec(
        "Drama 3", pin.repo, f"{description} (~{pin.download_gb:.0f} GB)",
        partial(_drama3_path, checkpoint), partial(_drama3_ready, checkpoint),
        partial(_drama3_download, checkpoint),
    )


def _snapshot_specs() -> dict[str, SnapshotSpec]:
    return {
        "drama3_sglang_checkpoint": _drama3_spec("fp8", "Drama 3 official FP8 checkpoint for SGLang-Omni"),
        "drama3_sglang_int4": _drama3_spec(
            "int4", "Drama 3 official int4 checkpoint (H3_DRAMA3_SGLANG_CHECKPOINT=int4)",
        ),
        "qwen_voice_design": SnapshotSpec(
            AUDIO_TOOLS_FAMILY, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen3-TTS VoiceDesign",
            partial(_audio_tools_path, "voice_design"), partial(_audio_tools_ready, "voice_design"),
            partial(_audio_tools_download, "voice_design"),
        ),
        "stable_audio3_medium": SnapshotSpec(
            AUDIO_TOOLS_FAMILY, "stabilityai/stable-audio-3-medium", "Stable Audio 3 Medium (gated)",
            partial(_audio_tools_path, "stable_audio"), partial(_audio_tools_ready, "stable_audio"),
            partial(_audio_tools_download, "stable_audio"),
        ),
    }


SNAPSHOT_SPECS = _snapshot_specs()


def _directory_size(path: Path) -> int:
    """Count regular files once; the Hugging Face cache links snapshots to blobs."""
    total = 0
    for folder, _dirs, files in os.walk(path):
        for name in files:
            file = os.path.join(folder, name)
            if not os.path.islink(file):
                total += os.path.getsize(file)
    return total


def _catalog_location(key: str, root: Path = Path("models")) -> tuple[str, str]:
    """Folder and file (or repository directory) shown for one catalog entry."""
    if key in MODEL_SPECS:
        return MODEL_SPECS[key].folder, MODEL_SPECS[key].local_name
    relative = SNAPSHOT_SPECS[key].path(root).relative_to(root)
    folder = relative.parent.as_posix()
    return ("" if folder == "." else folder), relative.name + "/"


def catalog_groups() -> dict[str, tuple[str, ...]]:
    """Group every model and its adapters under the family that uses them."""
    prefixes = {
        "MiniMax H3": (),
        "Qwen Image 2.1": ("qwen_image21_",),
        "LTX 2.5": ("ltx25_",),
        "Music 3": ("music3_",),
        "YuE2": ("yue2_",),
        "FLUX 3 Image Klein": ("flux3_",),
        "MOSS-Image": ("moss_image_",),
        "Fibo 2.0": ("fibo2_",),
        "Drama 3": ("drama3_",),
        AUDIO_TOOLS_FAMILY: (),
        "SeedVR2": ("seedvr2_",),
    }
    groups = {family: [] for family in prefixes}
    for key in MODEL_SPECS:
        family = next(
            (name for name, starts in prefixes.items() if starts and key.startswith(starts)),
            "MiniMax H3",
        )
        groups[family].append(key)
    for key, spec in SNAPSHOT_SPECS.items():
        groups[spec.family].append(key)
    return {family: tuple(keys) for family, keys in groups.items()}


def grouped_inventory(rows) -> list[list]:
    by_key = {row[0]: row for row in rows}
    return [[by_key[key] for key in keys if key in by_key]
            for keys in catalog_groups().values()]


def select_catalog(family: str, selected) -> list[str]:
    return list(dict.fromkeys([*(selected or []), *catalog_groups()[family]]))


def model_presets() -> dict[str, tuple[str, ...]]:
    qwen = (
        QWEN_IMAGE21_MODEL_CHOICES[DEFAULT_QWEN_IMAGE21_MODEL],
        QWEN_IMAGE21_TEXT_ENCODER_CHOICES[DEFAULT_QWEN_IMAGE21_TEXT_ENCODER],
        "qwen_image21_vae",
    )
    ltx = (LTX25_MODEL_CHOICES[DEFAULT_LTX25_MODEL], *LTX25_SHARED_MODEL_KEYS)
    moss = (
        MOSS_IMAGE_MODEL_CHOICES[DEFAULT_MOSS_IMAGE_MODEL],
        MOSS_IMAGE_TEXT_ENCODER_CHOICES[DEFAULT_MOSS_IMAGE_TEXT_ENCODER],
        MOSS_IMAGE_VAE_CHOICES[DEFAULT_MOSS_IMAGE_VAE],
    )
    fibo = (FIBO2_MODEL_CHOICES["Dev"][DEFAULT_FIBO2_PRECISION], FIBO2_TEXT_ENCODERS["generate"], "flux3_vae")
    return {
        "MiniMax H3 defaults": tuple(dict.fromkeys((
            *PRELOAD_MODEL_KEYS, "turbo_lora", "video_vae_lynnreal_int8",
            "semantic_bridge_v1",
        ))),
        "Qwen Image 2.1 defaults": qwen,
        "Qwen Image 2.1 + Turbo LoRAs": (
            *qwen, "qwen_image21_official_turbo_lora",
            "qwen_image21_lightning_lora",
            "qwen_image21_viggle_v02_lora", "qwen_image21_viggle_v03_lora",
        ),
        "LTX 2.5 defaults": ltx,
        "LTX 2.5 + IC-LoRAs": (*ltx, *LTX25_ICLORA_MODEL_KEYS),
        "Music 3 defaults": (
            MUSIC3_MODEL_CHOICES[DEFAULT_MUSIC3_MODEL], *MUSIC3_SHARED_MODEL_KEYS,
        ),
        "YuE2 defaults": (YUE2_MODEL_CHOICES[DEFAULT_YUE2_MODEL],),
        "FLUX 3 defaults": ("flux3_nvfp4", "flux3_text_encoder", "flux3_vae", "flux3_upsampler"),
        "MOSS-Image defaults": moss,
        "MOSS-Image + Turbo LoRAs": (
            *moss, *dict.fromkeys(key for key, _steps in MOSS_IMAGE_TURBO_MODES.values()),
        ),
        "Fibo 2.0 defaults": fibo,
        "Fibo 2.0 + editing": (
            *fibo, FIBO2_MODEL_CHOICES["Edit Turbo"][DEFAULT_FIBO2_PRECISION], FIBO2_TEXT_ENCODERS["edit"],
        ),
        "Drama 3 SGLang-Omni checkpoint": ("drama3_sglang_checkpoint",),
        "Drama 3 SGLang-Omni int4 checkpoint": ("drama3_sglang_int4",),
        "Qwen VoiceDesign + Stable Audio": ("qwen_voice_design", "stable_audio3_medium"),
    }


def preset_selection(names) -> list[str]:
    presets = model_presets()
    return list(dict.fromkeys(key for name in (names or []) for key in presets[name]))


class ModelManager:
    def __init__(self, data: Path, volume):
        self.data = Path(data)
        self.root = self.data / "models"
        self.manifest = self.data / "h3_model_manifest.json"
        self.config = self.data / "h3_models.json"
        self.volume = volume
        # Reload must never overlap an open file or another operation here.
        self.lock = RLock()

    def _keys(self, keys) -> tuple[str, ...]:
        selected = tuple(dict.fromkeys(keys or []))
        unknown = set(selected) - MODEL_SPECS.keys() - SNAPSHOT_SPECS.keys()
        if unknown:
            raise ValueError("Unknown models: " + ", ".join(sorted(unknown)))
        # Some profiles refer to the same physical checkpoint. Manage it once.
        unique = {}
        for key in selected:
            unique.setdefault(model_manifest_key(MODEL_SPECS[key]) if key in MODEL_SPECS else key, key)
        return tuple(unique.values())

    def _path(self, key: str) -> Path:
        if key in SNAPSHOT_SPECS:
            path = SNAPSHOT_SPECS[key].path(self.root)
            resolved, root = path.resolve(), self.root.resolve()
            # A repository directory is removed recursively, so never the root itself.
            if resolved == root or not resolved.is_relative_to(root):
                raise ValueError("Model path escapes the models directory")
            return path
        spec = MODEL_SPECS[key]
        path = self.root / spec.folder / spec.local_name
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Model path escapes the models directory")
        return path

    def _inventory(self):
        manifest = read_json(self.manifest, {"files": {}})
        if not isinstance(manifest, dict):
            manifest = {"files": {}}
        rows = []
        for key, spec in MODEL_SPECS.items():
            path = self._path(key)
            exists = path.is_file()
            ready = model_file_matches_manifest(self.root, manifest, spec)
            rows.append([
                key, spec.folder, spec.local_name,
                "Installed" if ready else "Unverified / incomplete" if exists else "Missing",
                round(path.stat().st_size / 1024**3, 3) if exists else 0,
                spec.repo_id,
            ])
        for key, spec in SNAPSHOT_SPECS.items():
            path = self._path(key)
            exists = path.is_dir()
            folder, name = _catalog_location(key, self.root)
            rows.append([
                key, folder, name,
                "Installed" if spec.ready(self.root) else "Unverified / incomplete" if exists else "Missing",
                round(_directory_size(path) / 1024**3, 3) if exists else 0,
                spec.repo_id,
            ])
        return rows

    def refresh(self):
        with self.lock:
            self.volume.reload()
            return self._inventory()

    def download(self, keys, progress=lambda *_args, **_kwargs: None):
        selected = self._keys(keys)
        with self.lock:
            self.volume.reload()
            if not selected:
                return "Select models to download.", self._inventory()
            for key in selected:
                self._path(key)
            failures = []
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                write_json_atomic(self.config, build_model_config(self.manifest.name))
                self.volume.commit()
                for index, key in enumerate(selected):
                    progress(index / len(selected), desc=f"Checking / downloading {key}")
                    try:
                        if key in SNAPSHOT_SPECS:
                            # The runtime's own downloader keeps its layout and completion marker.
                            SNAPSHOT_SPECS[key].download(self.root)
                        else:
                            # One worker keeps resource use modest on default CPU/RAM.
                            # Each file retains the shared resumable/version-aware logic.
                            sync_models(
                                root=self.root, manifest_path=self.manifest,
                                model_keys=(key,), download_workers=1, metadata_workers=1,
                                log_prefix="[storage-console]",
                            )
                    except Exception as exc:
                        failures.append(f"{key}: {exc}")
                    # Completed files survive a later timeout or container stop.
                    self.volume.commit()
            finally:
                # Preserve successful files even if another download failed.
                self.volume.commit()
            progress(1, desc="Batch complete")
            status = f"Batch complete: {len(selected) - len(failures)}/{len(selected)} files ready."
            if failures:
                status += "\n\nFailed (select and retry):\n" + "\n".join(failures)
            return status, self._inventory()

    def remove(self, keys, confirmed):
        selected = self._keys(keys)
        with self.lock:
            self.volume.reload()
            if not selected or not confirmed:
                return "Select models and confirm removal first.", self._inventory()
            # Validate every destination before changing any files.
            paths = [(key, self._path(key)) for key in selected]
            with model_manifest_lock(self.manifest):
                manifest = read_json(self.manifest, {"schema_version": 1, "files": {}})
                if not isinstance(manifest, dict):
                    manifest = {"schema_version": 1, "files": {}}
                if not isinstance(manifest.get("files"), dict):
                    manifest["files"] = {}
                try:
                    for key, path in paths:
                        if key in SNAPSHOT_SPECS:
                            # Repository snapshots are tracked by their own markers.
                            if path.is_dir():
                                shutil.rmtree(path)
                            continue
                        path.unlink(missing_ok=True)
                        # Forget a file only after it is really gone.
                        manifest["files"].pop(model_manifest_key(MODEL_SPECS[key]), None)
                finally:
                    write_json_atomic(self.manifest, manifest)
                    # Persist whatever was removed, even after a mid-batch error.
                    self.volume.commit()
            return f"Removed {len(selected)} selected model files.", self._inventory()


TITLE = "H3 Storage Console"


def build_ui(manager: ModelManager, media=None):
    """Models, plus the GPU-free Media library when ``media`` is (runtime, store)."""
    import gradio as gr

    with gr.Blocks(title=TITLE) as ui:
        gr.Markdown(
            f"# {TITLE}\n"
            "Prepare models and manage generated media on the shared Modal volume "
            "without starting the GPU endpoint. CPU and RAM use Modal defaults.\n\n"
            "Stop the GPU endpoint before changing models, or before importing, "
            "annotating or deleting media. Start it again afterwards so it sees the "
            "latest files."
        )
        with gr.Tabs():
            with gr.Tab("Models"):
                _build_models(ui, manager)
            if media is not None:
                from h3_ui.media_console import build_media_library

                runtime, store = media
                media_tab = gr.Tab("Media library")
                build_media_library(
                    media_tab, runtime, store, reload=partial(reload_volume, manager),
                )
    return ui.queue(default_concurrency_limit=1)


def reload_volume(manager: ModelManager) -> str:
    """See the GPU endpoint's newer commits without waiting behind a model batch."""
    if not manager.lock.acquire(blocking=False):
        return "A model batch is running. Reload after it finishes."
    try:
        manager.volume.reload()
    except Exception as exc:
        return f"Reload failed: {exc}. Try again in a moment."
    finally:
        manager.lock.release()
    return "Reloaded. Choose **Refresh library** to show the latest files."


def _build_models(ui, manager: ModelManager):
    import gradio as gr

    groups = catalog_groups()
    choices = [
        (f"{family} · {' / '.join(filter(None, _catalog_location(key, manager.root)))} ({key})", key)
        for family, keys in groups.items() for key in keys
    ]
    gr.Markdown(
        "Prepare models and LoRAs on CPU before opening the generation endpoint. "
        "Files are saved to the shared Modal volume. Gated models require accepting "
        "their Hugging Face licenses and a valid HF_TOKEN in the Modal Secret."
    )
    presets = gr.Dropdown(
        list(model_presets()), label="Presets", multiselect=True,
        value=["MiniMax H3 defaults"],
    )
    apply = gr.Button("Select preset models")
    selected = gr.Dropdown(
        choices, label="Models / LoRAs to manage", multiselect=True,
        value=preset_selection(["MiniMax H3 defaults"]),
        elem_id="model-selection",
    )
    with gr.Row():
        download = gr.Button("Download / update selected", variant="primary")
        refresh = gr.Button("Refresh inventory")
        select_installed = gr.Button("Select installed files")
        clear = gr.Button("Clear selection")
    with gr.Row():
        confirm = gr.Checkbox(label="Confirm deletion of selected files", value=False)
        remove = gr.Button("Remove selected", variant="stop")
    status = gr.Textbox(label="Batch status", interactive=False, lines=5)
    gr.Markdown("## Model catalog\nBrowse checkpoints, encoders, VAEs, and LoRAs by model family.")
    inventories = []
    family_buttons = []
    with gr.Tabs():
        for family, keys in groups.items():
            with gr.Tab(family):
                gr.Markdown(f"**{len(keys)} catalog entries**")
                family_buttons.append((family, gr.Button(f"Select all {family} models")))
                inventories.append(gr.Dataframe(
                    headers=["Key", "Folder", "File", "Status", "Size (GiB)", "Hugging Face repository"],
                    datatype=["str", "str", "str", "str", "number", "str"],
                    interactive=False, label=f"{family} inventory",
                ))

    def refresh_catalog():
        return grouped_inventory(manager.refresh())

    def download_selected(keys, progress=gr.Progress(track_tqdm=True)):
        message, rows = manager.download(keys, progress)
        return message, *grouped_inventory(rows)

    def remove_selected(keys, confirmed):
        message, rows = manager.remove(keys, confirmed)
        return message, *grouped_inventory(rows), False

    # Every volume operation shares a queue as well as the process lock.
    operation = dict(concurrency_id="model-volume", concurrency_limit=1)
    apply.click(preset_selection, presets, selected)
    clear.click(lambda: [], outputs=selected)
    for family, button in family_buttons:
        button.click(partial(select_catalog, family), selected, selected)
    download.click(download_selected, selected, [status, *inventories], **operation)
    remove.click(remove_selected, [selected, confirm], [status, *inventories, confirm], **operation)
    refresh.click(refresh_catalog, outputs=inventories, **operation)
    select_installed.click(
        lambda: [row[0] for row in manager.refresh() if row[3] != "Missing"],
        outputs=selected, **operation,
    )
    ui.load(refresh_catalog, outputs=inventories, **operation)


def serve(data: Path, volume, *, port: int, script_dir: Path) -> None:
    """Serve models and media with the main application's file routes, without ComfyUI."""
    import threading

    import uvicorn

    from h3_app import server as server_routes
    from h3_app.catalog import (
        AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, UVICORN_WEBSOCKET_OPTIONS, VIDEO_EXTENSIONS,
    )
    from h3_app.workspace_store import WorkspaceStore
    from h3_ui.media_console import media_runtime
    from h3_ui.styles import H3_SETUP_CSS

    manager = ModelManager(data, volume)
    runtime = media_runtime(data, script_dir)
    for folder in (runtime.input_dir, runtime.output_dir, runtime.outputs_dir):
        folder.mkdir(parents=True, exist_ok=True)
    # Same state directory as the GPU endpoint's H3_WORKSPACE_DIR.
    store = WorkspaceStore(Path(data) / "h3-workspace")
    ui = build_ui(manager, (runtime, store))
    app = server_routes.build_server(
        ui,
        [str(runtime.output_dir.resolve()), str(runtime.outputs_dir.resolve())],
        server_routes.ServerConfig(
            # No ComfyUI runs here; its proxy routes report it as unavailable.
            runtime.comfy_url, runtime.output_dir, runtime.outputs_dir, {},
            runtime.workflow_dir, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS,
            AUDIO_EXTENSIONS, H3_SETUP_CSS, runtime.gallery_thumbnails_dir,
        ),
    )
    server = uvicorn.Server(uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="warning",
        proxy_headers=True, forwarded_allow_ips="*", **UVICORN_WEBSOCKET_OPTIONS,
    ))
    threading.Thread(target=server.run, name="storage-console", daemon=True).start()
