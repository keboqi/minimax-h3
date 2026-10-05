"""CPU-only model management for the shared Modal volume.

No application runtime, ComfyUI, torch, or GPU dependencies are imported here.
"""
from __future__ import annotations

from pathlib import Path
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
    YUE2_MODEL_CHOICES,
    build_model_config,
    model_file_matches_manifest,
    model_manifest_key,
    model_manifest_lock,
    read_json,
    sync_models,
    write_json_atomic,
)


def catalog_groups() -> dict[str, tuple[str, ...]]:
    """Group every model and its adapters under the family that uses them."""
    prefixes = {
        "MiniMax H3": (),
        "Qwen Image 2.1": ("qwen_image21_",),
        "LTX 2.5": ("ltx25_",),
        "Music 3": ("music3_",),
        "YuE2": ("yue2_",),
        "SeedVR2": ("seedvr2_",),
    }
    groups = {family: [] for family in prefixes}
    for key in MODEL_SPECS:
        family = next(
            (name for name, starts in prefixes.items() if starts and key.startswith(starts)),
            "MiniMax H3",
        )
        groups[family].append(key)
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
    return {
        "MiniMax H3 defaults": tuple(dict.fromkeys((
            *PRELOAD_MODEL_KEYS, "turbo_lora", "video_vae_lynnreal_int8",
            "semantic_bridge_v1",
        ))),
        "Qwen Image 2.1 defaults": qwen,
        "Qwen Image 2.1 + Turbo LoRAs": (
            *qwen, "qwen_image21_viggle_v02_lora", "qwen_image21_viggle_v03_lora",
        ),
        "LTX 2.5 defaults": ltx,
        "LTX 2.5 + IC-LoRAs": (*ltx, *LTX25_ICLORA_MODEL_KEYS),
        "Music 3 defaults": (
            MUSIC3_MODEL_CHOICES[DEFAULT_MUSIC3_MODEL], *MUSIC3_SHARED_MODEL_KEYS,
        ),
        "YuE2 defaults": (YUE2_MODEL_CHOICES[DEFAULT_YUE2_MODEL],),
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
        unknown = set(selected) - MODEL_SPECS.keys()
        if unknown:
            raise ValueError("Unknown models: " + ", ".join(sorted(unknown)))
        # Some profiles refer to the same physical checkpoint. Manage it once.
        unique = {}
        for key in selected:
            unique.setdefault(model_manifest_key(MODEL_SPECS[key]), key)
        return tuple(unique.values())

    def _path(self, key: str) -> Path:
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
                        # One worker keeps resource use modest on default CPU/RAM.
                        # Each file retains the shared resumable/version-aware logic.
                        sync_models(
                            root=self.root, manifest_path=self.manifest,
                            model_keys=(key,), download_workers=1, metadata_workers=1,
                            log_prefix="[modal-model-manager]",
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
                        path.unlink(missing_ok=True)
                        manifest["files"].pop(model_manifest_key(MODEL_SPECS[key]), None)
                finally:
                    write_json_atomic(self.manifest, manifest)
            self.volume.commit()
            return f"Removed {len(selected)} selected model files.", self._inventory()


def build_ui(manager: ModelManager):
    import gradio as gr

    groups = catalog_groups()
    choices = [
        (f"{family} · {MODEL_SPECS[key].folder} / {MODEL_SPECS[key].local_name} ({key})", key)
        for family, keys in groups.items() for key in keys
    ]
    with gr.Blocks(title="H3 Model Manager") as ui:
        gr.Markdown(
            "# Model Manager\n"
            "Prepare models and LoRAs on CPU before opening the generation endpoint. "
            "Files are saved to the shared Modal volume. CPU and RAM use Modal defaults.\n\n"
            "Stop the GPU endpoint before changing models. Start it again after the batch "
            "finishes so it sees the latest files. Gated models require accepting their "
            "Hugging Face licenses and a valid HF_TOKEN in the Modal Secret."
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
    return ui.queue(default_concurrency_limit=1)
