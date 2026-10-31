#!/usr/bin/env python3
"""Download and prepare default models while the app starts; run_h3.sh launches it in the background.

Groups (``--groups``, comma separated):
  h3          MiniMax H3 tab's default Video workflow
  drama3      Drama 3: SGLang-Omni environment and checkpoint and the default ASR
              (Qwen VoiceDesign and Stable Audio 3 are small; they download on first use)
  qwen_image  Qwen Image 2.1 tab's default preset

Every step is optional: a failure is reported and the step that needs the model
downloads it on first use as before. Downloads share the model manifest lock with
the app, so an on-demand download waits for an overlapping background one.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

GROUPS = ("h3", "drama3", "qwen_image")
DEFAULT_GROUPS = ("h3",)
OFF_VALUES = {"", "0", "none", "off", "false", "no"}


def log(message: str) -> None:
    print(f"[h3-prepare] {message}", flush=True)


def parse_groups(value: str) -> tuple[str, ...]:
    if value.strip().lower() in OFF_VALUES:
        return ()
    groups = []
    for name in value.replace(" ", ",").split(","):
        name = name.strip().lower().replace("-", "_")
        name = {"minimax_h3": "h3", "qwen": "qwen_image", "drama": "drama3"}.get(name, name)
        if not name:
            continue
        if name == "all":
            return GROUPS
        if name not in GROUPS:
            raise argparse.ArgumentTypeError(
                f"unknown model group {name!r}; choose from {', '.join(GROUPS)}, all or none")
        if name not in groups:
            groups.append(name)
    return tuple(groups)


def h3_model_keys() -> tuple[str, ...]:
    """The files the MiniMax H3 tab's default Text to video run loads."""
    from h3_app.catalog import STANDARD_TURBO_LORA_SETTINGS, UI_DEFAULTS
    from h3_app.policy import h3_latent_upscaler_settings
    from h3_app.resources import resolve_decoders
    from h3_models import H3_TEXT_ENCODER_CHOICES, PRELOAD_MODEL_KEYS, PROFILE_MODEL_KEYS

    defaults = UI_DEFAULTS
    keys = [*PRELOAD_MODEL_KEYS, PROFILE_MODEL_KEYS[defaults["model_profile"].lower()][0],
            H3_TEXT_ENCODER_CHOICES[defaults["text_encoder"]]]
    keys += resolve_decoders(
        defaults["result_format"], defaults["image_vae"],
        use_trt_vae=defaults["use_trt_vae"], use_int8_vae=defaults["use_int8_vae"],
        use_lynnreal_vae=defaults["use_lynnreal_vae"], use_sol_vae=defaults["use_sol_vae"],
        video_encoder=defaults["video_encoder"], lightvae_decoder=defaults["lightvae_decoder"],
    ).optional_model_keys
    turbo = STANDARD_TURBO_LORA_SETTINGS.get(defaults["turbo_variant"])
    if defaults["generation_mode"] == "Turbo" and turbo is not None:
        keys.append(turbo.lora_attr)
    if defaults["semantic_bridge"]:
        keys.append("semantic_bridge_v1")
    if defaults["latent_upscale"]:
        keys.append(h3_latent_upscaler_settings(defaults["latent_upscaler_model"])[0])
    return tuple(dict.fromkeys(keys))


def qwen_image_model_keys() -> tuple[str, ...]:
    from h3_app.catalog import QWEN_IMAGE21_DEFAULTS
    from h3_app.model_service import qwen_image21_model_keys

    defaults = QWEN_IMAGE21_DEFAULTS
    return qwen_image21_model_keys(defaults["model"], defaults["text_encoder"], defaults["turbo_variant"])


def drama3_model_keys() -> tuple[str, ...]:
    from h3_app.workflows.drama3 import drama3_asr_model_keys

    return drama3_asr_model_keys()


def sync_model_files(keys: tuple[str, ...], runtime) -> None:
    from h3_models import resolve_hf_token, route_to_ram, sync_models

    # With H3_RAM_MODELS_DIR set (Colab), the preloaded defaults go to RAM while it has room.
    route_to_ram(keys)
    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=runtime.models_config.with_name("h3_model_manifest.json"),
        token=resolve_hf_token(),
        log_prefix="[h3-prepare]",
        model_keys=keys,
    )


def prepare_sglang(runtime, log_path: Path) -> None:
    from h3_app.drama3.serving import environment

    if os.name != "posix":
        raise RuntimeError("SGLang-Omni requires Linux")

    def run(command, **kwargs):
        with log_path.open("a", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, **kwargs)

    log(f"SGLang-Omni environment: installing if needed (log: {log_path})")
    environment.prepare(environment.default_venv(runtime.script_dir), run=run, log=log)


def prepare_drama3_checkpoint(runtime) -> None:
    from h3_app.drama3.serving.server import ServerSettings, ensure_model
    from h3_models import resolve_hf_token

    settings = ServerSettings.from_environment(runtime)
    log(f"Drama 3 {settings.checkpoint} checkpoint: checking")
    # With H3_RAM_MODELS_DIR set (Colab), the ~42 GB checkpoint goes to RAM while it has room.
    ensure_model(settings, token=resolve_hf_token(), log=log, ram=True)


def plan(groups: tuple[str, ...], runtime, log_dir: Path, *, files: bool = True,
         others: bool = True) -> list[tuple[str, object]]:
    """(label, callable) steps; the model-file steps share one parallel download.

    ``files`` and ``others`` select the model-file download and the remaining steps
    (environments and checkpoints): run_h3.sh starts the files before setup, with
    only huggingface_hub installed, and the rest once setup has built the app's env.
    """
    keys: list[str] = []
    labels: list[str] = []
    for group, label, key_source in (
        ("h3", "MiniMax H3", h3_model_keys),
        ("qwen_image", "Qwen Image 2.1", qwen_image_model_keys),
        ("drama3", "Drama 3 ASR", drama3_model_keys),
    ):
        if group in groups:
            keys += key_source()
            labels.append(label)
    steps: list[tuple[str, object]] = []
    if keys and files:
        unique = tuple(dict.fromkeys(keys))
        steps.append((f"{' + '.join(labels)} model files ({len(unique)})",
                      lambda: sync_model_files(unique, runtime)))
    if "drama3" in groups and others:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        steps += [
            ("SGLang-Omni environment", lambda: prepare_sglang(runtime, log_dir / f"sglang_{stamp}.log")),
            ("Drama 3 checkpoint", lambda: prepare_drama3_checkpoint(runtime)),
        ]
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", type=parse_groups, default=DEFAULT_GROUPS,
                        help="comma-separated groups: h3, drama3, qwen_image, all or none (default: h3)")
    parser.add_argument("--install-dir", type=Path, default=SCRIPT_DIR / "h3")
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--files-only", action="store_true", help="only download the groups' model files")
    only.add_argument("--skip-files", action="store_true", help="everything but the model files")
    args = parser.parse_args(argv)
    if not args.groups:
        log("Model preparation off")
        return 0

    from h3_app.config import RuntimeConfig

    install_dir = args.install_dir.expanduser().resolve()
    env = dict(os.environ)
    env.setdefault("COMFY_DIR", str(install_dir / "ComfyUI"))
    env.setdefault("MODELS_CONFIG", str(install_dir / "h3_models.json"))
    runtime = RuntimeConfig.from_environment(SCRIPT_DIR, env)
    log_dir = install_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    steps = plan(args.groups, runtime, log_dir, files=not args.skip_files, others=not args.files_only)
    if not steps:
        return 0
    log(f"Preparing {', '.join(args.groups)} in the background: {'; '.join(label for label, _ in steps)}")
    started = time.monotonic()
    failed = []
    with ThreadPoolExecutor(max_workers=len(steps)) as pool:
        futures = {pool.submit(step): label for label, step in steps}
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
                log(f"Ready: {label}")
            except Exception as exc:
                reason = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
                failed.append(label)
                log(f"WARNING: {label} not prepared ({reason}); it downloads on first use instead")
    elapsed = time.monotonic() - started
    if failed:
        log(f"Finished in {elapsed:.0f}s; not prepared: {', '.join(failed)}")
        return 1
    log(f"All selected models ready after {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
