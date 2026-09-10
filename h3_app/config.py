"""Immutable runtime paths and transport limits, loaded once at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RuntimeConfig:
    script_dir: Path
    comfy_url: str
    comfy_dir: Path
    models_config: Path
    outputs_dir: Path
    attention_backend: str = "sol"
    dense_attention_backend: str = "comfy-kitchen"
    memory_profile: str = "unknown"
    request_timeout: float = 60
    generation_timeout: float = 10800
    poll_seconds: float = 1
    gallery_limit: int = 200
    gallery_metadata_cache_limit: int = 512

    input_root: Path | None = None
    output_root: Path | None = None
    thumbnail_root: Path | None = None

    @property
    def input_dir(self):
        return (
            self.input_root if self.input_root is not None else self.comfy_dir / "input"
        )

    @property
    def output_dir(self):
        return (
            self.output_root
            if self.output_root is not None
            else self.comfy_dir / "output"
        )

    @property
    def gallery_thumbnails_dir(self):
        return (
            self.thumbnail_root
            if self.thumbnail_root is not None
            else self.outputs_dir / ".gallery_thumbnails"
        )

    @property
    def workflow_dir(self):
        return self.comfy_dir / "user" / "default" / "workflows" / "LTX 2.5"

    @property
    def prompt_system_path(self):
        return self.script_dir / "prompt.txt"

    @property
    def prompt_systems(self):
        return {
            "MiniMax H3": self.prompt_system_path,
            "MiniMax Music 3": self.script_dir / "prompt_music3.txt",
            "LTX-2.5": self.script_dir / "prompt_ltx25.txt",
        }

    @classmethod
    def from_environment(cls, script_dir: Path, env: Mapping[str, str] | None = None):
        env = os.environ if env is None else env
        script_dir = script_dir.resolve()
        candidates = [
            script_dir / "h3/ComfyUI",
            script_dir / "ComfyUI",
            Path.cwd() / "h3/ComfyUI",
            Path.cwd() / "ComfyUI",
        ]
        comfy = (
            Path(env["COMFY_DIR"]).expanduser().resolve()
            if env.get("COMFY_DIR")
            else next(
                (p.resolve() for p in candidates if (p / "main.py").is_file()),
                candidates[0].resolve(),
            )
        )
        gallery_limit = max(1, int(env.get("GRADIO_GALLERY_LIMIT", "200")))
        return cls(
            script_dir,
            env.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/"),
            comfy,
            Path(env.get("MODELS_CONFIG", str(comfy.parent / "h3_models.json")))
            .expanduser()
            .resolve(),
            Path(env.get("GRADIO_OUTPUT_DIR", str(comfy.parent / "gradio_outputs")))
            .expanduser()
            .resolve(),
            env.get("SERVER_ATTENTION_BACKEND", "sol").lower(),
            env.get("SERVER_DENSE_ATTENTION_BACKEND", "comfy-kitchen").lower(),
            env.get("SERVER_MEMORY_PROFILE", "unknown").lower(),
            float(env.get("REQUEST_TIMEOUT", "60")),
            float(env.get("GENERATION_TIMEOUT", "10800")),
            float(env.get("POLL_SECONDS", "1")),
            gallery_limit,
            max(
                gallery_limit,
                int(env.get("GRADIO_GALLERY_METADATA_CACHE_LIMIT", "512")),
            ),
        )
