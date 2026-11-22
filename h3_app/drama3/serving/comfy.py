"""Sequential Drama 3 serving through ComfyUI (ComfyUI-Drama3 nodes)."""

from __future__ import annotations

from pathlib import Path
import secrets
import threading
import uuid

import numpy as np
import requests

from h3_app.comfy import ComfyClient
from h3_app.errors import H3Error
from h3_app.execution import ExecutionRunner
from h3_app.jobs import CURRENT_JOB, JOBS
from h3_app.media import walk_saved_refs
from h3_app.model_service import ensure_models
from h3_app.workflows.drama3 import (
    DRAMA3_COMFY_REQUIRED_NODES,
    build_drama3_comfy_graph,
    drama3_comfy_model_keys,
)
from h3_gpu import CPU_ONLY_MESSAGE, is_cpu_only
from h3_models import DEFAULT_DRAMA3_COMFY_MODEL, DRAMA3_COMFY_CODEC_KEY, DRAMA3_COMFY_MODELS, MODEL_SPECS
from h3_requirements import DRAMA3_REF

from . import audio, server

COMFY_SAMPLE_RATE = 44100
TEMP_AUDIO_PREFIX = ".processing/drama3_comfy"


class ComfyDrama3Server:
    """Adapter presenting ComfyUI Drama 3 with the synchronous chunk speech interface.

    ComfyUI executes one prompt at a time, so callers must keep ``max_workers = 1``.
    Model, codec, and encoded reference voices stay cached inside ComfyUI across
    sequential chunks until another workflow needs VRAM.
    """

    label = "ComfyUI"
    max_workers = 1

    def __init__(
        self,
        runtime,
        *,
        model_key: str = DEFAULT_DRAMA3_COMFY_MODEL,
        experts: str = "in VRAM",
    ):
        if model_key not in DRAMA3_COMFY_MODELS:
            raise H3Error(f"Unsupported ComfyUI Drama 3 model: {model_key!r}.")
        self.runtime = runtime
        self.model_key = model_key
        self.experts = experts
        self.model_file = MODEL_SPECS[model_key].local_name
        self.codec_file = MODEL_SPECS[DRAMA3_COMFY_CODEC_KEY].local_name
        self._lock = threading.Lock()

    @property
    def take_engine_id(self) -> str:
        return f"comfyui:{self.model_key}"

    @property
    def refs_dir(self) -> Path:
        return Path(self.runtime.input_dir) / "drama3_comfy"

    def stage_reference(self, path: str | Path) -> Path:
        """Stage a reference clip inside ComfyUI's input directory by content digest."""
        from . import generate

        return generate.stage_reference(str(path), self.refs_dir)

    def _comfy_input_name(self, reference: Path | str | None) -> str | None:
        if reference is None:
            return None
        path = Path(reference)
        input_root = Path(self.runtime.input_dir).resolve()
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if not resolved.is_relative_to(input_root):
            resolved = self.stage_reference(path).resolve()
        return resolved.relative_to(input_root).as_posix()

    def ensure_ready(
        self,
        free_gpu,
        progress=lambda _message: None,
        *,
        hf_token=None,
        evict: bool = True,
        warmup: bool = False,
    ) -> None:
        """Stop SGLang-Omni, verify ComfyUI nodes and weights, optionally warm VRAM."""
        del free_gpu, hf_token  # Never unload ComfyUI's own loaded models.
        if is_cpu_only():
            raise H3Error(CPU_ONLY_MESSAGE)
        server.stop_all()
        if evict:
            server.reclaim_audio_tools()
        progress("Checking ComfyUI Drama 3 nodes")
        client = ComfyClient(self.runtime.comfy_url, timeout=self.runtime.request_timeout)
        try:
            available = set(client.object_info())
        except requests.RequestException as exc:
            raise H3Error(f"ComfyUI is not reachable at {self.runtime.comfy_url}: {exc}") from exc
        missing = sorted(set(DRAMA3_COMFY_REQUIRED_NODES) - available)
        if missing:
            raise H3Error(
                "Update setup and restart ComfyUI (ComfyUI-Drama3 revision "
                f"{DRAMA3_REF[:7]} is required). Missing nodes: {', '.join(missing)}"
            )
        progress(f"Checking Drama 3 model ({self.model_file})")
        ensure_models(
            drama3_comfy_model_keys(self.model_key),
            runtime=self.runtime,
            log_prefix="[drama3-comfy-on-demand]",
        )
        if warmup:
            progress("Warming up Drama 3 in ComfyUI")
            warmup_payload = self.payload(
                text=".",
                reference=None,
                transcript=None,
                max_new_tokens=1,
                sampling={"temperature": 1.0, "top_p": 0.9, "top_k": 0},
            )
            self._run_speech(warmup_payload, allow_empty=True)

    def payload(
        self,
        *,
        text: str,
        reference: Path | str | None,
        transcript: str | None,
        max_new_tokens: int,
        sampling: dict,
    ) -> dict:
        """Build sequential synthesis arguments for ``speech()`` with a fresh seed."""
        seed = sampling.get("seed")
        if seed is None:
            seed = secrets.randbelow(1 << 53)
        return {
            "text": text,
            "reference_audio": self._comfy_input_name(reference),
            "reference_transcript": str(transcript or ""),
            "seed": int(seed),
            "temperature": float(sampling.get("temperature", 1.0)),
            "top_p": float(sampling.get("top_p", 0.9)),
            "top_k": int(sampling.get("top_k", 0)),
            "max_frames": int(max_new_tokens),
        }

    def speech(self, payload: dict) -> bytes:
        """Synthesize one chunk through ComfyUI and return PCM16 44.1 kHz WAV bytes."""
        return self._run_speech(payload, allow_empty=False)

    def _run_speech(self, payload: dict, *, allow_empty: bool) -> bytes:
        with self._lock:
            prefix = f"{TEMP_AUDIO_PREFIX}/take_{uuid.uuid4().hex[:12]}"
            graph, save_node = build_drama3_comfy_graph(
                text=payload["text"],
                model_file=self.model_file,
                codec_file=self.codec_file,
                seed=payload["seed"],
                temperature=payload["temperature"],
                top_p=payload["top_p"],
                top_k=payload["top_k"],
                max_frames=payload["max_frames"],
                reference_audio=payload.get("reference_audio"),
                reference_transcript=payload.get("reference_transcript", ""),
                experts=self.experts,
                filename_prefix=prefix,
            )
            client = ComfyClient(self.runtime.comfy_url, timeout=self.runtime.request_timeout)
            runner = ExecutionRunner(self.runtime, client, JOBS)
            saved_path: Path | None = None
            active_prompt: str | None = None
            job = CURRENT_JOB.get()
            try:
                if job is not None:
                    job.check()
                # Submit without mutating job.ledger/replay_stage across multi-chunk loops;
                # attach job.check and bracket job.prompt_id so Interrupt cancels the chunk.
                prompt_id = runner.submit(graph, str(uuid.uuid4()), job=None)
                active_prompt = str(prompt_id)
                if job is not None:
                    job.prompt_id = active_prompt
                submission = getattr(prompt_id, "submission", None)
                if submission is not None:
                    if job is not None:
                        submission.check_cancelled = job.check
                    for _update in submission.progress():
                        pass
                    history = submission.history()
                else:
                    raise H3Error("ComfyUI submission did not provide progress tracking.")
                saved_path = self._resolve_saved_audio(history, save_node)
                if saved_path is None or not saved_path.is_file():
                    if allow_empty:
                        return b""
                    raise server.SpeechRequestError(
                        "ComfyUI Drama 3 did not write an output audio file.", 500, no_audio=True
                    )
                samples = audio.load_audio_file(saved_path, COMFY_SAMPLE_RATE, stereo=False)
                if samples.size == 0 or (
                    samples.size <= 2048 and float(np.max(np.abs(samples))) == 0.0
                ):
                    if allow_empty:
                        return b""
                    raise server.SpeechRequestError(
                        "ComfyUI Drama 3 generated no audio frames.", 500, no_audio=True
                    )
                return audio.encode_wav(samples, COMFY_SAMPLE_RATE)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 500
                body = exc.response.text[:500] if exc.response is not None else str(exc)
                raise server.SpeechRequestError(
                    f"{self.label} speech request failed (HTTP {status}): {body}",
                    status,
                    no_audio=server.NO_AUDIO_FRAMES in body,
                ) from exc
            finally:
                if job is not None and active_prompt is not None and job.prompt_id == active_prompt:
                    job.prompt_id = None
                if saved_path is not None:
                    saved_path.unlink(missing_ok=True)

    def _resolve_saved_audio(self, history: dict, save_node: str) -> Path | None:
        outputs = history.get("outputs", {}) or {}
        scoped = outputs.get(save_node)
        refs = list(walk_saved_refs(scoped)) or list(walk_saved_refs(outputs))
        if not refs:
            return None
        ref = refs[0]
        if ref["type"] == "temp":
            base = (Path(self.runtime.comfy_dir) / "temp").resolve()
        else:
            base = Path(self.runtime.output_dir).resolve()
        candidate = (base / ref["subfolder"] / ref["filename"]).resolve()
        if not candidate.is_relative_to(base):
            raise H3Error("ComfyUI returned an audio path outside its output directory.")
        return candidate
