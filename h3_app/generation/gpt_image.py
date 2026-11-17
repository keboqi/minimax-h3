"""Concurrent, rate-limited OpenAI GPT Image generation and editing."""

from __future__ import annotations

import base64
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
import io
import hashlib
from pathlib import Path
from threading import Lock
import time
from typing import Any, Callable, Iterator
import uuid

from PIL import Image
import requests

from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.gpt_image import (
    DEFAULT_GPT_IMAGE_MODEL,
    GPT_IMAGE_LABEL,
    GPT_IMAGE_MAX_REFERENCES,
    GPT_IMAGE_QUALITY,
    GPT_IMAGE_RATE_LIMIT,
    GPT_IMAGE_RATE_WINDOW_SECONDS,
    GptImageCredentials,
    GptImageEndpoint,
    gpt_image_credentials,
    validate_gpt_image_settings,
)
from h3_app.jobs import CURRENT_JOB, JobCancelled, check_cancelled, record_failure
from h3_app.llm import media_type
from h3_app.progress import ProgressCallback, no_progress
from h3_app.provenance import write_snapshot
from h3_app.status import StageTimings

from .results import GenerationUpdate

_PASSTHROUGH_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


@dataclass(frozen=True)
class GptImageRequest:
    prompt: str
    models: tuple[str, ...] = (DEFAULT_GPT_IMAGE_MODEL,)
    use_failback: bool = False
    quality: str = GPT_IMAGE_QUALITY
    width: int = 1920
    height: int = 1088
    batch_count: int = 1
    seed: int = -1
    reference_images: tuple[str, ...] = ()


class GptImageRateLimiter:
    """Thread-safe sliding window limiter: 2 requests/min per model per endpoint."""

    def __init__(
        self,
        *,
        limit: int = GPT_IMAGE_RATE_LIMIT,
        window_seconds: float = GPT_IMAGE_RATE_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limit = int(limit)
        self.window_seconds = float(window_seconds)
        self.clock = clock
        self.sleeper = sleeper
        self._lock = Lock()
        self._buckets: dict[tuple[str, str, bytes], deque[float]] = {}

    def _prune(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def available_slots(self, endpoint_url: str, model: str, *, api_key: str = "") -> int:
        key = (str(endpoint_url), str(model), hashlib.sha256(api_key.encode()).digest())
        with self._lock:
            bucket = self._buckets.get(key)
            if not bucket:
                return self.limit
            self._prune(bucket, self.clock())
            return max(0, self.limit - len(bucket))

    def acquire(
        self,
        endpoint_url: str,
        model: str,
        *,
        api_key: str = "",
        check_cancel: Callable[[], None] = check_cancelled,
    ) -> float:
        key = (str(endpoint_url), str(model), hashlib.sha256(api_key.encode()).digest())
        waited = 0.0
        while True:
            check_cancel()
            with self._lock:
                now = self.clock()
                bucket = self._buckets.setdefault(key, deque())
                self._prune(bucket, now)
                if len(bucket) < self.limit:
                    bucket.append(now)
                    return waited
                delay = max(0.01, bucket[0] + self.window_seconds - now)
            step = min(delay, 0.25)
            self.sleeper(step)
            waited += step

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


RATE_LIMITER = GptImageRateLimiter()


def validate_gpt_image(
    request: GptImageRequest,
    credentials: GptImageCredentials | None = None,
) -> tuple[tuple[str, ...], str, int, int, int, tuple[str, ...], GptImageCredentials]:
    prompt = str(request.prompt or "").strip()
    if not prompt:
        raise H3Error("Write an image prompt or edit instruction.")
    models, quality, width, height, batch_count, creds = validate_gpt_image_settings(
        models=request.models,
        use_failback=bool(request.use_failback),
        quality=request.quality,
        width=request.width,
        height=request.height,
        batch_count=request.batch_count,
        credentials=credentials,
    )
    references = tuple(str(path) for path in (request.reference_images or ()) if path)
    if len(references) > GPT_IMAGE_MAX_REFERENCES:
        raise H3Error(f"GPT Image supports up to {GPT_IMAGE_MAX_REFERENCES} reference images per edit.")
    for path in references:
        if not Path(path).is_file():
            raise H3Error(f"Reference image is missing: {Path(path).name}")
    return models, quality, width, height, batch_count, references, creds


def plan_gpt_image_slots(
    models: tuple[str, ...],
    endpoints: tuple[GptImageEndpoint, ...],
    batch_count: int,
    *,
    limiter: GptImageRateLimiter = RATE_LIMITER,
) -> list[tuple[GptImageEndpoint, str]]:
    """Assign each concurrent task an (endpoint, model) pair within 2 req/min per pair."""
    if not endpoints:
        raise H3Error("No GPT Image endpoint is configured.")
    max_slots = len(models) * len({(ep.url, ep.api_key) for ep in endpoints}) * GPT_IMAGE_RATE_LIMIT
    if batch_count < 1 or batch_count > max_slots:
        raise H3Error(f"GPT Image supports at most {max_slots} concurrent generations for this selection.")
    assigned: dict[tuple[str, str, str], int] = {}
    slots: list[tuple[GptImageEndpoint, str]] = []
    for index in range(batch_count):
        model = models[index % len(models)]
        candidates = [
            ep for ep in endpoints
            if assigned.get((ep.url, ep.api_key, model), 0) < GPT_IMAGE_RATE_LIMIT
        ]
        endpoint = min(
            candidates,
            key=lambda ep: (
                0 if limiter.available_slots(ep.url, model, api_key=ep.api_key) - assigned.get((ep.url, ep.api_key, model), 0) > 0 else 1,
                endpoints.index(ep),
                assigned.get((ep.url, ep.api_key, model), 0),
            ),
        )
        assigned[(endpoint.url, endpoint.api_key, model)] = assigned.get((endpoint.url, endpoint.api_key, model), 0) + 1
        slots.append((endpoint, model))
    return slots


def _encode_reference_file(path_str: str) -> tuple[str, bytes, str]:
    path = Path(path_str)
    try:
        if path.suffix.lower() in _PASSTHROUGH_IMAGE_SUFFIXES:
            return (path.name, path.read_bytes(), media_type(path))
        buffer = io.BytesIO()
        with Image.open(path) as img:
            img.save(buffer, format="PNG")
        return (f"{path.stem}.png", buffer.getvalue(), "image/png")
    except Exception as exc:
        raise H3Error(f"Could not read reference image {path.name}: {exc}") from exc


def _extract_image_bytes(response: Any) -> bytes:
    items = getattr(response, "data", None) or []
    if not items:
        raise H3Error("GPT Image returned no image data.")
    first = items[0]
    b64 = getattr(first, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(first, "url", None)
    if url:
        reply = requests.get(url, timeout=120)
        reply.raise_for_status()
        return reply.content
    raise H3Error("GPT Image returned an empty image payload.")


def _save_png(raw_bytes: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.{uuid.uuid4().hex[:8]}.tmp.png")
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.save(temporary, format="PNG")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _call_endpoint(
    endpoint: GptImageEndpoint,
    model: str,
    prompt: str,
    size_str: str,
    quality: str,
    encoded_references: list[tuple[str, bytes, str]],
) -> bytes:
    from openai import OpenAI

    # Every HTTP attempt must acquire a rate-limit slot, including failover.
    client = OpenAI(base_url=endpoint.url, api_key=endpoint.api_key, timeout=600.0, max_retries=0)
    if encoded_references:
        image_arg: Any = encoded_references[0] if len(encoded_references) == 1 else list(encoded_references)
        response = client.images.edit(
            model=model,
            image=image_arg,
            prompt=prompt,
            size=size_str,
            quality=quality,
            n=1,
        )
    else:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size_str,
            quality=quality,
            n=1,
        )
    return _extract_image_bytes(response)


def generate_gpt_image(
    request: GptImageRequest,
    runtime: RuntimeConfig,
    *,
    credentials: GptImageCredentials | None = None,
    limiter: GptImageRateLimiter = RATE_LIMITER,
    progress: ProgressCallback = no_progress,
) -> Iterator[GenerationUpdate]:
    """Generate or edit images concurrently across selected GPT Image models and endpoints."""
    models, quality, width, height, batch_count, references, creds = validate_gpt_image(
        request, credentials
    )
    endpoints = creds.endpoints(bool(request.use_failback))
    slots = plan_gpt_image_slots(models, endpoints, batch_count, limiter=limiter)

    def _run() -> Iterator[GenerationUpdate]:
        started = time.monotonic()
        timings = StageTimings(GPT_IMAGE_LABEL, started, "Preparing GPT Image request")
        prompt = str(request.prompt).strip()
        size_str = f"{width}x{height}"
        output_stamp = str(int(time.time()))
        output_nonce = uuid.uuid4().hex[:8]
        encoded_references = [_encode_reference_file(path) for path in references]
        job = CURRENT_JOB.get()
        check_cancel = job.check if job is not None else check_cancelled
        outputs_by_index: dict[int, str] = {}
        errors: list[tuple[int, Exception]] = []

        yield GenerationUpdate(
            None,
            f"Starting {batch_count} concurrent GPT Image request(s) across {', '.join(models)} ({size_str}, quality={quality})…",
        )
        timings.transition("Calling GPT Image API")

        def run_slot(index: int, endpoint: GptImageEndpoint, model: str) -> tuple[int, Path, GptImageEndpoint]:
            limiter.acquire(endpoint.url, model, api_key=endpoint.api_key, check_cancel=check_cancel)
            check_cancel()
            used_endpoint = endpoint
            try:
                raw = _call_endpoint(endpoint, model, prompt, size_str, quality, encoded_references)
            except JobCancelled:
                raise
            except Exception as primary_exc:
                alt_endpoint = next(
                    (ep for ep in endpoints if (ep.url, ep.api_key) != (endpoint.url, endpoint.api_key)),
                    None,
                )
                if alt_endpoint is not None:
                    check_cancel()
                    limiter.acquire(alt_endpoint.url, model, api_key=alt_endpoint.api_key, check_cancel=check_cancel)
                    check_cancel()
                    used_endpoint = alt_endpoint
                    try:
                        raw = _call_endpoint(
                            alt_endpoint, model, prompt, size_str, quality, encoded_references
                        )
                    except JobCancelled:
                        raise
                    except Exception as failback_exc:
                        raise H3Error(
                            f"{model} failed on {endpoint.name} ({primary_exc}) and {alt_endpoint.name} ({failback_exc})"
                        ) from failback_exc
                else:
                    raise H3Error(f"{model} ({endpoint.name}) failed: {primary_exc}") from primary_exc
            check_cancel()
            target = runtime.output_dir / f"gpt_image_{output_stamp}_{output_nonce}_{index + 1:05d}.png"
            _save_png(raw, target)
            return index, target, used_endpoint

        executor = ThreadPoolExecutor(max_workers=batch_count, thread_name_prefix="gpt-image")
        cancelled = False
        try:
            future_to_slot: dict[Future, tuple[int, GptImageEndpoint, str]] = {}
            for index, (endpoint, model) in enumerate(slots):
                ctx = copy_context()
                future = executor.submit(ctx.run, run_slot, index, endpoint, model)
                future_to_slot[future] = (index, endpoint, model)

            pending = set(future_to_slot)
            done_count = 0
            while pending:
                check_cancel()
                finished, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, planned_endpoint, model = future_to_slot[future]
                    done_count += 1
                    try:
                        slot_index, target, used_endpoint = future.result()
                    except JobCancelled:
                        cancelled = True
                        raise
                    except Exception as exc:
                        errors.append((index, exc))
                        progress((done_count, batch_count), desc=f"{model} failed")
                        ordered = [outputs_by_index[i] for i in sorted(outputs_by_index)]
                        yield GenerationUpdate(
                            ordered if ordered else None,
                            f"{done_count}/{batch_count} · {model} failed: {exc}",
                        )
                        continue
                    write_snapshot(
                        target,
                        {
                            "job_id": f"{output_stamp}_{output_nonce}",
                            "family": GPT_IMAGE_LABEL,
                            "settings": {
                                "model": model,
                                "models": list(models),
                                "endpoint": used_endpoint.name,
                                "planned_endpoint": planned_endpoint.name,
                                "use_failback": bool(request.use_failback),
                                "quality": quality,
                                "width": width,
                                "height": height,
                                "seed": request.seed,
                                "batch_count": batch_count,
                                "variant_index": slot_index,
                                "reference_count": len(references),
                            },
                        },
                    )
                    outputs_by_index[slot_index] = str(target)
                    ordered = [outputs_by_index[i] for i in sorted(outputs_by_index)]
                    progress((done_count, batch_count), desc=f"Generated {done_count}/{batch_count}")
                    yield GenerationUpdate(
                        ordered,
                        f"Completed {len(ordered)}/{batch_count} · {model} ({used_endpoint.name}) · {size_str}",
                    )
        except JobCancelled as exc:
            cancelled = True
            executor.shutdown(wait=False, cancel_futures=True)
            record_failure(exc)
            raise
        finally:
            if not cancelled:
                executor.shutdown(wait=True)
            timings.finish()

        final_outputs = [outputs_by_index[i] for i in sorted(outputs_by_index)]
        if errors:
            first_error = errors[0][1]
            record_failure(first_error)
            detail = "; ".join(str(err) for _, err in errors)
            yield GenerationUpdate(
                final_outputs,
                f"Error: {detail}\n\n{timings.summary()}",
            )
            return
        elapsed = time.monotonic() - started
        progress(1, desc="Complete")
        yield GenerationUpdate(
            final_outputs,
            f"GPT Image completed {len(final_outputs)}/{batch_count} image(s) in {elapsed:.1f}s\n\n{timings.summary()}",
        )

    return _run()
