"""Nano Banana and MAI-Image generation through hosted APIs."""

import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path
from queue import Empty, SimpleQueue
import uuid

import requests

from h3_app.errors import H3Error
from h3_app.hosted_image import (
    HOSTED_IMAGE_PROVIDERS, NANO_BANANA, RESOLUTION_PIXELS, validate_hosted_image_settings,
)
from h3_app.jobs import JobCancelled, check_cancelled, record_failure
from h3_app.llm import gemini
from h3_app.provenance import write_snapshot

from .gpt_image import (
    GptImageRateLimiter, _encode_reference_file, _extract_image_bytes, _rate_limit_delay, _save_png,
)
from .results import GenerationUpdate

RATE_LIMITERS = {engine: GptImageRateLimiter(limit=provider.requests_per_minute)
                 for engine, provider in HOSTED_IMAGE_PROVIDERS.items()}


def _gemini_image(endpoint, model, prompt, resolution, references):
    parts = [{"text": prompt}]
    parts.extend({"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode("ascii")}}
                 for _, data, mime in references)
    image_config = {"aspectRatio": "1:1"}
    if model != "gemini-3.1-flash-lite-image":
        image_config["imageSize"] = resolution
    response = requests.post(
        f"{endpoint.url}/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": endpoint.api_key},
        json={"contents": [{"role": "user", "parts": parts}],
              "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": image_config}},
        timeout=600,
    )
    if response.status_code == 429:
        response.raise_for_status()
    if not response.ok:
        raise gemini.error(response, "image generation")
    payload = response.json()
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("thought"):
                continue
            image = part.get("inlineData") or part.get("inline_data") or {}
            if image.get("data"):
                return base64.b64decode(image["data"])
    raise H3Error(f"Gemini returned no image ({gemini.empty_reply_reason(payload)}).")


def _mai_image(endpoint, model, prompt, resolution, references):
    from openai import OpenAI

    pixels = RESOLUTION_PIXELS[resolution]
    if endpoint.url.lower().endswith("/mai/v1"):
        # Foundry exposes its MAI-specific endpoint at this path.
        operation = "edits" if references else "generations"
        fields = {"model": model, "prompt": prompt, "width": pixels, "height": pixels}
        request = ({"data": fields, "files": [("image", image) for image in references]}
                   if references else {"json": fields})
        response = requests.post(f"{endpoint.url}/images/{operation}",
                                 headers={"api-key": endpoint.api_key}, timeout=600, **request)
        response.raise_for_status()
        items = response.json().get("data") or []
        if not items or not items[0].get("b64_json"):
            raise H3Error("MAI-Image returned no image.")
        return base64.b64decode(items[0]["b64_json"])
    with OpenAI(base_url=endpoint.url, api_key=endpoint.api_key, timeout=600, max_retries=0) as client:
        fields = {"model": model, "prompt": prompt, "size": f"{pixels}x{pixels}", "n": 1}
        if references:
            image = references[0] if len(references) == 1 else references
            response = client.images.edit(image=image, **fields)
        else:
            response = client.images.generate(**fields)
    return _extract_image_bytes(response, label="MAI-Image")


def generate_hosted_image(engine, model, resolution, batch_count, prompt, references, runtime):
    endpoint = validate_hosted_image_settings(engine, model, resolution, batch_count)
    provider = HOSTED_IMAGE_PROVIDERS[engine]
    prompt = str(prompt or "").strip()
    if not prompt:
        raise H3Error("Write an image prompt or edit instruction.")
    if len(references) > provider.max_references:
        raise H3Error(f"{provider.label} supports up to {provider.max_references} reference images per edit.")
    limiter = RATE_LIMITERS[engine]
    generate = _gemini_image if engine == NANO_BANANA else _mai_image
    job_id = uuid.uuid4().hex
    notices = SimpleQueue()
    count = int(batch_count)

    yield GenerationUpdate(None, f"{model} · Preparing {count} image request(s) at {resolution}…")
    encoded = [_encode_reference_file(path) for path in references]

    def run_one(index):
        for attempt in range(3):
            # Use a single quota key per provider across all models and concurrent jobs.
            limiter.acquire(endpoint.url, engine, api_key=endpoint.api_key,
                            on_wait=lambda delay: notices.put(f"{provider.label} · Waiting {delay:.0f}s for quota"))
            check_cancelled()
            try:
                raw = generate(endpoint, model, prompt, resolution, encoded)
                break
            except Exception as exc:
                delay = _rate_limit_delay(exc)
                if delay is None:
                    raise
                limiter.defer(endpoint.url, engine, delay, api_key=endpoint.api_key)
                if attempt == 2:
                    raise
        check_cancelled()
        target = runtime.output_dir / f"{engine}_{job_id}_{index + 1:05d}.png"
        width, height = _save_png(raw, target)
        write_snapshot(target, {"job_id": job_id, "family": provider.label, "settings": {
            "model": model, "resolution": resolution,
            "width": width, "height": height, "batch_count": count,
            "variant_index": index, "reference_count": len(references),
        }})
        return str(target)

    executor = ThreadPoolExecutor(max_workers=count, thread_name_prefix=engine)
    outputs = {}
    errors = []
    try:
        pending = {executor.submit(copy_context().run, run_one, index): index for index in range(count)}
        while pending:
            check_cancelled()
            done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            try:
                while True:
                    yield GenerationUpdate(None, notices.get_nowait())
            except Empty:
                pass
            for future in done:
                index = pending.pop(future)
                try:
                    outputs[index] = future.result()
                except JobCancelled:
                    raise
                except Exception as exc:
                    errors.append(exc)
            if done:
                yield GenerationUpdate([outputs[index] for index in sorted(outputs)] or None,
                                       f"{model} · Completed {count - len(pending)}/{count} requests")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    paths = [outputs[index] for index in sorted(outputs)]
    if errors:
        record_failure(errors[0])
        yield GenerationUpdate(paths, f"Error: {model} · {errors[0]}")
    else:
        yield GenerationUpdate(paths, f"{model} · Generated {len(paths)} image(s) at {resolution}")
