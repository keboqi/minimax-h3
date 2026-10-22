"""Gemini REST client: API keys, temporary file uploads and generateContent."""

from __future__ import annotations

import os
from pathlib import Path
import re
import time
from typing import Any

import requests

from h3_app.catalog import GEMINI_API_ROOT
from h3_app.errors import H3Error

from . import media_type


def api_key(temporary_key: str | None) -> str:
    """A key entered in the UI, else GEMINI_API_KEY; never stored."""
    key = str(temporary_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise H3Error(
            "Set GEMINI_API_KEY in the server environment or enter a temporary "
            "Gemini API key in Prompt enhancer."
        )
    return key


def error(response: requests.Response, action: str) -> H3Error:
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or response.text
    except (ValueError, AttributeError):
        detail = response.text
    detail = re.sub(r"\s+", " ", str(detail)).strip()[:800]
    return H3Error(f"Gemini {action} failed (HTTP {response.status_code}): {detail}")


def reply_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    return "".join(str(part.get("text", "")) for part in parts).strip()


def empty_reply_reason(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if candidates:
        return candidates[0].get("finishReason")
    return payload.get("promptFeedback", {}).get("blockReason", "no candidate")


class Session:
    """One HTTP session. Files uploaded through it are deleted when it closes."""

    def __init__(self, key: str):
        self.key = key
        self.http = requests.Session()
        self._uploaded: list[str] = []

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            for name in self._uploaded:
                try:
                    self.http.delete(f"{GEMINI_API_ROOT}/v1beta/{name}",
                                     headers={"x-goog-api-key": self.key}, timeout=30)
                except requests.RequestException:
                    pass
        finally:
            self.http.close()

    def upload(self, path: Path) -> dict[str, Any]:
        """Upload ``path`` with the resumable protocol; deleted when the session closes."""
        mime_type = media_type(path)
        size = path.stat().st_size
        started = self.http.post(
            f"{GEMINI_API_ROOT}/upload/v1beta/files",
            headers={
                "x-goog-api-key": self.key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
            json={"file": {"displayName": path.name[:512]}},
            timeout=60,
        )
        if not started.ok:
            raise error(started, f"upload initialization for {path.name}")
        upload_url = started.headers.get("x-goog-upload-url")
        if not upload_url:
            raise H3Error(f"Gemini did not return an upload URL for {path.name}.")
        with path.open("rb") as source:
            uploaded = self.http.post(
                upload_url,
                headers={
                    "Content-Length": str(size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                    "Content-Type": mime_type,
                },
                data=source,
                timeout=600,
            )
        if not uploaded.ok:
            raise error(uploaded, f"upload for {path.name}")
        file_info = uploaded.json().get("file", {})
        if not file_info.get("name") or not file_info.get("uri"):
            raise H3Error(f"Gemini returned incomplete file metadata for {path.name}.")
        self._uploaded.append(str(file_info["name"]))
        return file_info

    def wait_until_active(self, file_info: dict[str, Any]) -> dict[str, Any]:
        name = str(file_info["name"])
        deadline = time.monotonic() + 600
        while str(file_info.get("state", "ACTIVE")).upper() == "PROCESSING":
            if time.monotonic() >= deadline:
                raise H3Error(f"Gemini timed out while processing {name}.")
            time.sleep(2)
            response = self.http.get(f"{GEMINI_API_ROOT}/v1beta/{name}",
                                     headers={"x-goog-api-key": self.key}, timeout=60)
            if not response.ok:
                raise error(response, f"file status check for {name}")
            file_info = response.json()
        if str(file_info.get("state", "ACTIVE")).upper() == "FAILED":
            detail = file_info.get("error", {}).get("message", "unknown processing error")
            raise H3Error(f"Gemini could not process {name}: {detail}")
        return file_info

    def file_part(self, path: Path) -> dict[str, Any]:
        """Upload ``path``, wait until Gemini can read it and return its ``fileData`` part."""
        file_info = self.wait_until_active(self.upload(path))
        return {"fileData": {"mimeType": file_info.get("mimeType") or media_type(path),
                             "fileUri": file_info["uri"]}}

    def generate(self, model: str, parts: list[dict[str, Any]], *, generation_config: dict[str, Any],
                 system_prompt: str | None = None, timeout: float = 600,
                 action: str = "prompt generation") -> dict[str, Any]:
        """The raw generateContent reply for one user turn."""
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": parts}],
                                "generationConfig": generation_config}
        if system_prompt is not None:
            body = {"systemInstruction": {"parts": [{"text": system_prompt}]}, **body}
        response = self.http.post(
            f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        if not response.ok:
            raise error(response, action)
        return response.json()
