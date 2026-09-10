"""Small transport boundary; callers own policy and presentation."""

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ComfyClient:
    url: str
    timeout: float = 60
    session: Any = requests

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        response = self.session.get(
            self.url + path, timeout=kwargs.pop("timeout", self.timeout), **kwargs
        )
        response.raise_for_status()
        return response

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        response = self.session.post(
            self.url + path, timeout=kwargs.pop("timeout", self.timeout), **kwargs
        )
        response.raise_for_status()
        return response

    def websocket_url(self, client_id: str) -> str:
        from urllib.parse import quote, urlsplit, urlunsplit

        parsed = urlsplit(self.url)
        return urlunsplit(
            (
                "wss" if parsed.scheme == "https" else "ws",
                parsed.netloc,
                f"{parsed.path.rstrip('/')}/ws",
                f"clientId={quote(client_id)}",
                "",
            )
        )

    def object_info(self) -> dict:
        return self.get("/object_info").json()

    def unload(self) -> None:
        self.post("/free", json={"unload_models": True, "free_memory": True})
