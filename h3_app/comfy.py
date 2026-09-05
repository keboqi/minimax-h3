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
        response = self.session.get(self.url + path, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        response = self.session.post(self.url + path, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response
