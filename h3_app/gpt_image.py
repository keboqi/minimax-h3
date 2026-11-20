"""GPT Image configuration, .env/secret credentials, validation and concurrency rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import H3Error

GPT_IMAGE_ENGINE = "gpt_image"
GPT_IMAGE_LABEL = "GPT Image"
GPT_IMAGE_25_MODELS = ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare")
GPT_IMAGE_MODELS = ("gpt-image-2", *GPT_IMAGE_25_MODELS)
DEFAULT_GPT_IMAGE_MODEL = "gpt-image-2.5-flare"
GPT_IMAGE_QUALITY = "low"
GPT_IMAGE_QUALITIES = ("low", "medium")
GPT_IMAGE_SIZES = ((1920, 1088), (1088, 1920))
GPT_IMAGE_RESOLUTION_PRESETS = ("2M · 16:9 · 1920×1088", "2M · 9:16 · 1088×1920")
GPT_IMAGE_RATE_LIMIT = 2
GPT_IMAGE_RATE_WINDOW_SECONDS = 60.0
GPT_IMAGE_MAX_CONCURRENCY = len(GPT_IMAGE_MODELS) * 2 * GPT_IMAGE_RATE_LIMIT
GPT_IMAGE_MAX_REFERENCES = 16
GPT_IMAGE_ENV_NAMES = (
    "GPT_IMAGE_API_URL",
    "GPT_IMAGE_API_FAILBACK_URL",
    "GPT_IMAGE_API_KEY",
    "GPT_IMAGE_API_FAILBACK_KEY",
)

_OPERATION_SUFFIXES = (
    "/images/generations",
    "/images/edits",
    "/images/variations",
    "/chat/completions",
    "/responses",
    "/completions",
)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_dotenv_path() -> Path:
    configured = os.environ.get("H3_DOTENV_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _REPO_ROOT / ".env"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            tail = value[end + 1 :].strip()
            if not tail or tail.startswith("#"):
                return value[1:end].strip()
    for sep in (" #", "\t#"):
        if sep in value:
            value = value.split(sep, 1)[0].rstrip()
    return value


def read_dotenv(dotenv_path: Path | None = None) -> dict[str, str]:
    """Read key=value entries from a .env file without mutating os.environ."""
    path = _default_dotenv_path() if dotenv_path is None else Path(dotenv_path)
    try:
        if not path.is_file():
            return {}
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    entries: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        entries[key.upper()] = _strip_quotes(value)
    return entries


def gpt_image_env(
    name: str,
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> str:
    """Resolve an environment setting from ``env`` / ``os.environ``, falling back to ``.env``."""
    source = os.environ if env is None else env
    for candidate in (name, name.lower()):
        if candidate in source:
            return str(source[candidate] or "").strip()
    if env is not None and dotenv_path is None:
        return ""
    return read_dotenv(dotenv_path).get(name.upper(), "").strip()


def normalize_gpt_image_url(url: str) -> str:
    """Normalize a GPT Image API URL to its base endpoint path."""
    cleaned = str(url or "").strip().rstrip("/")
    lowered = cleaned.lower()
    for suffix in _OPERATION_SUFFIXES:
        if lowered.endswith(suffix):
            return cleaned[: -len(suffix)].rstrip("/")
    return cleaned


@dataclass(frozen=True)
class GptImageEndpoint:
    name: str
    url: str
    api_key: str


@dataclass(frozen=True)
class GptImageCredentials:
    api_url: str = ""
    api_key: str = ""
    failback_url: str = ""
    failback_key: str = ""

    @property
    def primary_available(self) -> bool:
        return bool(self.api_url and self.api_key)

    @property
    def failback_available(self) -> bool:
        return bool(self.failback_url and self.failback_key)

    @property
    def available(self) -> bool:
        return self.primary_available

    @property
    def distinct_failback(self) -> bool:
        return (
            self.primary_available
            and self.failback_available
            and (self.api_url, self.api_key) != (self.failback_url, self.failback_key)
        )

    def endpoints(self, use_failback: bool = False) -> tuple[GptImageEndpoint, ...]:
        if not self.primary_available:
            return ()
        items = [GptImageEndpoint("primary", self.api_url, self.api_key)]
        if use_failback and self.failback_available:
            items.append(GptImageEndpoint("failback", self.failback_url, self.failback_key))
        return tuple(items)


def gpt_image_credentials(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> GptImageCredentials:
    """Load GPT Image primary and failback URL/key pairs from environment and .env."""
    return GptImageCredentials(
        api_url=normalize_gpt_image_url(
            gpt_image_env("GPT_IMAGE_API_URL", env=env, dotenv_path=dotenv_path)
        ),
        api_key=gpt_image_env("GPT_IMAGE_API_KEY", env=env, dotenv_path=dotenv_path),
        failback_url=normalize_gpt_image_url(
            gpt_image_env("GPT_IMAGE_API_FAILBACK_URL", env=env, dotenv_path=dotenv_path)
        ),
        failback_key=gpt_image_env("GPT_IMAGE_API_FAILBACK_KEY", env=env, dotenv_path=dotenv_path),
    )


def gpt_image_available(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> bool:
    return gpt_image_credentials(env=env, dotenv_path=dotenv_path).available


def gpt_image_failback_available(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> bool:
    return gpt_image_credentials(env=env, dotenv_path=dotenv_path).failback_available


def normalize_gpt_image_models(models: Any) -> tuple[str, ...]:
    """Validate and normalize one or more GPT Image model selections in canonical order."""
    if isinstance(models, str):
        raw = [part.strip() for part in models.split(",") if part.strip()]
    elif isinstance(models, (list, tuple, set)):
        raw = [str(part).strip() for part in models if str(part).strip()]
    else:
        raise H3Error("Choose at least one GPT Image model.")
    if not raw:
        raise H3Error("Choose at least one GPT Image model.")
    invalid = [item for item in raw if item not in GPT_IMAGE_MODELS]
    if invalid:
        raise H3Error(f"Unsupported GPT Image model: {', '.join(invalid)}.")
    chosen = set(raw)
    return tuple(model for model in GPT_IMAGE_MODELS if model in chosen)


def gpt_image_max_concurrency(
    models: Any,
    use_failback: bool = False,
    *,
    credentials: GptImageCredentials | None = None,
) -> int:
    """Two requests per model on each distinct enabled endpoint."""
    selected = normalize_gpt_image_models(models)
    if credentials is not None:
        endpoint_count = len({(ep.url, ep.api_key) for ep in credentials.endpoints(use_failback)}) or 1
    else:
        endpoint_count = 2 if use_failback else 1
    return len(selected) * endpoint_count * GPT_IMAGE_RATE_LIMIT


def validate_gpt_image_canvas(width: Any, height: Any) -> tuple[int, int]:
    """Validate that the requested size is one of the allowed GPT Image dimensions."""
    try:
        valid = (
            not isinstance(width, bool)
            and not isinstance(height, bool)
            and math.isfinite(float(width))
            and math.isfinite(float(height))
            and float(width).is_integer()
            and float(height).is_integer()
        )
    except (TypeError, ValueError):
        valid = False
    if valid:
        pair = (int(float(width)), int(float(height)))
        if pair in GPT_IMAGE_SIZES:
            return pair
    raise H3Error("GPT Image supports only 1920×1088 and 1088×1920.")


def gpt_image_quality_choices(models: Sequence[str]) -> tuple[str, ...]:
    """Return the qualities supported by all selected models."""
    if models and all(model in GPT_IMAGE_25_MODELS for model in models):
        return GPT_IMAGE_QUALITIES
    return (GPT_IMAGE_QUALITY,)


def validate_gpt_image_quality(
    quality: Any, models: Sequence[str] = (DEFAULT_GPT_IMAGE_MODEL,)
) -> str:
    """Normalize quality and validate it for the selected models."""
    value = str(quality or "").strip().lower()
    if value not in gpt_image_quality_choices(models):
        raise H3Error("Choose low quality or select only GPT Image 2.5 models to use medium quality.")
    return value


def validate_gpt_image_settings(
    *,
    models: Any = (DEFAULT_GPT_IMAGE_MODEL,),
    use_failback: bool = False,
    quality: Any = GPT_IMAGE_QUALITY,
    width: Any = 1920,
    height: Any = 1088,
    batch_count: Any = 1,
    credentials: GptImageCredentials | None = None,
) -> tuple[tuple[str, ...], str, int, int, int, GptImageCredentials]:
    """Validate credentials, models, quality, canvas size and adapted concurrency."""
    creds = gpt_image_credentials() if credentials is None else credentials
    if not creds.primary_available:
        raise H3Error(
            "GPT Image is not available. Configure GPT_IMAGE_API_URL and GPT_IMAGE_API_KEY "
            "in .env or runtime secrets."
        )
    if use_failback and not creds.failback_available:
        raise H3Error(
            "GPT Image failback endpoint is not available. Configure GPT_IMAGE_API_FAILBACK_URL "
            "and GPT_IMAGE_API_FAILBACK_KEY in .env or runtime secrets, or turn Failback off."
        )
    normalized_models = normalize_gpt_image_models(models)
    normalized_quality = validate_gpt_image_quality(quality, normalized_models)
    canvas_width, canvas_height = validate_gpt_image_canvas(width, height)
    max_concurrency = gpt_image_max_concurrency(
        normalized_models, bool(use_failback), credentials=creds
    )
    try:
        valid_count = (
            not isinstance(batch_count, bool)
            and isinstance(batch_count, (int, float))
            and math.isfinite(float(batch_count))
            and int(batch_count) == batch_count
            and 1 <= int(batch_count) <= max_concurrency
        )
    except (TypeError, ValueError):
        valid_count = False
    if not valid_count:
        raise H3Error(
            f"GPT Image concurrency must be between 1 and {max_concurrency} "
            "for the selected models and failback setting."
        )
    return normalized_models, normalized_quality, canvas_width, canvas_height, int(batch_count), creds
