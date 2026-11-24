"""Models, output tiers, and credentials for the additional hosted image engines."""

from dataclasses import dataclass
import os

from .catalog import GEMINI_API_ROOT
from .errors import H3Error
from .gpt_image import GptImageEndpoint, gpt_image_credentials
from .llm.gemini import session_key

RESOLUTION_PIXELS = {"1K": 1024, "2K": 2048, "4K": 4096}
DEFAULT_ASPECT_RATIO = "1:1"
NANO_BANANA = "nano_banana"
MAI_IMAGE = "mai_image"
GEMINI_FLASH_LITE = "gemini-3.1-flash-lite-image"

# Gemini's named aspect ratios map to the provider's 1K dimensions.
GEMINI_DIMENSIONS = {
    "1:1": (1024, 1024),
    "16:9": (1376, 768), "9:16": (768, 1376),
    "4:3": (1200, 896), "3:4": (896, 1200),
    "3:2": (1264, 848), "2:3": (848, 1264),
    "5:4": (1152, 928), "4:5": (928, 1152),
    "21:9": (1584, 672),
    "4:1": (2048, 512), "1:4": (512, 2048),
    "8:1": (3072, 384), "1:8": (384, 3072),
}

# Keep both dimensions at least 768 px and the area within the selected tier.
MAI_DIMENSIONS = {
    "1:1": ((1024, 1024), (2048, 2048)),
    "16:9": ((1360, 768), (2720, 1536)), "9:16": ((768, 1360), (1536, 2720)),
    "4:3": ((1152, 864), (2304, 1728)), "3:4": ((864, 1152), (1728, 2304)),
    "3:2": ((1248, 832), (2496, 1664)), "2:3": ((832, 1248), (1664, 2496)),
}


def hosted_aspect_ratios(engine: str) -> tuple[str, ...]:
    return tuple(MAI_DIMENSIONS if engine == MAI_IMAGE else GEMINI_DIMENSIONS)


def hosted_image_dimensions(engine: str, resolution: str, aspect_ratio: str) -> tuple[int, int]:
    if engine == MAI_IMAGE:
        return MAI_DIMENSIONS[aspect_ratio][resolution == "2K"]
    width, height = GEMINI_DIMENSIONS[aspect_ratio]
    scale = RESOLUTION_PIXELS[resolution] // RESOLUTION_PIXELS["1K"]
    return width * scale, height * scale


@dataclass(frozen=True)
class HostedImageProvider:
    label: str
    native_prices: dict[str, dict[str, str]]
    requests_per_minute: int
    max_references: int
    credentials_hint: str

    @property
    def default_model(self) -> str:
        return next(iter(self.native_prices))

    def clamp_resolution(self, model: str, resolution: str) -> str:
        prices = self.native_prices[model]
        return resolution if resolution in prices else next(reversed(prices))

    def resolution_choices(self, model: str) -> tuple[tuple[str, str], ...]:
        return tuple((f"{tier} · {price}/image", tier) for tier, price in self.native_prices[model].items())


HOSTED_IMAGE_PROVIDERS = {
    NANO_BANANA: HostedImageProvider(
        "Nano Banana",
        {
            "gemini-nano-banana-2.1": {"1K": "~$0.0336", "2K": "~$0.0504", "4K": "~$0.113"},
            GEMINI_FLASH_LITE: {"1K": "~$0.0336"},
        },
        10, 14, "Set GEMINI_API_KEY or enter a session key in System.",
    ),
    MAI_IMAGE: HostedImageProvider(
        "MAI-Image",
        {
            "MAI-Image-2.7-Flash": {"1K": "~$0.01", "2K": "~$0.04"},
            "MAI-Image-2.7": {"1K": "~$0.035", "2K": "~$0.14"},
        },
        6, 5, "Set GPT_IMAGE_API_URL and GPT_IMAGE_API_KEY.",
    ),
}
API_IMAGE_ENGINES = ("gpt_image", *HOSTED_IMAGE_PROVIDERS)


def hosted_image_endpoint(engine: str) -> GptImageEndpoint | None:
    if engine == NANO_BANANA:
        key = session_key() or os.getenv("GEMINI_API_KEY", "").strip()
        return GptImageEndpoint("Gemini", GEMINI_API_ROOT, key) if key else None
    endpoint = next(iter(gpt_image_credentials().endpoints()), None)
    if (endpoint and engine == MAI_IMAGE
            and endpoint.url.lower().endswith("/openai/v1")
            and ".services.ai.azure.com" in endpoint.url.lower()):
        return GptImageEndpoint(endpoint.name, endpoint.url[: -len("/openai/v1")] + "/mai/v1", endpoint.api_key)
    return endpoint


def validate_hosted_image_settings(engine, model, resolution, batch_count, *, aspect_ratio=DEFAULT_ASPECT_RATIO):
    provider = HOSTED_IMAGE_PROVIDERS[engine]
    if not isinstance(model, str) or model not in provider.native_prices:
        raise H3Error(f"Choose a {provider.label} model.")
    prices = provider.native_prices[model]
    if not isinstance(resolution, str) or resolution not in prices:
        raise H3Error(f"{model} supports these native output sizes: {', '.join(prices)}.")
    if aspect_ratio not in hosted_aspect_ratios(engine):
        raise H3Error(f"Choose one of the aspect ratios supported by {provider.label}.")
    if (isinstance(batch_count, bool) or not isinstance(batch_count, (int, float))
            or batch_count not in range(1, provider.requests_per_minute + 1)):
        raise H3Error(f"Choose between 1 and {provider.requests_per_minute} concurrent images.")
    endpoint = hosted_image_endpoint(engine)
    if endpoint is None:
        raise H3Error(f"{provider.label} is unavailable. {provider.credentials_hint}")
    return endpoint
