"""Models, output tiers, and credentials for the additional hosted image engines."""

from dataclasses import dataclass
import os

from .catalog import GEMINI_API_ROOT
from .errors import H3Error
from .gpt_image import GptImageEndpoint, gpt_image_credentials

RESOLUTION_PIXELS = {"1K": 1024, "2K": 2048, "4K": 4096}
NANO_BANANA = "nano_banana"
MAI_IMAGE = "mai_image"


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
            "gemini-3.1-flash-lite-image": {"1K": "~$0.0336"},
        },
        10, 14, "Set GEMINI_API_KEY.",
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
        key = os.getenv("GEMINI_API_KEY", "").strip()
        return GptImageEndpoint("Gemini", GEMINI_API_ROOT, key) if key else None
    return next(iter(gpt_image_credentials().endpoints()), None)


def validate_hosted_image_settings(engine, model, resolution, batch_count):
    provider = HOSTED_IMAGE_PROVIDERS[engine]
    if not isinstance(model, str) or model not in provider.native_prices:
        raise H3Error(f"Choose a {provider.label} model.")
    prices = provider.native_prices[model]
    if not isinstance(resolution, str) or resolution not in prices:
        raise H3Error(f"{model} supports these native output sizes: {', '.join(prices)}.")
    if (isinstance(batch_count, bool) or not isinstance(batch_count, (int, float))
            or batch_count not in range(1, provider.requests_per_minute + 1)):
        raise H3Error(f"Choose between 1 and {provider.requests_per_minute} concurrent images.")
    endpoint = hosted_image_endpoint(engine)
    if endpoint is None:
        raise H3Error(f"{provider.label} is unavailable. {provider.credentials_hint}")
    return endpoint
