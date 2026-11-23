"""CPU Workshop dependencies, kept outside Studio's image build inputs.

Shared version pins still come from h3_requirements. Studio must never import this
module while building its image: Workshop-only edits should reuse that cache.
"""
from h3_requirements import (
    GRADIO_VERSION,
    HUGGINGFACE_HUB_REQUIREMENT,
    PLAYWRIGHT_REQUIREMENT,
    VIDEO_DOWNLOAD_REQUIREMENTS,
    WSPROTO_VERSION,
)


# Bare minimum needed to serve the Workshop UI on CPU for project review,
# media browsing, and backup/transfer without browser capture or URL downloaders.
WORKSHOP_MINIMAL_REQUIREMENTS = (
    # requests: the Drama 3 SGLang-Omni downloader module imports it.
    f"gradio=={GRADIO_VERSION}",
    HUGGINGFACE_HUB_REQUIREMENT,
    "requests>=2.32",
    # websocket-client, aiohttp: shared media controller and routes;
    # wsproto: the Gradio queue's WebSocket transport, as in Studio.
    "websocket-client>=1.8",
    "aiohttp>=3.11,<4",
    f"wsproto=={WSPROTO_VERSION}",
)

WORKSHOP_REQUIREMENTS = (
    *VIDEO_DOWNLOAD_REQUIREMENTS,
    PLAYWRIGHT_REQUIREMENT,
    *WORKSHOP_MINIMAL_REQUIREMENTS[:3],
    "json-repair>=0.60",
    "openai>=3.16.2,<4",  # Workshop's OpenAI-compatible prompt writer.
    *WORKSHOP_MINIMAL_REQUIREMENTS[3:],
)
