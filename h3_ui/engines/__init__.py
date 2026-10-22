"""Create engines that own their view, bindings and handlers.

Each engine module exposes ``FEATURE``; the workspace builds, binds and saves every
engine in :data:`FEATURES` the same way. Ids match :data:`h3_app.capabilities.ENGINES`.
The order sets the order of Media's "Use in" destinations.
"""

from . import fibo2, flux3, ltx, moss_image, music, qwen, yue2
from .feature import EngineDeps, EngineFeature

FEATURES: tuple[EngineFeature, ...] = (
    ltx.FEATURE, qwen.FEATURE, flux3.FEATURE, moss_image.FEATURE, fibo2.FEATURE, music.FEATURE, yue2.FEATURE,
)

__all__ = ["FEATURES", "EngineDeps", "EngineFeature"]
