"""Create engines that own their view, bindings and handlers.

Each engine module exposes ``FEATURE``; the workspace builds, binds and saves every
engine in :data:`FEATURES` the same way. Ids match :data:`h3_app.capabilities.ENGINES`.
"""

from . import fibo2, flux3, moss_image
from .feature import EngineDeps, EngineFeature

FEATURES: tuple[EngineFeature, ...] = (flux3.FEATURE, moss_image.FEATURE, fibo2.FEATURE)

__all__ = ["FEATURES", "EngineDeps", "EngineFeature"]
