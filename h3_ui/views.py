"""Compatibility re-exports; each feature owns its typed view construction."""

from .music_view import MusicView, build_music_view  # noqa: F401
from .qwen_view import QWEN_1K_RESOLUTION_PRESETS, QWEN_2K_RESOLUTION_PRESETS, QwenImage21View, build_qwen_image21_view  # noqa: F401
from .yue2_view import YuE2View, build_yue2_view  # noqa: F401
from .media_view import GalleryView, build_gallery_view  # noqa: F401
from .api_view import ApiView, build_api_view  # noqa: F401
