"""Viggle Remake's ComfyUI nodes, plus the holds a long Remake needs around them."""

from .hold_audio import NODE_CLASS_MAPPINGS as _HOLD_NODES
from .hold_audio import NODE_DISPLAY_NAME_MAPPINGS as _HOLD_NAMES
from .hold_frames import NODE_CLASS_MAPPINGS as _HOLD_FRAME_NODES
from .hold_frames import NODE_DISPLAY_NAME_MAPPINGS as _HOLD_FRAME_NAMES
from .nodes import NODE_CLASS_MAPPINGS as _REMAKE_NODES
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _REMAKE_NAMES

NODE_CLASS_MAPPINGS = {**_REMAKE_NODES, **_HOLD_NODES, **_HOLD_FRAME_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_REMAKE_NAMES, **_HOLD_NAMES, **_HOLD_FRAME_NAMES}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
