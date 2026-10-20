"""Local Fibo 2.0 Diffusers bridge; heavy dependencies live in a child process."""

from .nodes import H3Fibo2Generate

NODE_CLASS_MAPPINGS = {"H3Fibo2Generate": H3Fibo2Generate}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Fibo2Generate": "H3 Fibo 2.0 Generate / Edit"}
