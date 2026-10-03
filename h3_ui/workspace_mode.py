"""Server-selected rollout mode; legacy remains the default until GPU acceptance."""

import os


def workspace_enabled() -> bool:
    value = os.getenv("H3_UI_LAYOUT", "legacy").strip().lower()
    if value not in {"legacy", "workspace"}:
        raise ValueError("H3_UI_LAYOUT must be legacy or workspace.")
    return value == "workspace"
