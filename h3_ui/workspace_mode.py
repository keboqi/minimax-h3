"""Server-selected layout; workspace is the default and legacy is available."""

import os


def workspace_enabled() -> bool:
    value = os.getenv("H3_UI_LAYOUT", "workspace").strip().lower()
    if value not in {"legacy", "workspace"}:
        raise ValueError("H3_UI_LAYOUT must be legacy or workspace.")
    return value == "workspace"
