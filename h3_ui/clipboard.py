"""Browser code that copies a hidden prompt to the clipboard."""

from __future__ import annotations

import json


def copy_js(button_id: str, label: str) -> str:
    """Copy the input, falling back where the Clipboard API is unavailable, and confirm on the button."""
    return f"""async (prompt) => {{
  if (!prompt) return;
  try {{
    await navigator.clipboard.writeText(prompt);
  }} catch (_error) {{
    const area = document.createElement('textarea');
    area.value = prompt;
    document.body.appendChild(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }}
  const button = document.getElementById({json.dumps(button_id)});
  if (button) {{
    button.textContent = 'Copied';
    setTimeout(() => {{ button.textContent = {json.dumps(label)}; }}, 1500);
  }}
}}"""
