"""Native browser-backed persistence for non-sensitive Gradio settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gradio as gr


_PERSISTABLE_TYPES = (
    gr.Checkbox,
    gr.CheckboxGroup,
    gr.Dropdown,
    gr.Number,
    gr.Radio,
    gr.Slider,
)
_STORAGE_KEY = "minimax-h3:settings:v3"
_BROWSER_STATE_SECRET = "minimax-h3-ui-settings-v3"


def bind_browser_settings(
    demo: gr.Blocks,
    components: Mapping[str, Any],
    *,
    exclude: set[str] | None = None,
) -> gr.BrowserState:
    """Atomically restore and save supported settings with BrowserState.

    BrowserState is encrypted and stored by Gradio in the current browser's
    localStorage. Text, media, outputs, API keys, and transient controls are
    excluded by type or by the caller-provided names.
    """

    excluded = exclude or set()
    settings = [
        (name, component)
        for name, component in components.items()
        if name not in excluded and isinstance(component, _PERSISTABLE_TYPES)
    ]
    names = [name for name, _component in settings]
    controls = [component for _name, component in settings]
    defaults = {name: component.value for name, component in settings}

    browser_state = gr.BrowserState(
        default_value=defaults,
        storage_key=_STORAGE_KEY,
        secret=_BROWSER_STATE_SECRET,
    )

    def restore(saved: Any) -> tuple[Any, ...]:
        persisted = saved if isinstance(saved, Mapping) else {}
        return tuple(persisted.get(name, defaults[name]) for name in names)

    def remember(*values: Any) -> dict[str, Any]:
        return dict(zip(names, values, strict=True))

    demo.load(
        restore,
        inputs=browser_state,
        outputs=controls,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    gr.on(
        triggers=[component.change for component in controls],
        fn=remember,
        inputs=controls,
        outputs=browser_state,
        queue=False,
        show_progress="hidden",
        api_name=False,
        trigger_mode="always_last",
    )
    return browser_state
