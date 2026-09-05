"""Explicit, validated browser preferences with an in-place v3 migration."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import gradio as gr
from h3_app.settings import valid_preference, PRESET_FIELDS
from .settings_controller import SETTING_NAMES

# Keep the transport key/secret so existing encrypted v3 values remain readable.
_STORAGE_KEY = "minimax-h3:settings:v3"
_BROWSER_STATE_SECRET = "minimax-h3-ui-settings-v3"
SCHEMA_VERSION = 4
EXTRA_FIELDS = {
    "h3": (
        "ref_size",
        "local_prompt_base_model",
        "local_prompt_max_tokens",
        "local_prompt_temperature",
        "local_prompt_top_p",
        "local_prompt_greedy",
        "local_prompt_seed",
        "gemini_prompt_model",
        "prompt_writer_backend",
        "input_upscale_model",
        "input_upscale_seed",
        "input_upscale_force_offload",
        "input_upscale_frame_preset",
        "input_upscale_frame_width",
        "input_upscale_frame_height",
    ),
    "ltx25": (
        "mode",
        "model",
        "workflow",
        "duration",
        "fps",
        "width",
        "height",
        "seed",
        "cfg",
        "sampler",
        "image_strength",
        "middle_time",
        "middle_strength",
        "end_strength",
        "prompt_model",
    ),
    "music3": (
        "model",
        "duration",
        "seed",
        "tiled",
        "steps",
        "cfg",
        "ar_cfg",
        "top_k",
        "prompt_model",
    ),
    "gallery": (
        "postprocess",
        "upscale_resolution",
        "seedvr2_model",
        "post_seed",
        "force_offload",
        "split_upscale",
        "split_seconds",
    ),
}
PERSISTED_NAMES = frozenset(
    [
        *("h3." + name for name in SETTING_NAMES),
        *(
            prefix + "." + name
            for prefix, names in EXTRA_FIELDS.items()
            for name in names
        ),
    ]
)


def sanitize(component, value):
    choices = getattr(component, "choices", None)
    if choices is not None:
        choices = [
            choice[1] if isinstance(choice, (tuple, list)) else choice
            for choice in choices
        ]
        if component.value is None:
            choices.append(None)
    return valid_preference(
        value,
        component.value,
        choices=choices,
        minimum=getattr(component, "minimum", None),
        maximum=getattr(component, "maximum", None),
    )


def restore_preferences(saved, components):
    payload = saved if isinstance(saved, Mapping) else {}
    values = payload.get("values", payload)
    if not isinstance(values, Mapping):
        values = {}
    restored = {
        name: sanitize(component, values.get(name, component.value))
        for name, component in components.items()
        if name in PERSISTED_NAMES
    }
    old_memory = payload.get("mode_memory", {})
    modes = {}
    if isinstance(old_memory, Mapping) and isinstance(old_memory.get("modes"), Mapping):
        for mode in ("Normal", "Turbo"):
            candidate = old_memory["modes"].get(mode)
            if isinstance(candidate, Mapping):
                modes[mode] = {
                    name: sanitize(
                        components["h3." + name],
                        candidate.get(name, restored.get("h3." + name)),
                    )
                    for name in (*PRESET_FIELDS, "preset", "cache_mode")
                    if "h3." + name in components
                }
    active = restored.get("h3.generation_mode", "Turbo")
    memory = {"active": active, "modes": modes}
    if isinstance(old_memory, Mapping) and isinstance(
        old_memory.get("offload_preference"), bool
    ):
        memory["offload_preference"] = old_memory["offload_preference"]
    return restored, memory


def bind_browser_settings(demo, components, *, controller):
    selected = {
        name: component
        for name, component in components.items()
        if name in PERSISTED_NAMES
    }
    names, controls = list(selected), list(selected.values())
    defaults = {name: component.value for name, component in selected.items()}
    browser_state = gr.BrowserState(
        default_value={"schema_version": SCHEMA_VERSION, "values": defaults},
        storage_key=_STORAGE_KEY,
        secret=_BROWSER_STATE_SECRET,
    )

    def restore(saved):
        restored, memory = restore_preferences(saved, selected)
        return (*[restored[name] for name in names], memory)

    def remember(memory, *values):
        return {
            "schema_version": SCHEMA_VERSION,
            "values": dict(zip(names, values, strict=True)),
            "mode_memory": memory,
        }

    restored = demo.load(
        restore,
        inputs=browser_state,
        outputs=[*controls, controller.memory],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    restored.then(
        controller.refresh,
        inputs=[controller.memory, *controller.inputs],
        outputs=controller.outputs,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    for event in controller.events:
        event.then(
            remember,
            inputs=[controller.memory, *controls],
            outputs=browser_state,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
    others = [
        component.input
        for name, component in selected.items()
        if name not in {"h3." + n for n in SETTING_NAMES}
    ]
    gr.on(
        triggers=others,
        fn=remember,
        inputs=[controller.memory, *controls],
        outputs=browser_state,
        queue=False,
        show_progress="hidden",
        api_name=False,
        trigger_mode="always_last",
    )
    return browser_state
