"""Explicit, validated browser preferences with an in-place v3 migration."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import gradio as gr
from h3_app.settings import valid_preference, PRESET_FIELDS
from .settings_controller import SETTING_NAMES

# Keep the transport key/secret so existing encrypted v3 values remain readable.
_STORAGE_KEY = "minimax-h3:settings:v3"
_BROWSER_STATE_SECRET = "minimax-h3-ui-settings-v3"
SCHEMA_VERSION = 6
EXTRA_FIELDS = {
    "workspace": ("task", "engine"),
    "qwen_image21": (
        "mode",
        "model",
        "text_encoder",
        "width",
        "height",
        "reference_resolution",
        "edit_size",
        "seed",
        "batch_count",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "cache_device",
        "cache_dtype",
        "attention_backend",
        "accelerator",
        "turbo_variant",
        "preset",
        "prompt_model",
        "prompt_backend",
    ),
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
    "yue2": (
        "model",
        "mode",
        "duration",
        "seed",
        "steps",
        "cfg",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_abc_tokens",
        "abc_temperature",
        "abc_top_p",
        "abc_top_k",
        "abc_repetition_penalty",
        "abc_penalty_window",
        "tiled",
    ),
    "gallery": (
        "ltx25_model",
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
    if "workspace.task" in restored and "workspace.engine" in restored:
        from h3_app.capabilities import engines_for_task

        choices = engines_for_task(restored["workspace.task"])
        restored["workspace.engine"] = next(
            (item.id for item in choices if item.id == values.get("workspace.engine")),
            choices[0].id,
        )
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
    snapshot = gr.State({"schema_version": SCHEMA_VERSION, "values": defaults})

    def restore(saved, current):
        restored, memory = restore_preferences(saved, selected)
        current.clear()
        current.update(
            schema_version=SCHEMA_VERSION, values=restored, mode_memory=memory
        )
        updates = [restored[name] for name in names]
        if "workspace.engine" in names and isinstance(
            selected["workspace.engine"], gr.Dropdown
        ):
            from h3_app.capabilities import engines_for_task

            choices = engines_for_task(restored["workspace.task"])
            updates[names.index("workspace.engine")] = gr.update(
                choices=[(item.label, item.id) for item in choices],
                value=restored["workspace.engine"],
            )
        return (*updates, memory)

    restored = demo.load(
        restore,
        inputs=[browser_state, snapshot],
        outputs=[*controls, controller.memory],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    refreshed = restored.success(
        controller.refresh,
        inputs=[controller.memory, *controller.inputs],
        outputs=controller.outputs,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )

    def remember_namespace(namespace_names, *, h3=False):
        def remember(memory, current, session, *values):
            current["values"].update(dict(zip(namespace_names, values, strict=True)))
            if h3:
                memory = session.memory or memory
                current["values"].update(
                    {
                        "h3." + name: value
                        for name, value in (memory or {}).get("values", {}).items()
                        if "h3." + name in selected
                    }
                )
                current["mode_memory"] = {
                    key: value
                    for key, value in (memory or {}).items()
                    if key in {"active", "modes", "offload_preference"}
                }
            return deepcopy(current)

        return remember

    for namespace in sorted({name.split(".", 1)[0] for name in names}):
        namespace_names = [
            name
            for name in names
            if name.startswith(namespace + ".")
            and not (namespace == "h3" and name[3:] in SETTING_NAMES)
        ]
        triggers = [
            (
                selected[name].change
                if name == "workspace.engine"
                else selected[name].input
            )
            for name in namespace_names
        ]
        if namespace == "h3":
            triggers.append(controller.memory.change)
            # Unqueued callbacks do not dispatch State.change in this Gradio
            # path. Save after their explicit resolution refresh instead.
            for event in controller.events:
                if not event.get("queue", True):
                    event.then(
                        remember_namespace(namespace_names, h3=True),
                        inputs=[
                            controller.memory,
                            snapshot,
                            controller.session,
                            *(selected[name] for name in namespace_names),
                        ],
                        outputs=browser_state,
                        queue=True,
                        concurrency_id="h3-preferences",
                        concurrency_limit=1,
                        show_progress="hidden",
                        api_name=False,
                    )
        gr.on(
            triggers=triggers,
            fn=remember_namespace(namespace_names, h3=namespace == "h3"),
            inputs=[
                controller.memory,
                snapshot,
                controller.session,
                *(selected[name] for name in namespace_names),
            ],
            outputs=browser_state,
            queue=True,
            concurrency_id="h3-preferences",
            concurrency_limit=1,
            show_progress="hidden",
            api_name=False,
            trigger_mode="always_last",
        )
    browser_state.h3_restore_event = refreshed
    return browser_state
