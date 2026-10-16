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
# Runs only in the browser: drop every saved H3 preference, then reload so the
# page restores from the app defaults in a brand-new session.
RESET_BROWSER_SETTINGS_JS = """
() => {
  if (!confirm("Reset all saved UI settings in this browser to the app defaults? The page will reload.")) return;
  try {
    Object.keys(localStorage)
      .filter((key) => key.startsWith("minimax-h3:"))
      .forEach((key) => localStorage.removeItem(key));
  } catch (error) {
    console.error("Could not clear saved settings:", error);
  }
  location.reload();
}
"""
EXTRA_FIELDS = {
    "flux3": ("model", "width", "height", "edit_size", "seed", "steps", "expand_prompt",
              "prompt_backend", "prompt_model", "lightning_model"),
    "drama3": ("seed", "temperature", "top_p", "top_k", "max_frames", "split_long_text", "experts",
               "expert_cache_gb", "prompt_backend", "prompt_model", "lightning_model"),
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
        "lightning_model",
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
        "lightning_model",
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
        "lightning_model",
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
        "lightning_model",
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
        "lightning_model",
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
    # Older preferences can explicitly select a legacy decoder without the
    # LightVAE field. Do not combine that choice with the new default decoder.
    if "h3.lightvae_decoder" in restored and "h3.lightvae_decoder" not in values:
        if any(restored.get("h3." + name) is True for name in (
            "use_int8_vae", "use_lynnreal_vae", "use_trt_vae"
        )):
            restored["h3.lightvae_decoder"] = "none"
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


def create_browser_state():
    """Create the preference store before settings events that write to it."""
    return gr.BrowserState(
        default_value={"schema_version": SCHEMA_VERSION, "values": {}},
        storage_key=_STORAGE_KEY,
        secret=_BROWSER_STATE_SECRET,
    )


def bind_browser_settings(
    demo, components, *, controller, browser_state, navigation=None
):
    """Restore in one page-load response; save from each edit's own response.

    H3 settings and workspace navigation already return the saved payload with
    their updates. Other allowlisted controls save through one direct request.
    """
    selected = {
        name: component
        for name, component in components.items()
        if name in PERSISTED_NAMES
    }
    names = list(selected)
    defaults = {name: component.value for name, component in selected.items()}

    def restore(saved, memory, *values):
        restored, mode_memory = restore_preferences(saved, selected)
        session = values[-1]
        with session.lock:
            session.memory = None
            session.preferences = {
                "schema_version": SCHEMA_VERSION,
                "values": dict(restored),
                "mode_memory": deepcopy(mode_memory),
            }
            session.saved = deepcopy(session.preferences)
        merged = {}

        def put(component, update):
            previous = merged.get(component._id, (component, {}))[1]
            merged[component._id] = (component, {**previous, **update})

        # Unchanged values stay unsent so their change listeners stay idle.
        for name in names:
            if restored[name] != defaults[name]:
                put(selected[name], gr.update(value=restored[name]))
            else:
                put(selected[name], gr.skip())
        if "workspace.engine" in names and isinstance(
            selected["workspace.engine"], gr.Dropdown
        ):
            from h3_app.capabilities import engines_for_task

            choices = engines_for_task(restored["workspace.task"])
            put(
                selected["workspace.engine"],
                gr.update(choices=[(item.label, item.id) for item in choices]),
            )
        settings = list(values)
        for index, name in enumerate(controller.names):
            if "h3." + name in restored:
                settings[index] = restored["h3." + name]
        if navigation is not None:
            pairs = navigation.restore(
                restored.get("workspace.task", "Video"),
                restored.get("workspace.engine"),
                mode_memory,
                settings,
            )
        else:
            pairs = zip(
                controller.outputs,
                controller.update(mode_memory, *settings, action="refresh"),
            )
        for component, update in pairs:
            if not (isinstance(update, dict) and update.get("__type__") == "update"):
                update = gr.update(value=update)
            put(component, update)
        return [merged[key][1] for key in outputs]

    outputs = {component._id: component for component in selected.values()}
    for component in (
        navigation.outputs if navigation is not None else controller.outputs
    ):
        outputs.setdefault(component._id, component)
    demo.load(
        restore,
        inputs=[browser_state, controller.memory, *controller.inputs],
        outputs=list(outputs.values()),
        queue=False,
        show_progress="hidden",
        api_name=False,
    )

    def remember_namespace(namespace_names):
        def remember(session, *values):
            return session.remember(dict(zip(namespace_names, values, strict=True)))

        return remember

    # H3 settings save from SettingsController; navigation saves the workspace.
    owned = {"h3." + name for name in SETTING_NAMES} | {
        "workspace.task",
        "workspace.engine",
    }
    for namespace in sorted({name.split(".", 1)[0] for name in names}):
        namespace_names = [
            name
            for name in names
            if name.startswith(namespace + ".") and name not in owned
        ]
        if not namespace_names:
            continue
        gr.on(
            triggers=[selected[name].input for name in namespace_names],
            fn=remember_namespace(namespace_names),
            inputs=[
                controller.session,
                *(selected[name] for name in namespace_names),
            ],
            outputs=browser_state,
            queue=False,
            show_progress="hidden",
            api_name=False,
            trigger_mode="always_last",
        )
    return browser_state
