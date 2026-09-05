"""Session-local settings transitions and conditional presentation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping
import gradio as gr
from h3_app.settings import PRESET_FIELDS, transition_modes
from h3_app.contracts import GENERATION_COMPONENTS, GENERATION_FIELDS
from .presentation import generation_readiness

SETTING_NAMES = tuple(
    dict.fromkeys(
        (
            "preset",
            "generation_mode",
            *PRESET_FIELDS,
            "cache_mode",
            "mode",
            "result_format",
            "batch_count",
            "duration",
            "width",
            "height",
            "seed",
            "model_profile",
            "image_frames",
            "image_vae",
            "use_int8_vae",
            "use_trt_vae",
            "reuse_unchanged_inputs",
            "latent_upscale",
            "latent_upscaler_model",
            "latent_upscale_method",
            "latent_split_tile_width",
            "latent_split_tile_height",
            "latent_split_overlap_ratio",
            "latent_split_fade_ratio",
            "latent_split_chunk_frames",
            "latent_split_temporal_overlap_frames",
            "latent_split_seam_denoise",
            "latent_split_seam_polish",
            "ltx25_model",
            "generation_postprocess",
            "generation_seedvr2_model",
            "generation_force_offload",
            "generation_split_upscale",
            "generation_split_seconds",
            "generation_upscale_resolution",
            "fbcache_preset",
            "fbcache_threshold",
            "fbcache_start",
            "fbcache_end",
            "fbcache_max_hits",
            "fbcache_temporal_guard",
            "easycache_threshold",
            "easycache_start",
            "easycache_end",
            "easycache_verbose",
        )
    )
)
MEDIA_NAMES = (
    "prompt",
    "first",
    "last",
    *(f"ref_image_{i}" for i in range(1, 10)),
    *(f"ref_video_{i}" for i in range(1, 4)),
    *(f"ref_audio_{i}" for i in range(1, 4)),
)
ALIASES = dict(zip(GENERATION_COMPONENTS, GENERATION_FIELDS, strict=True))


@dataclass(frozen=True)
class SettingsServices:
    resolve: Callable
    describe: Callable
    cache_defaults: Callable


class SettingsController:
    def __init__(self, components: Mapping[str, Any], services: SettingsServices):
        self.components = components
        self.services = services
        self.names = SETTING_NAMES
        self.inputs = [components[name] for name in (*self.names, *MEDIA_NAMES)]
        self.memory = gr.State({"active": "Turbo", "modes": {}})
        self.ids = {
            component._id: name
            for name, component in components.items()
            if hasattr(component, "_id")
        }
        self.outputs = [components[name] for name in self.names] + [self.memory]
        self.groups = (
            "sla_settings",
            "sol_settings",
            "sol_quality_settings",
            "fbcache_settings",
            "easycache_settings",
            "finishing_section",
            "latent_upscale_settings",
        )
        self.outputs += [components[name] for name in self.groups]
        self.outputs += [
            components["settings_overview"],
            components["generation_readiness"],
            components["run"],
        ]

    def update(self, memory, *values, action="edit"):
        incoming = dict(zip((*self.names, *MEDIA_NAMES), values, strict=True))
        before = dict(incoming)
        if action in (*self.names, *MEDIA_NAMES) or action == "restore":
            previous = (memory or {}).get("values")
            if previous:
                incoming = {
                    **incoming,
                    **previous,
                    **{name: incoming[name] for name in MEDIA_NAMES},
                }
                if action in self.names:
                    incoming[action] = before[action]
        if action == "refresh" and (memory or {}).get("values"):
            incoming = {
                **incoming,
                **memory["values"],
                **{name: before[name] for name in (*MEDIA_NAMES, "width", "height")},
            }
        new_memory, current = transition_modes(memory, incoming, action)
        if action == "text_encoder":
            if incoming["text_encoder"] == "BF16":
                new_memory["offload_preference"] = incoming["stage_model_offload"]
                current["stage_model_offload"] = True
            else:
                current["stage_model_offload"] = (memory or {}).get(
                    "offload_preference", incoming["stage_model_offload"]
                )
        elif "offload_preference" in (memory or {}):
            new_memory["offload_preference"] = memory["offload_preference"]
        cache_updates = {}
        if action == "fbcache_preset":
            names = (
                "fbcache_threshold",
                "fbcache_start",
                "fbcache_end",
                "fbcache_max_hits",
            )
            cache_updates = dict(
                zip(
                    names,
                    self.services.cache_defaults(current["fbcache_preset"]),
                    strict=True,
                )
            )
            for name, update in cache_updates.items():
                if "value" in update:
                    current[name] = update["value"]
        new_memory["values"] = {name: current[name] for name in self.names}
        request_values = {
            ALIASES.get(key, key): value for key, value in current.items()
        }
        request_values["_settings_ready"] = True
        try:
            plan = self.services.resolve(request_values)
        except (ValueError, TypeError, RuntimeError):
            return (
                *(gr.update() for _ in self.names),
                new_memory,
                *(gr.update() for _ in self.groups),
                self.services.describe(request_values),
                "",
                gr.update(interactive=False),
            )
        fmt = current["result_format"]
        updates = []
        for name in self.names:
            props = {
                key: value
                for key, value in cache_updates.get(name, {}).items()
                if key != "__type__"
            }
            if current[name] != before[name]:
                props["value"] = current[name]
            if name in {
                "fbcache_threshold",
                "fbcache_start",
                "fbcache_end",
                "fbcache_max_hits",
            }:
                props["interactive"] = current["fbcache_preset"] == "Custom"
            if name == "stage_model_offload":
                props.update(
                    interactive=current["text_encoder"] != "BF16",
                    info="Required by the BF16 encoder; effective offload is On."
                    if current["text_encoder"] == "BF16"
                    else "Unload models between generation stages.",
                )
            if name == "turbo_variant":
                props["visible"] = current["generation_mode"] == "Turbo"
            if name in {"width", "height", "auto_megapixels"}:
                props["visible"] = fmt != "Audio"
            if name == "duration":
                props["visible"] = fmt != "Image"
            if name in {"image_frames", "image_vae"}:
                props["visible"] = fmt == "Image"
            if name == "batch_count":
                props["visible"] = fmt == "Video"
            if name == "latent_upscale":
                props["interactive"] = fmt != "Audio"
            if name == "generation_postprocess":
                props["interactive"] = fmt == "Video"
            updates.append(gr.update(**props))
        readiness = generation_readiness(
            current["mode"],
            current["prompt"],
            current["first"],
            current["last"],
            [current[n] for n in MEDIA_NAMES[3:]],
        )
        visibility = (
            current["attention_mode"] == "SLA",
            current["attention_mode"] in {"Sol-Attn", "Auto"},
            current["attention_mode"] in {"Sol-Attn", "Auto"},
            plan.effective.cache_mode == "FirstBlockCache",
            plan.effective.cache_mode == "EasyCache",
            fmt != "Audio",
            current["latent_upscale"] and fmt != "Audio",
        )
        return (
            *updates,
            new_memory,
            *(gr.update(visible=v) for v in visibility),
            self.services.describe(request_values),
            readiness.html if not plan.issues else "",
            gr.update(
                interactive=readiness.ready and not plan.issues,
                value=f"Generate {fmt.lower()}",
            ),
        )

    def bind(self):
        # Explicit action closures work for ordinary inputs and API invocation.
        # Generic EventData from a multi-trigger gr.on can have no target.
        self.events = []

        def handler(action):
            def dispatch(memory, *values):
                return self.update(memory, *values, action=action)

            return dispatch

        for name in (*self.names, *MEDIA_NAMES, "restore_preset"):
            trigger = (
                self.components[name].click
                if name == "restore_preset"
                # Textbox input can fire before its bound value has updated.
                # Change observes the committed value, including enhancer results.
                else self.components[name].change
                if name == "prompt"
                else self.components[name].input
            )
            self.events.append(
                trigger(
                    handler("restore" if name == "restore_preset" else name),
                    inputs=[self.memory, *self.inputs],
                    outputs=self.outputs,
                    queue=True,
                    concurrency_id="h3-settings",
                    concurrency_limit=1,
                    trigger_mode="always_last",
                    show_progress="hidden",
                    api_name=False,
                )
            )
        self.event = self.events[0]
        return self.event

    def refresh(self, memory, *values):
        return self.update(memory, *values, action="refresh")
