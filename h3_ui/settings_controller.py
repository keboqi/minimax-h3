"""Session-local settings transitions and conditional presentation."""

from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from threading import RLock
from typing import Any, Callable, Mapping
import gradio as gr
from h3_app.decoder_intent import VideoDecoder
from h3_app.settings import (
    PRESET_FIELDS,
    transition_modes,
)
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
            "use_lynnreal_vae",
            "use_trt_vae",
            "reuse_unchanged_inputs",
            "semantic_bridge",
            "semantic_bridge_alpha",
            "latent_upscale",
            "latent_upscaler_model",
            "latent_upscale_refine_lora",
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
    *(f"fl2va_audio_{i}" for i in range(1, 4)),
)
ALIASES = dict(zip(GENERATION_COMPONENTS, GENERATION_FIELDS, strict=True))


@dataclass(frozen=True)
class SettingsServices:
    resolve: Callable
    describe: Callable
    cache_defaults: Callable


class SessionSettings:
    """One authoritative transition stream per browser session."""

    def __init__(self):
        self.memory = None
        self.lock = RLock()

    def __deepcopy__(self, memo):
        clone = type(self)()
        clone.memory = deepcopy(self.memory, memo)
        return clone


class SettingsController:
    def __init__(self, components: Mapping[str, Any], services: SettingsServices):
        self.components = components
        self.services = services
        self.names = SETTING_NAMES
        self.inputs = [components[name] for name in (*self.names, *MEDIA_NAMES)]
        self.defaults = {name: components[name].value for name in self.names}
        self.session = gr.State(SessionSettings())
        self.inputs.append(self.session)
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
            "frame_group",
            "reference_group",
        )
        self.outputs += [components[name] for name in self.groups]
        self.outputs += [
            components["settings_overview"],
            components["generation_readiness"],
            components["run"],
        ]

    def update(self, memory, *values, action="edit"):
        if values and isinstance(values[-1], SessionSettings):
            session, values = values[-1], values[:-1]
            with session.lock:
                result = self._update(session.memory or memory, *values, action=action)
                session.memory = result[len(self.names)]
                return result
        return self._update(memory, *values, action=action)

    def _update(self, memory, *values, action="edit"):
        incoming = dict(zip((*self.names, *MEDIA_NAMES), values, strict=True))
        before = dict(incoming)
        if (
            action in (*self.names, *MEDIA_NAMES)
            or action == "restore"
            or action.startswith("decoder:")
        ):
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
        if action.startswith("decoder:"):
            incoming.update(
                VideoDecoder(action.removeprefix("decoder:")).workflow_flags()
            )
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
                *(gr.skip() for _ in self.names),
                new_memory,
                *(gr.skip() for _ in self.groups),
                self.services.describe(request_values),
                "",
                gr.update(interactive=False),
            )
        fmt = current["result_format"]
        updates = []
        previous_presentation = (memory or {}).get("presentation", {})
        presentation = {}
        previous_values = (memory or {}).get("values", before)
        for name in self.names:
            props = {
                key: value
                for key, value in cache_updates.get(name, {}).items()
                if key != "__type__"
            }
            already_entered = (
                name == action or action == "refresh" and name in {"width", "height"}
            ) and current[name] == before[name]
            # Only actual policy changes should write another control's value.
            # A resolution refresh can capture an in-progress numeric edit
            # before its input callback reaches the authoritative session.
            if current[name] != previous_values[name] and not already_entered:
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
                    info=(
                        "Required by the BF16 encoder; effective offload is On."
                        if current["text_encoder"] == "BF16"
                        else "Unload models between generation stages."
                    ),
                )
            if name == "semantic_bridge":
                props["interactive"] = "semantic_bridge" not in plan.inactive
            if name == "semantic_bridge_alpha":
                props["visible"] = bool(plan.effective.semantic_bridge)
            if name == "turbo_variant":
                props["visible"] = current["generation_mode"] == "Turbo"
            if name == "mode":
                props.update(
                    choices=["Text to video", "First / last frame", "Reference media"],
                    interactive=True,
                )
            if name in {"generation_mode", "steps", "scheduler"}:
                props["interactive"] = True
            if name in {"width", "height", "auto_megapixels"}:
                props["visible"] = fmt != "Audio"
            if (
                name in {"width", "height"}
                and self.components.get("aspect_ratio") is not None
            ):
                props["interactive"] = not (
                    fmt == "Image"
                    and current["mode"] == "First / last frame"
                    and bool(current["first"])
                )
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
            presentation[name] = {
                key: value for key, value in props.items() if key != "value"
            }
            props = {
                key: value
                for key, value in props.items()
                if key == "value"
                or previous_presentation.get(name, {}).get(key) != value
            }
            updates.append(gr.update(**props) if props else gr.skip())
        readiness = generation_readiness(
            current["mode"],
            current["prompt"],
            current["first"],
            current["last"],
            [current[n] for n in MEDIA_NAMES if n.startswith("ref_")],
        )
        visibility = (
            current["attention_mode"] == "SLA",
            current["attention_mode"] in {"Sol-Attn", "Auto"},
            current["attention_mode"] in {"Sol-Attn", "Auto"},
            plan.effective.cache_mode == "FirstBlockCache",
            plan.effective.cache_mode == "EasyCache",
            fmt != "Audio",
            current["latent_upscale"] and fmt != "Audio",
            current["mode"] == "First / last frame",
            current["mode"] == "Reference media",
        )
        new_memory["presentation"] = presentation
        new_memory["visibility"] = visibility
        old_visibility = (memory or {}).get("visibility", ())
        return (
            *updates,
            new_memory,
            *(
                (
                    gr.update(visible=v)
                    if i >= len(old_visibility) or old_visibility[i] != v
                    else gr.skip()
                )
                for i, v in enumerate(visibility)
            ),
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

        media_inputs = [self.components[name] for name in MEDIA_NAMES]

        def snapshot(memory, media, change=None):
            current = {
                **self.defaults,
                **(memory or {}).get("values", {}),
                **dict(zip(MEDIA_NAMES, media, strict=True)),
            }
            if change:
                current[change[0]] = change[1]
            return [current[name] for name in (*self.names, *MEDIA_NAMES)]

        def prompt_readiness(memory, *args):
            media, session = args[:-1], args[-1]
            if session.memory is None:
                return (
                    '<p class="h3-readiness" role="status">Restoring your settings…</p>',
                    gr.update(interactive=False),
                )
            # A draft edit does not change model policy. Keep its readiness
            # response small and avoid rewriting all settings components.
            current = dict(
                zip(
                    (*self.names, *MEDIA_NAMES),
                    snapshot(session.memory or memory, media),
                    strict=True,
                )
            )
            request_values = {
                ALIASES.get(key, key): value for key, value in current.items()
            }
            plan = self.services.resolve(request_values)
            readiness = generation_readiness(
                current["mode"],
                current["prompt"],
                current["first"],
                current["last"],
                [current[n] for n in MEDIA_NAMES if n.startswith("ref_")],
            )
            return (
                readiness.html if not plan.issues else "",
                gr.update(interactive=readiness.ready and not plan.issues),
            )

        self.components["prompt"].change(
            prompt_readiness,
            inputs=[self.memory, *media_inputs, self.session],
            outputs=[self.components["generation_readiness"], self.components["run"]],
            queue=False,
            trigger_mode="always_last",
            show_progress="hidden",
            api_name=False,
        )

        def select_decoder(value, memory, *args):
            media, session = args[:-1], args[-1]
            if session.memory is None:
                return tuple(gr.skip() for _ in self.outputs)
            return self.update(
                memory, *snapshot(memory, media), session, action="decoder:" + value
            )

        self.events.append(
            self.components["video_decoder"].input(
                select_decoder,
                inputs=[
                    self.components["video_decoder"],
                    self.memory,
                    *media_inputs,
                    self.session,
                ],
                outputs=self.outputs,
                queue=True,
                concurrency_limit=None,
                trigger_mode="always_last",
                show_progress="hidden",
                api_name=False,
            )
        )
        decoder_flags = [
            self.components[name]
            for name in ("use_int8_vae", "use_lynnreal_vae", "use_trt_vae")
        ]

        def sync_decoder(official, lynnreal, trt):
            try:
                return VideoDecoder.from_flags(official, lynnreal, trt).value
            except ValueError:
                return gr.skip()

        gr.on(
            triggers=[component.change for component in decoder_flags],
            fn=sync_decoder,
            inputs=decoder_flags,
            outputs=self.components["video_decoder"],
            queue=False,
            api_name=False,
            show_progress="hidden",
        )

        # These edits cannot change another control's value or presentation.
        # Presets, modes and policy selectors retain their complete updates.
        independent = {
            "duration",
            "seed",
            "batch_count",
            "width",
            "height",
            "steps",
            "cfg",
            "image_frames",
            "latent_upscale_refine_steps",
            "semantic_bridge_alpha",
            "generation_split_seconds",
            "easycache_threshold",
            "easycache_start",
            "easycache_end",
            "easycache_verbose",
            "fbcache_temporal_guard",
        }
        independent.update(
            name
            for name in self.names
            if isinstance(self.components[name], (gr.Slider, gr.Number))
        )

        def handler(action, output_indices):
            def dispatch(memory, *args):
                session, args = args[-1], args[:-1]
                if session.memory is None:
                    # Restoration writes numeric controls before its complete
                    # refresh. Those change events must not seed partial defaults.
                    return tuple(gr.skip() for _ in output_indices)
                change = (action, args[0]) if action in self.names else None
                media = args[1:] if change else args
                updates = self.update(
                    memory, *snapshot(memory, media, change), session, action=action
                )
                return tuple(updates[index] for index in output_indices)

            return dispatch

        for name in (
            *self.names,
            *(n for n in MEDIA_NAMES if n not in {"first", "prompt"}),
            "restore_preset",
        ):
            component = self.components[name]
            if name == "restore_preset":
                triggers = [component.click]
            elif isinstance(component, gr.Slider):
                # Slider input awaits the frontend binding; release also covers
                # its reset button. Change includes delayed preset writes, which
                # can otherwise overwrite a newer user edit.
                triggers = [component.input, component.release]
            elif isinstance(component, gr.Number):
                # Number input dispatches before its frontend binding commits.
                # Blur/submit observe the edited value without preset echoes.
                triggers = [component.blur, component.submit]
            else:
                triggers = [component.input]
            action = "restore" if name == "restore_preset" else name
            indices = (
                [len(self.names), *range(len(self.outputs) - 3, len(self.outputs))]
                if name in independent
                else list(range(len(self.outputs)))
            )
            self.events.append(
                gr.on(
                    triggers=triggers,
                    fn=handler(action, indices),
                    inputs=[
                        self.memory,
                        *([self.components[name]] if name in self.names else []),
                        *media_inputs,
                        self.session,
                    ],
                    outputs=[self.outputs[index] for index in indices],
                    queue=True,
                    concurrency_limit=None,
                    trigger_mode="always_last",
                    show_progress="hidden",
                    api_name=False,
                )
            )
        self.event = self.events[0]
        return self.event

    def refresh(self, memory, *values):
        return self.update(memory, *values, action="refresh")
