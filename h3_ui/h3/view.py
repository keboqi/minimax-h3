"""Typed MiniMax H3 view builder."""

from __future__ import annotations
from dataclasses import dataclass, fields
from html import escape
from typing import Any, Mapping
import gradio as gr
from .services import H3ViewServices
from ..step_cards import StepStrip, card_heading
from .sections.finishing import build_finishing_section
from .sections.input_upscale import build_input_upscale_section
from .sections.model import build_model_section
from .sections.output import build_output_section
from .sections.performance import build_performance_section
from .sections.prompt import build_prompt_section
from .sections.references import build_references_section
from .sections.results import build_results_section


@dataclass(frozen=True)
class H3View:
    aspect_ratio: gr.components.Component | None
    size_tier: gr.components.Component | None
    prompt_preview: gr.components.Component
    prompt_proposal: gr.components.Component
    prompt_previous: gr.components.Component
    accept_prompt: gr.components.Component
    keep_prompt: gr.components.Component
    undo_prompt: gr.components.Component
    reference_map: gr.components.Component
    reference_status: gr.components.Component
    repair_references: gr.components.Component
    reference_tag: gr.components.Component
    insert_reference_tag: gr.components.Component
    ltx25_model: gr.components.Component
    video_decoder: gr.components.Component
    settings_used: gr.HTML
    restore_preset: gr.components.Component
    sla_settings: gr.components.Component
    sol_settings: gr.components.Component
    sol_quality_settings: gr.components.Component
    fbcache_settings: gr.components.Component
    easycache_settings: gr.components.Component
    finishing_section: gr.components.Component
    attention_mode: gr.components.Component
    audio_output: gr.components.Component
    auto_megapixels: gr.components.Component
    batch_count: gr.components.Component
    cache_mode: gr.components.Component
    draft_resolution: gr.components.Component
    duration: gr.components.Component
    easycache_end: gr.components.Component
    easycache_start: gr.components.Component
    easycache_threshold: gr.components.Component
    easycache_verbose: gr.components.Component
    enhance_prompt_button: gr.components.Component
    enhance_prompt_status: gr.components.Component
    fast_resolution: gr.components.Component
    fbcache_end: gr.components.Component
    fbcache_max_hits: gr.components.Component
    fbcache_preset: gr.components.Component
    fbcache_start: gr.components.Component
    fbcache_temporal_guard: gr.components.Component
    fbcache_threshold: gr.components.Component
    first: gr.components.Component
    frame_group: gr.components.Component
    gemini_api_key: gr.components.Component
    gemini_prompt_model: gr.components.Component
    gemini_prompt_writer_group: gr.components.Component
    generation_force_offload: gr.components.Component
    generation_ltx25_note: gr.components.Component
    generation_mode: gr.components.Component
    generation_postprocess: gr.components.Component
    generation_postprocess_settings: gr.components.Component
    generation_readiness: gr.components.Component
    generation_seedvr2_model: gr.components.Component
    generation_upscale_resolution: gr.components.Component
    height: gr.components.Component
    help_text: gr.components.Component
    image_clear_selection: gr.components.Component
    image_frame_paths: gr.components.Component
    image_frames: gr.components.Component
    image_output: gr.components.Component
    image_output_group: gr.components.Component
    image_save_selected: gr.components.Component
    image_save_status: gr.components.Component
    image_saved_files: gr.components.Component
    image_select_all: gr.components.Component
    image_selection: gr.components.Component
    image_vae: gr.components.Component
    input_upscale_downloads: gr.components.Component
    input_upscale_force_offload: gr.components.Component
    input_upscale_frame_height: gr.components.Component
    input_upscale_frame_preset: gr.components.Component
    input_upscale_frame_width: gr.components.Component
    input_upscale_model: gr.components.Component
    input_upscale_run: gr.components.Component
    input_upscale_seed: gr.components.Component
    input_upscale_slots: gr.components.Component
    input_upscale_status: gr.components.Component
    large_resolution: gr.components.Component
    last: gr.components.Component
    latent_upscale: gr.components.Component
    latent_upscale_method: gr.components.Component
    latent_upscale_refine_lora: gr.components.Component
    latent_upscale_refine_steps: gr.components.Component
    latent_upscale_settings: gr.components.Component
    latent_upscaler_model: gr.components.Component
    openai_api_key: gr.components.Component
    openai_model: gr.components.Component
    openai_base_url: gr.components.Component
    openai_prompt_writer_group: gr.components.Component
    local_prompt_base_model: gr.components.Component
    local_prompt_greedy: gr.components.Component
    local_prompt_max_tokens: gr.components.Component
    local_prompt_seed: gr.components.Component
    local_prompt_temperature: gr.components.Component
    local_prompt_top_p: gr.components.Component
    local_prompt_writer_group: gr.components.Component
    mode: gr.components.Component
    model_profile: gr.components.Component
    output: gr.components.Component
    output_2: gr.components.Component
    output_3: gr.components.Component
    output_4: gr.components.Component
    preset: gr.components.Component
    prompt: gr.components.Component
    prompt_writer_backend: gr.components.Component
    fl2va_audio_1: gr.components.Component
    fl2va_audio_2: gr.components.Component
    fl2va_audio_3: gr.components.Component
    ref_audio_1: gr.components.Component
    ref_audio_2: gr.components.Component
    ref_audio_3: gr.components.Component
    ref_image_1: gr.components.Component
    ref_image_2: gr.components.Component
    ref_image_3: gr.components.Component
    ref_image_4: gr.components.Component
    ref_image_5: gr.components.Component
    ref_image_6: gr.components.Component
    ref_image_7: gr.components.Component
    ref_image_8: gr.components.Component
    ref_image_9: gr.components.Component
    ref_size: gr.components.Component
    ref_video_1: gr.components.Component
    ref_video_2: gr.components.Component
    ref_video_3: gr.components.Component
    reference_group: gr.components.Component
    refresh: gr.components.Component
    resolution_info: gr.components.Component
    result_format: gr.components.Component
    semantic_bridge: gr.components.Component
    semantic_bridge_alpha: gr.components.Component
    reuse_unchanged_inputs: gr.components.Component
    run: gr.components.Component
    scheduler: gr.components.Component
    seed: gr.components.Component
    settings_overview: gr.components.Component
    sla_preset: gr.components.Component
    sol_dense_steps: gr.components.Component
    sol_exact_mode: gr.components.Component
    sol_sink_tokens: gr.components.Component
    sol_step_off: gr.components.Component
    sol_tau: gr.components.Component
    sol_thresh_type: gr.components.Component
    stage_model_offload: gr.components.Component
    status: gr.components.Component
    steps: gr.components.Component
    stop: gr.components.Component
    text_encoder: gr.components.Component
    encoder_small_input: gr.components.Component
    turbo_variant: gr.components.Component
    trt_vae_compile: gr.components.Component
    use_int8_vae: gr.components.Component
    use_lynnreal_vae: gr.components.Component
    use_sol_vae: gr.components.Component
    video_encoder: gr.components.Component
    lightvae_decoder: gr.components.Component
    use_trt_vae: gr.components.Component
    width: gr.components.Component

    @property
    def values(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def unpack(self) -> tuple[Any, ...]:
        return tuple((getattr(self, name) for name in H3_COMPONENT_ORDER))


H3_COMPONENT_ORDER = (
    "attention_mode",
    "audio_output",
    "auto_megapixels",
    "batch_count",
    "cache_mode",
    "draft_resolution",
    "duration",
    "easycache_end",
    "easycache_start",
    "easycache_threshold",
    "easycache_verbose",
    "enhance_prompt_button",
    "enhance_prompt_status",
    "fast_resolution",
    "fbcache_end",
    "fbcache_max_hits",
    "fbcache_preset",
    "fbcache_start",
    "fbcache_temporal_guard",
    "fbcache_threshold",
    "first",
    "frame_group",
    "gemini_api_key",
    "gemini_prompt_model",
    "gemini_prompt_writer_group",
    "generation_force_offload",
    "generation_ltx25_note",
    "generation_mode",
    "generation_postprocess",
    "generation_postprocess_settings",
    "generation_readiness",
    "generation_seedvr2_model",
    "generation_upscale_resolution",
    "height",
    "help_text",
    "image_clear_selection",
    "image_frame_paths",
    "image_frames",
    "image_output",
    "image_output_group",
    "image_save_selected",
    "image_save_status",
    "image_saved_files",
    "image_select_all",
    "image_selection",
    "image_vae",
    "input_upscale_downloads",
    "input_upscale_force_offload",
    "input_upscale_frame_height",
    "input_upscale_frame_preset",
    "input_upscale_frame_width",
    "input_upscale_model",
    "input_upscale_run",
    "input_upscale_seed",
    "input_upscale_slots",
    "input_upscale_status",
    "large_resolution",
    "last",
    "latent_upscale",
    "latent_upscale_method",
    "latent_upscale_refine_lora",
    "latent_upscale_refine_steps",
    "latent_upscale_settings",
    "latent_upscaler_model",
    "openai_api_key",
    "openai_model",
    "openai_base_url",
    "openai_prompt_writer_group",
    "local_prompt_base_model",
    "local_prompt_greedy",
    "local_prompt_max_tokens",
    "local_prompt_seed",
    "local_prompt_temperature",
    "local_prompt_top_p",
    "local_prompt_writer_group",
    "mode",
    "model_profile",
    "output",
    "output_2",
    "output_3",
    "output_4",
    "preset",
    "prompt",
    "prompt_writer_backend",
    "fl2va_audio_1",
    "fl2va_audio_2",
    "fl2va_audio_3",
    "ref_audio_1",
    "ref_audio_2",
    "ref_audio_3",
    "ref_image_1",
    "ref_image_2",
    "ref_image_3",
    "ref_image_4",
    "ref_image_5",
    "ref_image_6",
    "ref_image_7",
    "ref_image_8",
    "ref_image_9",
    "ref_size",
    "ref_video_1",
    "ref_video_2",
    "ref_video_3",
    "reference_group",
    "refresh",
    "resolution_info",
    "result_format",
    "semantic_bridge",
    "semantic_bridge_alpha",
    "reuse_unchanged_inputs",
    "run",
    "scheduler",
    "seed",
    "settings_overview",
    "sla_preset",
    "sol_dense_steps",
    "sol_exact_mode",
    "sol_sink_tokens",
    "sol_step_off",
    "sol_tau",
    "sol_thresh_type",
    "stage_model_offload",
    "status",
    "steps",
    "stop",
    "text_encoder",
    "encoder_small_input",
    "turbo_variant",
    "trt_vae_compile",
    "use_int8_vae",
    "use_lynnreal_vae",
    "use_sol_vae",
    "video_encoder",
    "lightvae_decoder",
    "use_trt_vae",
    "width",
)


def build_h3_view(
    generation_view: gr.Row,
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    *,
    advanced_root=None,
) -> H3View:
    (
        model_section,
        prompt_section,
        references_section,
        input_upscale_section,
        preset,
        restore_preset,
        settings_overview,
        results_section,
        output_section,
        performance_controls,
        finishing_controls,
        finishing_section,
    ) = _build_workspace_h3(
        generation_view,
        defaults,
        services,
        advanced_root=advanced_root,
    )
    results_section.run.h3_canvas = output_section.canvas_controls
    from ..canvas_controls import bind_image_canvas_preview

    bind_image_canvas_preview(
        output_section,
        references_section.first,
        model_section.result_format,
        model_section.mode,
    )
    # Each control keeps its section field's name; the recipe controls live here.
    controls = {
        "preset": preset,
        "restore_preset": restore_preset,
        "settings_overview": settings_overview,
        "finishing_section": finishing_section,
    }
    for section in (
        model_section, prompt_section, references_section, input_upscale_section,
        results_section, output_section, performance_controls, finishing_controls,
    ):
        for field in fields(section):
            name = field.name
            if name in controls:
                raise ValueError(f"H3 sections define {name!r} twice.")
            controls[name] = getattr(section, name)
    return H3View(**{field.name: controls[field.name] for field in fields(H3View)})


def panel_heading(step: str, title: str, detail: str = "") -> str:
    """A numbered composer section heading: 01 Compose, 02 Preview & generate, 03 Fine-tune."""
    return (
        f'<div class="h3-panel-heading"><span class="h3-step" aria-hidden="true">{step}</span>'
        f"<div><h3>{escape(title)}</h3>" + (f"<p>{escape(detail)}</p>" if detail else "") + "</div></div>"
    )


# Badges each Fine-tune tab with how many of its settings differ from the recipe,
# read from the Next run card's data-changed-tabs. The badge is aria-hidden so the
# tab keeps its plain name; the card itself lists every change with its tab.
CHANGED_TABS_JS = """
const apply = () => {
  const card = element.querySelector('[data-changed-tabs]');
  const strip = document.getElementById('h3-settings-tabs');
  if (!card || !strip) return;
  let counts = {};
  try { counts = JSON.parse(card.dataset.changedTabs || '{}'); } catch (_) {}
  strip.querySelectorAll(':scope > .tab-wrapper [role="tab"]').forEach(tab => {
    const label = tab.dataset.h3Label ??= tab.textContent.trim();
    const count = counts[label] || 0;
    let badge = tab.querySelector('.h3-tab-changes');
    if (!count) { badge?.remove(); return; }
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'h3-tab-changes';
      badge.setAttribute('aria-hidden', 'true');
      tab.append(badge);
    }
    badge.textContent = String(count);
  });
};
new MutationObserver(apply).observe(element, {childList: true, subtree: true, attributes: true});
apply();
"""


# Fine-tune groups every setting beyond Compose's basics by what it changes.
MODEL_TAB, SIZE_TAB, WRITER_TAB, SPEED_TAB, MEMORY_TAB, FINISH_TAB, INPUTS_TAB = range(7)
SETTINGS_TABS = (
    ("Model & sampling", "Which checkpoint runs, Normal or Turbo, and how many steps it samples."),
    ("Size, seed & canvas", "Exact width and height, the start-frame size cap, a fixed seed and the image canvas."),
    ("Prompt writer", "The service Generate / enhance prompt uses, with its model and keys."),
    ("Speed & attention", "Attention backend and sampling acceleration. The defaults suit most jobs."),
    ("Memory & decoding", "Text encoder, model offloading, input reuse, and the video encoder and decoder."),
    # Gradio keeps a hidden tab's button, so audio hides only this tab's controls.
    ("Upscale & finish", "Raise resolution during sampling, or run an upscaling pass after generation. "
                         "Video and image results only."),
    ("Input upscaling", "Upscale uploaded frames or reference pictures with SeedVR2 before generating."),
)


@dataclass(frozen=True)
class FineTune:
    """The Fine-tune tab strip and the empty columns its sections fill."""

    tabs: StepStrip
    model: gr.Column
    sampling: gr.Column
    bridge: gr.Column
    size: gr.Column
    writer: gr.Column
    speed: gr.Column
    memory: gr.Column
    finish: gr.Column
    inputs: gr.Column


def _settings_tab(index, parts=1):
    """Fine-tune tab ``index`` with its heading and ``parts`` columns, filled in order.

    Tabs must be built in index order: the strip's links open them by position.
    """
    label, detail = SETTINGS_TABS[index]
    with gr.Tab(label, id=index, elem_classes=["h3-settings-tab"]):
        card_heading(label, detail)
        return tuple(gr.Column() for _ in range(parts))


def _build_fine_tune(root) -> FineTune:
    with root:
        gr.HTML(panel_heading(
            "03", "Fine-tune",
            "Everything beyond the basics, grouped by what it changes. Values follow the recipe; "
            "a tab's badge counts the settings you changed there.",
        ))
        tabs = StepStrip(gr.Tabs(
            selected=MODEL_TAB, elem_id="h3-settings-tabs",
            elem_classes=["h3-settings-tabs"], overflow_behavior="wrap",
        ))
        with tabs.tabs:
            # Base model, then Turbo/steps/scheduler, then Semantic Bridge.
            model, sampling, bridge = _settings_tab(MODEL_TAB, 3)
            (size,) = _settings_tab(SIZE_TAB)
            (writer,) = _settings_tab(WRITER_TAB)
            (speed,) = _settings_tab(SPEED_TAB)
            (memory,) = _settings_tab(MEMORY_TAB)
            (finish,) = _settings_tab(FINISH_TAB)
            (inputs,) = _settings_tab(INPUTS_TAB)
    return FineTune(tabs, model, sampling, bridge, size, writer, speed, memory, finish, inputs)


def _build_workspace_h3(generation_view, defaults, services, *, advanced_root=None):
    if advanced_root is None:
        raise ValueError("Workspace composition requires an advanced settings root.")
    fine = _build_fine_tune(advanced_root)

    def link(label, index):
        return lambda: fine.tabs.button(
            label, index, classes=("h3-settings-link",), reveal=True
        )

    with generation_view:
        with gr.Column(
            scale=6,
            min_width=320,
            elem_id="h3-composer",
            elem_classes=["h3-composer", "h3-run-panel"],
        ):
            gr.HTML(panel_heading("01", "Compose"))
            conditioning_root = gr.Group()
            references_section = build_references_section(defaults, services)
            prompt_section = build_prompt_section(
                defaults, services, writer_parent=fine.writer,
                writer_link=link("Writer service & keys →", WRITER_TAB),
            )
            with gr.Column(elem_id="h3-output-settings"):
                preset = gr.Radio(
                    ["Singularity", "Quality", "Balanced", "Fast"],
                    value="Singularity",
                    label="Recipe",
                    interactive=True,
                    info="Sets Fine-tune's sampling, attention, memory, refinement and decoding. Keeps your prompt, media and output size.",
                    elem_classes=["h3-recipe-cards"],
                )
                with gr.Row(elem_classes=["h3-settings-links"]):
                    restore_preset = gr.Button("Restore preset settings", size="sm")
                    link("Model & sampling →", MODEL_TAB)()
                model_section = build_model_section(
                    defaults,
                    services,
                    advanced_parent=fine.memory,
                    conditioning_parent=conditioning_root,
                    model_parent=fine.model,
                    bridge_parent=fine.bridge,
                )
                output_settings_section = gr.Column(elem_classes=["h3-essentials"])
                output_section = build_output_section(
                    defaults, services, output_settings_section,
                    details_parent=fine.size,
                    details_link=link("Exact size, seed & canvas →", SIZE_TAB),
                )
        with gr.Column(
            scale=6,
            min_width=320,
            elem_id="h3-preview",
            elem_classes=["h3-preview-panel"],
        ):
            results_root = gr.Column()
            action_root = gr.Column(
                elem_id="h3-generate-actions", elem_classes=["h3-actions-slot"]
            )
            settings_overview = gr.HTML(
                services.compact_settings_summary(
                    defaults["mode"],
                    defaults["model_profile"],
                    defaults["text_encoder"],
                    defaults["stage_model_offload"],
                    defaults["reuse_unchanged_inputs"],
                    defaults["use_int8_vae"],
                    defaults["generation_mode"],
                    defaults["turbo_variant"],
                    defaults["duration"],
                    defaults["width"],
                    defaults["height"],
                    defaults["steps"],
                    defaults["scheduler"],
                    defaults["attention_mode"],
                    defaults["sla_preset"],
                    defaults["cache_mode"],
                    defaults["latent_upscale"],
                    defaults["latent_upscaler_model"],
                    defaults["latent_upscale_refine_steps"],
                    defaults["postprocess"],
                    defaults["seedvr2_model"],
                    services.DEFAULT_LTX25_MODEL,
                    defaults["upscale_force_offload"],
                    defaults["result_format"],
                    defaults["image_frames"],
                    defaults["image_vae"],
                    defaults["latent_upscale_method"],
                    use_trt_vae=defaults["use_trt_vae"],
                    use_lynnreal_vae=defaults["use_lynnreal_vae"],
                    use_sol_vae=defaults["use_sol_vae"],
                    video_encoder=defaults["video_encoder"],
                    lightvae_decoder=defaults["lightvae_decoder"],
                ),
                elem_classes=["h3-settings-summary"],
                js_on_load=CHANGED_TABS_JS,
            )
            with results_root:
                results_section = build_results_section(
                    defaults, services, action_root=action_root
                )
    performance_controls = build_performance_section(
        defaults, services, fine.speed, sampling_parent=fine.sampling
    )
    finishing_controls = build_finishing_section(defaults, services, fine.finish)
    with fine.inputs:
        input_upscale_section = build_input_upscale_section(defaults, services)
    return (
        model_section,
        prompt_section,
        references_section,
        input_upscale_section,
        preset,
        restore_preset,
        settings_overview,
        results_section,
        output_section,
        performance_controls,
        finishing_controls,
        fine.finish,
    )
