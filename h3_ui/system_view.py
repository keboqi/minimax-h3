"""System controls and cached, read-only local model readiness."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
import time
import gradio as gr
from h3_app.capabilities import ENGINES
from .operations_view import page_heading
from .persistence import RESET_BROWSER_SETTINGS_JS


@dataclass(frozen=True)
class SystemView:
    health: gr.Markdown
    runtime: gr.HTML
    refresh: gr.Button
    unload_models: gr.Button
    memory_status: gr.Markdown
    maintenance: gr.Column


def build_system_view(root, initial_backend, initial_summary) -> SystemView:
    with root:
        with gr.Row(elem_classes=["h3-ops-hero"]):
            page_heading(
                "System",
                "Manage the backend, local models, and browser settings.",
                "M6 4h12a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z "
                "M6 13h12a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2Z "
                "M8 7.5h1m-1 9h1",
            )
            with gr.Row(scale=0, elem_classes=["h3-ops-actions"]):
                gr.Button("Open ComfyUI ↗", link="/comfyui/", scale=0)
                refresh = gr.Button("Refresh backend", variant="primary", scale=0)
        with gr.Row(equal_height=False, elem_classes=["h3-ops-split"]):
            with gr.Column(elem_classes=["h3-ops-panel", "h3-ops-main"], min_width=0):
                gr.Markdown(
                    "### Backend & VRAM\nCheck the runtime and free GPU memory.",
                    elem_classes=["h3-ops-section-title"],
                )
                runtime = gr.HTML(initial_summary, elem_classes=["h3-ops-runtime"])
                with gr.Accordion("Runtime diagnostics", open=False):
                    health = gr.Markdown(initial_backend, elem_classes=["h3-runtime-diagnostics"])
                with gr.Row(elem_classes=["h3-ops-actions"]):
                    unload_models = gr.Button("Unload all models / free VRAM", scale=0)
                memory_status = gr.Markdown()
            with gr.Column(elem_classes=["h3-ops-panel", "h3-ops-aside"], min_width=0):
                gr.Markdown("### Maintenance", elem_classes=["h3-ops-section-title"])
                maintenance = gr.Column(min_width=0)
                with gr.Accordion("Browser settings", open=False):
                    gr.Markdown(
                        "Clears settings saved in this browser and reloads the workspace with app defaults.",
                        elem_classes=["h3-ops-note"],
                    )
                    gr.Button(
                        "Reset all settings to defaults",
                        variant="stop",
                        elem_id="h3-reset-browser-settings",
                    ).click(None, js=RESET_BROWSER_SETTINGS_JS, api_name=False)
    return SystemView(health, runtime, refresh, unload_models, memory_status, maintenance)


def build_readiness_view(root, *, model_specs, models_root):
    cached = None
    checked_at = 0

    def inspect(refresh=False):
        nonlocal cached, checked_at
        if cached is not None and not refresh and time.monotonic() - checked_at < 30:
            return cached
        rows = []
        for name, spec in model_specs.items():
            path = Path(models_root) / spec.folder / spec.local_name
            try:
                present = path.is_file() and path.stat().st_size > 0
            except OSError:
                present = False
            state = "Present locally" if present else "Downloads when used"
            rows.append(
                f"<tr><td>{escape(name)}</td><td>{escape(spec.local_name)}</td><td>{state}</td></tr>"
            )
        cached = (
            '<div class="h3-ops-table h3-model-inventory" tabindex="0"'
            ' role="region" aria-label="Local model file presence">'
            '<table aria-label="Local model file presence"><thead><tr>'
            '<th scope="col">Component</th><th scope="col">File</th><th scope="col">Presence</th>'
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
        checked_at = time.monotonic()
        return cached

    with root, gr.Column(elem_classes=["h3-ops-panel"]):
        with gr.Row(elem_classes=["h3-ops-panel-header"]):
            gr.Markdown(
                "### Model readiness\nCheck which files are already local. Missing files download when used.",
                elem_classes=["h3-ops-section-title"],
            )
            refresh = gr.Button("Refresh local model presence", scale=0)
        with gr.Accordion("Model inventory", open=False) as inventory_panel:
            gr.Markdown("Model presence is cached for 30 seconds.", elem_classes=["h3-ops-note"])
            inventory = gr.HTML(inspect())
        with gr.Accordion("Engine capabilities", open=False):
            gr.Markdown(
                "Execution checks required nodes, model compatibility, and TensorRT engine "
                "currency before submitting a workflow.",
                elem_classes=["h3-ops-note"],
            )
            gr.HTML(
                '<dl class="h3-engine-capabilities">'
                + "".join(
                    f"<div><dt>{escape(engine.label)}</dt><dd>{escape(engine.description)}</dd></div>"
                    for engine in ENGINES
                )
                + "</dl>"
            )
        refresh.click(
            lambda: (inspect(True), gr.update(open=True)),
            outputs=[inventory, inventory_panel],
            queue=False,
            api_name=False,
        )
