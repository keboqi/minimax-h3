"""Cached, read-only local readiness evidence without model downloads."""

from html import escape
from pathlib import Path
import time
import gradio as gr
from h3_app.capabilities import ENGINES


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
            '<div class="h3-job-table"><table><caption>Local model file presence</caption>'
            "<thead><tr><th>Component</th><th>File</th><th>Presence</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
        checked_at = time.monotonic()
        return cached

    with root:
        gr.Markdown(
            "## Engine capabilities\n"
            + "\n".join(
                f"- **{engine.label}**: {engine.description}" for engine in ENGINES
            )
        )
        gr.Markdown(
            "Model presence is cached for 30 seconds. Execution checks required nodes, model compatibility "
            "and TensorRT engine currency before submitting a workflow. Files missing locally download only when used."
        )
        refresh = gr.Button("Refresh local model presence")
        with gr.Accordion("Model inventory", open=False):
            inventory = gr.HTML(inspect())
        refresh.click(
            lambda: inspect(True), outputs=inventory, queue=False, api_name=False
        )
