"""Render the execution plan; never independently decide execution policy."""

from __future__ import annotations

from html import escape
from h3_app.settings import ResolvedSettings

LABELS = {
    "semantic_bridge": "Semantic Bridge",
    "steps": "Base sampling steps",
    "text_encoder": "Text encoder",
    "encoder_small_input": "Qwen small input attention",
    "stage_model_offload": "Stage offload",
    "turbo_variant": "Turbo implementation",
    "attention_mode": "Attention",
    "sla_preset": "SLA preset",
    "scheduler": "Scheduler",
    "latent_upscale_refine_steps": "Refinement steps",
    "auto_megapixels": "Start-frame cap",
    "sol_tau": "Sol tau",
    "sol_thresh_type": "Sol threshold",
    "sol_exact_mode": "Sol prefix mode",
    "sol_dense_steps": "Dense final blocks",
    "cache_mode": "Acceleration",
    "dimensions": "Dimensions",
    "image_frames": "Image frames",
}


def render_settings(plan: ResolvedSettings, extras: dict | None = None) -> str:
    extras = extras or {}
    effective = plan.effective
    output, sampling, finishing = (
        effective.output,
        effective.sampling,
        effective.finishing,
    )
    esc = lambda value: escape(str(value), quote=True)
    differences = plan.differences()
    title = effective.preset + (" · Modified" if differences else "")
    fmt = output.result_format
    length = (
        f"{output.image_frames} image frame(s)"
        if fmt == "Image"
        else f"{output.duration:g} seconds"
    )
    dimensions = f"{output.width}×{output.height}"
    voice_count = sum(bool(getattr(effective, f"fl2va_audio_{i}")) for i in range(1, 4))
    if effective.mode == "First / last frame" and voice_count:
        length += f" · {voice_count} FL2VA voice reference(s)"
    seed = (
        "Independent random seeds"
        if output.batch_count > 1
        else "Random seed"
        if output.seed < 0
        else f"Seed {output.seed}"
    )

    def detail(label, value):
        return f'<div class="h3-setup-detail"><dt>{esc(label)}</dt><dd>{esc(value)}</dd></div>'

    changes = "".join(
        detail(LABELS.get(key, key), f"{before} → {after}")
        for key, (before, after) in differences.items()
    )
    changes = (
        f'<details class="h3-setup-disclosure"><summary>{len(differences)} changes from {esc(effective.preset)}</summary><dl class="h3-setup-changes">{changes}</dl></details>'
        if differences
        else ""
    )
    adjustments = "".join(
        f"<li><strong>{esc(LABELS.get(a.field, a.field))}: {esc(a.effective)}</strong> — {esc(a.reason)}</li>"
        for a in plan.adjustments
    )
    if sampling.text_encoder == "BF16" and not any(
        a.field == "stage_model_offload" for a in plan.adjustments
    ):
        adjustments += "<li><strong>Stage offload: On</strong> — Required by the BF16 text encoder.</li>"
    adjustments = (
        f'<details class="h3-setup-disclosure"><summary>Automatically adjusted / required</summary><ul>{adjustments}</ul></details>'
        if adjustments
        else ""
    )
    issues = "".join(f"<li>{esc(issue)}</li>" for issue in plan.issues)
    issues = (
        f'<div role="alert"><strong>Before generating</strong><ul>{issues}</ul></div>'
        if issues
        else ""
    )
    canvas = (
        f"{output.width // 2}×{output.height // 2}"
        if finishing.latent_upscale
        else dimensions
    )
    enhancement = (
        f"Native 2× refinement · {sampling.latent_upscale_refine_steps} refinement steps"
        if finishing.latent_upscale
        else "Native refinement off"
    )
    if finishing.postprocess != "None":
        enhancement += f" · {finishing.postprocess}"
    output_details = detail("Result", f"{fmt} · {length}") + detail("Seed policy", seed)
    if fmt != "Audio":
        output_details += detail("Generation canvas", canvas) + detail(
            "H3 output dimensions", dimensions
        )
        if finishing.postprocess != "None":
            output_details += detail(
                "Finished dimensions", "Resolved during post-processing"
            )
    technical = detail("Base model", effective.model_profile) + detail(
        "Text encoder", sampling.text_encoder
    )
    technical += detail(
        "Qwen attention",
        "Small input (PyTorch/basic)" if sampling.encoder_small_input else "Server backend",
    )
    technical += detail(
        "Stage offload", "On" if sampling.stage_model_offload else "Off"
    )
    technical += detail(
        "Generation",
        effective.generation_mode
        + (
            f" / {sampling.turbo_variant}"
            if effective.generation_mode == "Turbo"
            else ""
        ),
    )
    technical += detail("Scheduler", sampling.scheduler)
    technical += detail(
        "Semantic Bridge",
        f"Experimental v1 · strength {effective.semantic_bridge_alpha:g} · per token"
        if effective.semantic_bridge else "Off",
    )
    technical += detail(
        "Attention",
        "Automatic · resolved during preparation"
        if sampling.attention_mode == "Auto"
        else sampling.attention_mode,
    )
    if sampling.attention_mode == "SLA":
        technical += detail("SLA preset", sampling.sla_preset)
    technical += detail("Acceleration", effective.cache_mode)
    technical += detail(
        "Decoder",
        output.image_vae
        if fmt == "Image"
        else "TensorRT"
        if effective.use_trt_vae
        else "INT8 ConvRot"
        if effective.use_int8_vae
        else "FP16",
    )
    if finishing.latent_upscale:
        technical += detail(
            "Refinement", extras.get("latent_upscale_method", "Full-frame")
        )
        technical += detail("Latent upscaler", extras.get("latent_upscaler_model", ""))
        if "Split" in str(extras.get("latent_upscale_method", "")):
            technical += detail(
                "Split refinement",
                f"{extras.get('latent_split_tile_width')}×{extras.get('latent_split_tile_height')}px tiles · {extras.get('latent_split_chunk_frames')}f chunks+{extras.get('latent_split_temporal_overlap_frames')}f · polish {extras.get('latent_split_seam_polish')}",
            )
    metrics = detail("Output", f"{fmt} · {length} · {output.batch_count} variant(s)")
    if fmt != "Audio":
        metrics += detail("H3 output", dimensions)
    metrics += detail(
        "Base sampling", f"{effective.generation_mode} · {sampling.steps} steps"
    )
    return (
        f'<section class="h3-setup-card" aria-label="Next run" data-settings-ready="{str(bool(extras.get("_settings_ready"))).lower()}">'
        f'<div class="h3-setup-heading"><strong>Next run</strong><span>{esc(title)}</span></div>'
        f'<dl class="h3-setup-metrics">{metrics}</dl>'
        f'<div class="h3-setup-context"><span>Base model: {esc(effective.model_profile)}</span><span>{esc(seed)}</span>'
        + (f"<span>{esc(enhancement)}</span>" if fmt != "Audio" else "")
        + "</div>"
        + changes
        + adjustments
        + issues
        + '<details class="h3-setup-disclosure"><summary>Execution details</summary>'
        + f'<div class="h3-setup-detail-grid"><dl>{output_details}</dl><dl>{technical}</dl></div></details></section>'
    )
