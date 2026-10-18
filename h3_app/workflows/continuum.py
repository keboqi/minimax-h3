"""Pure ComfyUI graph for one Sequence (H3 Continuum) review operation.

Node IDs are semantic and fixed: upstream fingerprints the node IDs and inputs
of every route feeding its sampler, so the same request must build the same
upstream topology on every operation for saved chunks to stay reusable. Save
tokens only reach the output node, which is downstream of the sampler.
"""

from __future__ import annotations

from typing import Any, Mapping

from h3_app.catalog import CORE_SAMPLER_NODE, H3_NVENC_SAVE_NODE
from h3_app.continuum_policy import (
    CONTINUITY_CHOICES,
    CONTINUUM_FINALIZE_NODE,
    CONTINUUM_REFERENCES_NODE,
    CONTINUUM_REQUIRED_NODES,
    CONTINUUM_SAMPLER_NODE,
    GENERATION_MODE_REVIEW,
    MAX_REFERENCE_IMAGES,
    RUN_STORAGE_SAVE,
    serialize_prompts,
)
from h3_app.errors import H3Error
from h3_app.generation.continuum_requests import ContinuumAction, ContinuumRequest

SAMPLER_NAME = "res_multistep"
SAMPLER_ID = "continuum_sampler"
STATUS_ID = "continuum_status"
REPORT_ID = "continuum_report"
SAVE_ID = "continuum_save"


def _link(node_id: str, slot: int = 0) -> list[Any]:
    return [node_id, slot]


def required_continuum_nodes(request: ContinuumRequest) -> set[str]:
    required = set(CONTINUUM_REQUIRED_NODES) | {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        CORE_SAMPLER_NODE,
        "BasicScheduler",
        "PrimitiveStringMultiline",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        H3_NVENC_SAVE_NODE,
    }
    if request.first_image or any(request.reference_images):
        required.add("LoadImage")
    return required


def build_continuum_graph(
    request: ContinuumRequest,
    action: ContinuumAction,
    *,
    project_id: str,
    model_name: str,
    text_encoder_name: str,
    video_vae_name: str,
    audio_vae_name: str,
    staged_inputs: Mapping[str, str],
    available_nodes: set[str],
    filename_prefix: str = "h3/sequence",
) -> dict[str, Any]:
    """Build one upstream review operation, decoded, assembled and saved.

    `staged_inputs` maps retained source paths to ComfyUI input names.
    """
    missing = required_continuum_nodes(request) - set(available_nodes)
    if missing:
        raise H3Error(
            "Sequence mode needs H3 Continuum v3.9.8 nodes: "
            + ", ".join(sorted(missing))
            + ". Re-run setup_h3.py (or redeploy Modal) and restart ComfyUI."
        )
    graph: dict[str, Any] = {}

    def add(node_id: str, class_type: str, **inputs: Any) -> str:
        graph[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    def staged(path: str) -> str:
        try:
            return staged_inputs[path]
        except KeyError as exc:
            raise H3Error("A retained Sequence source was not staged.") from exc

    unet = add("continuum_unet", "UNETLoader", unet_name=model_name, weight_dtype="default")
    clip = add(
        "continuum_clip",
        "CLIPLoader",
        clip_name=text_encoder_name,
        type="minimax",
        device="default",
    )
    video_vae = add("continuum_video_vae", "VAELoader", vae_name=video_vae_name)
    audio_vae = add("continuum_audio_vae", "VAELoader", vae_name=audio_vae_name)
    sampler = add("continuum_sampler_select", CORE_SAMPLER_NODE, sampler_name=SAMPLER_NAME)
    sigmas = add(
        "continuum_sigmas",
        "BasicScheduler",
        model=_link(unet),
        scheduler=request.scheduler,
        steps=int(request.steps),
        denoise=1.0,
    )
    prompt = add(
        "continuum_prompt",
        "PrimitiveStringMultiline",
        value=serialize_prompts(request.prompts),
    )
    inputs: dict[str, Any] = {
        "model": _link(unet),
        "clip": _link(clip),
        "video_vae": _link(video_vae),
        "sampler": _link(sampler),
        "sigmas": _link(sigmas),
        "sequence_prompt": _link(prompt),
        # List is explicit even for identical rows; Auto could reinterpret text.
        "prompt_mode": "List",
        "chunks": int(request.chunk_count),
        "chunk_seconds": float(request.chunk_seconds),
        "aspect": "Auto from First Image",
        "preset": "Draft — 0.30 MP",
        "custom_mp": 0.3,
        "continuity": CONTINUITY_CHOICES[request.continuity],
        "base_seed": int(request.seed),
        "audio_continuity": True,
        "diagnostics": "Basic",
        "strict_compatibility": False,
        "debug": False,
        "show_preview": False,
        "run_storage": RUN_STORAGE_SAVE,
        "run_name": "",
        "reference_size": "Match Output",
        "project_id": project_id,
        "video_reference_size": "Efficient - 0.4 MP",
        "continuation_backend": "Standard",
        "generation_mode": GENERATION_MODE_REVIEW,
        "size_source": "Manual",
        "width": int(request.width),
        "height": int(request.height),
        **action.sampler_controls(),
    }
    if request.mode == "First / last frame" and request.first_image:
        first = add("continuum_first_image", "LoadImage", image=staged(request.first_image))
        inputs["first_frame"] = _link(first)
    if request.mode == "Reference media" and any(request.reference_images):
        selectors = request.reference_selectors()
        reference_inputs: dict[str, Any] = {"reference_use": request.reference_use}
        for index in range(1, MAX_REFERENCE_IMAGES + 1):
            reference_inputs[f"reference_r{index}_chunks"] = selectors[index - 1]
            path = request.reference_images[index - 1]
            if path:
                # Fixed slots: @R4 stays slot 4 even when slots 1-3 are empty.
                loaded = add(f"continuum_reference_{index}", "LoadImage", image=staged(path))
                reference_inputs[f"reference_image_{index}"] = _link(loaded)
        references = add("continuum_references", CONTINUUM_REFERENCES_NODE, **reference_inputs)
        inputs["reference_images"] = _link(references)
    add(SAMPLER_ID, CONTINUUM_SAMPLER_NODE, **inputs)
    add(STATUS_ID, "PreviewAny", source=_link(SAMPLER_ID, 3))
    # The sampler emits one latent per physical group; Core maps decoding
    # over those lists and Finalize consumes them with the assembly plan.
    images = add("continuum_decode_video", "VAEDecode", samples=_link(SAMPLER_ID, 0), vae=_link(video_vae))
    audio = add("continuum_decode_audio", "VAEDecodeAudio", samples=_link(SAMPLER_ID, 1), vae=_link(audio_vae))
    final = add(
        "continuum_finalize",
        CONTINUUM_FINALIZE_NODE,
        images=_link(images),
        audio=_link(audio),
        assembly_plan=_link(SAMPLER_ID, 2),
        driving_audio=_link(SAMPLER_ID, 4),
        exact_total_duration=True,
        audio_seam="Auto",
        video_seam="Auto",
        buffer_backend="Auto",
        diagnostics="Basic",
    )
    add(REPORT_ID, "PreviewAny", source=_link(final, 2))
    video = add(
        "continuum_video",
        "CreateVideo",
        images=_link(final, 0),
        audio=_link(final, 1),
        fps=24.0,
        bit_depth=8,
    )
    add(
        SAVE_ID,
        H3_NVENC_SAVE_NODE,
        video=_link(video),
        filename_prefix=filename_prefix,
        preset="p4",
        constant_quality=23,
    )
    return graph


def history_text(history: Mapping[str, Any], node_id: str) -> str:
    """Return the text a PreviewAny node published in ComfyUI history."""
    output = (history.get("outputs") or {}).get(node_id) or {}
    for key in ("text", "string", "value"):
        value = output.get(key)
        if isinstance(value, (list, tuple)):
            return "\n".join(str(item) for item in value if item is not None)
        if isinstance(value, str):
            return value
    return ""
