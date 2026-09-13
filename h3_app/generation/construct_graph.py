"""H3 construct graph phase."""

from __future__ import annotations

from typing import Any

from .preparation import PreparedH3
from .requests import H3Request
from .services import GenerationServices


def construct_graph(
    request: H3Request, prepared: PreparedH3, services: GenerationServices
) -> dict[str, Any]:
    if request.media.mode == "Reference media":
        graph = services.workflows.build_ref2va_graph(
            prompt=request.media.prompt,
            reference_images=prepared.refs_i,
            reference_videos=prepared.refs_v,
            reference_audios=prepared.refs_a,
            width=prepared.resolved_width,
            height=prepared.resolved_height,
            duration=float(request.output.duration),
            result_format=request.output.result_format,
            image_frames=prepared.requested_image_frames,
            image_vae=prepared.selected_image_vae,
            steps=prepared.effective_steps,
            seed=prepared.actual_seed,
            scheduler=prepared.effective_scheduler,
            ref_image_size=request.media.ref_image_size,
            turbo_lora_name=prepared.turbo_lora_name,
            turbo_variant=prepared.selected_turbo,
            turbo_strength=prepared.turbo_strength,
            use_sol=prepared.effective_sol,
            sol_tau=float(request.sampling.sol_tau),
            use_sage=prepared.effective_sage,
            use_sla=prepared.effective_sla,
            sol_thresh_type=request.sampling.sol_thresh_type,
            sla_preset=prepared.effective_sla_preset,
            sol_exact_mode=request.sampling.sol_exact_mode,
            sol_dense_steps=int(request.sampling.sol_dense_steps),
            sol_step_off=float(request.sampling.sol_step_off),
            sol_sink_tokens=int(request.sampling.sol_sink_tokens),
            cache_mode=prepared.effective_cache_mode,
            fbcache_preset=str(request.sampling.fbcache_preset),
            fbcache_threshold=float(request.sampling.fbcache_threshold),
            fbcache_start=float(request.sampling.fbcache_start),
            fbcache_end=float(request.sampling.fbcache_end),
            fbcache_max_hits=int(request.sampling.fbcache_max_hits),
            fbcache_temporal_guard=bool(request.sampling.fbcache_temporal_guard),
            easycache_threshold=float(request.sampling.easycache_threshold),
            easycache_start=float(request.sampling.easycache_start),
            easycache_end=float(request.sampling.easycache_end),
            easycache_verbose=bool(request.sampling.easycache_verbose),
            model_name=prepared.selected_model,
            models=prepared.models,
            available_nodes=prepared.available,
            use_int8_vae=bool(request.output.use_int8_vae),
            use_trt_vae=bool(request.output.use_trt_vae),
            latent_upscale_model_name=prepared.latent_upscale_model_name,
            latent_upscale_precision=prepared.latent_upscale_precision,
            latent_upscale_refine_steps=int(
                request.finishing.latent_upscale_refine_steps
            ),
            latent_split_config=prepared.latent_split_config,
            text_encoder_name=prepared.selected_text_encoder,
            encoder_small_input=request.sampling.encoder_small_input,
            reuse_unchanged_inputs=bool(request.media.reuse_unchanged_inputs),
            stage_model_offload=prepared.effective_stage_offload,
            smart_stage_offload=prepared.smart_stage_offload,
        )
    else:
        graph = services.workflows.build_fl2va_graph(
            voice_reference_audios=prepared.voice_refs,
            semantic_bridge=request.sampling.semantic_bridge,
            semantic_bridge_alpha=request.sampling.semantic_bridge_alpha,
            prompt=request.media.prompt,
            first_image=request.media.first_image,
            last_image=request.media.last_image,
            width=prepared.resolved_width,
            height=prepared.resolved_height,
            duration=float(request.output.duration),
            result_format=request.output.result_format,
            image_frames=prepared.requested_image_frames,
            image_vae=prepared.selected_image_vae,
            steps=prepared.effective_steps,
            seed=prepared.actual_seed,
            scheduler=prepared.effective_scheduler,
            turbo_lora_name=prepared.turbo_lora_name,
            turbo_variant=prepared.selected_turbo,
            turbo_strength=prepared.turbo_strength,
            use_sol=prepared.effective_sol,
            sol_tau=float(request.sampling.sol_tau),
            use_sage=prepared.effective_sage,
            use_sla=prepared.effective_sla,
            sol_thresh_type=request.sampling.sol_thresh_type,
            sla_preset=prepared.effective_sla_preset,
            sol_exact_mode=request.sampling.sol_exact_mode,
            sol_dense_steps=int(request.sampling.sol_dense_steps),
            sol_step_off=float(request.sampling.sol_step_off),
            sol_sink_tokens=int(request.sampling.sol_sink_tokens),
            cache_mode=prepared.effective_cache_mode,
            fbcache_preset=str(request.sampling.fbcache_preset),
            fbcache_threshold=float(request.sampling.fbcache_threshold),
            fbcache_start=float(request.sampling.fbcache_start),
            fbcache_end=float(request.sampling.fbcache_end),
            fbcache_max_hits=int(request.sampling.fbcache_max_hits),
            fbcache_temporal_guard=bool(request.sampling.fbcache_temporal_guard),
            easycache_threshold=float(request.sampling.easycache_threshold),
            easycache_start=float(request.sampling.easycache_start),
            easycache_end=float(request.sampling.easycache_end),
            easycache_verbose=bool(request.sampling.easycache_verbose),
            model_name=prepared.selected_model,
            models=prepared.models,
            available_nodes=prepared.available,
            use_int8_vae=bool(request.output.use_int8_vae),
            use_trt_vae=bool(request.output.use_trt_vae),
            latent_upscale_model_name=prepared.latent_upscale_model_name,
            latent_upscale_precision=prepared.latent_upscale_precision,
            latent_upscale_refine_steps=int(
                request.finishing.latent_upscale_refine_steps
            ),
            latent_split_config=prepared.latent_split_config,
            text_encoder_name=prepared.selected_text_encoder,
            encoder_small_input=request.sampling.encoder_small_input,
            reuse_unchanged_inputs=bool(request.media.reuse_unchanged_inputs),
            stage_model_offload=prepared.effective_stage_offload,
            smart_stage_offload=prepared.smart_stage_offload,
        )

    return graph
