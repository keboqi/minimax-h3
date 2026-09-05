"""Legacy workflow contracts extracted from the production entry point."""
from __future__ import annotations
import gradio_app as app

def selftest() -> None:
    assert app.MODEL_PROFILE_CHOICES == ["Speed", "Quality", "Original"]
    assert app.GEMINI_PROMPT_MODELS == (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    )
    assert app.LIGHTNING_API_ROOT == "https://lightning.ai/api/v1/"
    assert app.LIGHTNING_PROMPT_MODEL == "openai/gpt-5.6-luna"
    assert app.PROMPT_WRITER_BACKENDS == (
        "Local MiniMax-H3 8B",
        "Gemini",
        "Lightning AI",
    )
    with (
        app.unittest.mock.patch.dict(
            app.os.environ, {"LIGHTNING_API_KEY": "selftest-lightning-key"}
        ),
        app.unittest.mock.patch("openai.OpenAI") as openai_client,
    ):
        completion = app.unittest.mock.Mock()
        completion.choices = [
            app.unittest.mock.Mock(
                message=app.unittest.mock.Mock(content="Rewritten Lightning prompt")
            )
        ]
        openai_client.return_value.chat.completions.create.return_value = completion
        rewritten, lightning_status = app._enhance_h3_prompt_with_lightning(
            prompt="A moonlit tracking shot",
            temporary_api_key="",
            mode="Text to video",
            first_image=None,
            last_image=None,
            ref_image_1=None,
            ref_image_2=None,
            ref_image_3=None,
            ref_image_4=None,
            ref_image_5=None,
            ref_image_6=None,
            ref_image_7=None,
            ref_image_8=None,
            ref_image_9=None,
            ref_video_1=None,
            ref_video_2=None,
            ref_video_3=None,
            ref_audio_1=None,
            ref_audio_2=None,
            ref_audio_3=None,
            duration=6,
            width=768,
            height=512,
        )
        assert rewritten == "Rewritten Lightning prompt"
        assert app.LIGHTNING_PROMPT_MODEL in lightning_status
        openai_client.assert_called_once_with(
            base_url=app.LIGHTNING_API_ROOT,
            api_key="selftest-lightning-key",
            timeout=600.0,
        )
        request = openai_client.return_value.chat.completions.create.call_args.kwargs
        assert request["model"] == app.LIGHTNING_PROMPT_MODEL
        assert request["messages"][0]["role"] == "system"
        assert "A moonlit tracking shot" in request["messages"][1]["content"][0]["text"]
    preflight_message, preflight_update = app.generation_preflight(
        "Text to video", "", None, None
    )
    assert "write a prompt" in preflight_message
    assert preflight_update["interactive"] is False
    preflight_message, preflight_update = app.generation_preflight(
        "First / last frame", "A tracking shot", "first.png", None
    )
    assert "Ready to generate" in preflight_message
    assert preflight_update["interactive"] is True
    preflight_message, preflight_update = app.generation_preflight(
        "Reference media", "Match the subject", None, None, None
    )
    assert "add at least one reference" in preflight_message
    assert preflight_update["interactive"] is False
    escaped_readiness = app.generation_readiness_state(
        "Text to video", "<script>prompt</script>", None, None
    )
    assert escaped_readiness.ready is True
    assert "<script>" not in escaped_readiness.html
    assert 'role="status"' in escaped_readiness.html
    assert "&lt;script&gt;" not in escaped_readiness.html  # Prompt is never echoed.
    escaped_backend = app.backend_status_html("<backend error>")
    assert "<backend error>" not in escaped_backend
    assert "&lt;backend error&gt;" in escaped_backend
    assert 'role="alert"' in escaped_backend
    assert "h3-action-dock" in app.H3_UI_CSS
    assert "button:focus-visible" in app.H3_UI_CSS
    assert "@media (max-width: 600px)" in app.H3_UI_CSS
    with app.unittest.mock.patch(
        f"{app.__name__}.backend_status", return_value="Connected self-test"
    ):
        ui_demo = app.build_ui()
    ui_config = ui_demo.get_config_file()
    components_by_id = {
        component["id"]: component for component in ui_config["components"]
    }
    main_tabs = next(
        component
        for component in ui_config["components"]
        if component["type"] == "tabs"
        and component.get("props", {}).get("elem_id") == "h3-main-tabs"
    )

    def find_layout_node(
        node: dict[str, app.Any], component_id: int
    ) -> dict[str, app.Any] | None:
        if node.get("id") == component_id:
            return node
        for child in node.get("children", []):
            match = find_layout_node(child, component_id)
            if match is not None:
                return match
        return None

    tabs_layout = find_layout_node(ui_config["layout"], main_tabs["id"])
    assert tabs_layout is not None
    tab_nodes = tabs_layout["children"]
    assert [components_by_id[node["id"]]["props"]["label"] for node in tab_nodes] == [
        "MiniMax H3",
        "LTX 2.5",
        "MiniMax Music 3",
        "Gallery",
        "API",
    ]
    assert [
        components_by_id[node["children"][0]["id"]]["type"] for node in tab_nodes
    ] == ["row", "group", "group", "group", "group"]
    with app.tempfile.TemporaryDirectory() as output_temp:
        output_root = app.Path(output_temp)
        staging_root = output_root / "h3" / "image_staging"
        staging_root.mkdir(parents=True)
        video_path = output_root / "video.mp4"
        image_path = staging_root / "frame.png"
        video_path.write_bytes(b"video")
        image_path.write_bytes(b"image")
        history = {
            "outputs": {
                "video": {"filename": "video.mp4", "type": "output"},
                "image": {
                    "filename": "frame.png",
                    "subfolder": "h3/image_staging",
                    "type": "output",
                },
                "escape": {
                    "filename": "outside.mp4",
                    "subfolder": "../",
                    "type": "output",
                },
            }
        }
        with app.unittest.mock.patch(f"{app.__name__}.OUTPUT_DIR", output_root):
            assert app._history_output_candidates(history, app.VIDEO_EXTENSIONS) == [
                video_path.resolve()
            ]
            assert app._history_output_candidates(
                history,
                app.IMAGE_EXTENSIONS,
                directory=staging_root,
            ) == [image_path.resolve()]
            assert app._recent_output_candidates(
                output_root,
                app.VIDEO_EXTENSIONS,
                app.time.time(),
            ) == []  # Unowned fallback scans must not claim another job's output.
    with app.tempfile.TemporaryDirectory() as enhancer_temp:
        enhancer_root = app.Path(enhancer_temp)
        first_path = enhancer_root / "first.png"
        last_path = enhancer_root / "last.png"
        video_path = enhancer_root / "reference.mp4"
        for path in (first_path, last_path, video_path):
            path.write_bytes(b"test")
        last_only = app._active_prompt_media(
            "First / last frame", None, str(last_path), (), (), ()
        )
        assert last_only == [("<Picture 1> (last frame)", last_path)]
        both_frames = app._active_prompt_media(
            "First / last frame", str(first_path), str(last_path), (), (), ()
        )
        assert [label for label, _ in both_frames] == [
            "<Picture 1> (first frame)",
            "<Picture 2> (last frame)",
        ]
        references = app._active_prompt_media(
            "Reference media",
            None,
            None,
            (None, str(first_path)),
            (str(video_path),),
            (),
        )
        assert [label for label, _ in references] == ["<Picture 2>", "<Video 1>"]
    music_graph = app.build_music3_graph(
        model_choice=app.DEFAULT_MUSIC3_MODEL,
        caption="Global Metadata: test song",
        lyrics="[Instrumental]",
        max_duration=30,
        seed=7,
        steps=30,
        cfg=1.7,
        ar_cfg=1.7,
        top_k=50,
        tiled_decode=True,
    )
    music_classes = app.graph_class_types(music_graph)
    assert app.required_music3_nodes(True) == music_classes
    music_encode = next(
        node
        for node in music_graph.values()
        if node["class_type"] == "MiniMaxMusic3TextEncode"
    )
    assert music_encode["inputs"]["max_duration"] == 30.0
    assert music_encode["inputs"]["top_k"] == 50
    music_save = next(
        node for node in music_graph.values() if node["class_type"] == "SaveAudioMP3"
    )
    assert music_save["inputs"]["quality"] == "V0"
    assert "format" not in music_save["inputs"]
    rewritten_html = app._rewrite_comfy_text(
        (
            '<html><head></head><body><script src="/assets/app.js">'
            "</script></body></html>"
        ),
        "text/html; charset=utf-8",
    )
    assert '<base href="/comfyui/">' in rewritten_html
    assert 'src="/comfyui/assets/app.js"' in rewritten_html
    rewritten_css = app._rewrite_comfy_text(
        'src:url("/assets/font.woff2")',
        "text/css",
    )
    assert 'url("/comfyui/assets/font.woff2")' in rewritten_css
    assert app._rewrite_comfy_text("const route = '/api';", "application/javascript") == (
        "const route = '/api';"
    )
    assert (
        app._comfy_upstream_path(
            "api/userdata/workflows/LTX 2.5/example.json",
            (b"/comfyui/api/userdata/workflows%2FLTX%202.5%2Fexample.json"),
        )
        == "api/userdata/workflows%2FLTX%202.5%2Fexample.json"
    )
    assert (
        app._comfy_upstream_path(
            "assets/app.js",
            b"/comfyui/assets/app.js",
        )
        == "assets/app.js"
    )
    existing_base = app._rewrite_comfy_text(
        '<html><head><base href="/custom/"></head></html>',
        "text/html",
    )
    assert existing_base.count("<base ") == 1
    filtered_headers = app._proxy_headers(
        {
            "Connection": "keep-alive, x-private",
            "X-Private": "drop",
            "X-Test": "ok",
        }
    )
    assert filtered_headers == {"X-Test": "ok"}
    cookie_response = app._append_set_cookies(
        app.Response(),
        app.httpx.Headers(
            [
                ("set-cookie", "one=1; Path=/"),
                ("set-cookie", "two=2; Path=/"),
            ]
        ),
    )
    assert cookie_response.headers.getlist("set-cookie") == [
        "one=1; Path=/",
        "two=2; Path=/",
    ]
    fake = app.ModelConfig(
        profiles={
            "speed": app.ModelProfile(
                label="Speed",
                fl2va="fl2va_speed.safetensors",
                ref2va="ref2va_speed.safetensors",
            ),
            "quality": app.ModelProfile(
                label="Quality",
                fl2va="fl2va_quality_convrot.safetensors",
                ref2va="ref2va_quality_convrot.safetensors",
            ),
            "original": app.ModelProfile(
                label="Original",
                fl2va="minimax_h3_fl2va_pruned_bf16.safetensors",
                ref2va="minimax_h3_ref2va_pruned_bf16.safetensors",
            ),
        },
        default_profile="speed",
        text_encoder="text.safetensors",
        video_vae="video_vae.safetensors",
        audio_vae="audio_vae.safetensors",
        video_vae_int8="video_vae_int8_convrot.safetensors",
        video_vae_int8_source="test",
        video_vae_trt_encoder="minimax_h3_vae_encoder.onnx",
        video_vae_trt_decoder="minimax_h3_vae_decoder.onnx",
        video_vae_trt_source="test",
        image_vae_500k="minimax_h3_single_frame_decoder_500k.safetensors",
        image_vae_500k_source="test",
        turbo_lora="minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
        turbo_source="test",
        turbo_ref_lora="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        turbo_ref_source="ref2v-test",
        turbo_8step_lora="minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        turbo_8step_source="test",
        turbo_8step_ref_lora="minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        turbo_8step_ref_source="ref2v-test",
        larry_turbo_lora="minimax_h3_turbo_v4_step600_ema.safetensors",
        larry_turbo_source="test",
        larry_turbo_ref_lora="minimax_h3_turbo_v4_step600_ema.safetensors",
        larry_turbo_ref_source="shared-fl2va-test",
        seedvr2_dit="seedvr2_7b_nvfp4.safetensors",
        seedvr2_dit_source="test",
        seedvr2_models={
            "3B NVFP4": "seedvr2_3b_nvfp4.safetensors",
            "3B INT8": "seedvr2_3b_int8_convrot.safetensors",
            "7B NVFP4": "seedvr2_7b_nvfp4.safetensors",
            "7B Sharp NVFP4": "seedvr2_7b_sharp_nvfp4.safetensors",
        },
        seedvr2_vae="seedvr2_ema_vae_fp16.safetensors",
        seedvr2_vae_source="test",
    )
    hybrid_graph = app.Graph()
    trt_decode_ref = ["trt-vae", 0]
    assert (
        app.h3_conditioning_video_vae(
            hybrid_graph,
            fake,
            trt_decode_ref,
            use_trt_vae=True,
            has_visual_conditioning=False,
        )
        == trt_decode_ref
    )
    regular_encode_ref = app.h3_conditioning_video_vae(
        hybrid_graph,
        fake,
        trt_decode_ref,
        use_trt_vae=True,
        has_visual_conditioning=True,
    )
    assert regular_encode_ref != trt_decode_ref
    hybrid_loader = next(iter(hybrid_graph.nodes.values()))
    assert hybrid_loader["class_type"] == "VAELoader"
    assert hybrid_loader["inputs"] == {"vae_name": fake.video_vae}

    trt_graph = app.Graph()
    app.add_model_stack(
        trt_graph,
        fake.profile("speed").fl2va,
        fake,
        turbo_lora_name=None,
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Off",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes={"MiniMaxH3TRTVAELoader"},
        use_trt_vae=True,
    )
    trt_loader = next(
        node
        for node in trt_graph.nodes.values()
        if node["class_type"] == "MiniMaxH3TRTVAELoader"
    )
    assert trt_loader["inputs"] == {
        "decoder": "minimax_h3_vae_decoder.engine",
        "encoder": "None",
    }

    available = app.required_nodes_for(
        "Text to video",
        True,
        "FirstBlockCache",
        use_turbo=True,
    ) | app.required_nodes_for("Reference media", True, "EasyCache", True)
    available |= app.required_nodes_for(
        "Text to video",
        False,
        "Off",
        latent_upscale=True,
    )
    available |= app.required_nodes_for(
        "Text to video",
        False,
        "Off",
        latent_upscale=True,
        latent_upscale_method=app.H3_LATENT_UPSCALE_SPLIT,
    )
    available.add("SpectrumApplyMiniMaxH3")
    available.add(app.CHUNK_FEED_FORWARD_NODE)
    available.add(app.SAGE_ATTENTION_NODE)
    available.add(app.SLA_ATTENTION_NODE)
    available |= {
        app.LARRY_TURBO_LORA_NODE,
        app.LARRY_TURBO_SAMPLER_NODE,
        app.LIGHTX2V_BYPASS_LORA_NODE,
        app.H3_SIGMA_SHIFT_NODE,
        app.H3_SINGLE_FRAME_VAE_LOADER_NODE,
        app.H3_IMAGE_SLICES_NODE,
        app.H3_STAGE_OFFLOAD_NODE,
        app.H3_CONDITIONING_CACHE_NODE,
        app.H3_STAGE_OFFLOAD_POLICY_NODE,
    }
    assert app.h3_text_encoder_settings(fake, "NVFP4 / AWQ") == (
        "text_encoder",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        False,
    )
    assert app.h3_text_encoder_settings(fake, "BF16") == (
        "text_encoder_bf16",
        "qwen3vl_32b_minimax_h3_bf16.safetensors",
        True,
    )
    assert app.text_encoder_offload_update("BF16")["value"] is True
    assert app.text_encoder_offload_update("BF16")["interactive"] is False
    assert app.H3_STAGE_OFFLOAD_NODE in app.required_nodes_for(
        "Text to video", False, "Off", stage_model_offload=True
    )
    assert {
        app.H3_STAGE_OFFLOAD_NODE,
        app.H3_CONDITIONING_CACHE_NODE,
        app.H3_STAGE_OFFLOAD_POLICY_NODE,
    } <= app.required_nodes_for(
        "Text to video",
        False,
        "Off",
        stage_model_offload=True,
        smart_stage_offload=True,
    )
    assert fake.turbo_lora_for("Text to video", app.LIGHTX2V_4STEP_TURBO) == fake.turbo_lora
    assert (
        fake.turbo_lora_for("Reference media", app.LIGHTX2V_4STEP_TURBO)
        == fake.turbo_ref_lora
    )
    assert (
        fake.turbo_lora_for("Text to video", app.LIGHTX2V_8STEP_TURBO)
        == fake.turbo_8step_lora
    )
    assert (
        fake.turbo_lora_for("Reference media", app.LIGHTX2V_8STEP_TURBO)
        == fake.turbo_8step_ref_lora
    )
    with app.unittest.mock.patch(
        f"{app.__name__}.stale_model_keys", return_value=[]
    ) as stale_turbo_models:
        assert (
            app.ensure_turbo_lora(
                fake,
                app.LIGHTX2V_8STEP_TURBO,
                "Reference media",
            )
            is False
        )
    assert stale_turbo_models.call_args.kwargs["model_keys"] == (
        "turbo_8step_ref_lora",
    )
    assert fake.turbo_lora_for("Text to video", app.LARRY_TURBO) == fake.larry_turbo_lora
    reference_updates = app.mode_layout_updates("Reference media")
    assert reference_updates[3].get("interactive") is True
    assert "value" not in reference_updates[3]
    # Avoid staging files in selftest; build prompt-only T2V and check graph wiring.
    graph = app.build_fl2va_graph(
        prompt="test",
        first_image=None,
        last_image=None,
        width=864,
        height=480,
        duration=5,
        steps=18,
        seed=1,
        scheduler="simple",
        turbo_lora_name="minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=True,
        sol_tau=1.0,
        sol_thresh_type="exact",
        sol_exact_mode="exact_kv_and_rows",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="FirstBlockCache",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        model_name=fake.profile("speed").fl2va,
        models=fake,
        available_nodes=available,
        text_encoder_name="qwen3vl_32b_minimax_h3_bf16.safetensors",
        stage_model_offload=True,
        smart_stage_offload=True,
    )
    classes = {node["class_type"] for node in graph.values()}
    expected = {
        "MiniMaxH3ImageToVideo",
        "SamplerCustomAdvanced",
        app.H3_NVENC_SAVE_NODE,
        app.SOL_ATTENTION_NODE,
        app.FUSED_MODULATION_NODE,
        "H3FirstBlockCache",
        app.LIGHTX2V_BYPASS_LORA_NODE,
    }
    missing = expected - classes
    if missing:
        raise SystemExit(f"Selftest failed; missing nodes: {missing}")
    clip_loader = next(
        node for node in graph.values() if node["class_type"] == "CLIPLoader"
    )
    assert clip_loader["inputs"]["clip_name"] == (
        "qwen3vl_32b_minimax_h3_bf16.safetensors"
    )
    assert (
        sum(node["class_type"] == app.H3_STAGE_OFFLOAD_NODE for node in graph.values()) == 1
    )
    cache_node = next(
        node
        for node in graph.values()
        if node["class_type"] == app.H3_CONDITIONING_CACHE_NODE
    )
    expected_cache_key = app.h3_conditioning_cache_key(
        "fl2va",
        "test",
        "qwen3vl_32b_minimax_h3_bf16.safetensors",
        [],
    )
    assert cache_node["inputs"]["cache_key"] == expected_cache_key
    policy_id, policy = next(
        (node_id, node)
        for node_id, node in graph.items()
        if node["class_type"] == app.H3_STAGE_OFFLOAD_POLICY_NODE
    )
    assert policy["inputs"]["cache_key"] == expected_cache_key
    assert expected_cache_key != app.h3_conditioning_cache_key(
        "fl2va",
        "changed",
        "qwen3vl_32b_minimax_h3_bf16.safetensors",
        [],
    )
    final_offload = next(
        node for node in graph.values() if node["class_type"] == app.H3_STAGE_OFFLOAD_NODE
    )
    assert final_offload["inputs"]["enabled"] == [policy_id, 4]

    image_graph = app.build_fl2va_graph(
        prompt="image result test",
        first_image=None,
        last_image=None,
        width=864,
        height=480,
        duration=5,
        steps=4,
        seed=2,
        scheduler="simple",
        turbo_lora_name=None,
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=0,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Off",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        model_name=fake.profile("speed").fl2va,
        models=fake,
        available_nodes=available,
        result_format="Image",
        image_frames=20,
    )
    image_conditioning = next(
        node
        for node in image_graph.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    )
    assert image_conditioning["inputs"]["length"] == 22
    assert {"ImageFromBatch", "SaveImage"} <= {
        node["class_type"] for node in image_graph.values()
    }

    assert app.normalize_result_format("image") == "Image"
    assert app.image_sampling_length(1) == 5
    assert app.image_sampling_length(5) == 5
    assert app.image_sampling_length(6) == 22
    assert app.image_sampling_length(20) == 22
    assert app.single_frame_image_sampling_length(1) == 5
    assert app.selected_image_sampling_length(20, app.OFFICIAL_IMAGE_VAE) == 22
    try:
        app.selected_image_sampling_length(2, app.SINGLE_FRAME_IMAGE_VAE)
    except app.H3Error as exc:
        assert "exactly one output image" in str(exc)
    else:
        raise AssertionError("Single-frame 500K accepted multiple output images")
    single_frame_update = app.image_vae_frame_updates(app.SINGLE_FRAME_IMAGE_VAE)
    assert "value" not in single_frame_update  # Preserve the inactive preference.
    assert single_frame_update["interactive"] is False
    assert {"VAEDecode", "ImageFromBatch", "SaveImage"} <= app.required_nodes_for(
        "Text to video", False, "Off", result_format="Image"
    )
    assert {"VAEDecodeAudio", "SaveAudioMP3"} <= app.required_nodes_for(
        "Text to video", False, "Off", result_format="Audio"
    )

    single_frame_graph = app.build_fl2va_graph(
        prompt="single-frame image result test",
        first_image=None,
        last_image=None,
        width=864,
        height=480,
        duration=5,
        steps=4,
        seed=3,
        scheduler="simple",
        turbo_lora_name=None,
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=0,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Off",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        model_name=fake.profile("speed").fl2va,
        models=fake,
        available_nodes=available,
        result_format="Image",
        image_frames=1,
        image_vae=app.SINGLE_FRAME_IMAGE_VAE,
    )
    single_classes = {node["class_type"] for node in single_frame_graph.values()}
    single_conditioning = next(
        node
        for node in single_frame_graph.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    )
    assert single_conditioning["inputs"]["length"] == 5
    assert {app.H3_SINGLE_FRAME_VAE_LOADER_NODE, app.H3_IMAGE_SLICES_NODE} <= single_classes
    assert "ImageFromBatch" not in single_classes
    single_loader = next(
        node
        for node in single_frame_graph.values()
        if node["class_type"] == app.H3_SINGLE_FRAME_VAE_LOADER_NODE
    )
    assert single_loader["inputs"] == {
        "base_vae_name": fake.video_vae,
        "decoder_name": fake.image_vae_500k,
    }
    assert {app.H3_SINGLE_FRAME_VAE_LOADER_NODE, app.H3_IMAGE_SLICES_NODE} <= (
        app.required_nodes_for(
            "Text to video",
            False,
            "Off",
            result_format="Image",
            image_vae=app.SINGLE_FRAME_IMAGE_VAE,
        )
    )

    image_result_graph = app.Graph()
    app.finish_sampling(
        image_result_graph,
        model_ref=["model", 0],
        conditioning_ref=["conditioning", 0],
        latent_ref=["latent", 0],
        video_vae_ref=["video_vae", 0],
        audio_vae_ref=["audio_vae", 0],
        seed=1,
        steps=4,
        scheduler="simple",
        turbo_variant=None,
        filename_prefix="h3/image_staging/selftest",
        result_format="Image",
        image_frames=3,
    )
    image_result_classes = {
        node["class_type"] for node in image_result_graph.nodes.values()
    }
    assert {"VAEDecode", "ImageFromBatch", "SaveImage"} <= image_result_classes
    assert not image_result_classes & {"VAEDecodeAudio", "CreateVideo", "SaveVideo"}
    image_slice = next(
        node
        for node in image_result_graph.nodes.values()
        if node["class_type"] == "ImageFromBatch"
    )
    assert image_slice["inputs"]["length"] == 3

    audio_result_graph = app.Graph()
    app.finish_sampling(
        audio_result_graph,
        model_ref=["model", 0],
        conditioning_ref=["conditioning", 0],
        latent_ref=["latent", 0],
        video_vae_ref=["video_vae", 0],
        audio_vae_ref=["audio_vae", 0],
        seed=1,
        steps=4,
        scheduler="simple",
        turbo_variant=None,
        filename_prefix="audio/h3_selftest",
        result_format="Audio",
    )
    audio_result_classes = {
        node["class_type"] for node in audio_result_graph.nodes.values()
    }
    assert {"VAEDecodeAudio", "SaveAudioMP3"} <= audio_result_classes
    assert not audio_result_classes & {"VAEDecode", "CreateVideo", "SaveVideo"}

    original_image_output_dir = vars(app)["OUTPUT_DIR"]
    with app.tempfile.TemporaryDirectory() as image_temp:
        image_root = app.Path(image_temp)
        staging = image_root / "h3" / "image_staging"
        staging.mkdir(parents=True)
        staged_paths = []
        image_refs = []
        for index in range(3):
            staged = staging / f"packet_{index:05d}.png"
            staged.write_bytes(f"frame-{index}".encode())
            staged_paths.append(staged)
            image_refs.append(
                {
                    "filename": staged.name,
                    "subfolder": "h3/image_staging",
                    "type": "output",
                }
            )
        vars(app)["OUTPUT_DIR"] = image_root
        try:
            resolved_frames = app.resolve_image_outputs(
                {"outputs": {"save": {"images": image_refs}}}, app.time.time(), 3
            )
            assert resolved_frames == staged_paths
            saved_frames, _ = app.save_selected_image_frames(
                resolved_frames, ["Frame 1", "Frame 3"]
            )
            assert [app.Path(path).name for path in saved_frames] == [
                "frame_001.png",
                "frame_003.png",
            ]
            assert all(app.Path(path).is_file() for path in saved_frames)
        finally:
            vars(app)["OUTPUT_DIR"] = original_image_output_dir

    original_ui_generate = vars(app)["generate"]

    def fake_image_generate(**_args: app.Any):
        yield None, "working"
        yield ["frame-a.png", "frame-b.png"], "complete"

    vars(app)["generate"] = fake_image_generate
    try:
        image_args = [None] * len(app.GENERATION_FIELDS)
        image_args[app.GENERATION_FIELDS.index("result_format")] = "Image"
        image_args[app.GENERATION_FIELDS.index("image_frames")] = 2
        image_args[app.GENERATION_FIELDS.index("seed")] = -1
        ui_updates = list(app.generate_for_ui(1, *image_args))
    finally:
        vars(app)["generate"] = original_ui_generate
    assert len(ui_updates) == 2
    assert all(len(update) == 12 for update in ui_updates)
    assert ui_updates[-1][7] == ["frame-a.png", "frame-b.png"]
    assert app.video_batch_seeds(7, 1) == [7]

    captured_batch_args: list[tuple[app.Any, ...]] = []
    original_generate_signature = app.inspect.signature(original_ui_generate)

    def fake_batch_generate(**batch_args: app.Any):
        captured_batch_args.append(tuple(batch_args[key] for key in app.GENERATION_FIELDS))
        yield None, "working"
        yield f"video-{len(captured_batch_args)}.mp4", "complete"

    fake_batch_generate.__signature__ = original_generate_signature
    generate_parameters = list(original_generate_signature.parameters)
    batch_args = [None] * generate_parameters.index("progress")
    seed_index = generate_parameters.index("seed")
    reuse_index = generate_parameters.index("reuse_unchanged_inputs")
    batch_args[seed_index] = 7
    batch_args[reuse_index] = False
    batch_args[-2] = "Video"
    batch_args[-1] = 5
    original_random_sample = app.random.sample
    app.random.sample = lambda _population, count: list(range(101, 101 + count))
    vars(app)["generate"] = fake_batch_generate
    try:
        assert app.video_batch_seeds(7, 4) == [101, 102, 103, 104]
        batch_updates = list(app.generate_for_ui(3, *batch_args))
    finally:
        vars(app)["generate"] = original_ui_generate
        app.random.sample = original_random_sample
    assert [args[seed_index] for args in captured_batch_args] == [101, 102, 103]
    assert all(args[reuse_index] is False for args in captured_batch_args)
    assert len(batch_updates) == 6
    assert all(len(update) == 12 for update in batch_updates)

    latent_graph = app.build_fl2va_graph(
        prompt="latent upscale test",
        first_image=None,
        last_image=None,
        width=1024,
        height=1024,
        duration=5,
        steps=8,
        seed=2,
        scheduler="beta",
        turbo_lora_name=None,
        turbo_variant=app.LIGHTX2V_8STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=0,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Off",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        model_name=fake.profile("speed").fl2va,
        models=fake,
        available_nodes=available,
        latent_upscale_model_name="minimax_h3_latent_upscaler_3d_bf16.safetensors",
        latent_upscale_precision="bf16",
        latent_upscale_refine_steps=2,
        stage_model_offload=True,
    )
    latent_conditioning = [
        node
        for node in latent_graph.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    ]
    assert {
        (node["inputs"]["width"], node["inputs"]["height"])
        for node in latent_conditioning
    } == {(512, 512), (1024, 1024)}
    latent_samplers = [
        (node_id, node)
        for node_id, node in latent_graph.items()
        if node["class_type"] == "SamplerCustomAdvanced"
    ]
    assert len(latent_samplers) == 2
    assert (
        sum(
            node["class_type"] == app.H3_STAGE_OFFLOAD_NODE
            for node in latent_graph.values()
        )
        == 4
    )
    split_id = next(
        node_id
        for node_id, node in latent_graph.items()
        if node["class_type"] == "SplitSigmas"
    )
    scheduler_id = next(
        node_id
        for node_id, node in latent_graph.items()
        if node["class_type"] == "BasicScheduler"
    )
    noise_id = next(
        node_id
        for node_id, node in latent_graph.items()
        if node["class_type"] == "RandomNoise"
    )
    assert latent_graph[split_id]["inputs"]["step"] == 6
    assert latent_samplers[0][1]["inputs"]["sigmas"] == app.Graph.out(scheduler_id)
    assert latent_samplers[1][1]["inputs"]["sigmas"] == app.Graph.out(split_id, 1)
    assert latent_samplers[0][1]["inputs"]["noise"] == app.Graph.out(noise_id)
    assert latent_samplers[1][1]["inputs"]["noise"] == app.Graph.out(noise_id)
    separate_id = next(
        node_id
        for node_id, node in latent_graph.items()
        if node["class_type"] == app.H3_SEPARATE_AV_LATENT_NODE
    )
    upscaler_id = next(
        node_id
        for node_id, node in latent_graph.items()
        if node["class_type"] == app.H3_LATENT_UPSCALER_NODE
    )
    combine_id = next(
        node_id
        for node_id, node in latent_graph.items()
        if node["class_type"] == app.H3_COMBINE_AV_LATENT_NODE
    )
    assert latent_graph[upscaler_id]["inputs"] == {
        "latent": app.Graph.out(separate_id, 0),
        "model_name": "minimax_h3_latent_upscaler_3d_bf16.safetensors",
        "mode": "scale by multiplier",
        "align": 32,
        "enable_temporal_chunking": True,
        "force_unload": True,
        "device": "cuda",
        "precision": "bf16",
        "mode.scale": 2.0,
    }
    assert latent_graph[combine_id]["inputs"]["video_latent"] == app.Graph.out(upscaler_id)
    assert latent_graph[combine_id]["inputs"]["audio_latent"] == app.Graph.out(
        separate_id, 1
    )
    initial_sampler_id = latent_samplers[0][0]
    audio_decode = next(
        node for node in latent_graph.values() if node["class_type"] == "VAEDecodeAudio"
    )
    audio_offload_ref = audio_decode["inputs"]["samples"]
    assert audio_offload_ref[1] == 3
    audio_offload = latent_graph[audio_offload_ref[0]]
    assert audio_offload["class_type"] == app.H3_STAGE_OFFLOAD_NODE
    initial_offload_ref = audio_offload["inputs"]["additional_latent"]
    initial_offload = latent_graph[initial_offload_ref[0]]
    assert initial_offload["class_type"] == app.H3_STAGE_OFFLOAD_NODE
    assert initial_offload["inputs"]["latent"] == app.Graph.out(initial_sampler_id)
    assert app.h3_latent_upscale_dimensions(1024, 1024) == (512, 512, 1024, 1024)
    assert app.h3_latent_upscale_dimensions(864, 480) == (448, 256, 896, 512)
    split_config = app.resolve_h3_split_upscale_config(
        app.H3_LATENT_UPSCALE_SPLIT,
        tile_width=512,
        tile_height=640,
        overlap_ratio=0.25,
        fade_ratio=0.50,
        chunk_frames=73,
        temporal_overlap_frames=22,
        seam_denoise=0.75,
        seam_polish="auto",
    )
    assert split_config == app.H3SplitUpscaleConfig(
        tile_width=512,
        tile_height=640,
        overlap_ratio=0.25,
        fade_ratio=0.50,
        chunk_frames=73,
        temporal_overlap_frames=22,
        seam_denoise=0.75,
        seam_polish="auto",
    )
    assert (
        app.resolve_h3_split_upscale_config(
            app.H3_LATENT_UPSCALE_STANDARD,
            tile_width=1,
            tile_height=1,
            overlap_ratio=2,
            fade_ratio=2,
            chunk_frames=1,
            temporal_overlap_frames=2,
            seam_denoise=2,
            seam_polish="invalid",
        )
        is None
    )
    assert {
        app.H3_SPLIT_TEMPORAL_PARAMS_NODE,
        app.H3_SPLIT_SPATIAL_PARAMS_NODE,
        app.H3_SPLIT_UPSCALE_NODE,
    } <= app.required_nodes_for(
        "Text to video",
        False,
        "Off",
        latent_upscale=True,
        latent_upscale_method=app.H3_LATENT_UPSCALE_SPLIT,
    )
    split_graph_builder = app.Graph()
    app.finish_sampling(
        split_graph_builder,
        model_ref=["model", 0],
        conditioning_ref=["target-conditioning", 0],
        latent_ref=["target-latent", 0],
        video_vae_ref=["video-vae", 0],
        audio_vae_ref=["audio-vae", 0],
        seed=11,
        steps=8,
        scheduler="beta",
        turbo_variant=None,
        filename_prefix="h3/split-selftest",
        initial_conditioning_ref=["initial-conditioning", 0],
        initial_latent_ref=["initial-latent", 0],
        latent_upscale_model_name=("minimax_h3_latent_upscaler_3d_bf16.safetensors"),
        latent_upscale_precision="bf16",
        latent_upscale_refine_steps=2,
        latent_split_config=split_config,
    )
    split_graph = split_graph_builder.nodes
    split_temporal_id = next(
        node_id
        for node_id, node in split_graph.items()
        if node["class_type"] == app.H3_SPLIT_TEMPORAL_PARAMS_NODE
    )
    split_spatial_id = next(
        node_id
        for node_id, node in split_graph.items()
        if node["class_type"] == app.H3_SPLIT_SPATIAL_PARAMS_NODE
    )
    split_upscale_id = next(
        node_id
        for node_id, node in split_graph.items()
        if node["class_type"] == app.H3_SPLIT_UPSCALE_NODE
    )
    split_sigmas_id = next(
        node_id
        for node_id, node in split_graph.items()
        if node["class_type"] == "SplitSigmas"
    )
    split_noise_id = next(
        node_id
        for node_id, node in split_graph.items()
        if node["class_type"] == "RandomNoise"
    )
    split_sampler_id = next(
        node_id
        for node_id, node in split_graph.items()
        if node["class_type"] == app.CORE_SAMPLER_NODE
    )
    split_combine_id = next(
        node_id
        for node_id, node in split_graph.items()
        if node["class_type"] == app.H3_COMBINE_AV_LATENT_NODE
    )
    assert (
        sum(
            node["class_type"] == "SamplerCustomAdvanced"
            for node in split_graph.values()
        )
        == 1
    )
    assert split_graph[split_temporal_id]["inputs"] == {
        "chunk_frames": 73,
        "temporal_overlap_frames": 22,
        "anchor_strength": 0.999,
        "motion_anchor_frames": "22",
        "identity_anchor_frames": 24,
    }
    assert split_graph[split_spatial_id]["inputs"] == {
        "tile_width": 512,
        "tile_height": 640,
        "overlap_ratio": 0.25,
        "fade_ratio": 0.5,
        "min_tile_size": 256,
        "seam_denoise": 0.75,
    }
    assert split_graph[split_upscale_id]["inputs"] == {
        "model": ["model", 0],
        "conditioning": ["target-conditioning", 0],
        "latent": app.Graph.out(split_combine_id),
        "noise": app.Graph.out(split_noise_id),
        "sampler": app.Graph.out(split_sampler_id),
        "sigmas": app.Graph.out(split_sigmas_id, 1),
        "cfg": 1.0,
        "temporal_split_param": app.Graph.out(split_temporal_id),
        "spatial_split_param": app.Graph.out(split_spatial_id),
        "seam_polish": "auto",
        "color_match": True,
    }

    sol_nodes = [
        node for node in graph.values() if node["class_type"] == app.SOL_ATTENTION_NODE
    ]
    assert len(sol_nodes) == 1
    assert sol_nodes[0]["inputs"]["thresh_type"] == "exact"
    assert sol_nodes[0]["inputs"]["sink_conditioning"] == "exact_kv_and_rows"
    assert sol_nodes[0]["inputs"]["dense_blocks"] == "-1"
    assert sol_nodes[0]["inputs"]["min_tokens"] == app.AUTO_SOL_TOKEN_THRESHOLD
    assert sol_nodes[0]["inputs"]["int8_qk"] is False

    sage_graph = app.Graph()
    app.add_model_stack(
        sage_graph,
        fake.profile("speed").fl2va,
        fake,
        turbo_lora_name=None,
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Off",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes=available,
        use_sage=True,
    )
    sage_nodes = [
        node
        for node in sage_graph.nodes.values()
        if node["class_type"] == app.SAGE_ATTENTION_NODE
    ]
    assert len(sage_nodes) == 1
    assert sage_nodes[0]["inputs"]["sage_attention"] == "auto"
    assert sage_nodes[0]["inputs"]["allow_compile"] is False
    assert not any(
        node["class_type"] == app.SOL_ATTENTION_NODE for node in sage_graph.nodes.values()
    )

    assert app.resolve_sla_preset("Fast") == ("Fast", app.SLA_PRESET_INPUTS["Fast"])
    assert app.resolve_sla_preset("Balance") == ("Balanced", app.SLA_PRESET_INPUTS["Balanced"])
    assert app.resolve_sla_preset("Quality") == ("Quality", app.SLA_PRESET_INPUTS["Quality"])

    sla_graph = app.Graph()
    app.add_model_stack(
        sla_graph,
        fake.profile("speed").fl2va,
        fake,
        turbo_lora_name=None,
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Off",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes=available,
        use_sla=True,
        sla_preset="Quality",
    )
    sla_nodes = [
        node
        for node in sla_graph.nodes.values()
        if node["class_type"] == app.SLA_ATTENTION_NODE
    ]
    assert len(sla_nodes) == 1
    assert sla_nodes[0]["inputs"] == {
        "model": sla_nodes[0]["inputs"]["model"],
        "sparsity_ratio": 0.85,
        "block_size": "64",
        "min_seq_len": 8192,
        "dense_last_steps": 1,
        "protect_audio": True,
        "enabled": True,
    }
    assert not any(
        node["class_type"] in {app.SOL_ATTENTION_NODE, app.SAGE_ATTENTION_NODE}
        for node in sla_graph.nodes.values()
    )

    cache_nodes = [
        node for node in graph.values() if node["class_type"] == "H3FirstBlockCache"
    ]
    assert len(cache_nodes) == 1
    assert cache_nodes[0]["inputs"]["preset"] == "Fast"
    assert cache_nodes[0]["inputs"]["residual_diff_threshold"] == 0.10
    assert cache_nodes[0]["inputs"]["start_percent"] == 0.10
    assert cache_nodes[0]["inputs"]["end_percent"] == 0.95
    assert cache_nodes[0]["inputs"]["max_consecutive_cache_hits"] == 2
    assert cache_nodes[0]["inputs"]["temporal_guard"] is True

    # Turbo LoRA -> fused modulation -> FirstBlockCache -> Sol preserves every
    # wrapper's execution boundary. The inverse cache/Sol ordering reproduces
    # the runtime failure where Sol's executor.original() bypasses the cache.
    turbo_id = next(
        node_id
        for node_id, node in graph.items()
        if node["class_type"] == app.LIGHTX2V_BYPASS_LORA_NODE
    )
    fused_id = next(
        node_id
        for node_id, node in graph.items()
        if node["class_type"] == app.FUSED_MODULATION_NODE
    )
    cache_id = next(
        node_id
        for node_id, node in graph.items()
        if node["class_type"] == "H3FirstBlockCache"
    )
    sol_id = next(
        node_id
        for node_id, node in graph.items()
        if node["class_type"] == app.SOL_ATTENTION_NODE
    )
    assert graph[fused_id]["inputs"]["model"] == [turbo_id, 0]
    assert graph[fused_id]["inputs"]["enabled"] is True
    assert graph[cache_id]["inputs"]["model"] == [fused_id, 0]
    assert graph[sol_id]["inputs"]["model"] == [cache_id, 0]
    assert graph[cache_id]["inputs"]["model"] != [sol_id, 0]

    spectrum_graph = app.Graph()
    app.add_model_stack(
        spectrum_graph,
        fake.profile("quality").fl2va,
        fake,
        turbo_lora_name=None,
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=True,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Spectrum",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes=available,
    )
    spectrum_id = next(
        node_id
        for node_id, node in spectrum_graph.nodes.items()
        if node["class_type"] == "SpectrumApplyMiniMaxH3"
    )
    spectrum_sol_id = next(
        node_id
        for node_id, node in spectrum_graph.nodes.items()
        if node["class_type"] == app.SOL_ATTENTION_NODE
    )
    spectrum_chunk_id = next(
        node_id
        for node_id, node in spectrum_graph.nodes.items()
        if node["class_type"] == app.CHUNK_FEED_FORWARD_NODE
    )
    spectrum_inputs = spectrum_graph.nodes[spectrum_id]["inputs"]
    assert spectrum_inputs["model"] == [spectrum_chunk_id, 0]
    assert spectrum_graph.nodes[spectrum_chunk_id]["inputs"]["model"] == [
        spectrum_sol_id,
        0,
    ]
    assert spectrum_inputs["offline_smoothing_replay"] is True
    assert spectrum_inputs["audio_blend_weight"] == 0.0
    assert spectrum_inputs["offline_archive_storage"] == "system_ram"
    assert spectrum_inputs["model_aware_mode"] == "off"
    assert spectrum_inputs["model_aware_risk_threshold"] == 0.65
    assert not any(
        node["class_type"] == "H3FirstBlockCache"
        for node in spectrum_graph.nodes.values()
    )

    save_nodes = [
        node for node in graph.values() if node["class_type"] == app.H3_NVENC_SAVE_NODE
    ]
    assert len(save_nodes) == 1
    assert save_nodes[0]["inputs"]["preset"] == "p4"
    assert save_nodes[0]["inputs"]["constant_quality"] == 23

    ltx25_graph = app.build_ltx25_graph(
        prompt="a test shot",
        negative_prompt="artifacts",
        first_image=None,
        width=960,
        height=544,
        duration=5,
        fps=24,
        seed=9,
        cfg=1.0,
        sampler_name="euler_ancestral",
        image_strength=0.7,
    )
    ltx25_nodes = list(ltx25_graph.values())
    ltx25_classes = {node["class_type"] for node in ltx25_nodes}
    assert app.required_ltx25_nodes() <= ltx25_classes
    assert not ({"LoadImage", "LTXVAddGuide"} & ltx25_classes)
    ltx25_unet = next(
        node for node in ltx25_nodes if node["class_type"] == "UNETLoader"
    )
    assert ltx25_unet["inputs"]["unet_name"] == (
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    )
    ltx25_video_latent = next(
        node for node in ltx25_nodes if node["class_type"] == "EmptyLTXVLatentVideo"
    )
    assert ltx25_video_latent["inputs"]["length"] == 121
    ltx25_sigmas = next(
        node for node in ltx25_nodes if node["class_type"] == "ManualSigmas"
    )
    assert ltx25_sigmas["inputs"]["sigmas"] == app.LTX25_SIGMAS
    ltx25_save = next(node for node in ltx25_nodes if node["class_type"] == "SaveVideo")
    assert ltx25_save["inputs"]["filename_prefix"].startswith("ltx25/")
    assert app.ltx25_frame_length(5, 24) == 121
    assert len(app.LTX25_WORKFLOWS) == 10
    assert len({entry["id"] for entry in app.LTX25_WORKFLOWS.values()}) == 10
    assert {
        entry["filename"] for entry in app.LTX25_WORKFLOWS.values()
    } == set(app.LTX25_WORKFLOW_FILENAMES)
    assert all(
        entry["filename"].startswith("LTX-2.5_") and entry["filename"].endswith(".json")
        for entry in app.LTX25_WORKFLOWS.values()
    )
    for label in app.LTX25_WORKFLOWS:
        assert set(app.ltx25_workflow_model_keys(label)) <= set(app.MODEL_SPECS)
        assert "/ltx25-workflows/" in app.render_ltx25_workflow_details(label)
    audio_label = next(
        label for label, entry in app.LTX25_WORKFLOWS.items() if entry.get("audio_only")
    )
    assert not {
        "ltx25_video_vae",
        "ltx25_video_vae_full",
    } & set(app.ltx25_workflow_model_keys(audio_label))
    video_label = next(
        label for label, entry in app.LTX25_WORKFLOWS.items() if not entry.get("audio_only")
    )
    assert {
        "ltx25_video_vae",
        "ltx25_video_vae_full",
    } <= set(app.ltx25_workflow_model_keys(video_label))
    inventory = app.render_ltx25_official_model_inventory()
    assert all(app.MODEL_SPECS[key].repo_id in inventory for key in app.LTX25_ICLORA_MODEL_KEYS)
    assert set(app.LTX25_ICLORA_MODEL_KEYS) <= set(app.ltx25_official_inventory_keys())
    original_stage_file = vars(app)["stage_file"]
    try:
        vars(app)["stage_file"] = lambda path, category: f"{category}/{app.Path(path).name}"
        ltx25_keyframe_graph = app.build_ltx25_graph(
            prompt="a guided test shot",
            negative_prompt="artifacts",
            first_image="start.png",
            middle_image="middle.png",
            middle_time=2.5,
            end_image="end.png",
            width=960,
            height=544,
            duration=5,
            fps=24,
            seed=10,
            cfg=1.0,
            sampler_name="euler_ancestral",
            image_strength=0.8,
            middle_strength=0.65,
            end_strength=0.9,
        )
    finally:
        vars(app)["stage_file"] = original_stage_file
    keyframe_nodes = ltx25_keyframe_graph.values()
    guides = [node for node in keyframe_nodes if node["class_type"] == "LTXVAddGuide"]
    assert app.required_ltx25_nodes(image_to_video=True) <= {
        node["class_type"] for node in keyframe_nodes
    }
    assert [node["inputs"]["frame_idx"] for node in guides] == [0, 60, -1]
    assert [node["inputs"]["strength"] for node in guides] == [0.8, 0.65, 0.9]
    guide_ids = [
        node_id
        for node_id, node in ltx25_keyframe_graph.items()
        if node["class_type"] == "LTXVAddGuide"
    ]
    assert guides[1]["inputs"]["positive"] == app.Graph.out(guide_ids[0], 0)
    assert guides[1]["inputs"]["negative"] == app.Graph.out(guide_ids[0], 1)
    assert guides[1]["inputs"]["latent"] == app.Graph.out(guide_ids[0], 2)
    assert guides[2]["inputs"]["positive"] == app.Graph.out(guide_ids[1], 0)
    assert guides[2]["inputs"]["negative"] == app.Graph.out(guide_ids[1], 1)
    assert guides[2]["inputs"]["latent"] == app.Graph.out(guide_ids[1], 2)
    keyframe_guider = next(
        node for node in keyframe_nodes if node["class_type"] == "CFGGuider"
    )
    assert keyframe_guider["inputs"]["positive"] == app.Graph.out(guide_ids[2], 0)
    assert keyframe_guider["inputs"]["negative"] == app.Graph.out(guide_ids[2], 1)
    keyframe_av_latent = next(
        node for node in keyframe_nodes if node["class_type"] == "LTXVConcatAVLatent"
    )
    assert keyframe_av_latent["inputs"]["video_latent"] == app.Graph.out(guide_ids[2], 2)

    seedvr2_graph = app.build_seedvr2_upscale_graph(
        source_video="h3_gradio/seedvr2_upscale/source.mp4",
        seed=7,
        models=fake,
        fps=48.0,
    )
    seedvr2_nodes = list(seedvr2_graph.values())
    assert app.required_seedvr2_upscale_nodes() <= {
        node["class_type"] for node in seedvr2_nodes
    }
    seedvr2_scale = next(
        node for node in seedvr2_nodes if node["class_type"] == "ImageScaleBy"
    )
    assert seedvr2_scale["inputs"]["scale_by"] == 2.0
    seedvr2_vae_nodes = [
        node
        for node in seedvr2_nodes
        if node["class_type"] in {"VAEEncodeTiled", "VAEDecodeTiled"}
    ]
    assert len(seedvr2_vae_nodes) == 2
    assert all(node["inputs"]["tile_size"] == 1024 for node in seedvr2_vae_nodes)
    seedvr2_sampler = next(
        node for node in seedvr2_nodes if node["class_type"] == "KSampler"
    )
    seedvr2_chunks = next(
        node for node in seedvr2_nodes if node["class_type"] == "SeedVR2TemporalChunk"
    )
    assert seedvr2_chunks["inputs"]["chunking_mode"] == "auto"
    assert app.node_stage("VAEEncodeTiled", app.graph_class_types(seedvr2_graph)) == (
        "Encoding H3 video for SeedVR2"
    )
    assert app.node_stage("SeedVR2TemporalChunk", app.graph_class_types(seedvr2_graph)) == (
        "Splitting SeedVR2 video into VRAM-safe chunks"
    )
    assert seedvr2_sampler["inputs"]["steps"] == 1
    assert seedvr2_sampler["inputs"]["denoise"] == 1.0
    seedvr2_video = next(
        node for node in seedvr2_nodes if node["class_type"] == "CreateVideo"
    )
    assert seedvr2_video["inputs"]["fps"] == 48.0
    for seedvr2_choice, expected_name in fake.seedvr2_models.items():
        choice_graph = app.build_seedvr2_upscale_graph(
            source_video="source.mp4",
            seed=7,
            models=fake,
            model_choice=seedvr2_choice,
        )
        choice_loader = next(
            node for node in choice_graph.values() if node["class_type"] == "UNETLoader"
        )
        assert choice_loader["inputs"]["unet_name"] == expected_name

    seedvr2_image_graph = app.build_seedvr2_image_upscale_graph(
        source_images=[
            ("first", "h3_gradio/input_image_upscale/first.png", 2.5),
            ("picture_2", "h3_gradio/input_image_upscale/picture.png", 1.25),
        ],
        seed=7,
        models=fake,
        model_choice=app.DEFAULT_SEEDVR2_MODEL,
        output_token="inputtest",
    )
    seedvr2_image_nodes = list(seedvr2_image_graph.values())
    assert app.required_seedvr2_image_upscale_nodes() <= {
        node["class_type"] for node in seedvr2_image_nodes
    }
    assert sum(node["class_type"] == "VAELoader" for node in seedvr2_image_nodes) == 1
    assert sum(node["class_type"] == "UNETLoader" for node in seedvr2_image_nodes) == 1
    assert [
        node["inputs"]["seed"]
        for node in seedvr2_image_nodes
        if node["class_type"] == "KSampler"
    ] == [7, 8]
    assert [
        node["inputs"]["scale_by"]
        for node in seedvr2_image_nodes
        if node["class_type"] == "ImageScaleBy"
    ] == [2.5, 1.25]
    assert {
        node["inputs"]["filename_prefix"]
        for node in seedvr2_image_nodes
        if node["class_type"] == "SaveImage"
    } == {
        "h3/input_upscale/inputtest_first",
        "h3/input_upscale/inputtest_picture_2",
    }
    assert tuple(app.INPUT_IMAGE_UPSCALE_SLOTS[:3]) == (
        "First frame",
        "Last frame",
        "Picture 1",
    )
    assert app.INPUT_IMAGE_FRAME_PRESETS["1920 × 1920"] == (1920, 1920)
    assert app.input_image_frame_preset_updates("3840 × 3840", 1920, 1920) == (3840, 3840)
    from PIL import Image

    with app.tempfile.TemporaryDirectory() as upscale_dimensions_temp:
        dimension_root = app.Path(upscale_dimensions_temp)
        portrait = dimension_root / "portrait.png"
        square = dimension_root / "square.png"
        landscape = dimension_root / "landscape.png"
        Image.new("RGB", (500, 700)).save(portrait)
        Image.new("RGB", (2048, 2048)).save(square)
        Image.new("RGB", (800, 400)).save(landscape)
        assert app.input_image_upscale_dimensions(str(portrait), 1920, 1920)[:4] == (
            500,
            700,
            1371,
            1920,
        )
        assert app.input_image_upscale_dimensions(str(square), 1920, 1920) == (
            2048,
            2048,
            2048,
            2048,
            1.0,
        )
        assert app.input_image_upscale_dimensions(str(landscape), 1920, 1920)[:4] == (
            800,
            400,
            1920,
            960,
        )

    ltx25_upscale_graph = app.build_ltx25_upscale_graph(
        source_video="h3_gradio/ltx25_upscale/source.mp4",
        seed=11,
        model_choice="INT8 ConvRot",
        prompt="a detailed test scene",
        width=864,
        height=480,
        fps=24.0,
    )
    ltx25_upscale_nodes = list(ltx25_upscale_graph.values())
    assert app.required_ltx25_upscale_nodes() <= {
        node["class_type"] for node in ltx25_upscale_nodes
    }
    ltx25_upscale_loader = next(
        node
        for node in ltx25_upscale_nodes
        if node["class_type"] == "LTXICLoRALoaderModelOnly"
    )
    assert ltx25_upscale_loader["inputs"]["lora_name"] == (
        "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    )
    ltx25_upscale_unet = next(
        node for node in ltx25_upscale_nodes if node["class_type"] == "UNETLoader"
    )
    assert ltx25_upscale_unet["inputs"]["unet_name"] == (
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    )
    ltx25_upscale_latent = next(
        node
        for node in ltx25_upscale_nodes
        if node["class_type"] == "EmptyLTXVLatentVideo"
    )
    assert ltx25_upscale_latent["inputs"]["width"] == 1728
    assert ltx25_upscale_latent["inputs"]["height"] == 960
    assert any(node["class_type"] == "LTXVCropGuides" for node in ltx25_upscale_nodes)
    assert app.COMFY_UPSCALE_OPTIONS == {app.SEEDVR2_UPSCALE, app.LTX25_UPSCALE}
    assert app.UVICORN_WEBSOCKET_OPTIONS == {
        "ws": "wsproto",
        "ws_per_message_deflate": False,
    }
    assert app.POSTPROCESS_OPTIONS == [
        app.SEEDVR2_UPSCALE,
        app.LTX25_UPSCALE,
        app.SWIFTVR_UPSCALE,
        "48 fps interpolation",
    ]
    assert app.GENERATION_POSTPROCESS_OPTIONS == [
        "None",
        app.SEEDVR2_UPSCALE,
        app.LTX25_UPSCALE,
        app.SWIFTVR_UPSCALE,
    ]

    assert app.resolution_choice_values("9:16 · 768×1344", "large")[:2] == (768, 1344)
    assert app.resolution_choice_values("1:1 · 1024×1024", "large")[:2] == (1024, 1024)
    assert set(app.RESOLUTION_TIERS) == {"draft", "fast", "large"}
    assert app.preset_values("Quality")[0] == 20
    assert app.preset_values("Balanced")[0] == 18
    assert app.preset_values("Fast")[0] == 15
    assert all(len(values) == 11 for values in app.SAMPLING_PRESETS.values())
    assert app.preset_values("Fast")[6:11] == (
        "1 MP",
        app.LIGHTX2V_4STEP_TURBO,
        "SLA",
        "Fast",
        2,
    )
    assert app.preset_values("Balanced")[6:11] == (
        "2 MP",
        app.LARRY_TURBO,
        "SLA",
        "Balanced",
        2,
    )
    assert app.preset_values("Quality")[6:11] == (
        "4 MP",
        app.LIGHTX2V_8STEP_TURBO,
        "SLA",
        "Quality",
        2,
    )
    for preset_name, encoder in app.SAMPLING_PRESET_TEXT_ENCODERS.items():
        preset_defaults = app.preset_values(preset_name)
        assert preset_defaults[11] == encoder
        offload_update = preset_defaults[12]
        assert offload_update["value"] is (encoder == "BF16")
        assert offload_update["interactive"] is (encoder != "BF16")
    assert app.preset_values("Fast", "Turbo")[0] == 4
    assert app.preset_values("Balanced", "Turbo")[0] == 6
    assert app.preset_values("Quality", "Turbo")[0] == 8
    assert app.preset_values("unknown") == app.preset_values("Balanced")
    assert app.UI_DEFAULTS["steps"] == app.turbo_steps_for(app.UI_DEFAULTS["turbo_variant"])
    assert app.UI_DEFAULTS["width"] == 864 and app.UI_DEFAULTS["height"] == 480
    assert app.UI_DEFAULTS["reuse_unchanged_inputs"] is True
    assert 'api_name="/generate_video"' in app.api_guide()

    original_input_dir = vars(app)["INPUT_DIR"]
    with app.tempfile.TemporaryDirectory() as staging_temp:
        staging_root = app.Path(staging_temp)
        staged_input_root = staging_root / "comfy-input"
        source = staging_root / "reference.png"
        source.write_bytes(b"same input bytes")
        vars(app)["INPUT_DIR"] = staged_input_root
        try:
            with app.unittest.mock.patch("builtins.print") as cache_print:
                cached_first = app.stage_file(str(source), "reference_images", reuse=True)
                cached_second = app.stage_file(str(source), "reference_images", reuse=True)
            assert cached_first == cached_second
            assert cache_print.call_count == 2
            assert "Stored" in cache_print.call_args_list[0].args[0]
            assert "Reusing" in cache_print.call_args_list[1].args[0]
            assert (
                staged_input_root / cached_first
            ).read_bytes() == b"same input bytes"

            (staged_input_root / cached_first).write_bytes(b"")
            with app.unittest.mock.patch("builtins.print") as repair_print:
                repaired = app.stage_file(str(source), "reference_images", reuse=True)
            assert repaired == cached_first
            assert "Stored" in repair_print.call_args.args[0]
            assert (staged_input_root / repaired).read_bytes() == b"same input bytes"

            uncached_first = app.stage_file(str(source), "reference_images", reuse=False)
            uncached_second = app.stage_file(str(source), "reference_images", reuse=False)
            assert uncached_first != uncached_second

            source.write_bytes(b"changed input bytes")
            with app.unittest.mock.patch("builtins.print"):
                changed = app.stage_file(str(source), "reference_images", reuse=True)
            assert changed != cached_first

            video_source = staging_root / "reference.mov"
            video_source.write_bytes(b"video input bytes")

            def fake_ffmpeg(command: list[str], **_kwargs: app.Any):
                app.Path(command[-1]).write_bytes(b"transcoded video")
                return app.unittest.mock.Mock(returncode=0, stderr="")

            with (
                app.unittest.mock.patch("subprocess.run", side_effect=fake_ffmpeg) as run,
                app.unittest.mock.patch("builtins.print"),
            ):
                video_first = app.stage_file(
                    str(video_source),
                    "reference_videos",
                    transcode_video=True,
                    reuse=True,
                )
                video_second = app.stage_file(
                    str(video_source),
                    "reference_videos",
                    transcode_video=True,
                    reuse=True,
                )
            assert video_first == video_second
            assert run.call_count == 1
            assert (staged_input_root / video_first).read_bytes() == b"transcoded video"

            failed_video = staging_root / "failed.mov"
            failed_video.write_bytes(b"failed video bytes")
            with app.unittest.mock.patch(
                "subprocess.run", side_effect=OSError("ffmpeg unavailable")
            ):
                try:
                    app.stage_file(
                        str(failed_video),
                        "reference_videos",
                        transcode_video=True,
                        reuse=True,
                    )
                except app.H3Error as exc:
                    assert "Could not stage input file" in str(exc)
                else:
                    raise AssertionError("Expected failed video staging to raise")
            video_cache_dir = staged_input_root / "h3_gradio" / "reference_videos"
            assert not list(video_cache_dir.glob(".*.tmp.mp4"))
        finally:
            vars(app)["INPUT_DIR"] = original_input_dir
    captured_free_call: dict[str, app.Any] = {}
    original_api_post = vars(app)["api_post"]
    original_backend_status = vars(app)["backend_status"]

    def fake_api_post(path: str, **kwargs: app.Any) -> None:
        captured_free_call["path"] = path
        captured_free_call["kwargs"] = kwargs

    vars(app)["api_post"] = fake_api_post
    vars(app)["backend_status"] = lambda: "refreshed backend"
    try:
        app.unload_comfy_models()
        unload_message, unload_status = app.unload_all_models()
    finally:
        vars(app)["api_post"] = original_api_post
        vars(app)["backend_status"] = original_backend_status
    assert captured_free_call == {
        "path": "/free",
        "kwargs": {"json": {"unload_models": True, "free_memory": True}},
    }
    assert unload_message == "All models unloaded and cached VRAM released."
    assert unload_status == "refreshed backend"
    captured_api_call: dict[str, app.Any] = {}
    original_generate = vars(app)["generate"]
    original_download_url = vars(app)["absolute_video_download_url"]

    def fake_generate(*args: app.Any, **kwargs: app.Any):
        captured_api_call["args"] = args
        captured_api_call["kwargs"] = kwargs
        yield "video.mp4", "complete"

    def fake_download_url(video: str, _request: app.Any) -> str:
        return f"https://example.test/downloads/{video}"

    vars(app)["generate"] = fake_generate
    vars(app)["absolute_video_download_url"] = fake_download_url
    try:
        assert list(app.generate_with_ui_defaults("API prompt", object())) == [
            ("https://example.test/downloads/video.mp4", "complete")
        ]
    finally:
        vars(app)["generate"] = original_generate
        vars(app)["absolute_video_download_url"] = original_download_url
    assert captured_api_call["args"] == ()
    api_kwargs = captured_api_call["kwargs"]
    assert api_kwargs["prompt"] == "API prompt"
    assert api_kwargs["mode"] == "Text to video"
    assert api_kwargs["model_profile"] == "Speed"
    assert api_kwargs["turbo_variant"] == app.DEFAULT_TURBO
    for key, expected in app.UI_DEFAULTS.items():
        assert api_kwargs[key] == expected
    assert all(api_kwargs[f"ref_image_{index}"] is None for index in range(1, 10))
    assert all(api_kwargs[f"ref_video_{index}"] is None for index in range(1, 4))
    assert all(api_kwargs[f"ref_audio_{index}"] is None for index in range(1, 4))
    assert app.video_download_path(app.OUTPUT_DIR / "h3" / "result video.mp4") == (
        "/downloads/comfy/h3/result%20video.mp4"
    )
    original_output_dir = vars(app)["OUTPUT_DIR"]
    original_outputs_dir = vars(app)["OUTPUTS_DIR"]
    original_thumbnails_dir = vars(app)["GALLERY_THUMBNAILS_DIR"]
    original_gallery_thumbnail = vars(app)["gallery_thumbnail"]
    original_gallery_video_resolution = vars(app)["gallery_video_resolution"]
    with app.tempfile.TemporaryDirectory() as gallery_temp:
        gallery_root = app.Path(gallery_temp)
        comfy_test_output = gallery_root / "comfy"
        gradio_test_output = gallery_root / "gradio"
        comfy_test_output.mkdir()
        gradio_test_output.mkdir()
        fallback_video = comfy_test_output / "fallback.mp4"
        fallback_video.write_bytes(b"test")
        vars(app)["OUTPUT_DIR"] = comfy_test_output
        vars(app)["OUTPUTS_DIR"] = gradio_test_output
        vars(app)["GALLERY_THUMBNAILS_DIR"] = gradio_test_output / ".thumbs"
        vars(app)["gallery_thumbnail"] = lambda _video: None
        vars(app)["gallery_video_resolution"] = lambda _video: (864, 480)
        try:
            h3_video = comfy_test_output / "h3" / "minimax.mp4"
            ltx25_video = comfy_test_output / "ltx25" / "ltx.mp4"
            h3_video.parent.mkdir()
            ltx25_video.parent.mkdir()
            h3_video.write_bytes(b"h3")
            ltx25_video.write_bytes(b"ltx25")
            assert len(app.gallery_video_paths(limit=1)) == 1
            assert len(app.gallery_video_paths(limit=None)) == 3
            discovered_families = {
                app.generated_video_family(video) for video in app.gallery_video_paths()
            }
            assert {"MiniMax H3", "LTX-2.5", "Post-processed"} <= discovered_families
            h3_video.unlink()
            ltx25_video.unlink()
            gallery_items, gallery_paths, gallery_detail = app.refresh_gallery()

            class FakeGalleryRequest:
                class request:
                    base_url = "https://example.test/"

            class FakeSelectEvent:
                index = 0

            gallery_play_url, gallery_download_link, selected_video = (
                app.select_gallery_video(
                    gallery_paths,
                    FakeGalleryRequest(),  # type: ignore[arg-type]
                    FakeSelectEvent(),  # type: ignore[arg-type]
                )
            )
            unconfirmed_delete = app.delete_selected_gallery_video(selected_video, False)
            fallback_exists_after_unconfirmed = fallback_video.exists()
            confirmed_delete = app.delete_selected_gallery_video(selected_video, True)
            fallback_exists_after_delete = fallback_video.exists()

            empty_video_1 = comfy_test_output / "empty-1.mp4"
            empty_video_2 = gradio_test_output / "empty-2.mp4"
            empty_video_1.write_bytes(b"test")
            empty_video_2.write_bytes(b"test")
            unconfirmed_empty = app.empty_generated_gallery(None, False)
            empty_exists_after_unconfirmed = (
                empty_video_1.exists() and empty_video_2.exists()
            )
            confirmed_empty = app.empty_generated_gallery(None, True)
            empty_exists_after_delete = empty_video_1.exists() or empty_video_2.exists()
            try:
                app.managed_video_path(gallery_root / "outside.mp4", require_file=False)
                raise AssertionError("Unmanaged gallery path was accepted")
            except app.H3Error:
                pass
        finally:
            vars(app)["OUTPUT_DIR"] = original_output_dir
            vars(app)["OUTPUTS_DIR"] = original_outputs_dir
            vars(app)["GALLERY_THUMBNAILS_DIR"] = original_thumbnails_dir
            vars(app)["gallery_thumbnail"] = original_gallery_thumbnail
            vars(app)["gallery_video_resolution"] = original_gallery_video_resolution
        assert len(gallery_items) == 1
        assert "864×480" in gallery_items[0][1]
        assert gallery_paths == [str(fallback_video)]
        assert gallery_play_url == str(fallback_video)
        assert selected_video == str(fallback_video)
        assert gallery_download_link.endswith(
            "/downloads/comfy/fallback.mp4?download=1)"
        )
        assert "**Resolution:** 864×480" in gallery_download_link
        assert "1 generated video" in gallery_detail
        assert "1 thumbnail" in gallery_detail
        assert fallback_exists_after_unconfirmed is True
        assert "Confirm permanent deletion" in unconfirmed_delete[2]
        assert fallback_exists_after_delete is False
        assert "Deleted `fallback.mp4`" in confirmed_delete[2]
        assert empty_exists_after_unconfirmed is True
        assert "Confirm permanent deletion" in unconfirmed_empty[2]
        assert empty_exists_after_delete is False
        assert "Deleted 2 generated videos" in confirmed_empty[2]
    assert (
        app.estimate_packed_tokens("Text to video", 1344, 768, 5)
        >= app.AUTO_SOL_TOKEN_THRESHOLD
    )
    kitchen_policy = app.resolve_sol_policy(
        "Kitchen", "Text to video", 608, 352, 2, None, None
    )
    assert kitchen_policy[0] is False and kitchen_policy[2] == "forced Comfy Kitchen"
    sage_policy = app.resolve_sol_policy("Sage 2", "Text to video", 608, 352, 2, None, None)
    assert sage_policy[0] is False and sage_policy[2] == "forced Sage 2"
    sla_policy = app.resolve_sol_policy("SLA", "Text to video", 608, 352, 2, None, None)
    assert sla_policy[0] is False and sla_policy[2] == "forced SLA"
    assert (
        app.resolve_sol_policy("Auto", "Text to video", 608, 352, 2, None, None)[0] is False
    )
    assert (
        app.resolve_sol_policy(
            "Auto", "Text to video", 608, 352, 2, None, None, use_turbo=True
        )[0]
        is False
    )
    reference_turbo_sol = app.resolve_sol_policy(
        "Auto", "Reference media", 608, 352, 2, None, None, use_turbo=True
    )
    assert reference_turbo_sol[0] is True
    assert reference_turbo_sol[2] == "Auto Turbo: reference mode"
    turbo_sol_enabled, _, turbo_sol_reason = app.resolve_sol_policy(
        "Auto", "Text to video", 1344, 768, 5, None, None, use_turbo=True
    )
    assert turbo_sol_enabled is True
    assert turbo_sol_reason.startswith("Auto Turbo:")
    assert app.validate_resolution(865, 481) == (864, 480)
    assert app.validate_resolution(2048, 2048) == (2048, 2048)
    assert app.generation_resolution(
        1344,
        768,
        result_format="Audio",
        latent_upscale=False,
        mode="Text to video",
        first_image=None,
    ) == (32, 32)
    assert app.generation_resolution(
        865,
        481,
        result_format="Video",
        latent_upscale=True,
        mode="Text to video",
        first_image=None,
    ) == (896, 512)
    auto_landscape = app.resolution_for_aspect_ratio(4096, 2304)
    auto_portrait = app.resolution_for_aspect_ratio(2304, 4096)
    assert app.resolution_for_aspect_ratio(1024, 1024) == (992, 992)
    assert auto_landscape[0] % 32 == 0 and auto_landscape[1] % 32 == 0
    assert auto_portrait[0] % 32 == 0 and auto_portrait[1] % 32 == 0
    assert auto_landscape[0] * auto_landscape[1] < 4_000_000
    assert auto_portrait[0] * auto_portrait[1] < 4_000_000
    assert abs(auto_landscape[0] / auto_landscape[1] - 16 / 9) < 0.1
    assert abs(auto_portrait[0] / auto_portrait[1] - 9 / 16) < 0.1
    one_mp_landscape = app.resolution_for_aspect_ratio(
        4096, 2304, pixel_cap=app.auto_resolution_pixel_cap("1 MP")
    )
    two_mp_landscape = app.resolution_for_aspect_ratio(
        4096, 2304, pixel_cap=app.auto_resolution_pixel_cap("2 MP")
    )
    assert one_mp_landscape[0] * one_mp_landscape[1] < 1_000_000
    assert two_mp_landscape[0] * two_mp_landscape[1] < 2_000_000
    assert app.auto_resolution_pixel_cap("4 MP") == 4_000_000 - 1
    assert app.auto_resolution_pixel_cap("8 MP") == 8_000_000 - 1
    assert app.UI_DEFAULTS["model_profile"] == "Speed"
    assert app.UI_DEFAULTS["text_encoder"] == "NVFP4 / AWQ"
    assert app.UI_DEFAULTS["stage_model_offload"] is False
    fast_defaults = app.preset_values("Fast")
    assert app.DEFAULT_AUTO_RESOLUTION_MEGAPIXELS == fast_defaults[6]
    assert app.UI_DEFAULTS["turbo_variant"] == fast_defaults[7]
    assert app.UI_DEFAULTS["attention_mode"] == fast_defaults[8]
    assert app.UI_DEFAULTS["sla_preset"] == fast_defaults[9]
    assert app.UI_DEFAULTS["latent_upscale_refine_steps"] == fast_defaults[10]
    assert app.resolution_for_aspect_ratio(4096, 2304, preserve_native=True) == (4096, 2304)
    assert app.resolution_for_aspect_ratio(
        4010, 2250, preserve_native=True, alignment=64
    ) == (4032, 2240)
    assert app.resolution_control_updates(865, 481, True, "Video")[0:2] == (896, 512)
    assert "32×32" in app.resolution_control_updates(1344, 768, False, "Audio")[2]
    with app.tempfile.TemporaryDirectory() as resolution_temp:
        from PIL import Image

        native_start = app.Path(resolution_temp) / "native-start.png"
        Image.new("RGB", (2048, 1152)).save(native_start)
        assert app.generation_resolution(
            864,
            480,
            result_format="Image",
            latent_upscale=False,
            mode="First / last frame",
            first_image=str(native_start),
        ) == (2048, 1152)
        assert app.auto_resolution_from_start_frame(
            str(native_start), 864, 480, "Image", False
        )[:2] == (2048, 1152)
        one_mp_auto = app.auto_resolution_from_start_frame(
            str(native_start), 864, 480, "Video", False, "1 MP"
        )[:2]
        assert one_mp_auto[0] * one_mp_auto[1] < 1_000_000
    unchanged = app.auto_resolution_from_start_frame(None, 640, 480)
    assert unchanged[:2] == (640, 480)
    assert app.frame_length(5) == 124
    assert app.frame_length(15) == 362
    assert app.websocket_url("client id").startswith("ws://")
    assert "clientId=client%20id" in app.websocket_url("client id")
    assert app.node_stage("SamplerCustomAdvanced") == "Generating video and audio"
    assert app.node_stage("VAEDecode") == "Decoding output"
    assert app.node_stage(app.H3_NVENC_SAVE_NODE) == "Saving video with NVENC"
    rendered_progress = app.progress_status(
        "Generating video and audio",
        started=app.time.monotonic(),
        completed_nodes=7,
        total_nodes=12,
        step=2,
        step_total=4,
        configured_steps=4,
    )
    assert "Sampler step 2/4 (50%)" in rendered_progress
    assert "Workflow nodes 7/12" in rendered_progress
    expanded_progress = app.progress_status(
        "Generating video and audio",
        started=app.time.monotonic(),
        step=3,
        step_total=12,
        configured_steps=6,
    )
    assert "Overall generation progress 3/12 (25%)" in expanded_progress
    assert "Sampling schedule 6 steps (UI setting)" in expanded_progress
    assert "Sampler step 3/12" not in expanded_progress

    with app.unittest.mock.patch("builtins.print") as timing_print:
        stage_timings = app.StageTimings("test job", 100.0, "Preparing request")
        stage_timings.transition("Loading models", now=102.0)
        stage_timings.transition("Loading models", now=103.0)
        timing_summary = stage_timings.summary(now=105.5)
        stage_timings.finish(now=106.0)
    assert stage_timings.durations == {
        "Preparing request": 2.0,
        "Loading models": 3.5,
    }
    assert timing_summary == (
        "Step times: Preparing request 2.0s · Loading models 3.5s"
    )
    assert timing_print.call_count == 3

    class FakeProgressSocket:
        def __init__(self) -> None:
            self.messages = iter(
                [
                    app.json.dumps(
                        {
                            "type": "executing",
                            "data": {"prompt_id": "test-job", "node": "1"},
                        }
                    ),
                    app.json.dumps(
                        {
                            "type": "progress",
                            "data": {
                                "prompt_id": "test-job",
                                "node": "1",
                                "value": 3,
                                "max": 4,
                            },
                        }
                    ),
                    app.json.dumps(
                        {
                            "type": "executing",
                            "data": {"prompt_id": "test-job", "node": None},
                        }
                    ),
                ]
            )

        def settimeout(self, _timeout: float) -> None:
            pass

        def recv(self) -> str:
            return next(self.messages)

    live_updates = list(
        app.stream_comfy_progress(
            FakeProgressSocket(),  # type: ignore[arg-type]
            "test-job",
            {"1": {"class_type": "SamplerCustomAdvanced", "inputs": {}}},
            app.time.monotonic(),
        )
    )
    assert live_updates[0][0] == "Generating video and audio"
    assert live_updates[1][3:] == (3, 4)
    turbo_defaults = app.generation_mode_defaults("Turbo")
    assert app.DEFAULT_TURBO == app.LIGHTX2V_4STEP_TURBO
    assert turbo_defaults[1]["value"] == 4
    assert turbo_defaults[1]["interactive"] is True
    assert turbo_defaults[2:] == ("simple", "Spectrum", "SLA")
    larry_defaults = app.generation_mode_defaults("Turbo", app.LARRY_TURBO)
    assert larry_defaults[1]["value"] == 6
    assert larry_defaults[1]["interactive"] is True
    assert larry_defaults[2:] == ("simple", "Spectrum", "SLA")
    lightx_defaults = app.generation_mode_defaults("Turbo", app.LIGHTX2V_4STEP_TURBO)
    assert lightx_defaults[1]["value"] == 4
    assert lightx_defaults[1]["interactive"] is True
    assert lightx_defaults[2:] == ("simple", "Spectrum", "SLA")
    lightx_8step_defaults = app.generation_mode_defaults("Turbo", app.LIGHTX2V_8STEP_TURBO)
    assert lightx_8step_defaults[1]["value"] == 8
    assert lightx_8step_defaults[1]["interactive"] is True
    assert lightx_8step_defaults[2:] == ("simple", "Spectrum", "SLA")
    normal_defaults = app.generation_mode_defaults("Normal")
    assert normal_defaults[1]["value"] == 18
    assert normal_defaults[1]["interactive"] is True
    assert normal_defaults[2:] == ("simple", "Spectrum", "SLA")
    assert app.resolve_cache_policy("Off", use_turbo=True) == ("Off", None)
    turbo_spectrum, turbo_spectrum_note = app.resolve_cache_policy(
        "Spectrum", use_turbo=True
    )
    assert turbo_spectrum == "Spectrum" and turbo_spectrum_note
    turbo_easycache, turbo_easycache_note = app.resolve_cache_policy(
        "EasyCache", use_turbo=True
    )
    assert turbo_easycache == "EasyCache" and turbo_easycache_note
    turbo_firstblock, turbo_firstblock_note = app.resolve_cache_policy(
        "FirstBlockCache", use_turbo=True
    )
    assert turbo_firstblock == "FirstBlockCache" and turbo_firstblock_note
    assert app.SERVER_DENSE_ATTENTION_BACKEND == "comfy-kitchen"

    quality_turbo_graph = app.build_fl2va_graph(
        prompt="test",
        first_image=None,
        last_image=None,
        width=864,
        height=480,
        duration=5,
        steps=4,
        seed=2,
        scheduler="simple",
        turbo_lora_name="minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Spectrum",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        model_name=fake.profile("quality").fl2va,
        models=fake,
        available_nodes=available,
        use_int8_vae=True,
    )
    quality_unets = [
        node
        for node in quality_turbo_graph.values()
        if node["class_type"] == "UNETLoader"
    ]
    assert len(quality_unets) == 1
    assert quality_unets[0]["inputs"]["unet_name"] == fake.profile("quality").fl2va
    quality_video_vae = next(
        node
        for node in quality_turbo_graph.values()
        if node["class_type"] == "VAELoader"
        and node["inputs"]["vae_name"] == fake.video_vae_int8
    )
    assert quality_video_vae["inputs"]["vae_name"] == fake.video_vae_int8
    turbo_nodes = [
        node
        for node in quality_turbo_graph.values()
        if node["class_type"] == app.LIGHTX2V_BYPASS_LORA_NODE
    ]
    assert len(turbo_nodes) == 1
    assert turbo_nodes[0]["inputs"]["strength"] == 1.0
    assert turbo_nodes[0]["inputs"]["lora_name"].endswith(
        "v1.2_768p_comfyui_bf16.safetensors"
    )
    quality_fused_id = next(
        node_id
        for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == app.FUSED_MODULATION_NODE
    )
    quality_turbo_id = next(
        node_id
        for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == app.LIGHTX2V_BYPASS_LORA_NODE
    )
    assert quality_turbo_graph[quality_fused_id]["inputs"]["model"] == [
        quality_turbo_id,
        0,
    ]
    chunk_nodes = [
        node
        for node in quality_turbo_graph.values()
        if node["class_type"] == app.CHUNK_FEED_FORWARD_NODE
    ]
    assert len(chunk_nodes) == 1
    assert chunk_nodes[0]["inputs"]["chunks"] == 2
    assert chunk_nodes[0]["inputs"]["min_tokens"] == app.AUTO_SOL_TOKEN_THRESHOLD
    lightx_spectrum_id = next(
        node_id
        for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == "SpectrumApplyMiniMaxH3"
    )
    lightx_chunk_id = next(
        node_id
        for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == app.CHUNK_FEED_FORWARD_NODE
    )
    assert quality_turbo_graph[lightx_spectrum_id]["inputs"]["model"] == [
        lightx_chunk_id,
        0,
    ]
    assert (
        quality_turbo_graph[lightx_spectrum_id]["inputs"]["offline_archive_storage"]
        == "system_ram"
    )
    shift_id = next(
        node_id
        for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == app.H3_SIGMA_SHIFT_NODE
    )
    assert quality_turbo_graph[shift_id]["inputs"] == {
        "model": [lightx_spectrum_id, 0],
        "shift_video": 6.0,
        "shift_audio": 3.0,
    }
    quality_sampler = next(
        node
        for node in quality_turbo_graph.values()
        if node["class_type"] == app.CORE_SAMPLER_NODE
    )
    assert quality_sampler["inputs"]["sampler_name"] == "euler"
    assert not any(
        node["class_type"] == app.LARRY_TURBO_SAMPLER_NODE
        for node in quality_turbo_graph.values()
    )

    larry_graph = app.Graph()
    larry_model, _, larry_video_vae, larry_audio_vae = app.add_model_stack(
        larry_graph,
        fake.profile("speed").fl2va,
        fake,
        turbo_lora_name=fake.larry_turbo_lora,
        turbo_variant=app.LARRY_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Spectrum",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes=available,
    )
    app.finish_sampling(
        larry_graph,
        model_ref=larry_model,
        conditioning_ref=["conditioning", 0],
        latent_ref=["latent", 0],
        video_vae_ref=larry_video_vae,
        audio_vae_ref=larry_audio_vae,
        seed=3,
        steps=6,
        scheduler="simple",
        turbo_variant=app.LARRY_TURBO,
        filename_prefix="h3/larry_test",
    )
    larry_loader = next(
        node
        for node in larry_graph.nodes.values()
        if node["class_type"] == app.LARRY_TURBO_LORA_NODE
    )
    assert larry_loader["inputs"]["strength"] == 1.0
    assert larry_loader["inputs"]["low_vram"] is False
    larry_loader_id = next(
        node_id
        for node_id, node in larry_graph.nodes.items()
        if node["class_type"] == app.LARRY_TURBO_LORA_NODE
    )
    larry_spectrum_id = next(
        node_id
        for node_id, node in larry_graph.nodes.items()
        if node["class_type"] == "SpectrumApplyMiniMaxH3"
    )
    assert larry_graph.nodes[larry_spectrum_id]["inputs"]["model"] == [
        larry_loader_id,
        0,
    ]
    assert larry_model == [larry_spectrum_id, 0]
    assert (
        larry_graph.nodes[larry_spectrum_id]["inputs"]["offline_archive_storage"]
        == "system_ram"
    )
    assert app.FUSED_MODULATION_NODE not in {
        node["class_type"] for node in larry_graph.nodes.values()
    }
    assert app.FUSED_MODULATION_NODE not in app.turbo_required_nodes(app.LARRY_TURBO)
    assert app.FUSED_MODULATION_NODE in app.turbo_required_nodes(app.LIGHTX2V_4STEP_TURBO)
    assert app.FUSED_MODULATION_NODE in app.turbo_required_nodes(app.LIGHTX2V_8STEP_TURBO)
    assert app.H3_SIGMA_SHIFT_NODE in app.turbo_required_nodes(
        app.LIGHTX2V_4STEP_TURBO, fake.turbo_lora
    )
    assert app.H3_SIGMA_SHIFT_NODE not in app.turbo_required_nodes(
        app.LIGHTX2V_4STEP_TURBO, fake.turbo_ref_lora
    )
    assert app.H3_SIGMA_SHIFT_NODE in app.turbo_required_nodes(
        app.LIGHTX2V_8STEP_TURBO, fake.turbo_8step_lora
    )
    assert app.turbo_sampler_name(app.LIGHTX2V_4STEP_TURBO, fake.turbo_lora) == "euler"
    assert app.turbo_sampler_name(app.LIGHTX2V_4STEP_TURBO, fake.turbo_ref_lora) == "euler"
    assert app.turbo_sampler_name(app.LIGHTX2V_8STEP_TURBO, fake.turbo_8step_lora) == "euler"
    assert app.turbo_sampler_name(app.LIGHTX2V_8STEP_TURBO, None) == "res_multistep"
    assert app.turbo_uses_custom_nodes(app.LARRY_TURBO) is True
    assert app.turbo_uses_custom_nodes(app.LIGHTX2V_4STEP_TURBO) is False
    assert app.turbo_uses_custom_nodes(app.LIGHTX2V_8STEP_TURBO) is False
    assert any(
        node["class_type"] == app.LARRY_TURBO_SAMPLER_NODE
        for node in larry_graph.nodes.values()
    )
    assert not any(
        node["class_type"] == app.CORE_SAMPLER_NODE for node in larry_graph.nodes.values()
    )

    def turbo_route_graph(profile_name: str, variant: str) -> app.Graph:
        route_graph = app.Graph()
        profile = fake.profile(profile_name)
        app.add_model_stack(
            route_graph,
            profile.fl2va,
            fake,
            turbo_lora_name=fake.turbo_lora_for("Text to video", variant),
            turbo_variant=variant,
            turbo_strength=app.turbo_strength_for(variant),
            use_sol=False,
            sol_tau=1.0,
            sol_thresh_type="diag",
            sol_exact_mode="off",
            sol_dense_steps=1,
            sol_step_off=0.0,
            sol_sink_tokens=0,
            cache_mode="Off",
            fbcache_preset="Fast",
            fbcache_threshold=0.10,
            fbcache_start=0.10,
            fbcache_end=0.95,
            fbcache_max_hits=2,
            fbcache_temporal_guard=True,
            easycache_threshold=0.10,
            easycache_start=0.15,
            easycache_end=0.85,
            easycache_verbose=False,
            available_nodes=available,
        )
        return route_graph

    for profile_name in ("speed", "quality", "original"):
        larry_route = turbo_route_graph(profile_name, app.LARRY_TURBO)
        larry_route_loader = next(
            node
            for node in larry_route.nodes.values()
            if node["class_type"] == app.LARRY_TURBO_LORA_NODE
        )
        assert larry_route_loader["inputs"]["low_vram"] is False

        for lightx_variant in (
            app.LIGHTX2V_4STEP_TURBO,
            app.LIGHTX2V_8STEP_TURBO,
        ):
            lightx_route = turbo_route_graph(profile_name, lightx_variant)
            route_classes = {node["class_type"] for node in lightx_route.nodes.values()}
            assert app.LIGHTX2V_BYPASS_LORA_NODE in route_classes
            assert app.CORE_LORA_LOADER_NODE not in route_classes
            assert app.FUSED_MODULATION_NODE in route_classes
            if lightx_variant == app.LIGHTX2V_8STEP_TURBO:
                shift_node = next(
                    node
                    for node in lightx_route.nodes.values()
                    if node["class_type"] == app.H3_SIGMA_SHIFT_NODE
                )
                assert shift_node["inputs"]["shift_video"] == 6.0
                assert shift_node["inputs"]["shift_audio"] == 3.0

    assert app.LIGHTX2V_BYPASS_LORA_NODE in app.turbo_required_nodes(app.LIGHTX2V_4STEP_TURBO)
    assert app.CORE_LORA_LOADER_NODE not in app.turbo_required_nodes(app.LIGHTX2V_4STEP_TURBO)

    ref_turbo_graph = app.Graph()
    app.add_model_stack(
        ref_turbo_graph,
        fake.profile("quality").ref2va,
        fake,
        turbo_lora_name=fake.turbo_ref_lora,
        turbo_variant=app.LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="EasyCache",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes=available,
    )
    ref_unet = next(
        node
        for node in ref_turbo_graph.nodes.values()
        if node["class_type"] == "UNETLoader"
    )
    ref_lora = next(
        node
        for node in ref_turbo_graph.nodes.values()
        if node["class_type"] == app.LIGHTX2V_BYPASS_LORA_NODE
    )
    assert ref_unet["inputs"]["unet_name"] == fake.profile("quality").ref2va
    assert ref_lora["inputs"]["lora_name"] == fake.turbo_ref_lora
    assert ref_lora["inputs"]["strength"] == 1.0
    assert app.H3_SIGMA_SHIFT_NODE not in {
        node["class_type"] for node in ref_turbo_graph.nodes.values()
    }
    ref_easycache = next(
        node
        for node in ref_turbo_graph.nodes.values()
        if node["class_type"] == "EasyCache"
    )
    assert ref_easycache["inputs"]["reuse_threshold"] == 0.10

    sched_nodes = [
        node
        for node in quality_turbo_graph.values()
        if node["class_type"] == "BasicScheduler"
    ]
    assert len(sched_nodes) == 1
    assert sched_nodes[0]["inputs"]["steps"] == 4
    assert sched_nodes[0]["inputs"]["scheduler"] == "simple"
    print(
        f"Selftest OK: {len(graph)} nodes, 5s=124 frames, "
        f"15s=362 frames, tiered resolution presets valid, Sol exact valid, "
        f"Sol Auto/Turbo policy valid, Spectrum default + Sol/ConvRot order valid, "
        f"zero-copy Sol + FirstBlockCache composition valid, "
        f"LightX fused modulation + Larry compatibility + ConvRot FFN chunking valid, "
        f"Spectrum v0.2.23 legacy Turbo composition + block-cache guard valid, "
        f"MMH3 Split Upscale controls + three-node graph contract valid, "
        f"selectable Larry/LightX2V Turbo on "
        f"FL2VA/Ref2VA + synchronized editable Turbo steps valid, "
        f"video/image/audio result branches + image selection saving valid, "
        f"H3 NVENC save wiring valid, prompt API download URL valid, "
        f"gallery resolution/fallback/deletion guards + VRAM unload valid, "
        f"10 official LTX-2.5 workflow mappings valid, MiniMax Music 3 graph valid, "
        f"/comfyui proxy rewrites valid"
    )
