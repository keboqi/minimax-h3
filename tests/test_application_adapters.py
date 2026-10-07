"""Focused CPU contracts retained from the former launcher self-test."""

import inspect as _inspect
import json as _json
import unittest
import unittest.mock

import h3_app.catalog as _catalog
import h3_app.contracts as _contracts
import h3_app.policy as _policy
import h3_app.prompt_service as _prompt_service
import h3_app.status as _status
import h3_app.workflows.h3 as _h3_workflow
import h3_models as _models
import h3_requirements as _requirements
from h3_ui import application as app
from tests.workflow_cases import MODELS as fake


class ApplicationAdaptersTests(unittest.TestCase):
    def setUp(self):
        catalog = unittest.mock.patch.object(
            app, "_catalog_inventory", return_value=None
        )
        catalog.start()
        self.addCleanup(catalog.stop)

    def test_h3_lightning_writer(self):
        with (
            unittest.mock.patch.dict(
                app.os.environ, {"LIGHTNING_API_KEY": "selftest-lightning-key"}
            ),
            unittest.mock.patch("openai.OpenAI") as openai_client,
        ):
            completion = unittest.mock.Mock()
            completion.choices = [
                unittest.mock.Mock(
                    message=unittest.mock.Mock(content="Rewritten Lightning prompt")
                )
            ]
            openai_client.return_value.chat.completions.create.return_value = completion
            rewritten, lightning_status = (
                _prompt_service._enhance_h3_prompt_with_lightning(
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
                    runtime=app._runtime_config(),
                )
            )
            self.assertEqual(rewritten, "Rewritten Lightning prompt")
            self.assertIn(app.LIGHTNING_PROMPT_MODEL, lightning_status)
            openai_client.assert_called_once_with(
                base_url=_catalog.LIGHTNING_API_ROOT,
                api_key="selftest-lightning-key",
                timeout=600.0,
            )
            request = (
                openai_client.return_value.chat.completions.create.call_args.kwargs
            )
            self.assertEqual(request["model"], app.LIGHTNING_PROMPT_MODEL)
            self.assertEqual(request["messages"][0]["role"], "system")
            self.assertIn(
                "A moonlit tracking shot", request["messages"][1]["content"][0]["text"]
            )

    def test_active_prompt_media_labels(self):
        with app.tempfile.TemporaryDirectory() as enhancer_temp:
            enhancer_root = app.Path(enhancer_temp)
            first_path = enhancer_root / "first.png"
            last_path = enhancer_root / "last.png"
            video_path = enhancer_root / "reference.mp4"
            for path in (first_path, last_path, video_path):
                path.write_bytes(b"test")
            last_only = _prompt_service._active_prompt_media(
                "First / last frame", None, str(last_path), (), (), ()
            )
            self.assertEqual(last_only, [("<Picture 1> (last frame)", last_path)])
            both_frames = _prompt_service._active_prompt_media(
                "First / last frame", str(first_path), str(last_path), (), (), ()
            )
            self.assertEqual(
                [label for label, _ in both_frames],
                ["<Picture 1> (first frame)", "<Picture 2> (last frame)"],
            )
            references = _prompt_service._active_prompt_media(
                "Reference media",
                None,
                None,
                (None, str(first_path)),
                (str(video_path),),
                (),
            )
            self.assertEqual(
                [label for label, _ in references], ["<Picture 2>", "<Video 1>"]
            )

    def test_mode_specific_model_provisioning(self):
        self.assertEqual(
            app.h3_text_encoder_settings(fake, "NVFP4 / AWQ"),
            ("text_encoder", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", False),
        )
        self.assertEqual(
            app.h3_text_encoder_settings(fake, "BF16"),
            ("text_encoder_bf16", "qwen3vl_32b_minimax_h3_bf16.safetensors", True),
        )
        self.assertIn(
            _catalog.H3_STAGE_OFFLOAD_NODE,
            _h3_workflow.required_nodes_for(
                "Text to video", False, "Off", stage_model_offload=True
            ),
        )
        self.assertLessEqual(
            {
                _catalog.H3_STAGE_OFFLOAD_NODE,
                _catalog.H3_CONDITIONING_CACHE_NODE,
                _catalog.H3_STAGE_OFFLOAD_POLICY_NODE,
            },
            _h3_workflow.required_nodes_for(
                "Text to video",
                False,
                "Off",
                stage_model_offload=True,
                smart_stage_offload=True,
            ),
        )
        self.assertEqual(
            fake.turbo_lora_for("Text to video", _catalog.LIGHTX2V_4STEP_TURBO),
            fake.turbo_lora,
        )
        self.assertEqual(
            fake.turbo_lora_for("Reference media", _catalog.LIGHTX2V_4STEP_TURBO),
            fake.turbo_ref_lora,
        )
        self.assertEqual(
            fake.turbo_lora_for("Text to video", _catalog.LIGHTX2V_8STEP_TURBO),
            fake.turbo_8step_lora,
        )
        self.assertEqual(
            fake.turbo_lora_for("Reference media", _catalog.LIGHTX2V_8STEP_TURBO),
            fake.turbo_8step_ref_lora,
        )
        with unittest.mock.patch(
            "h3_app.model_service.stale_model_keys", return_value=[]
        ) as stale_turbo_models:
            self.assertIs(
                app.ensure_turbo_lora(
                    fake, _catalog.LIGHTX2V_8STEP_TURBO, "Reference media"
                ),
                False,
            )
        self.assertEqual(
            stale_turbo_models.call_args.kwargs["model_keys"], ("turbo_8step_ref_lora",)
        )
        self.assertEqual(
            fake.turbo_lora_for("Text to video", _catalog.LARRY_TURBO),
            fake.larry_turbo_lora,
        )

    def test_save_selected_image_frames(self):
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
                self.assertEqual(resolved_frames, staged_paths)
                saved_frames, _ = app.save_selected_image_frames(
                    resolved_frames, ["Frame 1", "Frame 3"]
                )
                self.assertEqual(
                    [app.Path(path).name for path in saved_frames],
                    ["frame_001.png", "frame_003.png"],
                )
                self.assertTrue(
                    all((app.Path(path).is_file() for path in saved_frames))
                )
            finally:
                vars(app)["OUTPUT_DIR"] = original_image_output_dir

    def test_ui_image_and_batch_results(self):
        original_ui_generate = vars(app)["generate"]

        def fake_image_generate(**_args: app.Any):
            yield None, "working"
            yield ["frame-a.png", "frame-b.png"], "complete"

        vars(app)["generate"] = fake_image_generate
        try:
            image_args = [None] * len(_contracts.GENERATION_FIELDS)
            image_args[_contracts.GENERATION_FIELDS.index("result_format")] = "Image"
            image_args[_contracts.GENERATION_FIELDS.index("image_frames")] = 2
            image_args[_contracts.GENERATION_FIELDS.index("seed")] = -1
            ui_updates = list(app.generate_for_ui(1, *image_args))
        finally:
            vars(app)["generate"] = original_ui_generate
        self.assertEqual(len(ui_updates), 2)
        self.assertTrue(all((len(update) == 12 for update in ui_updates)))
        self.assertEqual(ui_updates[-1][7], ["frame-a.png", "frame-b.png"])
        self.assertEqual(app.video_batch_seeds(7, 1), [7])
        captured_batch_args: list[tuple[app.Any, ...]] = []
        original_generate_signature = _inspect.signature(original_ui_generate)

        def fake_batch_generate(**batch_args: app.Any):
            captured_batch_args.append(
                tuple(batch_args[key] for key in _contracts.GENERATION_FIELDS)
            )
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
            self.assertEqual(app.video_batch_seeds(7, 4), [101, 102, 103, 104])
            batch_updates = list(app.generate_for_ui(3, *batch_args))
        finally:
            vars(app)["generate"] = original_ui_generate
            app.random.sample = original_random_sample
        self.assertEqual(
            [args[seed_index] for args in captured_batch_args], [101, 102, 103]
        )
        self.assertTrue(
            all((args[reuse_index] is False for args in captured_batch_args))
        )
        self.assertEqual(len(batch_updates), 6)
        self.assertTrue(all((len(update) == 12 for update in batch_updates)))

    def test_official_ltx_workflow_inventory(self):
        self.assertEqual(_policy.ltx25_frame_length(5, 24), 121)
        self.assertEqual(len(app.LTX25_WORKFLOWS), 10)
        self.assertEqual(
            len({entry["id"] for entry in app.LTX25_WORKFLOWS.values()}), 10
        )
        self.assertEqual(
            {entry["filename"] for entry in app.LTX25_WORKFLOWS.values()},
            set(_requirements.LTX25_WORKFLOW_FILENAMES),
        )
        self.assertTrue(
            all(
                (
                    entry["filename"].startswith("LTX-2.5_")
                    and entry["filename"].endswith(".json")
                    for entry in app.LTX25_WORKFLOWS.values()
                )
            )
        )
        for label in app.LTX25_WORKFLOWS:
            self.assertLessEqual(
                set(app.ltx25_workflow_model_keys(label)), set(app.MODEL_SPECS)
            )
            self.assertIn("/ltx25-workflows/", app.render_ltx25_workflow_details(label))
        audio_label = next(
            label
            for label, entry in app.LTX25_WORKFLOWS.items()
            if entry.get("audio_only")
        )
        self.assertFalse(
            {"ltx25_video_vae", "ltx25_video_vae_full"}
            & set(app.ltx25_workflow_model_keys(audio_label))
        )
        video_label = next(
            label
            for label, entry in app.LTX25_WORKFLOWS.items()
            if not entry.get("audio_only")
        )
        self.assertLessEqual(
            {"ltx25_video_vae", "ltx25_video_vae_full"},
            set(app.ltx25_workflow_model_keys(video_label)),
        )
        inventory = app.render_ltx25_official_model_inventory()
        self.assertTrue(
            all(
                (
                    app.MODEL_SPECS[key].repo_id in inventory
                    for key in _models.LTX25_ICLORA_MODEL_KEYS
                )
            )
        )
        self.assertLessEqual(
            set(_models.LTX25_ICLORA_MODEL_KEYS),
            set(app.ltx25_official_inventory_keys()),
        )

    def test_input_image_upscale_dimensions(self):
        self.assertEqual(
            tuple(app.INPUT_IMAGE_UPSCALE_SLOTS[:3]),
            ("First frame", "Last frame", "Picture 1"),
        )
        self.assertEqual(app.INPUT_IMAGE_FRAME_PRESETS["1920 × 1920"], (1920, 1920))
        self.assertEqual(
            app.input_image_frame_preset_updates("3840 × 3840", 1920, 1920),
            (3840, 3840),
        )
        from PIL import Image

        with app.tempfile.TemporaryDirectory() as upscale_dimensions_temp:
            dimension_root = app.Path(upscale_dimensions_temp)
            portrait = dimension_root / "portrait.png"
            square = dimension_root / "square.png"
            landscape = dimension_root / "landscape.png"
            Image.new("RGB", (500, 700)).save(portrait)
            Image.new("RGB", (2048, 2048)).save(square)
            Image.new("RGB", (800, 400)).save(landscape)
            self.assertEqual(
                app.input_image_upscale_dimensions(str(portrait), 1920, 1920)[:4],
                (500, 700, 1371, 1920),
            )
            self.assertEqual(
                app.input_image_upscale_dimensions(str(square), 1920, 1920),
                (2048, 2048, 2048, 2048, 1.0),
            )
            self.assertEqual(
                app.input_image_upscale_dimensions(str(landscape), 1920, 1920)[:4],
                (800, 400, 1920, 960),
            )

    def test_staging_reuse_and_transcoding(self):
        original_input_dir = vars(app)["INPUT_DIR"]
        with app.tempfile.TemporaryDirectory() as staging_temp:
            staging_root = app.Path(staging_temp)
            staged_input_root = staging_root / "comfy-input"
            source = staging_root / "reference.png"
            source.write_bytes(b"same input bytes")
            vars(app)["INPUT_DIR"] = staged_input_root
            try:
                with unittest.mock.patch("builtins.print") as cache_print:
                    cached_first = app.stage_file(
                        str(source), "reference_images", reuse=True
                    )
                    cached_second = app.stage_file(
                        str(source), "reference_images", reuse=True
                    )
                self.assertEqual(cached_first, cached_second)
                self.assertEqual(cache_print.call_count, 2)
                self.assertIn("Stored", cache_print.call_args_list[0].args[0])
                self.assertIn("Reusing", cache_print.call_args_list[1].args[0])
                self.assertEqual(
                    (staged_input_root / cached_first).read_bytes(), b"same input bytes"
                )

                (staged_input_root / cached_first).write_bytes(b"")
                with unittest.mock.patch("builtins.print") as repair_print:
                    repaired = app.stage_file(
                        str(source), "reference_images", reuse=True
                    )
                self.assertEqual(repaired, cached_first)
                self.assertIn("Stored", repair_print.call_args.args[0])
                self.assertEqual(
                    (staged_input_root / repaired).read_bytes(), b"same input bytes"
                )

                uncached_first = app.stage_file(
                    str(source), "reference_images", reuse=False
                )
                uncached_second = app.stage_file(
                    str(source), "reference_images", reuse=False
                )
                self.assertNotEqual(uncached_first, uncached_second)

                source.write_bytes(b"changed input bytes")
                with unittest.mock.patch("builtins.print"):
                    changed = app.stage_file(
                        str(source), "reference_images", reuse=True
                    )
                self.assertNotEqual(changed, cached_first)

                video_source = staging_root / "reference.mov"
                video_source.write_bytes(b"video input bytes")

                def fake_ffmpeg(command: list[str], **_kwargs: app.Any):
                    app.Path(command[-1]).write_bytes(b"transcoded video")
                    return unittest.mock.Mock(returncode=0, stderr="")

                with (
                    unittest.mock.patch.object(
                        app.staging, "run_media_process", side_effect=fake_ffmpeg
                    ) as run,
                    unittest.mock.patch("builtins.print"),
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
                self.assertEqual(video_first, video_second)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(
                    (staged_input_root / video_first).read_bytes(), b"transcoded video"
                )

                failed_video = staging_root / "failed.mov"
                failed_video.write_bytes(b"failed video bytes")
                with unittest.mock.patch(
                    "h3_app.staging.run_media_process",
                    side_effect=OSError("ffmpeg unavailable"),
                ):
                    try:
                        app.stage_file(
                            str(failed_video),
                            "reference_videos",
                            transcode_video=True,
                            reuse=True,
                        )
                    except app.H3Error as exc:
                        self.assertIn("Could not stage input file", str(exc))
                    else:
                        raise AssertionError("Expected failed video staging to raise")
                video_cache_dir = staged_input_root / "h3_gradio" / "reference_videos"
                self.assertFalse(list(video_cache_dir.glob(".*.tmp.mp4")))
            finally:
                vars(app)["INPUT_DIR"] = original_input_dir

    def test_unload_and_prompt_api(self):
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
        self.assertEqual(
            captured_free_call,
            {
                "path": "/free",
                "kwargs": {"json": {"unload_models": True, "free_memory": True}},
            },
        )
        self.assertEqual(
            unload_message, "All models unloaded and cached VRAM released."
        )
        self.assertEqual(unload_status, "refreshed backend")
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
            self.assertEqual(
                list(app.generate_with_ui_defaults("API prompt", object())),
                [("https://example.test/downloads/video.mp4", "complete")],
            )
        finally:
            vars(app)["generate"] = original_generate
            vars(app)["absolute_video_download_url"] = original_download_url
        self.assertEqual(captured_api_call["args"], ())
        api_kwargs = captured_api_call["kwargs"]
        self.assertEqual(api_kwargs["prompt"], "API prompt")
        self.assertEqual(api_kwargs["mode"], "Text to video")
        self.assertEqual(api_kwargs["model_profile"], "Singularity")
        self.assertEqual(api_kwargs["turbo_variant"], _catalog.DEFAULT_TURBO)
        for key, expected in app.UI_DEFAULTS.items():
            self.assertEqual(api_kwargs[key], expected)
        self.assertTrue(
            all((api_kwargs[f"ref_image_{index}"] is None for index in range(1, 10)))
        )
        self.assertTrue(
            all((api_kwargs[f"ref_video_{index}"] is None for index in range(1, 4)))
        )
        self.assertTrue(
            all((api_kwargs[f"ref_audio_{index}"] is None for index in range(1, 4)))
        )
        self.assertEqual(
            app.video_download_path(app.OUTPUT_DIR / "h3" / "result video.mp4"),
            "/downloads/comfy/h3/result%20video.mp4",
        )

    def test_gallery_selection_and_deletion(self):
        original_output_dir = vars(app)["OUTPUT_DIR"]
        original_outputs_dir = vars(app)["OUTPUTS_DIR"]
        original_thumbnails_dir = vars(app)["GALLERY_THUMBNAILS_DIR"]
        original_gallery_thumbnail = vars(app)["gallery_thumbnail"]
        original_gallery_video_resolution = vars(app.gallery_store)[
            "gallery_video_resolution"
        ]
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
            vars(app.gallery_store)["gallery_video_resolution"] = (
                lambda _video, **_kwargs: (864, 480)
            )
            try:
                h3_video = comfy_test_output / "h3" / "minimax.mp4"
                ltx25_video = comfy_test_output / "ltx25" / "ltx.mp4"
                h3_video.parent.mkdir()
                ltx25_video.parent.mkdir()
                h3_video.write_bytes(b"h3")
                ltx25_video.write_bytes(b"ltx25")
                self.assertEqual(len(app.gallery_video_paths(limit=1)), 1)
                self.assertEqual(len(app.gallery_video_paths(limit=None)), 3)
                discovered_families = {
                    app.generated_video_family(video)
                    for video in app.gallery_video_paths()
                }
                self.assertLessEqual(
                    {"MiniMax H3", "LTX-2.5", "Post-processed"}, discovered_families
                )
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
                unconfirmed_delete = app.delete_selected_gallery_video(
                    selected_video, False
                )
                fallback_exists_after_unconfirmed = fallback_video.exists()
                confirmed_delete = app.delete_selected_gallery_video(
                    selected_video, True
                )
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
                empty_exists_after_delete = (
                    empty_video_1.exists() or empty_video_2.exists()
                )
                try:
                    app.managed_video_path(
                        gallery_root / "outside.mp4", require_file=False
                    )
                    raise AssertionError("Unmanaged gallery path was accepted")
                except app.H3Error:
                    pass
            finally:
                vars(app)["OUTPUT_DIR"] = original_output_dir
                vars(app)["OUTPUTS_DIR"] = original_outputs_dir
                vars(app)["GALLERY_THUMBNAILS_DIR"] = original_thumbnails_dir
                vars(app)["gallery_thumbnail"] = original_gallery_thumbnail
                vars(app.gallery_store)["gallery_video_resolution"] = (
                    original_gallery_video_resolution
                )
            self.assertEqual(len(gallery_items), 1)
            self.assertIn("fallback.mp4", gallery_items[0][1])
            self.assertNotIn("864×480", gallery_items[0][1])
            self.assertEqual(gallery_paths, [str(fallback_video)])
            self.assertEqual(gallery_play_url, str(fallback_video))
            self.assertEqual(selected_video, str(fallback_video))
            self.assertTrue(
                gallery_download_link.endswith(
                    "/downloads/comfy/fallback.mp4?download=1)"
                )
            )
            self.assertIn("**Resolution:** 864×480", gallery_download_link)
            self.assertIn("Showing 1 of 1 generated videos", gallery_detail)
            self.assertIn("1 thumbnail", gallery_detail)
            self.assertIs(fallback_exists_after_unconfirmed, True)
            self.assertIn("Confirm permanent deletion", unconfirmed_delete[2])
            self.assertIs(fallback_exists_after_delete, False)
            self.assertIn("Deleted `fallback.mp4`", confirmed_delete[2])
            self.assertIs(empty_exists_after_unconfirmed, True)
            self.assertIn("Confirm permanent deletion", unconfirmed_empty[2])
            self.assertIs(empty_exists_after_delete, False)
            self.assertIn("Deleted 2 generated videos", confirmed_empty[2])

    def test_attention_routing_policy(self):
        self.assertGreaterEqual(
            app.estimate_packed_tokens("Text to video", 1344, 768, 5),
            app.AUTO_SOL_TOKEN_THRESHOLD,
        )
        kitchen_policy = app.resolve_sol_policy(
            "Kitchen", "Text to video", 608, 352, 2, None, None
        )
        self.assertTrue(
            kitchen_policy[0] is False and kitchen_policy[2] == "forced Comfy Kitchen"
        )
        sage_policy = app.resolve_sol_policy(
            "Sage 2", "Text to video", 608, 352, 2, None, None
        )
        self.assertTrue(sage_policy[0] is False and sage_policy[2] == "forced Sage 2")
        sla_policy = app.resolve_sol_policy(
            "SLA", "Text to video", 608, 352, 2, None, None
        )
        self.assertTrue(sla_policy[0] is False and sla_policy[2] == "forced SLA")
        self.assertIs(
            app.resolve_sol_policy("Auto", "Text to video", 608, 352, 2, None, None)[0],
            False,
        )
        self.assertIs(
            app.resolve_sol_policy(
                "Auto", "Text to video", 608, 352, 2, None, None, use_turbo=True
            )[0],
            False,
        )
        reference_turbo_sol = app.resolve_sol_policy(
            "Auto", "Reference media", 608, 352, 2, None, None, use_turbo=True
        )
        self.assertIs(reference_turbo_sol[0], True)
        self.assertEqual(reference_turbo_sol[2], "Auto Turbo: reference mode")
        turbo_sol_enabled, _, turbo_sol_reason = app.resolve_sol_policy(
            "Auto", "Text to video", 1344, 768, 5, None, None, use_turbo=True
        )
        self.assertIs(turbo_sol_enabled, True)
        self.assertTrue(turbo_sol_reason.startswith("Auto Turbo:"))
        self.assertEqual(app.validate_resolution(865, 481), (864, 480))
        self.assertEqual(app.validate_resolution(2048, 2048), (2048, 2048))

    def test_resolution_controls(self):
        auto_landscape = app.resolution_for_aspect_ratio(4096, 2304)
        auto_portrait = app.resolution_for_aspect_ratio(2304, 4096)
        self.assertEqual(app.resolution_for_aspect_ratio(1024, 1024), (992, 992))
        self.assertTrue(auto_landscape[0] % 32 == 0 and auto_landscape[1] % 32 == 0)
        self.assertTrue(auto_portrait[0] % 32 == 0 and auto_portrait[1] % 32 == 0)
        self.assertLess(auto_landscape[0] * auto_landscape[1], 4000000)
        self.assertLess(auto_portrait[0] * auto_portrait[1], 4000000)
        self.assertLess(abs(auto_landscape[0] / auto_landscape[1] - 16 / 9), 0.1)
        self.assertLess(abs(auto_portrait[0] / auto_portrait[1] - 9 / 16), 0.1)
        one_mp_landscape = app.resolution_for_aspect_ratio(
            4096, 2304, pixel_cap=app.auto_resolution_pixel_cap("1 MP")
        )
        two_mp_landscape = app.resolution_for_aspect_ratio(
            4096, 2304, pixel_cap=app.auto_resolution_pixel_cap("2 MP")
        )
        self.assertLess(one_mp_landscape[0] * one_mp_landscape[1], 1000000)
        self.assertLess(two_mp_landscape[0] * two_mp_landscape[1], 2000000)
        self.assertEqual(app.auto_resolution_pixel_cap("4 MP"), 4000000 - 1)
        self.assertEqual(app.auto_resolution_pixel_cap("8 MP"), 8000000 - 1)
        self.assertEqual(app.UI_DEFAULTS["model_profile"], "Singularity")
        self.assertEqual(app.UI_DEFAULTS["text_encoder"], "NVFP4 / AWQ")
        self.assertIs(app.UI_DEFAULTS["stage_model_offload"], False)
        self.assertEqual(
            app.resolution_for_aspect_ratio(4096, 2304, preserve_native=True),
            (4096, 2304),
        )
        self.assertEqual(
            app.resolution_for_aspect_ratio(
                4010, 2250, preserve_native=True, alignment=64
            ),
            (4032, 2240),
        )
        self.assertEqual(
            app.resolution_control_updates(865, 481, True, "Video")[0:2], (896, 512)
        )
        self.assertIn(
            "32×32", app.resolution_control_updates(1344, 768, False, "Audio")[2]
        )
        unchanged = app.auto_resolution_from_start_frame(None, 640, 480)
        self.assertEqual(unchanged[:2], (640, 480))
        self.assertEqual(_policy.frame_length(5), 124)
        self.assertEqual(_policy.frame_length(15), 362)

    def test_streamed_progress_and_timings(self):
        self.assertTrue(app.websocket_url("client id").startswith("ws://"))
        self.assertIn("clientId=client%20id", app.websocket_url("client id"))
        self.assertEqual(
            _status.node_stage("SamplerCustomAdvanced"), "Generating video and audio"
        )
        self.assertEqual(_status.node_stage("VAEDecode"), "Decoding output")
        self.assertEqual(
            _status.node_stage(_catalog.H3_NVENC_SAVE_NODE), "Saving video with NVENC"
        )
        rendered_progress = app.progress_status(
            "Generating video and audio",
            started=app.time.monotonic(),
            completed_nodes=7,
            total_nodes=12,
            step=2,
            step_total=4,
            configured_steps=4,
        )
        self.assertIn("Sampler step 2/4 (50%)", rendered_progress)
        self.assertIn("Workflow nodes 7/12", rendered_progress)
        expanded_progress = app.progress_status(
            "Generating video and audio",
            started=app.time.monotonic(),
            step=3,
            step_total=12,
            configured_steps=6,
        )
        self.assertIn("Overall generation progress 3/12 (25%)", expanded_progress)
        self.assertIn("Sampling schedule 6 steps (UI setting)", expanded_progress)
        self.assertNotIn("Sampler step 3/12", expanded_progress)
        with unittest.mock.patch("builtins.print") as timing_print:
            stage_timings = _status.StageTimings("test job", 100.0, "Preparing request")
            stage_timings.transition("Loading models", now=102.0)
            stage_timings.transition("Loading models", now=103.0)
            timing_summary = stage_timings.summary(now=105.5)
            stage_timings.finish(now=106.0)
        self.assertEqual(
            stage_timings.durations, {"Preparing request": 2.0, "Loading models": 3.5}
        )
        self.assertEqual(
            timing_summary, "Step times: Preparing request 2.0s · Loading models 3.5s"
        )
        self.assertEqual(timing_print.call_count, 3)

        class FakeProgressSocket:
            def __init__(self) -> None:
                self.messages = iter(
                    [
                        _json.dumps(
                            {
                                "type": "executing",
                                "data": {"prompt_id": "test-job", "node": "1"},
                            }
                        ),
                        _json.dumps(
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
                        _json.dumps(
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
        self.assertEqual(live_updates[0][0], "Generating video and audio")
        self.assertEqual(live_updates[1][3:], (3, 4))

    def test_turbo_cache_policy(self):
        self.assertEqual(
            _policy.resolve_cache_policy("Off", use_turbo=True), ("Off", None)
        )
        turbo_spectrum, turbo_spectrum_note = _policy.resolve_cache_policy(
            "Spectrum", use_turbo=True
        )
        self.assertTrue(turbo_spectrum == "Spectrum" and turbo_spectrum_note)
        turbo_easycache, turbo_easycache_note = _policy.resolve_cache_policy(
            "EasyCache", use_turbo=True
        )
        self.assertTrue(turbo_easycache == "EasyCache" and turbo_easycache_note)
        turbo_firstblock, turbo_firstblock_note = _policy.resolve_cache_policy(
            "FirstBlockCache", use_turbo=True
        )
        self.assertTrue(turbo_firstblock == "FirstBlockCache" and turbo_firstblock_note)
        self.assertEqual(app.SERVER_DENSE_ATTENTION_BACKEND, "comfy-kitchen")
