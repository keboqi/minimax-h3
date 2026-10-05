"""CQ image recipe, lazy provisioning, and gallery execution contracts."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app import model_service
from h3_app.workflows.upscale import (
    build_cq_image_enhance_graph, required_cq_image_enhance_nodes,
)
from h3_models import CQ_IMAGE_MODEL_KEYS, MODEL_SPECS, PRELOAD_MODEL_KEYS
import h3_ui.application as app


class CqImageTests(unittest.TestCase):
    def test_image_recipe_has_one_frame_and_dedicated_models(self):
        graph = build_cq_image_enhance_graph(
            source_image="source.png", seed=42, target_width=1920,
            target_height=1080, output_token="unique",
        )
        by_type = {}
        for node in graph.values():
            by_type.setdefault(node["class_type"], []).append(node["inputs"])
        self.assertEqual(set(by_type), required_cq_image_enhance_nodes())
        self.assertEqual(by_type["UNETLoader"][0]["unet_name"], MODEL_SPECS["ltx25_dev_int8"].local_name)
        adapters = by_type["LoraLoaderModelOnly"]
        self.assertEqual([(n["lora_name"], n["strength_model"]) for n in adapters], [
            (MODEL_SPECS["ltx25_distillation_lora"].local_name, 0.5),
            (MODEL_SPECS["ltx25_cq_image_enhancer"].local_name, 1.0),
        ])
        self.assertEqual(by_type["VAELoader"][0]["vae_name"], MODEL_SPECS["ltx25_video_vae"].local_name)
        self.assertEqual([n["text"] for n in by_type["CLIPTextEncode"]], ["", ""])
        latent = by_type["EmptyLTXVLatentVideo"][0]
        self.assertEqual(latent["length"], 1)
        self.assertEqual((latent["width"], latent["height"]), (1920, 1088))
        self.assertEqual(by_type["LTXAddVideoICLoRAGuide"][0]["latent_downscale_factor"], 1)
        self.assertEqual(by_type["RandomNoise"][0]["noise_seed"], 42)
        self.assertEqual((by_type["ImageScale"][-1]["width"], by_type["ImageScale"][-1]["height"]), (1920, 1080))
        self.assertEqual(by_type["SaveImage"][0]["filename_prefix"], "h3/input_upscale/unique_gallery")
        self.assertFalse(any("Audio" in name or "VideoComponents" in name for name in by_type))

    def test_provisioning_is_lazy_and_checks_all_required_files(self):
        runtime = SimpleNamespace(comfy_dir=Path("comfy"), models_config=Path("config/models.json"))
        self.assertTrue(set(CQ_IMAGE_MODEL_KEYS).isdisjoint(PRELOAD_MODEL_KEYS))
        with (
            patch.object(model_service, "stale_model_keys", return_value=[]) as stale,
            patch.object(model_service, "sync_models") as sync,
        ):
            self.assertFalse(model_service.ensure_cq_image_enhance_models(runtime=runtime))
            self.assertEqual(stale.call_args.kwargs["model_keys"], CQ_IMAGE_MODEL_KEYS)
            sync.assert_not_called()
        with (
            patch.object(model_service, "stale_model_keys", return_value=["ltx25_cq_image_enhancer"]),
            patch.object(model_service, "resolve_hf_token", return_value=None),
            patch.object(model_service, "sync_models") as sync,
            patch.object(model_service, "model_file_is_ready", return_value=True) as ready,
        ):
            self.assertTrue(model_service.ensure_cq_image_enhance_models(runtime=runtime))
            self.assertEqual(sync.call_args.kwargs["model_keys"], CQ_IMAGE_MODEL_KEYS)
            self.assertEqual(ready.call_count, len(CQ_IMAGE_MODEL_KEYS))
            ready.return_value = False
            with self.assertRaisesRegex(app.H3Error, "did not produce"):
                model_service.ensure_cq_image_enhance_models(runtime=runtime)

    def test_gallery_routes_cq_images_and_records_fitted_resolution(self):
        for dimensions, expected in [
            ((1024, 768, 1920, 1440, 1.875), (1920, 1440)),
            ((1920, 1920, 1920, 1920, 1.0), (1920, 1920)),
            ((3840, 2160, 3840, 2160, 1.0), (1920, 1080)),
        ]:
            with self.subTest(dimensions=dimensions), ExitStack() as stack:
                returns = {
                    "managed_gallery_image_path": Path("source.png"),
                    "input_image_upscale_dimensions": dimensions,
                    "object_info": required_cq_image_enhance_nodes(),
                    "ensure_cq_image_enhance_models": False,
                    "ensure_seedvr2_upscale_models": False,
                    "load_model_config": Mock(),
                    "stage_file": "staged.png", "submit_prompt": "cq-job",
                    "poll_comfy_progress": [], "wait_for_history": {},
                    "resolve_seedvr2_input_upscale_outputs": {"gallery": Path("enhanced.png")},
                    "write_snapshot": None, "unload_comfy_models": None,
                    "gallery_media_processed_result": "complete",
                }
                mocks = {name: stack.enter_context(patch.object(app, name, return_value=value)) for name, value in returns.items()}
                build = stack.enter_context(patch.object(app, "build_cq_image_enhance_graph", wraps=build_cq_image_enhance_graph))
                updates = list(app.postprocess_selected_gallery_media(
                    "Image", "source.png", app.LTX25_CQ_IMAGE_ENHANCER,
                    42, app.DEFAULT_SEEDVR2_MODEL, app.DEFAULT_LTX25_MODEL,
                    "ignored", True, True, 5.0, "1920 × 1920",
                    request=Mock(), progress=Mock(),
                ))
                self.assertEqual(updates[-1], "complete")
                self.assertEqual((build.call_args.kwargs["target_width"], build.call_args.kwargs["target_height"]), expected)
                output_token = build.call_args.kwargs["output_token"]
                self.assertTrue(output_token.startswith("cq_image_"))
                self.assertEqual(
                    mocks["resolve_seedvr2_input_upscale_outputs"].call_args.args[2],
                    output_token,
                )
                mocks["ensure_cq_image_enhance_models"].assert_called_once_with()
                mocks["ensure_seedvr2_upscale_models"].assert_not_called()
                mocks["load_model_config"].assert_not_called()
                mocks["unload_comfy_models"].assert_called_once_with()
                snapshot = mocks["write_snapshot"].call_args.args[1]
                self.assertEqual(snapshot["settings"]["target_resolution"], f"{expected[0]}×{expected[1]}")
                self.assertEqual(snapshot["settings"]["method"], app.LTX25_CQ_IMAGE_ENHANCER)

    def test_missing_nodes_prevent_download_and_submission(self):
        with (
            patch.object(app, "managed_gallery_image_path", return_value=Path("source.png")),
            patch.object(app, "input_image_upscale_dimensions", return_value=(1024, 768, 1920, 1440, 1.875)),
            patch.object(app, "object_info", return_value={}),
            patch.object(app, "ensure_cq_image_enhance_models") as provision,
            patch.object(app, "submit_prompt") as submit,
        ):
            updates = list(app.postprocess_selected_gallery_image(
                "source.png", app.LTX25_CQ_IMAGE_ENHANCER, 42,
                app.DEFAULT_SEEDVR2_MODEL, False, "1920 × 1920",
                request=Mock(), progress=Mock(),
            ))
            self.assertIn("requires current ComfyUI nodes", str(updates[-1]))
            provision.assert_not_called()
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
