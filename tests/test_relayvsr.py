"""Native-scale RelayVSR graph and Media routing contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image

from h3_app.catalog import GENERATION_POSTPROCESS_OPTIONS, POSTPROCESS_OPTIONS, RELAYVSR_OPTIONS
from h3_app.errors import H3Error
from h3_app.workflows.relayvsr import (
    build_relayvsr_graph, check_relayvsr_size, relayvsr_model_keys, required_relayvsr_nodes,
)
from h3_app.workflows.upscale import build_upscale_graph, required_upscale_nodes
from h3_models import LAZY_POSTPROCESS_MODEL_KEYS, MODEL_SPECS, PRELOAD_MODEL_KEYS
from h3_ui.media_controller import MediaController


def node(graph, kind):
    return next((key, value["inputs"]) for key, value in graph.items() if value["class_type"] == kind)


class RelayVSRTests(unittest.TestCase):
    def test_video_uses_native_scale_and_preserves_timing_audio(self):
        for option, scale in RELAYVSR_OPTIONS.items():
            with self.subTest(option=option):
                graph, steps = build_upscale_graph(
                    option=option, source_video="source.mp4", seed=42, models=Mock(),
                    width=853, height=479, target_width=1920, target_height=1080, fps=23.976,
                )
                self.assertEqual(steps, 1)
                self.assertEqual({n["class_type"] for n in graph.values()}, required_upscale_nodes(option))
                self.assertFalse(any("Scale" in n["class_type"] for n in graph.values()))
                components, _ = node(graph, "GetVideoComponents")
                _, video = node(graph, "CreateVideo")
                self.assertEqual(video["fps"], 23.976)
                self.assertEqual(video["audio"], [components, 1])
                _, conditioning = node(graph, "RelayVSRConditioning")
                _, decoded = node(graph, "VSRDecode")
                self.assertEqual(conditioning["images"], [components, 0])
                self.assertEqual(decoded["images"], conditioning["images"])
                self.assert_pair(graph, option, scale)

    def assert_pair(self, graph, option, scale):
        dit, vae = relayvsr_model_keys(option)
        self.assertEqual(node(graph, "UNETLoader")[1]["unet_name"], MODEL_SPECS[dit].local_name)
        self.assertEqual(node(graph, "VAELoader")[1]["vae_name"], f"relayvsr_flash_vae_x{scale}_fp32.safetensors")
        for key in (dit, vae):
            self.assertIn(key, LAZY_POSTPROCESS_MODEL_KEYS)
            self.assertNotIn(key, PRELOAD_MODEL_KEYS)
        self.assertIn(option, POSTPROCESS_OPTIONS)
        self.assertNotIn(option, GENERATION_POSTPROCESS_OPTIONS)

    def test_image_uses_one_still_without_resize_or_video_nodes(self):
        for option, scale in RELAYVSR_OPTIONS.items():
            with self.subTest(option=option):
                graph = build_relayvsr_graph(option=option, source="source.png", seed=3, image=True, output_token="test")
                self.assertEqual({n["class_type"] for n in graph.values()}, required_relayvsr_nodes(image=True))
                loaded, inputs = node(graph, "LoadImage")
                self.assertEqual(inputs["image"], "source.png")
                self.assertEqual(node(graph, "RelayVSRConditioning")[1]["images"], [loaded, 0])
                self.assertEqual(node(graph, "SaveImage")[1]["filename_prefix"], "h3/input_upscale/test_gallery")
                self.assert_pair(graph, option, scale)

    def test_image_controller_ignores_target_preset_and_records_exact_dimensions(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (53, 31)).save(source)
            for option, scale in RELAYVSR_OPTIONS.items():
                with self.subTest(option=option):
                    services = SimpleNamespace(
                        object_info=Mock(return_value=dict.fromkeys(required_relayvsr_nodes(image=True))),
                        ensure_model_keys=Mock(return_value=True),
                        input_image_upscale_dimensions=Mock(side_effect=AssertionError("must not resize")),
                        stage_file=Mock(return_value="source.png"),
                        submit_prompt=Mock(return_value="job"), poll_comfy_progress=Mock(return_value=[]),
                        wait_for_history=Mock(return_value={}),
                        resolve_seedvr2_input_upscale_outputs=Mock(return_value={"gallery": source}),
                        write_snapshot=Mock(), unload_comfy_models=Mock(),
                    )
                    controller = MediaController(services)
                    controller.managed_gallery_image_path = Mock(return_value=source)
                    controller.gallery_media_progress_result = lambda status: status
                    controller.gallery_media_processed_result = Mock(return_value="done")
                    result = list(controller.postprocess_selected_gallery_image(
                        str(source), option, 42, "unused", True, "unused preset", request=Mock(), progress=Mock(),
                    ))
                    self.assertEqual(result[-1], "done")
                    services.ensure_model_keys.assert_called_once_with(relayvsr_model_keys(option))
                    services.input_image_upscale_dimensions.assert_not_called()
                    services.unload_comfy_models.assert_called_once()
                    snapshot = services.write_snapshot.call_args.args[1]["settings"]
                    self.assertEqual(snapshot["source_resolution"], "53×31")
                    self.assertEqual(snapshot["target_resolution"], f"{53 * scale}×{31 * scale}")
                    self.assertEqual(snapshot["model"], option)

    def test_video_controller_ignores_target_preset_and_records_exact_dimensions(self):
        for option, scale in RELAYVSR_OPTIONS.items():
            with self.subTest(option=option):
                source = Path("source.mp4")
                services = SimpleNamespace(
                    load_model_config=Mock(side_effect=AssertionError("RelayVSR needs no H3 configuration")),
                    object_info=Mock(return_value=dict.fromkeys(required_relayvsr_nodes())),
                    ensure_model_keys=Mock(return_value=False),
                    probe_video_metadata=Mock(return_value=SimpleNamespace(width=853, height=479, fps=29.97)),
                    upscale_target_dimensions=Mock(side_effect=AssertionError("must not use target presets")),
                    stage_file=Mock(return_value="source.mp4"), build_upscale_graph=Mock(wraps=build_upscale_graph),
                    submit_prompt=Mock(return_value="job"), poll_comfy_progress=Mock(return_value=[]),
                    wait_for_history=Mock(return_value={}), resolve_output=Mock(return_value=Path("result.mp4")),
                    write_snapshot=Mock(),
                )
                controller = MediaController(services)
                controller.managed_video_path = Mock(return_value=source)
                controller.gallery_media_progress_result = lambda status: status
                controller.gallery_media_processed_result = Mock(return_value="done")
                result = list(controller._postprocess_selected_video(
                    str(source), option, 42, "unused", "unused", "stale LTX prompt", False, "unused preset",
                    request=Mock(), progress=Mock(),
                ))
                self.assertEqual(result[-1], "done", result)
                services.ensure_model_keys.assert_called_once_with(relayvsr_model_keys(option))
                services.upscale_target_dimensions.assert_not_called()
                services.load_model_config.assert_not_called()
                settings = services.write_snapshot.call_args.args[1]["settings"]
                self.assertEqual(settings["target_resolution"], f"{853 * scale}×{479 * scale}")
                self.assertEqual(settings["prompt"], "")
                self.assertEqual(services.build_upscale_graph.call_args.kwargs["fps"], 29.97)

    def test_oversized_output_is_refused_only_past_the_limit(self):
        check_relayvsr_size("RelayVSR 4x", 2048, 1080)
        check_relayvsr_size("RelayVSR 2x", 3840, 4096)
        for option, width, height in (("RelayVSR 4x", 3840, 2160), ("RelayVSR 4x", 1080, 2049),
                                      ("RelayVSR 2x", 4097, 2160)):
            with self.subTest(option=option, width=width, height=height), self.assertRaisesRegex(H3Error, "8192"):
                check_relayvsr_size(option, width, height)

    def test_oversized_video_fails_before_downloading_or_submitting(self):
        services = SimpleNamespace(
            object_info=Mock(return_value=dict.fromkeys(required_relayvsr_nodes())),
            ensure_model_keys=Mock(side_effect=AssertionError("must not download")),
            probe_video_metadata=Mock(return_value=SimpleNamespace(width=3840, height=2160, fps=30)),
            submit_prompt=Mock(side_effect=AssertionError("must not submit")),
        )
        controller = MediaController(services)
        controller.managed_video_path = Mock(return_value=Path("source.mp4"))
        controller.gallery_media_progress_result = lambda status: status
        result = list(controller._postprocess_selected_video(
            "source.mp4", "RelayVSR 4x", 42, "unused", "unused", "", False, "unused preset",
            request=Mock(), progress=Mock(),
        ))
        self.assertIn("15360×8640", result[-1])

    def test_oversized_image_fails_before_downloading_or_submitting(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "large.png"
            Image.new("RGB", (2049, 16)).save(source)
            services = SimpleNamespace(
                object_info=Mock(return_value=dict.fromkeys(required_relayvsr_nodes(image=True))),
                ensure_model_keys=Mock(side_effect=AssertionError("must not download")),
            )
            controller = MediaController(services)
            controller.managed_gallery_image_path = Mock(return_value=source)
            controller.gallery_media_progress_result = lambda status: status
            result = list(controller.postprocess_selected_gallery_image(
                str(source), "RelayVSR 4x", 42, "unused", False, "unused preset",
                request=Mock(), progress=Mock(),
            ))
            self.assertIn("8196×64", result[-1])

    def test_missing_native_nodes_fail_before_downloading_or_submitting(self):
        option = next(iter(RELAYVSR_OPTIONS))
        services = SimpleNamespace(
            object_info=Mock(return_value={}), ensure_model_keys=Mock(), submit_prompt=Mock(),
        )
        controller = MediaController(services)
        controller.managed_video_path = Mock(return_value=Path("source.mp4"))
        controller.gallery_media_progress_result = lambda status: status
        result = list(controller._postprocess_selected_video(
            "source.mp4", option, 42, "unused", "unused", "", False, "unused",
            request=Mock(), progress=Mock(),
        ))
        self.assertIn("Missing ComfyUI nodes", result[-1])
        self.assertIn("RelayVSRConditioning", result[-1])
        services.ensure_model_keys.assert_not_called()
        services.submit_prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
