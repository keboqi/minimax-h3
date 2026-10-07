"""Progress and gallery families follow the media actually produced."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h3_app.catalog import H3_NVENC_SAVE_NODE
from h3_app.config import RuntimeConfig
from h3_app.gallery_store import generated_image_family
from h3_app.outputs import OutputContext, resolve_seedvr2_input_upscale_outputs
from h3_app.provenance import write_snapshot
from h3_app.status import graph_class_types, node_stage
from h3_app.workflows.qwen import build_qwen_image21_graph
from h3_app.workflows.upscale import build_cq_image_enhance_graph


class ProgressLabelTests(unittest.TestCase):
    def test_qwen_generation_edit_and_turbo_report_image_stages(self):
        for references in ([], ["reference.png"]):
            for turbo in ("Off", "Viggle Turbo v0.3 (9-step)"):
                with self.subTest(references=references, turbo=turbo):
                    graph = build_qwen_image21_graph(
                        model_choice="BF16", text_encoder_choice="BF16",
                        prompt="blue sky", negative_prompt="",
                        reference_images=references, width=1024, height=1024,
                        reference_resolution=0, match_input_size=False,
                        seed=42, steps=40 if turbo == "Off" else 9, cfg=1.0,
                        sampler_name="euler", scheduler="simple",
                        cache_device="auto", cache_dtype="default",
                        attention_backend="pytorch attention", accelerator="Off",
                        output_stamp="123", output_nonce="fixture",
                        turbo_variant=turbo,
                    )
                    classes = graph_class_types(graph)
                    stages = {node_stage(name, classes) for name in classes}
                    self.assertIn("Generating image", stages)
                    self.assertIn("Saving image", stages)
                    self.assertFalse(any("video" in stage or "audio" in stage for stage in stages))
                    # Selector nodes prepare sampling; they do not sample.
                    if "KSamplerSelect" in classes:
                        self.assertEqual(node_stage("KSamplerSelect", classes), "Preparing sampler")

    def test_cq_recipe_reports_image_latents_and_source(self):
        graph = build_cq_image_enhance_graph(
            source_image="source.png", seed=1, target_width=1024,
            target_height=1024, output_token="cq_image_fixture",
        )
        classes = graph_class_types(graph)
        stages = {node_stage(name, classes) for name in classes}
        self.assertIn("Preparing image latents", stages)
        self.assertIn("Encoding source image for CQ enhancement", stages)
        self.assertIn("Generating image", stages)
        self.assertFalse(any("video" in stage or "audio" in stage for stage in stages))

    def test_seedvr2_stills_encode_images_and_video_exports_keep_video_stages(self):
        classes = {"SeedVR2Preprocess", "VAEEncodeTiled", "KSampler", "SaveImage"}
        self.assertEqual(node_stage("VAEEncodeTiled", classes), "Encoding image for SeedVR2")
        self.assertEqual(node_stage("KSampler", classes), "Generating image")
        for video_node in ("CreateVideo", "SaveVideo", H3_NVENC_SAVE_NODE):
            video_classes = classes | {video_node}
            self.assertEqual(node_stage("KSampler", video_classes), "Generating video and audio")
            self.assertEqual(node_stage("VAEEncodeTiled", video_classes), "Encoding H3 video for SeedVR2")
            self.assertEqual(
                node_stage("SeedVR2TemporalChunk", video_classes),
                "Splitting SeedVR2 video into VRAM-safe chunks",
            )
            self.assertEqual(node_stage("SaveImage", video_classes), "Saving image")

    def test_audio_samplers_and_save_nodes_use_correct_media_labels(self):
        for encoder, expected in (
            ("MiniMaxMusic3TextEncode", "Generating music"),
            ("YuE2GenerateMusic", "Generating YuE2 audio"),
        ):
            for sampler in ("KSampler", "SamplerCustomAdvanced"):
                self.assertEqual(node_stage(sampler, {encoder}), expected)
        self.assertEqual(node_stage("SaveAudioMP3"), "Saving audio")
        self.assertEqual(node_stage("SaveVideo"), "Saving video")
        self.assertEqual(node_stage("CLIPTextEncode"), "Encoding prompt")

    def test_cq_family_recognizes_new_and_legacy_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = replace(RuntimeConfig.from_environment(root, {}), output_root=root)
            output_dir = root / "h3/input_upscale"
            output_dir.mkdir(parents=True)
            legacy = output_dir / "legacy_gallery_00001.png"
            legacy.touch()
            write_snapshot(legacy, {"family": "CQ image enhancement"})
            for image, expected in (
                (legacy, "CQ image enhancement"),
                (output_dir / "cq_image_new_gallery_00001.png", "CQ image enhancement"),
                (output_dir / "seedvr_gallery_00001.png", "SeedVR2"),
            ):
                self.assertEqual(generated_image_family(image, runtime=runtime), expected)

    def test_cq_prefix_resolves_from_history_and_disk_fallback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = replace(RuntimeConfig.from_environment(root, {}), output_root=root)
            output_dir = root / "h3/input_upscale"
            output_dir.mkdir(parents=True)
            image = output_dir / "cq_image_fixture_gallery_submission_00001.png"
            image.touch()
            history = {"outputs": {"save": {"images": [{
                "filename": image.name, "subfolder": "h3/input_upscale",
                "type": "output",
            }]}}}
            for saved in (history, {}):
                self.assertEqual(resolve_seedvr2_input_upscale_outputs(
                    saved, 0, "cq_image_fixture", ["gallery"],
                    context=OutputContext(runtime, "submission"),
                ), {"gallery": image})


if __name__ == "__main__":
    unittest.main()
