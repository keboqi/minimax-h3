"""Storyboard flow scenes: where the continuation nodes sit in both H3 graph shapes."""

import unittest

from h3_app.catalog import (
    H3_COMBINE_AV_LATENT_NODE,
    H3_CONTINUATION_PREFIX_NODE,
    H3_LOAD_AV_LATENT_NODE,
    H3_SAVE_AV_LATENT_NODE,
)
from h3_app.errors import H3Error
from h3_app.workflows import h3
from tests.workflow_cases import H3_DEFAULTS


def nodes(graph, kind):
    return [(node_id, node["inputs"]) for node_id, node in graph.items() if node["class_type"] == kind]


STORYBOARD = {
    "output_prefix": "h3/.storyboard/sb/scene_take",
    "latent_output_prefix": "h3/.storyboard/sb/scene_take",
    "continuation_frames": 39,
}


class StoryboardGraphTests(unittest.TestCase):
    def test_first_scene_only_saves_its_latent_and_uses_its_own_folder(self):
        graph = h3.build_fl2va_graph(**{**H3_DEFAULTS, **STORYBOARD})
        self.assertEqual(nodes(graph, H3_CONTINUATION_PREFIX_NODE), [])
        (save_id, save), = nodes(graph, H3_SAVE_AV_LATENT_NODE)
        (sampler_id, _), = nodes(graph, "SamplerCustomAdvanced")
        self.assertEqual(save, {"latent": [sampler_id, 0], "filename_prefix": STORYBOARD["latent_output_prefix"]})
        video = next(inputs for _, inputs in nodes(graph, "H3SaveVideoNVENC"))
        self.assertEqual(video["filename_prefix"], STORYBOARD["output_prefix"])

    def test_single_stage_flow_prefixes_the_sampled_latent(self):
        graph = h3.build_fl2va_graph(
            **{**H3_DEFAULTS, **STORYBOARD, "continuation_source": "h3/.storyboard/sb/prev.safetensors"}
        )
        (prefix_id, prefix), = nodes(graph, H3_CONTINUATION_PREFIX_NODE)
        (load_id, load), = nodes(graph, H3_LOAD_AV_LATENT_NODE)
        self.assertEqual(load["path"], "h3/.storyboard/sb/prev.safetensors")
        self.assertEqual(prefix["source"], [load_id, 0])
        self.assertEqual(prefix["context_frames"], 39)
        self.assertEqual(graph[prefix["latent"][0]]["class_type"], "MiniMaxH3ImageToVideo")
        (_, sampler), = nodes(graph, "SamplerCustomAdvanced")
        self.assertEqual(sampler["latent_image"], [prefix_id, 0])

    def test_latent_upscale_flow_prefixes_both_stages_and_saves_both_latents(self):
        graph = h3.build_ref2va_graph(
            **{key: value for key, value in H3_DEFAULTS.items()
               if key not in {"first_image", "last_image", "latent_upscale_model_name",
                              "semantic_bridge", "semantic_bridge_alpha", "voice_reference_audios"}},
            **STORYBOARD,
            reference_images=[], reference_videos=[], reference_audios=[], ref_image_size="Original",
            latent_upscale_model_name="upscale.pth",
            continuation_source="h3/.storyboard/sb/prev.safetensors",
            continuation_stage_source="h3/.storyboard/sb/prev_stage1.safetensors",
        )
        prefixes = dict(nodes(graph, H3_CONTINUATION_PREFIX_NODE))
        loads = dict(nodes(graph, H3_LOAD_AV_LATENT_NODE))
        self.assertEqual(sorted(load["path"] for load in loads.values()), [
            "h3/.storyboard/sb/prev.safetensors", "h3/.storyboard/sb/prev_stage1.safetensors"])
        samplers = dict(nodes(graph, "SamplerCustomAdvanced"))
        stage_one = next(i for i in samplers.values() if i["latent_image"][0] in prefixes)
        stage_one_prefix = prefixes[stage_one["latent_image"][0]]
        self.assertEqual(loads[stage_one_prefix["source"][0]]["path"], "h3/.storyboard/sb/prev_stage1.safetensors")
        refine = next(i for i in samplers.values() if i is not stage_one)
        refine_prefix = prefixes[refine["latent_image"][0]]
        self.assertEqual(graph[refine_prefix["latent"][0]]["class_type"], H3_COMBINE_AV_LATENT_NODE)
        self.assertEqual(loads[refine_prefix["source"][0]]["path"], "h3/.storyboard/sb/prev.safetensors")
        saved = sorted(inputs["filename_prefix"] for _, inputs in nodes(graph, H3_SAVE_AV_LATENT_NODE))
        self.assertEqual(saved, [STORYBOARD["latent_output_prefix"], STORYBOARD["latent_output_prefix"] + "_stage1"])

    def test_latent_upscale_flow_needs_a_stage_one_source(self):
        with self.assertRaisesRegex(H3Error, "stage-one latent"):
            h3.build_fl2va_graph(**{
                **H3_DEFAULTS, **STORYBOARD, "latent_upscale_model_name": "upscale.pth",
                "continuation_source": "h3/.storyboard/sb/prev.safetensors",
            })

    def test_continuation_is_video_only(self):
        with self.assertRaisesRegex(H3Error, "video results only"):
            h3.build_fl2va_graph(**{
                **H3_DEFAULTS, **STORYBOARD, "result_format": "Audio",
                "continuation_source": "h3/.storyboard/sb/prev.safetensors",
            })


if __name__ == "__main__":
    unittest.main()
