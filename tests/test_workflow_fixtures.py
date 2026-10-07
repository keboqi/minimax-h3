"""Pure graph builders must match the reviewed baseline snapshots."""

import json
import re
import unittest
from pathlib import Path
from h3_app.workflows import h3, ltx, music, upscale
from tests.workflow_cases import CASES


BUILDERS = {
    "build_fl2va_graph": h3.build_fl2va_graph,
    "build_ltx25_graph": ltx.build_ltx25_graph,
    "build_music3_graph": music.build_music3_graph,
    "build_seedvr2_upscale_graph": upscale.build_seedvr2_upscale_graph,
    "build_seedvr2_image_upscale_graph": upscale.build_seedvr2_image_upscale_graph,
    "build_ltx25_upscale_graph": upscale.build_ltx25_upscale_graph,
}


def normalize(value):
    if isinstance(value, dict):
        return {
            key: re.sub(r"_[0-9][0-9_a-f]*$", "_<submission>", item)
            if key == "filename_prefix" and isinstance(item, str)
            else normalize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


class WorkflowFixtureTests(unittest.TestCase):
    def test_preflight_requirements_match_generated_nodes(self):
        upscale_requirements = {
            "build_seedvr2_upscale_graph": upscale.required_seedvr2_upscale_nodes,
            "build_seedvr2_image_upscale_graph": upscale.required_seedvr2_image_upscale_nodes,
            "build_ltx25_upscale_graph": upscale.required_ltx25_upscale_nodes,
        }
        for index, (name, arguments) in enumerate(CASES):
            if name == "build_fl2va_graph":
                continue
            variants = [arguments]
            if name == "build_music3_graph":
                variants = [
                    {**arguments, "tiled_decode": tiled} for tiled in (False, True)
                ]
            for inputs in variants:
                with self.subTest(
                    index=index, builder=name, tiled=inputs.get("tiled_decode")
                ):
                    if name == "build_music3_graph":
                        required = music.required_music3_nodes(inputs["tiled_decode"])
                    elif name == "build_ltx25_graph":
                        required = ltx.required_ltx25_nodes(
                            image_to_video=any(
                                inputs.get(key)
                                for key in ("first_image", "middle_image", "end_image")
                            ),
                            reference_images=bool(inputs.get("reference_sheet")),
                        )
                    else:
                        required = upscale_requirements[name]()
                    graph = BUILDERS[name](**inputs)
                    self.assertEqual(
                        required, {node["class_type"] for node in graph.values()}
                    )

    def test_workflow_graphs(self):
        expected = json.loads(
            (Path(__file__).parent / "fixtures/workflow_graphs.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(CASES), len(expected))
        for index, ((name, arguments), baseline) in enumerate(zip(CASES, expected)):
            with self.subTest(index=index, builder=name):
                graph = BUILDERS[name](**arguments)
                self.assertEqual(
                    {"function": name, "graph": normalize(graph)}, baseline
                )


if __name__ == "__main__":
    unittest.main()
