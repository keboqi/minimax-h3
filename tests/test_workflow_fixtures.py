"""Graph fixtures and the legacy compatibility entry point are discoverable."""

import asyncio
import json
import re
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import gradio_app as app
from tests.legacy_selftest import selftest


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
    def test_workflow_graphs_and_legacy_contracts(self):
        expected = json.loads(
            (Path(__file__).parent / "fixtures/workflow_graphs.json").read_text(
                encoding="utf-8"
            )
        )
        names = {row["function"] for row in expected}
        actual = []

        def capture(name, fn):
            def run(*args, **kwargs):
                result = fn(*args, **kwargs)
                actual.append({"function": name, "graph": normalize(result)})
                return result

            return run

        loops = []
        policy = asyncio.get_event_loop_policy()
        create_loop = policy.new_event_loop

        def tracked_loop():
            loop = create_loop()
            loops.append(loop)
            return loop

        self.addCleanup(
            lambda: [
                loop.close()
                for loop in loops
                if not loop.is_closed() and not loop.is_running()
            ]
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(policy, "new_event_loop", side_effect=tracked_loop)
            )
            for name in names:
                stack.enter_context(
                    patch.object(
                        app, name, side_effect=capture(name, getattr(app, name))
                    )
                )
            selftest()
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
