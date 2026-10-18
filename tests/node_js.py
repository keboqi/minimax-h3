"""Evaluate browser-side Gradio event functions (``js=`` with ``fn=None``) in Node."""

import json
import shutil
import subprocess
import unittest


def run_js(source, *args):
    """Call ``source`` with JSON ``args``; skips the test where Node.js is not installed."""
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("Node.js evaluates browser-side event functions")
    script = f"process.stdout.write(JSON.stringify(({source})(...{json.dumps(args)})))"
    return json.loads(subprocess.run([node, "-e", script], check=True, capture_output=True,
                                     text=True, encoding="utf-8", timeout=30).stdout)
