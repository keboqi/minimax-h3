"""Run CPU service/workflow contracts; opt in to UI-only browser acceptance."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Also run settings, references, resolution and workspace browser checks",
    )
    args = parser.parse_args(argv)
    with TemporaryDirectory(prefix="h3-test-state-") as state:
        os.environ["H3_WORKSPACE_DIR"] = state
        try:
            return run(args)
        finally:
            from h3_app.jobs import JOBS

            JOBS.close()


def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    suite = unittest.defaultTestLoader.discover(
        str(root / "tests"), top_level_dir=str(root)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if args.browser:
        for script in (
            "browser_queue_transport.py",
            "browser_settings.py",
            "browser_voice_refs.py",
            "browser_auto_resolution.py",
            "browser_image_library.py",
            "browser_workspace.py",
            "browser_media_performance.py",
            "browser_media_deletion.py",
        ):
            subprocess.run(
                [sys.executable, "-m", "tests." + Path(script).stem],
                cwd=root,
                check=True,
                timeout=600,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
