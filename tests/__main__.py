"""Run CPU contracts and legacy suites; opt in to UI-only browser acceptance."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Also run settings and voice-reference browser checks",
    )
    args = parser.parse_args()
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
        for script in ("browser_settings.py", "browser_voice_refs.py"):
            subprocess.run(
                [sys.executable, str(root / "tests" / script)],
                cwd=root,
                check=True,
                timeout=300,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
