"""Tests for the MiniMax H3 application."""

import os

# Tests never call the free Gemma helpers; a test that covers them turns them back on.
os.environ.setdefault("H3_GEMMA", "0")
