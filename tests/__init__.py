"""Tests for the MiniMax H3 application."""

import os

# Tests never call the free Gemma helpers; a test that covers them turns them back on.
os.environ.setdefault("H3_GEMMA", "0")
# Nor Gemini Decision checks or prompt-fit scoring; tests that cover them patch the request.
os.environ.setdefault("H3_DECISION", "0")
