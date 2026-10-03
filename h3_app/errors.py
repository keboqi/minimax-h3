"""Extracted errors boundary."""

from __future__ import annotations


class H3Error(RuntimeError):
    def __init__(self, message, *, code="generation_error", field=None, recovery=None):
        super().__init__(message)
        self.code = code
        self.field = field
        self.recovery = recovery
