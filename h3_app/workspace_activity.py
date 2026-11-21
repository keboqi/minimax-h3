"""Track detached work that can access the workspace outside a request."""

from contextlib import contextmanager
from functools import wraps
from threading import Lock

_lock = Lock()
_active = 0


def workspace_task(work):
    @wraps(work)
    def run(*args, **kwargs):
        global _active
        with _lock:
            _active += 1
        try:
            return work(*args, **kwargs)
        finally:
            with _lock:
                _active -= 1

    return run


@contextmanager
def idle_workspace():
    """Keep new detached work from starting while an idle workspace reloads."""
    with _lock:
        yield _active == 0
