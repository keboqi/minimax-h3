"""Bounded, shared preview work so browsing can render before decoding finishes."""

from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import copy_context
from threading import Lock
from time import monotonic
from pathlib import Path
from urllib.parse import urlsplit, unquote
from gradio.data_classes import FileData


def browser_preview_updates(updates):
    """Serialize URL previews explicitly so Gradio never downloads them itself."""
    serialized = []
    for update in updates[:3]:
        value = update.get("value")
        serialized.append({
            **update,
            "value": FileData(
                path=value, url=value,
                orig_name=Path(unquote(urlsplit(value).path)).name,
            ).model_dump() if value else None,
        })
    return (*serialized, *updates[3:])

_WORKERS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="h3-posters")
_LOCK = Lock()
_TASKS = {}


def preview_results(paths, create, *, kind, cache_root, timeout=None):
    futures = []
    submitted = []
    with _LOCK:
        now = monotonic()
        for key, (future, expires) in list(_TASKS.items()):
            if future.done() and (expires <= now or len(_TASKS) > 512):
                del _TASKS[key]
        for path in paths:
            try:
                info = path.stat()
            except OSError:
                futures.append(None)
                continue
            key = (kind, str(cache_root), str(path.resolve()), info.st_mtime_ns, info.st_size)
            task = _TASKS.get(key)
            if task is None:
                future = _WORKERS.submit(copy_context().run, create, path)
                # Keep failures briefly too, so polling does not retry broken files.
                _TASKS[key] = (future, float("inf"))

                futures.append(future)
                submitted.append((key, future))
                # Register outside the lock: an already completed future invokes
                # its callback immediately.
            else:
                futures.append(task[0])
    for key, future in submitted:
        def expire(done, key=key):
            with _LOCK:
                if key in _TASKS:
                    _TASKS[key] = (done, monotonic() + 30)
        future.add_done_callback(expire)
    wait([future for future in futures if future is not None], timeout=timeout)
    return [(future.result() if future is not None and future.done() else None,
             future is not None and not future.done()) for future in futures]
