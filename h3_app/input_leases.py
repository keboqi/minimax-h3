"""Crash-safe private upload leases, protected by cross-process file locks."""

from pathlib import Path
import shutil
import tempfile
import uuid

from filelock import FileLock, Timeout

ROOT = Path(tempfile.gettempdir()) / "h3-input-leases"
MARKER = ".h3-input-lease"


def cleanup_orphans():
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in ROOT.iterdir():
        if not path.is_dir() or not (path / MARKER).is_file() or path.is_symlink():
            continue
        lock = FileLock(str(path) + ".lock", thread_local=False)
        try:
            with lock.acquire(timeout=0):
                if path.resolve().parent == ROOT.resolve():
                    shutil.rmtree(path)
        except (Timeout, OSError):
            continue


def create_lease():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / uuid.uuid4().hex
    lock = FileLock(str(path) + ".lock", thread_local=False)
    lock.acquire()
    try:
        path.mkdir()
        (path / MARKER).write_text("H3 private input lease v1", encoding="utf-8")
        return str(path), lock
    except BaseException:
        lock.release()
        raise


def release_lease(path, lock):
    if path and Path(path).resolve().parent == ROOT.resolve():
        shutil.rmtree(path, ignore_errors=True)
    if lock:
        lock.release()
