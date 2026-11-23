"""Read the frozen fitted dialogue saved beside a completed dub."""

from contextlib import contextmanager
import json
from pathlib import Path
import zipfile

import numpy as np

from .pcm import FITTED_AUDIO_RATE as FITTED_AUDIO_RATE
from .results import player_folder


@contextmanager
def open_alignment(output: Path):
    """Read the manifest and keep the archive open only for the requested clips."""
    folder = player_folder(output)
    try:
        saved = json.loads((folder / "alignment.json").read_text(encoding="utf-8"))
        with np.load(folder / "alignment.npz", allow_pickle=False) as clips:
            yield saved, clips
    except (EOFError, zipfile.BadZipFile) as exc:
        raise ValueError("Invalid fitted dialogue archive") from exc
