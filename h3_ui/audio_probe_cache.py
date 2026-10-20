"""Remember Gradio's per-file audio checks so repeated audio outputs stay cheap.

Each time a filepath reaches an Audio output, Gradio looks for ffmpeg on PATH and runs
ffprobe on the file, which costs about 35 ms on Windows. Streaming handlers that repeat
finished clips in every frame (Drama 3 voice batches) paid that for every clip in every
frame. A file's answer only changes with the file, so it is kept per path, size and
modification time.
"""

from functools import lru_cache
import os

from gradio import processing_utils

_PLAYABLE_LIMIT = 4096


def install():
    """Wrap Gradio's ffmpeg lookup and playability probe once per process."""
    if getattr(processing_utils, "_h3_probe_cache", False):
        return
    probe = processing_utils.audio_is_playable

    @lru_cache(maxsize=_PLAYABLE_LIMIT)
    def playable(path, size, modified):
        return probe(path)

    def audio_is_playable(audio_filepath):
        try:
            stat = os.stat(audio_filepath)
        except OSError:
            return probe(audio_filepath)
        return playable(os.path.abspath(audio_filepath), stat.st_size, stat.st_mtime_ns)

    lookup = processing_utils.ffmpeg_installed
    found = []

    def ffmpeg_installed():
        # Only a hit is kept: ffmpeg may be installed while the app runs.
        if not found and lookup():
            found.append(True)
        return bool(found)

    processing_utils.ffmpeg_installed = ffmpeg_installed
    processing_utils.audio_is_playable = audio_is_playable
    processing_utils._h3_probe_cache = True
