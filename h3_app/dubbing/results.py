"""A finished dub's result player, saved beside its outputs so Media can show it again.

Each dub keeps its player tracks in ``<outputs>/dubbing/.dub-player/<stem>/``, named after
the dub's ``<stem>.mp3`` and ``<stem>.mp4``. ``player.json`` there records the player;
paths are found again by file name, so a volume mounted at another path still works.
Kept apart from the pipeline so the GPU-free Library Console can import it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import uuid

# The settings snapshot's family for dub outputs.
DUB_FAMILY = "Dub & Translate"
# The result player's separate tracks sit beside the outputs in this folder, which Media skips.
PLAYER_DIR = ".dub-player"
PLAYER_FILE = "player.json"


@dataclass
class Player:
    """What the result player plays: the speech tracks and background apart, so either voice and the
    music can be switched while it plays, and each line's original and translated subtitle times."""

    video: Path | None
    dub: Path
    original: Path
    music: Path | None
    duration: float
    lines: list[dict]
    target_language: str = ""
    source_language: str = ""
    word_timing: bool = False


TRACKS = ("video", "dub", "original", "music")


def player_folder(output: str | Path) -> Path:
    """The player folder of a dub output (its ``.mp3`` or ``.mp4``)."""
    output = Path(output)
    return output.parent / PLAYER_DIR / output.stem


def save_player(folder: Path, player: Player) -> None:
    folder = Path(folder)
    destination = folder / PLAYER_FILE
    temporary = folder / f"{PLAYER_FILE}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(asdict(player), ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def player_from_data(data: dict, folder: Path) -> Player:
    """A saved player whose tracks are looked up by name in its folder, then beside the outputs."""
    folder = Path(folder)
    data = dict(data)
    for field in TRACKS:
        if not data.get(field):
            data[field] = None
            continue
        name = Path(str(data[field]).replace("\\", "/")).name
        found = next((path for path in (folder / name, folder.parent.parent / name) if path.is_file()), None)
        data[field] = found
    return Player(**{key: value for key, value in data.items() if key in Player.__dataclass_fields__})


def load_player(output: str | Path) -> Player | None:
    """The saved player of a dub output, or None when it has none or its tracks are gone.

    Dubs from before ``player.json`` was written still have the player in ``alignment.json``.
    """
    folder = player_folder(output)
    data = None
    for name, key in ((PLAYER_FILE, None), ("alignment.json", "player")):
        try:
            saved = json.loads((folder / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data = saved.get(key) if key else saved
        if isinstance(data, dict):
            break
        data = None
    if data is None:
        return None
    try:
        player = player_from_data(data, folder)
    except (TypeError, ValueError):
        return None
    return player if player.dub is not None and player.original is not None else None


@dataclass(frozen=True)
class SavedResult:
    audio: Path | None
    video: Path | None
    subtitles: tuple[Path, ...]
    player: Player | None

    @property
    def translated_subtitles(self) -> Path | None:
        return next((path for path in self.subtitles if not path.name.endswith(".original.srt")), None)


def load_result(output: str | Path) -> SavedResult | None:
    """Find this dub's outputs, including surviving audio if its video was removed."""
    output = Path(output)
    audio, video = output.with_suffix(".mp3"), output.with_suffix(".mp4")
    audio = audio if audio.is_file() else None
    video = video if video.is_file() else None
    if audio is None and video is None:
        return None
    # The dot boundary avoids collecting another dub whose stem starts with this one.
    subtitles = tuple(sorted(path for path in output.parent.iterdir()
                             if path.name.startswith(output.stem + ".")
                             and path.suffix == ".srt" and path.is_file()))
    return SavedResult(audio, video, subtitles, load_player(output))


def prune_players(output_dir: Path) -> None:
    """Drop player folders whose dub was deleted from Media: neither its ``.mp3`` nor ``.mp4`` is left."""
    import shutil

    root = Path(output_dir) / PLAYER_DIR
    for folder in root.iterdir() if root.is_dir() else ():
        if folder.is_dir() and not any((Path(output_dir) / f"{folder.name}{suffix}").exists()
                                       for suffix in (".mp3", ".mp4")):
            shutil.rmtree(folder, ignore_errors=True)
