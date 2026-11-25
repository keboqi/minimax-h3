"""Shared storyboard operations over the workspace store, plus export."""

from __future__ import annotations

import copy
import hashlib
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Any, Callable, Iterable
from weakref import WeakValueDictionary

from h3_app.burned_subtitles import Line, ass_subtitles, font_size, word_pieces
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.gallery_store import gallery_thumbnail_path
from h3_app.media_review import review_path
from h3_app.processes import run_media_process

from . import clip, model, subtitles

TAKE_FOLDER = "h3/.storyboard"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
# Scene fields a sync sets from the plan; the rest are the storyboard's own (takes, take in use).
SYNC_FIELDS = ("title", "prompt", "seconds", "link", "mode", "cast", "overlap", "seed", "takes_per_run")
_TRANSLATION_POOL = ThreadPoolExecutor(max_workers=10, thread_name_prefix="storyboard-translation")


def take_files(take: dict[str, Any]) -> set[str]:
    """Every file a take owns: its video, saved latents, and for a scene from Media its tail clip."""
    return {value for value in (take.get("video"), take.get("latent"), take.get("stage_latent"), take.get("tail"))
            if value}


def file_digest(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SceneFiles:
    """Stores scene media once per storyboard.

    An incoming file is the stored file it is named after (the editor hands back
    Gradio's cached copies of files it shows), else a used file with the same content,
    else a new copy. A branch's scenes point at the original's media folder, so any
    file the storyboard uses counts. ``discard`` removes the copies made so far.
    """

    def __init__(self, directory: Path, board: dict[str, Any]):
        self.directory = directory
        self.used = {Path(path).name: path for scene in board["scenes"] for path in model.media_paths(scene)}
        self.digests: dict[str, str] | None = None
        self.copied: list[Path] = []

    def keep(self, item) -> str:
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        source = Path(str(item))
        if source.name in self.used and Path(self.used[source.name]).is_file():
            return self.used[source.name]
        stored = self.directory / source.name
        if stored.is_file():
            return str(stored)
        if not source.is_file():
            raise H3Error(f"{source.name} is no longer available. Add it again.")
        digest = file_digest(source)
        if self.digests is None:
            self.digests = {}
            for path in self.used.values():
                if Path(path).is_file():
                    self.digests.setdefault(file_digest(path), path)
        if digest in self.digests:
            return self.digests[digest]
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{model.new_id()}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        self.copied.append(destination)
        self.used[destination.name] = self.digests[digest] = str(destination)
        return str(destination)

    def value(self, field, value):
        """A frame field's stored path, or a reference field's list of them."""
        if field in ("first_image", "last_image"):
            return self.keep(value) if value else None
        items = value if isinstance(value, (list, tuple)) else [value] if value else []
        return [self.keep(item) for item in items if item]

    def discard(self):
        for path in self.copied:
            path.unlink(missing_ok=True)


class StoryboardService:
    def __init__(self, store, runtime: Callable[[], RuntimeConfig]):
        self.store = store
        self.runtime = runtime
        # One process edits a storyboard at a time; each change is read-modify-write.
        # Share the lock per WorkspaceStore so transient StoryboardService helpers
        # (e.g. cleanup/transfer) serialize against the live server instance.
        lock = getattr(store, "_storyboard_service_lock", None)
        if lock is None:
            lock = RLock()
            try:
                setattr(store, "_storyboard_service_lock", lock)
            except AttributeError:
                pass
        self.lock = lock
        self._export_media: dict[int, set[str]] = {}
        self._translation_locks = WeakValueDictionary()

    # Paths -----------------------------------------------------------------
    def cast_dir(self, storyboard_id: str) -> Path:
        return self.store.root / "storyboards" / storyboard_id / "cast"

    def media_dir(self, storyboard_id: str) -> Path:
        """Scene frames and references; served like takes, so the editor can show them."""
        return self.output_path(f"{TAKE_FOLDER}/{storyboard_id}/media")

    def take_prefix(self, storyboard_id: str, scene_id: str, take_id: str) -> str:
        """Output-relative prefix shared by a take's video and latents."""
        return f"{TAKE_FOLDER}/{storyboard_id}/{scene_id[:8]}_{take_id[:8]}"

    def output_path(self, relative: str) -> Path:
        root = self.runtime().output_dir.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise H3Error("A storyboard file is outside the output directory.")
        return path

    # Documents -------------------------------------------------------------
    def list(self, owner: str) -> list[dict[str, Any]]:
        return self.store.storyboards(owner)

    def load(self, owner: str, storyboard_id: str) -> dict[str, Any]:
        return self.store.storyboard(owner, storyboard_id)

    def save(self, owner: str, storyboard_id: str, board: dict[str, Any]) -> dict[str, Any]:
        for scene in board["scenes"]:
            model.validate_scene(scene)
        if not board["scenes"]:
            raise H3Error("A storyboard keeps at least one scene.")
        self.store.save_storyboard(owner, storyboard_id, board)
        return board

    def change(self, owner: str, storyboard_id: str, edit: Callable[[dict[str, Any]], Any]):
        """Apply ``edit`` to the stored document under the lock and save it."""
        with self.lock:
            board = self.load(owner, storyboard_id)
            result = edit(board)
            if result is not False:
                self.save(owner, storyboard_id, board)
            return board, result

    def create(self, owner, name, profile, *, scenes=(), cast=()) -> str:
        """A new storyboard; ``cast`` is (name, picture) pairs copied into it."""
        storyboard_id = model.new_id()
        board = model.new_storyboard(name, profile, scenes)
        for member_name, image in cast:
            board["cast"].append(self._copy_cast(storyboard_id, member_name, image))
        self.save(owner, storyboard_id, board)
        return storyboard_id

    def _copy_cast(self, storyboard_id, name, image, *, from_branch_source=False) -> dict[str, str]:
        name = str(name).strip()
        source = Path(image)
        if not name:
            raise H3Error("Give each cast picture a name.")
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise H3Error("Cast pictures are PNG, JPEG or WebP images.")
        directory = self.cast_dir(storyboard_id)
        directory.mkdir(parents=True, exist_ok=True)
        filename = source.name if from_branch_source else f"{model.new_id()}{source.suffix.lower()}"
        destination = directory / filename
        shutil.copy2(source, destination)
        return {"name": name, "image": str(destination)}

    def add_cast(self, owner, storyboard_id, name, image):
        def edit(board):
            if len(board["cast"]) >= model.MAX_CAST:
                raise H3Error(f"A storyboard has at most {model.MAX_CAST} cast pictures.")
            if any(member["name"] == str(name).strip() for member in board["cast"]):
                raise H3Error("That cast name is already used.")
            board["cast"].append(self._copy_cast(storyboard_id, name, image))
        return self.change(owner, storyboard_id, edit)[0]

    def remove_cast(self, owner, storyboard_id, name):
        def edit(board):
            board["cast"] = [member for member in board["cast"] if member["name"] != name]
            for scene in board["scenes"]:
                scene["cast"] = [member for member in scene["cast"] if member != name]
        return self.change(owner, storyboard_id, edit)[0]

    def rename(self, owner, storyboard_id, name):
        def edit(board):
            board["name"] = (str(name).strip() or board["name"])[:120]
        return self.change(owner, storyboard_id, edit)[0]

    def set_subtitle_settings(self, owner, storyboard_id, *, auto_subtitles=None, subtitle_language=None):
        def edit(board):
            if auto_subtitles is not None:
                board["auto_subtitles"] = auto_subtitles
            if subtitle_language is not None:
                board["subtitle_language"] = subtitle_language
        return self.change(owner, storyboard_id, edit)[0]

    def set_score_settings(self, owner, storyboard_id, *, auto_score: bool):
        def edit(board):
            board["auto_score"] = bool(auto_score)
        return self.change(owner, storyboard_id, edit)[0]

    def apply_profile(self, owner, storyboard_id, profile):
        def edit(board):
            board["profile"] = profile
        return self.change(owner, storyboard_id, edit)[0]

    # Scenes ----------------------------------------------------------------
    @staticmethod
    def scene_index(board, scene_id) -> int:
        for index, scene in enumerate(board["scenes"]):
            if scene["id"] == scene_id:
                return index
        raise H3Error("That scene is no longer in this storyboard.")

    def update_scene(self, owner, storyboard_id, scene_id, **fields):
        allowed = {"title", "prompt", "seconds", "link", "cast", "overlap", "seed", "takes_per_run",
                   *model.VIDEO_FIELDS}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown scene fields: {sorted(unknown)}")

        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
            if set(fields) & set(model.VIDEO_FIELDS) and not model.is_media(scene):
                raise H3Error("Only a scene from Media has In, Out and Fit.")
            for name in ("video_in", "video_out"):
                if name in fields and fields[name] is not None:
                    fields[name] = round(max(0.0, float(fields[name])), 3)
            scene.update(fields)
            names = {member["name"] for member in board["cast"]}
            scene["cast"] = [name for name in scene["cast"] if name in names]
        return self.change(owner, storyboard_id, edit)[0]

    def add_scene(self, owner, storyboard_id, after=None):
        def edit(board):
            position = len(board["scenes"]) if after is None else self.scene_index(board, after) + 1
            previous = board["scenes"][position - 1] if position else None
            # A new scene continues the one before it, in the same mode and with the same cast.
            scene = model.new_scene(
                f"Scene {len(board['scenes']) + 1}",
                seconds=previous["seconds"] if previous else 5,
                cast=previous["cast"] if previous else (),
                mode=model.scene_media(previous)["mode"] if previous else model.TEXT,
            )
            board["scenes"].insert(position, scene)
            return scene["id"]
        return self.change(owner, storyboard_id, edit)

    def add_media_scene(self, owner, storyboard_id, after=None):
        """A scene that plays a video from Media, after ``after`` or at the end; returns (board, id)."""
        def edit(board):
            position = len(board["scenes"]) if after is None else self.scene_index(board, after) + 1
            scene = model.new_media_scene()
            board["scenes"].insert(position, scene)
            return scene["id"]
        return self.change(owner, storyboard_id, edit)

    def set_scene_video(self, owner, storyboard_id, scene_id, source):
        """Copy a video into the storyboard as a Media scene's clip; In and Out start at its ends."""
        info = clip.probe(source)
        with self.lock:
            board = self.load(owner, storyboard_id)
            scene = board["scenes"][self.scene_index(board, scene_id)]
            if not model.is_media(scene):
                raise H3Error("Only a scene from Media plays a video.")
            files = SceneFiles(self.media_dir(storyboard_id), board)
            before = self._scene_files(board)
            try:
                stored = files.keep(source)
                if stored == scene.get("video"):
                    return board
                scene.update(video=stored, video_info=info, video_in=0.0, video_out=None,
                             title=scene["title"] if scene["title"] != "Video from Media"
                             else model.title_from(Path(str(source)).stem.replace("_", " "), 5) or scene["title"])
                self.save(owner, storyboard_id, board)
            except Exception:
                files.discard()
                raise
            self._collect_media(before - self._scene_files(board))
            return board

    def match_video_shape(self, owner, storyboard_id, scene_id):
        """Give the storyboard this Media scene's shape, so its video fits without stretching or cropping."""
        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
            info = scene.get("video_info") if model.is_media(scene) else None
            if not info:
                raise H3Error("Choose a video first.")
            if model.generated_takes(board):
                raise H3Error("Scenes are already generated at the storyboard's size, so it stays as it is.")
            width, height = model.shaped_like(board["profile"], info["width"], info["height"])
            board["profile"] = {**board["profile"], "width": width, "height": height}
        return self.change(owner, storyboard_id, edit)[0]

    def prepare_media_scene(self, owner, storyboard_id, scene_id):
        """Fit a Media scene's clip to the canvas and cut its tail; the result is the scene's take."""
        board = self.load(owner, storyboard_id)
        index = self.scene_index(board, scene_id)
        scene = board["scenes"][index]
        if not model.is_media(scene):
            raise H3Error("Only a scene from Media is prepared.")
        model.check_scene(board, index)
        inputs = model.scene_inputs(board, index)
        start, end = model.video_range(scene)
        width, height = model.canvas(board["profile"])
        frames = model.video_frames(scene)
        take_id = model.new_id()
        prefix = self.output_path(self.take_prefix(storyboard_id, scene["id"], take_id))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        video = prefix.with_name(prefix.name + ".mp4")
        tail = prefix.with_name(prefix.name + "_tail.mp4")
        try:
            clip.conform(scene["video"], video, start=start, end=end, fit=scene.get("fit", model.RESIZE),
                         width=width, height=height, has_audio=bool((scene.get("video_info") or {}).get("audio")))
            tail_path = str(clip.cut_tail(video, tail, frames=frames)) if frames >= model.TAIL_FRAMES else None
        except Exception:
            video.unlink(missing_ok=True)
            tail.unlink(missing_ok=True)
            raise

        def edit(current):
            target = current["scenes"][self.scene_index(current, scene_id)]
            current["takes"][take_id] = {
                "id": take_id, "scene": scene_id, "created": time.time(), "video": str(video),
                "latent": None, "stage_latent": None, "tail": tail_path, "prefix_frames": 0, "seed": 0,
                "job_id": None, "inputs": inputs,
            }
            # A Media scene keeps one take: the clip as it is prepared now.
            replaced = [item for item in target["takes"] if item != take_id]
            target["takes"] = [take_id]
            target["in_use"] = take_id
            kept = {item for version in current.get("versions", ()) for saved in version.get("scenes", ())
                    for item in (*saved.get("takes", ()), saved.get("in_use")) if item}
            return [current["takes"].pop(item) for item in replaced if item not in kept and item in current["takes"]]

        with self.lock:
            board, dropped = self.change(owner, storyboard_id, edit)
            used = self._used_take_files()
            for take in dropped:
                self._unlink_take_files(take, used)
        return board

    def set_mode(self, owner, storyboard_id, scene_id, mode):
        """Switch a scene's conditioning; References starts with the whole cast ticked."""
        if mode not in model.MODES:
            raise H3Error(f"Unknown scene mode {mode!r}.")

        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
            if scene.get("mode", model.TEXT) != mode and mode == model.REFERENCES and not scene["cast"]:
                scene["cast"] = [member["name"] for member in board["cast"]]
            scene["mode"] = mode
        return self.change(owner, storyboard_id, edit)[0]

    def set_scene_media(self, owner, storyboard_id, scene_id, field, value):
        """Replace one of a scene's frames or reference lists, copying new files in.

        The editor hands back Gradio's cached copies of files it already shows; their
        names match the stored ones, so those are kept instead of copied again. A
        branch's scenes still point at the original's media folder, so any stored file
        the storyboard already uses counts. Returns False when nothing changed.
        """
        if field not in model.MEDIA_FIELDS:
            raise ValueError(f"Unknown scene media {field!r}.")

        with self.lock:
            board = self.load(owner, storyboard_id)
            files = SceneFiles(self.media_dir(storyboard_id), board)
            before = self._scene_files(board)
            try:
                stored = files.value(field, value)
                scene = board["scenes"][self.scene_index(board, scene_id)]
                if model.scene_media(scene)[field] == stored:
                    return False
                scene[field] = stored
                self.save(owner, storyboard_id, board)
            except Exception:
                files.discard()
                raise
            self._collect_media(before - self._scene_files(board))
            return True

    def toggle_link(self, owner, storyboard_id, scene_id):
        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
            scene["link"] = model.CUT if scene["link"] == model.FLOW else model.FLOW
        return self.change(owner, storyboard_id, edit)[0]

    def duplicate_scene(self, owner, storyboard_id, scene_id):
        def edit(board):
            index = self.scene_index(board, scene_id)
            scene = copy.deepcopy(board["scenes"][index])
            scene.update(id=model.new_id(), title=f"{scene['title']} (copy)", takes=[], in_use=None)
            board["scenes"].insert(index + 1, scene)
            return scene["id"]
        return self.change(owner, storyboard_id, edit)

    def move_scene(self, owner, storyboard_id, scene_id, offset):
        def edit(board):
            index = self.scene_index(board, scene_id)
            target = max(0, min(len(board["scenes"]) - 1, index + int(offset)))
            board["scenes"].insert(target, board["scenes"].pop(index))
        return self.change(owner, storyboard_id, edit)[0]

    def delete_scene(self, owner, storyboard_id, scene_id):
        with self.lock:
            board = self.load(owner, storyboard_id)
            if len(board["scenes"]) == 1:
                raise H3Error("A storyboard keeps at least one scene.")
            before = self._scene_files(board)
            board["scenes"].pop(self.scene_index(board, scene_id))
            board["queue"] = [item for item in board["queue"] if item != scene_id]
            self.save(owner, storyboard_id, board)
            self._collect_media(before - self._scene_files(board))
            return board

    def sync(self, owner, storyboard_id, scenes, *, name=None, profile=None, cast=None) -> dict[str, Any]:
        """Make the storyboard match planned ``scenes``, keeping the takes of scenes that stay.

        Each planned scene is a dict of ``SYNC_FIELDS``, an optional ``key`` and ``media``
        (media field → uploaded file or files). A scene whose key matches a stored one keeps
        its id, takes and take in use, gets the given fields and exactly the given media;
        fields it leaves out keep their values. Stored scenes the plan leaves out are removed
        (their takes stay recorded), and scenes follow the plan's order. ``name``, ``profile``
        and ``cast`` ((name, picture) pairs) replace the storyboard's only when given. Whether
        a scene is out of date then follows from its inputs as always. Returns what changed.
        """
        keys = [item["key"] for item in scenes if item.get("key")]
        if len(keys) != len(set(keys)):
            raise H3Error("Each scene key can be used only once.")
        with self.lock:
            board = self.load(owner, storyboard_id)
            files = SceneFiles(self.media_dir(storyboard_id), board)
            before = self._scene_files(board)
            try:
                changes = self._sync(board, storyboard_id, scenes, name, profile, cast, files)
                self.save(owner, storyboard_id, board)
            except Exception:
                files.discard()
                raise
            self._collect_media(before - self._scene_files(board))
        return changes

    def _sync(self, board, storyboard_id, scenes, name, profile, cast, files):
        if name is not None:
            board["name"] = (str(name).strip() or board["name"])[:120]
        settings_changed = profile is not None and profile != board["profile"]
        if settings_changed:
            board["profile"] = profile
        if cast is not None:
            board["cast"] = self._sync_cast(board, storyboard_id, cast, files)
        names = {member["name"] for member in board["cast"]}
        stored = {scene["key"]: scene for scene in board["scenes"] if scene.get("key")}
        planned, added, updated = [], [], []
        for number, item in enumerate(scenes, 1):
            key = item.get("key")
            scene = stored.pop(key, None) if key else None
            before = copy.deepcopy(scene)
            if scene is None:
                title = item.get("title") or model.title_from(item.get("prompt", "")) or f"Scene {number}"
                scene = model.new_scene(title)
                if key:
                    scene["key"] = key
            for field in SYNC_FIELDS:
                if field in item:
                    scene[field] = list(item[field]) if field == "cast" else item[field]
            media = item.get("media") or {}
            for field in model.MEDIA_FIELDS:
                scene[field] = files.value(field, media.get(field))
            scene["cast"] = [member for member in scene["cast"] if member in names]
            label = key or scene["title"]
            if before is None:
                added.append(label)
            elif before != scene:
                updated.append(label)
            planned.append(scene)
        kept = {scene["id"] for scene in planned}
        removed = [scene.get("key") or scene["title"] for scene in board["scenes"] if scene["id"] not in kept]
        board["scenes"] = planned
        board["queue"] = [scene_id for scene_id in board["queue"] if scene_id in kept]
        return {"added": added, "updated": updated, "removed": removed, "settings_changed": settings_changed}

    def _sync_cast(self, board, storyboard_id, cast, files):
        """The planned cast; a member whose name and picture are unchanged keeps its stored picture."""
        if len(cast) > model.MAX_CAST:
            raise H3Error(f"A storyboard has at most {model.MAX_CAST} cast pictures.")
        stored = {member["name"]: member for member in board["cast"]}
        members = []
        for member_name, image in cast:
            member_name = str(member_name).strip()
            if any(member["name"] == member_name for member in members):
                raise H3Error("That cast name is already used.")
            current = stored.get(member_name)
            if (current and Path(current["image"]).is_file() and Path(image).is_file()
                    and file_digest(current["image"]) == file_digest(image)):
                members.append(current)
                continue
            member = self._copy_cast(storyboard_id, member_name, image)
            files.copied.append(Path(member["image"]))
            members.append(member)
        return members

    # Takes -----------------------------------------------------------------
    def record_take(self, owner, storyboard_id, scene_id, take_id, *, video, latents, seed, inputs, job_id=None):
        prefix_frames = int(latents.get("prefix_frames", 0))

        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
            board["takes"][take_id] = {
                "id": take_id, "scene": scene_id, "created": time.time(), "video": str(video),
                "latent": latents["latent"], "stage_latent": latents.get("stage_latent"),
                "prefix_frames": prefix_frames, "seed": int(seed), "job_id": job_id, "inputs": inputs,
            }
            scene["takes"].append(take_id)
            # A take made from the scene as it is now replaces one made from older inputs;
            # later takes of the same inputs leave the choice in use alone.
            current = board["takes"].get(scene["in_use"]) if scene["in_use"] else None
            if current is None or current["inputs"] != inputs or current.get("omitted"):
                scene["in_use"] = take_id
        return self.change(owner, storyboard_id, edit)[0]

    def add_manual_take(
        self,
        owner,
        storyboard_id,
        scene_id,
        source,
        *,
        source_kind="upload",
        source_name=None,
        fit=model.RESIZE,
    ) -> tuple[dict[str, Any], str]:
        """Conform an uploaded or Media library video to the canvas and add it as a scene's new take."""
        if source_kind not in model.MANUAL_TAKE_SOURCES:
            raise ValueError(f"Unknown manual take source {source_kind!r}.")
        if fit not in model.FITS:
            raise H3Error("Fit is resize or crop.")
        board = self.load(owner, storyboard_id)
        scene = board["scenes"][self.scene_index(board, scene_id)]
        if model.is_media(scene):
            raise H3Error("A scene from Media uses its Video input instead of extra takes.")
        width, height = model.canvas(board["profile"])
        info = clip.probe(source)
        if clip.frame_count(0.0, info["seconds"]) < 1:
            raise H3Error("This video is too short to use as a take.")

        take_id = model.new_id()
        prefix = self.output_path(self.take_prefix(storyboard_id, scene_id, take_id))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        video = prefix.with_name(prefix.name + ".mp4")
        tail = prefix.with_name(prefix.name + "_tail.mp4")
        try:
            clip.conform(
                source, video, start=0.0, end=info["seconds"], fit=fit,
                width=width, height=height, has_audio=bool(info.get("audio")),
            )
            actual_frames = clip.video_frame_total(video)
            if actual_frames < 1:
                raise H3Error("Couldn't read any video frames from this clip.")
            tail_path = str(clip.cut_tail(video, tail, frames=actual_frames)) if actual_frames >= model.TAIL_FRAMES else None

            def edit(current):
                index = self.scene_index(current, scene_id)
                target = current["scenes"][index]
                if model.is_media(target):
                    raise H3Error("A scene from Media uses its Video input instead of extra takes.")
                inputs = model.scene_inputs(current, index)
                flow = model.flows_from_previous(current, index)
                current_take = model.active_take(current, target)
                inherited = int(current_take.get("prefix_frames") or 0) if current_take else 0
                prefix_frames = inherited if flow and inherited > 0 and actual_frames > inherited else 0
                current["takes"][take_id] = {
                    "id": take_id, "scene": scene_id, "created": time.time(), "video": str(video),
                    "latent": None, "stage_latent": None, "tail": tail_path,
                    "prefix_frames": prefix_frames, "seed": None, "job_id": None, "inputs": inputs,
                    "source": source_kind,
                    "source_name": Path(str(source_name or source)).name,
                    "duration": round(actual_frames / model.FPS, 3),
                    "frames": actual_frames,
                }
                target["takes"].append(take_id)
                if current_take is None or current_take["inputs"] != inputs or current_take.get("omitted"):
                    target["in_use"] = take_id
                return take_id

            saved, _ = self.change(owner, storyboard_id, edit)
        except Exception:
            video.unlink(missing_ok=True)
            tail.unlink(missing_ok=True)
            raise
        return saved, take_id

    def set_take_overlap(self, owner, storyboard_id, scene_id, take_id, trim_overlap: bool) -> dict[str, Any]:
        """Toggle whether an imported take skips the continuing scene's opening overlap frames."""
        def edit(board):
            index = self.scene_index(board, scene_id)
            scene = board["scenes"][index]
            if model.is_media(scene):
                raise H3Error("Only imported takes can toggle opening overlap trim.")
            if take_id not in scene["takes"]:
                raise H3Error("That take belongs to another scene.")
            take = board["takes"].get(take_id)
            if not model.is_imported_take(take):
                raise H3Error("Only imported takes can toggle opening overlap trim.")
            if not trim_overlap:
                take["prefix_frames"] = 0
                return
            if not model.flows_from_previous(board, index):
                raise H3Error("Only a continuing scene has an opening overlap to trim.")
            overlap = int(scene["overlap"])
            frames = int(take.get("frames") or 0)
            if not frames and take.get("video") and Path(take["video"]).is_file():
                frames = clip.video_frame_total(take["video"])
            if frames <= overlap:
                raise H3Error("This take isn't longer than the scene's opening overlap.")
            take["prefix_frames"] = overlap

        return self.change(owner, storyboard_id, edit)[0]

    def use_take(self, owner, storyboard_id, scene_id, take_id):
        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
            if take_id not in scene["takes"]:
                raise H3Error("That take belongs to another scene.")
            scene["in_use"] = take_id
        return self.change(owner, storyboard_id, edit)[0]

    def save_subtitles(self, owner, storyboard_id, take_id, lines, *, error=None):
        def edit(board):
            take = board["takes"].get(take_id)
            if take is None:
                return
            if lines is not None:
                take["subtitles"] = lines
                take["subtitle_revision"] = time.time_ns()
            take["subtitle_error"] = error
        self.change(owner, storyboard_id, edit)

    def translate_subtitles(self, owner, storyboard_id, takes, translation: subtitles.Translation, *, force=False):
        """Translate the supplied take snapshot in board order, reusing its cached caption text."""
        with self.lock:
            locks = [self._translation_locks.setdefault((storyboard_id, take_id, translation.language), RLock())
                     for take_id in sorted({take["id"] for take in takes})]
        with ExitStack() as stack:
            for lock in locks:
                stack.enter_context(lock)
            return self._translate_subtitles(owner, storyboard_id, takes, translation, force)

    def _translate_subtitles(self, owner, storyboard_id, takes, translation, force):
        # Another scene preview or export may have saved these captions while we waited.
        current = self.load(owner, storyboard_id)["takes"]
        results, pending, texts = {}, {}, []
        for take in takes:
            lines = subtitles.take_lines(take)
            if not lines:
                continue
            take_id = take["id"]
            cached = None if force else subtitles.cached_translation(current.get(take_id, take), translation.language, lines)
            if cached is not None:
                results[take_id] = cached
                continue
            pending[take_id] = (subtitles.source_digest(lines), len(texts), len(lines))
            texts.extend(line["text"] for line in lines)
        if not pending:
            return results

        batch_size = subtitles.TRANSLATION_BATCH
        batches = (texts[start:start + batch_size] for start in range(0, len(texts), batch_size))
        # map returns results in submission order even when requests finish out of order.
        translated = list(chain.from_iterable(_TRANSLATION_POOL.map(translation.translate, batches)))

        def save(board):
            for take_id, (source, start, count) in pending.items():
                lines = translated[start:start + count]
                results[take_id] = lines
                take = board["takes"].get(take_id)
                if take is None or subtitles.source_digest(subtitles.take_lines(take)) != source:
                    continue
                cache = take.setdefault("subtitle_translations", {})
                cache[translation.language] = {"source": source, "texts": lines, "model": translation.model}

        self.change(owner, storyboard_id, save)
        return results

    def delete_take(self, owner, storyboard_id, scene_id, take_id):
        with self.lock:
            board = self.load(owner, storyboard_id)
            scene = board["scenes"][self.scene_index(board, scene_id)]
            if take_id not in scene["takes"]:
                raise H3Error("That take belongs to another scene.")
            if scene["in_use"] == take_id:
                raise H3Error("Switch to another take before deleting this one.")
            scene["takes"] = [item for item in scene["takes"] if item != take_id]
            in_version = any(
                take_id in item.get("takes", ()) or item.get("in_use") == take_id
                for version in board.get("versions", ())
                for item in version.get("scenes", ())
            )
            removed = None if in_version else board["takes"].pop(take_id, None)
            self.save(owner, storyboard_id, board)
            if removed is not None:
                self._unlink_take_files(removed, self._used_take_files())
            return board

    def _resolve_owner(self, owner: str | None, storyboard_id: str) -> str:
        if owner:
            return str(owner)
        with self.store.connect() as db:
            row = db.execute("SELECT owner FROM storyboards WHERE id=?", (storyboard_id,)).fetchone()
        return str(row["owner"]) if row and row["owner"] else "local"

    @staticmethod
    def _prune_board_history_takes(board: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]], bool]:
        takes = board.get("takes") if isinstance(board.get("takes"), dict) else {}
        in_use_ids = {
            str(scene["in_use"])
            for scene in (board.get("scenes") or ())
            if isinstance(scene, dict) and scene.get("in_use") and str(scene["in_use"]) in takes
        }
        dropped_ids = [tid for tid in list(takes) if str(tid) not in in_use_ids]
        dropped_takes: list[dict[str, Any]] = []
        changed = bool(dropped_ids)
        for tid in dropped_ids:
            dropped = takes.pop(tid)
            if isinstance(dropped, dict):
                dropped_takes.append(dropped)

        for scene in board.get("scenes") or ():
            if not isinstance(scene, dict):
                continue
            filtered = [tid for tid in (scene.get("takes") or ()) if str(tid) in in_use_ids]
            if filtered != list(scene.get("takes") or ()):
                scene["takes"] = filtered
                changed = True
            if scene.get("in_use") and str(scene["in_use"]) not in in_use_ids:
                scene["in_use"] = None
                changed = True

        for version in board.get("versions") or ():
            if not isinstance(version, dict):
                continue
            for vscene in version.get("scenes") or ():
                if not isinstance(vscene, dict):
                    continue
                filtered = [tid for tid in (vscene.get("takes") or ()) if str(tid) in in_use_ids]
                if filtered != list(vscene.get("takes") or ()):
                    vscene["takes"] = filtered
                    changed = True
                if vscene.get("in_use") and str(vscene["in_use"]) not in in_use_ids:
                    vscene["in_use"] = None
                    changed = True

        partial = board.get("_partial_takes")
        if isinstance(partial, dict):
            if partial.get("omitted_takes"):
                partial["omitted_takes"] = []
                changed = True
            if not partial.get("omitted_versions") and not any(
                isinstance(t, dict) and t.get("omitted") for t in takes.values()
            ):
                board.pop("_partial_takes", None)
                changed = True

        return in_use_ids, dropped_takes, changed

    def _resolved_take_paths(self, values: Iterable[str]) -> set[Path]:
        resolved_paths: set[Path] = set()
        for value in values:
            try:
                raw_path = Path(value)
                resolved_paths.add((raw_path if raw_path.is_absolute() else self.output_path(value)).resolve())
            except (OSError, H3Error):
                continue
        return resolved_paths

    @staticmethod
    def _unlink_file(path: Path) -> tuple[int, int]:
        try:
            if not path.is_file() or path.is_symlink():
                return 0, 0
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            return 1, size
        except OSError:
            return 0, 0

    def _unlink_take_artifacts(self, path: Path, *, runtime, has_thumbs: bool) -> tuple[int, int]:
        targets = [path, Path(f"{path}.settings.json"), review_path(path)]
        if has_thumbs:
            try:
                targets.append(gallery_thumbnail_path(path, runtime=runtime))
            except OSError:
                pass
        removed_files = 0
        freed_bytes = 0
        for target in targets:
            count, size = self._unlink_file(target)
            removed_files += count
            freed_bytes += size
        return removed_files, freed_bytes

    def clear_history_takes(self, owner: str | None, storyboard_id: str) -> dict[str, Any]:
        """Remove alternate takes and unused take files, keeping only each scene's take in use."""
        with self.lock:
            board = self.load(owner, storyboard_id)
            sb_owner = self._resolve_owner(owner, storyboard_id)
            in_use_ids, dropped_takes, changed = self._prune_board_history_takes(board)
            if changed:
                self.save(sb_owner, storyboard_id, board)

            used = self._used_take_files()
            used_resolved = self._resolved_take_paths(used)
            rt = self.runtime()
            has_thumbs = getattr(rt, "gallery_thumbnails_dir", None) is not None
            removed_files = 0
            freed_bytes = 0

            for take in dropped_takes:
                for value in take_files(take):
                    try:
                        raw_path = Path(value)
                        path = raw_path if raw_path.is_absolute() else self.output_path(value)
                        resolved = path.resolve()
                    except (OSError, H3Error):
                        continue
                    if value not in used and resolved not in used_resolved:
                        count, size = self._unlink_take_artifacts(path, runtime=rt, has_thumbs=has_thumbs)
                        removed_files += count
                        freed_bytes += size

            take_dir = self.output_path(f"{TAKE_FOLDER}/{storyboard_id}")
            if take_dir.is_dir():
                keep_resolved = set(used_resolved)
                take_dir_resolved = take_dir.resolve()
                for p in used_resolved:
                    if p.parent == take_dir_resolved:
                        keep_resolved.add(Path(f"{p}.settings.json").resolve())
                        keep_resolved.add(review_path(p).resolve())
                keep_resolved.add((take_dir / "preview.mp4").resolve())
                keep_resolved.add((take_dir / "preview.vtt").resolve())
                try:
                    entries = list(take_dir.iterdir())
                except OSError:
                    entries = []
                for entry in entries:
                    try:
                        if not entry.is_dir() and not entry.is_symlink() and entry.resolve() not in keep_resolved:
                            count, size = self._unlink_file(entry)
                            removed_files += count
                            freed_bytes += size
                    except OSError:
                        continue

            return {
                "board": board,
                "name": str(board.get("name") or storyboard_id),
                "kept_takes": len(in_use_ids),
                "removed_takes": len(dropped_takes),
                "removed_files": removed_files,
                "freed_bytes": freed_bytes,
            }

    def set_queue(self, owner, storyboard_id, scene_ids):
        def edit(board):
            board["queue"] = list(scene_ids)
        return self.change(owner, storyboard_id, edit)[0]

    # Versions --------------------------------------------------------------
    @contextmanager
    def export_source(self, owner, storyboard_id):
        """Keep the export's scene media alive until its version has been saved."""
        with self.lock:
            board = self.load(owner, storyboard_id)
            paths = {path for scene in board["scenes"] for path in model.media_paths(scene)}
            self._export_media[id(board)] = paths
        try:
            yield board
        finally:
            with self.lock:
                del self._export_media[id(board)]
                self._collect_media(paths)

    def save_version(self, owner, storyboard_id, label="", *, source=None):
        """Save the supplied document's scene list, or the current one for a manual version."""
        def edit(board):
            version = {
                "id": model.new_id(), "created": time.time(),
                "label": (str(label).strip() or f"Version {len(board['versions']) + 1}")[:80],
                **model.snapshot(board if source is None else source),
            }
            board["versions"].append(version)
            return version["id"]
        return self.change(owner, storyboard_id, edit)

    def branch(self, owner, storyboard_id, version_id=None, name=None) -> str:
        """A new storyboard from a saved version (or the current one), sharing take files."""
        # Publish the branch's file references before deletion can collect shared files.
        with self.lock:
            return self._branch(owner, storyboard_id, version_id, name)

    def _branch(self, owner, storyboard_id, version_id, name):
        board = self.load(owner, storyboard_id)
        source = board
        if version_id is not None:
            source = next((v for v in board["versions"] if v["id"] == version_id), None)
            if source is None:
                raise H3Error("That version is no longer saved.")
        branch_id = model.new_id()
        branched = model.new_storyboard(name or f"{board['name']} (copy)", copy.deepcopy(board["profile"]),
                                        copy.deepcopy(source["scenes"]))
        branched["takes"] = copy.deepcopy(board["takes"])
        branched["auto_subtitles"] = board.get("auto_subtitles", False)
        branched["auto_score"] = bool(board.get("auto_score", False))
        if "subtitle_language" in board:
            branched["subtitle_language"] = board["subtitle_language"]
        for member in source["cast"]:
            branched["cast"].append(self._copy_cast(branch_id, member["name"], member["image"], from_branch_source=True))
        self.save(owner, branch_id, branched)
        return branch_id

    @staticmethod
    def _scene_files(board):
        return {path for snapshot in (board, *board.get("versions", ()))
                for scene in snapshot["scenes"] for path in model.media_paths(scene)}

    def delete(self, owner, storyboard_id):
        with self.lock:
            board = self.load(owner, storyboard_id)
            self.store.delete_storyboard(owner, storyboard_id)
            shutil.rmtree(self.store.root / "storyboards" / storyboard_id, ignore_errors=True)
            # Branches share take and scene media files, so keep any another storyboard still uses.
            used = set().union(*self._export_media.values())
            for other_owner, other_id in self._all_ids():
                other = self.store.storyboard(other_owner, other_id)
                for take in other["takes"].values():
                    used |= take_files(take)
                used.update(self._scene_files(other))
            for value in self._scene_files(board) - used:
                Path(value).unlink(missing_ok=True)
            for take in board["takes"].values():
                self._unlink_take_files(take, used)
            # Media this storyboard dropped earlier but a branch still kept also lives here.
            media = self.media_dir(storyboard_id)
            if media.is_dir():
                for path in media.iterdir():
                    if path.is_file() and str(path) not in used:
                        path.unlink(missing_ok=True)
            preview = self.preview_path(storyboard_id)
            preview.unlink(missing_ok=True)
            preview.with_suffix(".vtt").unlink(missing_ok=True)
            for folder in (media, preview.parent):
                try:
                    folder.rmdir()  # Only when no branch still keeps takes or media there.
                except OSError:
                    pass

    def _collect_media(self, dropped: set[str]) -> None:
        """Remove scene media no storyboard, saved version or branch uses any more."""
        if not dropped:
            return
        used = set().union(*self._export_media.values())
        for other_owner, other_id in self._all_ids():
            used.update(self._scene_files(self.store.storyboard(other_owner, other_id)))
        for value in dropped - used:
            path = Path(value).resolve()
            # Only files this feature stored; anything else was never ours to remove.
            if path.parent.name == "media" and path.parent.parent.parent == self.output_path(TAKE_FOLDER):
                path.unlink(missing_ok=True)

    def _used_take_files(self) -> set[str]:
        used = set()
        for other_owner, other_id in self._all_ids():
            other = self.store.storyboard(other_owner, other_id)
            for take in other["takes"].values():
                used |= take_files(take)
        return used

    def _unlink_take_files(self, take: dict[str, Any], used: set[str]) -> None:
        for value in take_files(take):
            if value not in used:
                path = Path(value) if Path(value).is_absolute() else self.output_path(value)
                path.unlink(missing_ok=True)
                Path(str(path) + ".settings.json").unlink(missing_ok=True)
                review_path(path).unlink(missing_ok=True)

    def _all_ids(self):
        with self.store.connect() as db:
            return [(row["owner"], row["id"]) for row in db.execute("SELECT owner, id FROM storyboards")]

    # Export ----------------------------------------------------------------
    def clips(self, board, *, include_outdated=True) -> list[dict[str, Any]]:
        """The take in use per scene with the frames to drop at its start."""
        states = model.scene_states(board)
        clips = []
        for index, (scene, state) in enumerate(zip(board["scenes"], states)):
            take = board["takes"].get(scene["in_use"]) if scene["in_use"] else None
            if take is None:
                raise H3Error(f"Scene {index + 1} has no take yet.")
            if state == model.OUTDATED and not include_outdated:
                raise H3Error(f"Scene {index + 1} is outdated. Regenerate it or include outdated scenes.")
            video = take.get("video")
            if not video or not Path(video).is_file():
                if take.get("omitted"):
                    raise H3Error(f"Scene {index + 1}'s take wasn't transferred to this server.")
                raise H3Error(f"Scene {index + 1}'s take is missing from disk.")
            flows = model.flows_from_previous(board, index)
            trim = take["prefix_frames"] if flows else 0
            # A take continuing a video from Media or an imported take redraws the clip's ending
            # through the VAE, so its overlap is blended in from the real frames rather than cut to:
            # fine detail such as eyes would otherwise jump at the join.
            prev_scene = board["scenes"][index - 1] if index > 0 else None
            prev_take = model.active_take(board, prev_scene) if prev_scene else None
            blend = bool(flows and trim and (model.is_media(prev_scene) or (prev_take and not prev_take.get("latent"))))
            clips.append({"video": take["video"], "trim_frames": trim, "blend": blend})
        return clips

    def render(self, board, destination: Path, *, include_outdated=True, timeout=3600,
               subtitle_preview: Path | None = None, burn_subtitles: bool = False,
               translations: dict[str, list[str]] | None = None, translation_language: str = "") -> Path:
        """Join the takes, trimming repeated overlap and optionally burning captions into the picture."""
        clips = self.clips(board, include_outdated=include_outdated)
        # A video from Media or imported take is fitted to the canvas, which a generated take's size may differ from.
        width, height = model.canvas(board["profile"])
        mixed = any(
            model.is_media(scene) or ((take := model.active_take(board, scene)) and not take.get("latent"))
            for scene in board["scenes"]
        )
        fit = f",scale={width}:{height}:flags=lanczos,setsar=1" if mixed else ""
        sound = f",aresample={clip.AUDIO_RATE},aformat=channel_layouts=stereo" if mixed else ""
        arguments, chains, videos, sounds = [], [], "", ""
        translations = translations or {}
        take_lines = [subtitles.with_translation(subtitles.take_lines(board["takes"][scene["in_use"]]),
                                                 translations.get(scene["in_use"]))
                      for scene in board["scenes"]]
        with_subtitles = any(take_lines)
        captions = []
        elapsed_frames = 0

        def cut(label, output, start, end=None):
            until = f":end_frame={end}" if end is not None else ""
            return f"{label}trim=start_frame={start}{until},setpts=PTS-STARTPTS{fit},format=yuv420p[{output}]"

        for index, item in enumerate(clips):
            video = Path(item["video"]).resolve() if burn_subtitles else item["video"]
            arguments += ["-i", str(video)]
            start = int(item["trim_frames"])
            following = clips[index + 1] if index + 1 < len(clips) else None
            hand_off = int(following["trim_frames"]) if following and following.get("blend") else 0
            parts = [f"b{index}"] + ([f"h{index}"] if item.get("blend") else []) + ([f"t{index}"] if hand_off else [])
            source = f"[{index}:v]"
            if len(parts) > 1:
                chains.append(source + f"split={len(parts)}" + "".join(f"[s{part}]" for part in parts))
                source = None
            total = clip.video_frame_total(item["video"]) if hand_off or with_subtitles else None
            if hand_off and total <= hand_off:
                raise H3Error(f"Scene {index + 1}'s clip is shorter than the overlap the next scene continues.")
            chains.append(cut(source or f"[sb{index}]", f"b{index}", start, total - hand_off if hand_off else None))
            if item.get("blend"):
                chains.append(cut(f"[sh{index}]", f"h{index}", 0, start))
                # Ease from the real frames (A) into the redrawn overlap (B); the frame after it is all new.
                d = start + 1
                chains.append(f"[t{index - 1}][h{index}]blend=all_expr=A+(B-A)*(N+1)*(N+1)*({3 * d}-2*(N+1))/{d ** 3}"
                              f"[x{index}]")
                videos += f"[x{index}]"
            if hand_off:
                chains.append(cut(f"[st{index}]", f"t{index}", total - hand_off))
            videos += f"[b{index}]"
            chains.append(f"[{index}:a]atrim=start={start / model.FPS:.6f},asetpts=PTS-STARTPTS{sound}[a{index}]")
            sounds += f"[a{index}]"
            if with_subtitles:
                captions.extend(subtitles.shifted_lines(take_lines[index], start / model.FPS, total / model.FPS,
                                                        (elapsed_frames - start) / model.FPS))
                elapsed_frames += total - start
        count = videos.count("[")
        graph = (";".join(chains) + f";{videos}concat=n={count}:v=1:a=0[v]"
                 f";{sounds}concat=n={len(clips)}:v=0:a=1[a]")
        srt_text = subtitles.subtitle_text(captions, web=False)
        has_translations = any(line.get("translation") for line in captions)
        if burn_subtitles and not srt_text:
            raise H3Error("The takes in use have no subtitles to burn in.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.stem}.{model.new_id()[:8]}.partial{destination.suffix}")
        with ExitStack() as temporary:
            tracks = []
            cwd = None
            if burn_subtitles:
                # A relative filter path works on Windows as well as Unix.
                cwd = temporary.enter_context(TemporaryDirectory(prefix="h3-storyboard-captions-"))
                lines = [Line(None, line["text"], line["start"], line["end"],
                              word_pieces(line["text"], line.get("words"))) for line in captions]
                below = [Line(None, line["translation"], line["start"], line["end"])
                         for line in captions if line.get("translation")]
                script = ass_subtitles(lines, width=width, height=height, below=below, linger=0)
                (Path(cwd) / "subtitles.ass").write_text(script, encoding="utf-8")
                graph += ";[v]ass=subtitles.ass[burned]"
            elif srt_text:
                folder = temporary.enter_context(TemporaryDirectory(prefix="h3-storyboard-captions-"))
                srt = Path(folder) / "storyboard.srt"
                srt.write_text(srt_text, encoding="utf-8")
                arguments += ["-i", str(srt)]
                tracks = ["-map", f"{len(clips)}:s:0", "-c:s", "mov_text", "-disposition:s:0", "0"]
                if has_translations:
                    bilingual = Path(folder) / "bilingual.srt"
                    size = font_size(100, width)
                    bilingual.write_text(subtitles.subtitle_text(
                        captions, web=False, translation_size=(size, round(size * 0.75))),
                        encoding="utf-8")
                    arguments += ["-i", str(bilingual)]
                    tracks += ["-map", f"{len(clips) + 1}:s:0", "-disposition:s:1", "0",
                               "-metadata:s:s:0", "handler_name=Original subtitles",
                               "-metadata:s:s:1", f"handler_name=Original + {translation_language}"]
            try:
                process = run_media_process(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *arguments,
                     "-filter_complex", graph, "-map", "[burned]" if burn_subtitles else "[v]", "-map", "[a]", *tracks,
                     "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(partial.resolve())],
                    timeout=timeout, partial_outputs=(partial,), cwd=cwd,
                )
            except FileNotFoundError as exc:
                partial.unlink(missing_ok=True)
                raise H3Error("FFmpeg isn't installed in this environment.") from exc
        if process.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise H3Error(f"Joining the scenes failed: {process.stderr.strip()[-600:] or 'ffmpeg reported no output'}")
        partial.replace(destination)
        if subtitle_preview is not None and srt_text:
            partial_track = partial.with_suffix(".vtt")
            try:
                partial_track.write_text(subtitles.subtitle_text(captions, web=True), encoding="utf-8")
                partial_track.replace(subtitle_preview)
            finally:
                partial_track.unlink(missing_ok=True)
        elif subtitle_preview is not None:
            subtitle_preview.unlink(missing_ok=True)
        return destination

    def preview_path(self, storyboard_id) -> Path:
        return self.output_path(f"{TAKE_FOLDER}/{storyboard_id}/preview.mp4")

    def export_path(self, board, *, burn_subtitles=False) -> Path:
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in board["name"])[:60] or "storyboard"
        suffix = "_subtitled" if burn_subtitles else ""
        return self.output_path(f"h3/storyboard_{stem}_{time.strftime('%Y%m%d_%H%M%S')}{suffix}.mp4")
