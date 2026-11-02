"""Owner-scoped storyboard operations over the workspace store, plus export."""

from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.processes import run_media_process

from . import model

TAKE_FOLDER = "h3/.storyboard"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


class StoryboardService:
    def __init__(self, store, runtime: Callable[[], RuntimeConfig]):
        self.store = store
        self.runtime = runtime
        # One process edits a storyboard at a time; each change is read-modify-write.
        self.lock = RLock()

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

    def _copy_cast(self, storyboard_id, name, image) -> dict[str, str]:
        name = str(name).strip()
        source = Path(image)
        if not name:
            raise H3Error("Give each cast picture a name.")
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise H3Error("Cast pictures are PNG, JPEG or WebP images.")
        directory = self.cast_dir(storyboard_id)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{model.new_id()}{source.suffix.lower()}"
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
        allowed = {"title", "prompt", "seconds", "link", "cast", "overlap", "seed", "takes_per_run"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown scene fields: {sorted(unknown)}")

        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
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
        names match the stored ones, so those are kept instead of copied again.
        Returns False when nothing changed.
        """
        if field not in model.MEDIA_FIELDS:
            raise ValueError(f"Unknown scene media {field!r}.")
        directory = self.media_dir(storyboard_id)

        def keep(item):
            if isinstance(item, dict):
                item = item.get("path") or item.get("name")
            source = Path(str(item))
            stored = directory / source.name
            if stored.is_file():
                return str(stored)
            if not source.is_file():
                raise H3Error(f"{source.name} is no longer available. Add it again.")
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{model.new_id()}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            return str(destination)

        if field in ("first_image", "last_image"):
            stored = keep(value) if value else None
        else:
            items = value if isinstance(value, (list, tuple)) else [value] if value else []
            stored = [keep(item) for item in items if item]

        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
            if model.scene_media(scene)[field] == stored:
                return False
            scene[field] = stored
            return True
        return self.change(owner, storyboard_id, edit)[1]

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
        def edit(board):
            if len(board["scenes"]) == 1:
                raise H3Error("A storyboard keeps at least one scene.")
            board["scenes"].pop(self.scene_index(board, scene_id))
        return self.change(owner, storyboard_id, edit)[0]

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
            if scene["in_use"] is None or scene["in_use"] not in board["takes"]:
                scene["in_use"] = take_id
        return self.change(owner, storyboard_id, edit)[0]

    def use_take(self, owner, storyboard_id, scene_id, take_id):
        def edit(board):
            scene = board["scenes"][self.scene_index(board, scene_id)]
            if take_id not in scene["takes"]:
                raise H3Error("That take belongs to another scene.")
            scene["in_use"] = take_id
        return self.change(owner, storyboard_id, edit)[0]

    def set_queue(self, owner, storyboard_id, scene_ids):
        def edit(board):
            board["queue"] = list(scene_ids)
        return self.change(owner, storyboard_id, edit)[0]

    # Versions --------------------------------------------------------------
    def save_version(self, owner, storyboard_id, label=""):
        def edit(board):
            version = {
                "id": model.new_id(), "created": time.time(),
                "label": (str(label).strip() or f"Version {len(board['versions']) + 1}")[:80],
                **model.snapshot(board),
            }
            board["versions"].append(version)
            return version["id"]
        return self.change(owner, storyboard_id, edit)

    def branch(self, owner, storyboard_id, version_id=None, name=None) -> str:
        """A new storyboard from a saved version (or the current one), sharing take files."""
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
        for member in source["cast"]:
            branched["cast"].append(self._copy_cast(branch_id, member["name"], member["image"]))
        self.save(owner, branch_id, branched)
        return branch_id

    def delete(self, owner, storyboard_id):
        with self.lock:
            board = self.load(owner, storyboard_id)
            self.store.delete_storyboard(owner, storyboard_id)
            shutil.rmtree(self.store.root / "storyboards" / storyboard_id, ignore_errors=True)
            # Branches share take and scene media files, so keep any another storyboard still uses.
            used = set()
            for other_owner, other_id in self._all_ids():
                other = self.store.storyboard(other_owner, other_id)
                for take in other["takes"].values():
                    used |= {take["video"], take["latent"], take.get("stage_latent")}
                for scene in other["scenes"]:
                    used |= set(model.media_paths(scene))
            for scene in board["scenes"]:
                for value in model.media_paths(scene):
                    if value not in used:
                        Path(value).unlink(missing_ok=True)
            for take in board["takes"].values():
                for value in (take["video"], take["latent"], take.get("stage_latent")):
                    if value and value not in used:
                        path = Path(value) if Path(value).is_absolute() else self.output_path(value)
                        path.unlink(missing_ok=True)
                        Path(str(path) + ".settings.json").unlink(missing_ok=True)
            preview = self.preview_path(storyboard_id)
            preview.unlink(missing_ok=True)
            try:
                preview.parent.rmdir()  # Only when no branch still keeps takes there.
            except OSError:
                pass

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
            if not Path(take["video"]).is_file():
                raise H3Error(f"Scene {index + 1}'s take is missing from disk.")
            trim = take["prefix_frames"] if model.flows_from_previous(board, index) else 0
            clips.append({"video": take["video"], "trim_frames": trim})
        return clips

    def render(self, board, destination: Path, *, include_outdated=True, timeout=3600) -> Path:
        """Join the takes: flows drop their repeated overlap, cuts join hard."""
        clips = self.clips(board, include_outdated=include_outdated)
        arguments, chains, labels = [], [], ""
        for index, clip in enumerate(clips):
            arguments += ["-i", clip["video"]]
            frames = int(clip["trim_frames"])
            chains.append(
                f"[{index}:v]trim=start_frame={frames},setpts=PTS-STARTPTS[v{index}];"
                f"[{index}:a]atrim=start={frames / model.FPS:.6f},asetpts=PTS-STARTPTS[a{index}]"
            )
            labels += f"[v{index}][a{index}]"
        graph = ";".join(chains) + f";{labels}concat=n={len(clips)}:v=1:a=1[v][a]"
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.stem}.{model.new_id()[:8]}.partial{destination.suffix}")
        process = run_media_process(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *arguments,
             "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(partial)],
            timeout=timeout, partial_outputs=(partial,),
        )
        if process.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise H3Error(f"Joining the scenes failed: {process.stderr.strip()[-600:] or 'ffmpeg reported no output'}")
        partial.replace(destination)
        return destination

    def preview_path(self, storyboard_id) -> Path:
        return self.output_path(f"{TAKE_FOLDER}/{storyboard_id}/preview.mp4")

    def export_path(self, board) -> Path:
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in board["name"])[:60] or "storyboard"
        return self.output_path(f"h3/storyboard_{stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
