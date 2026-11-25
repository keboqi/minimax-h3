"""Tests for cross-server project export, import, routes, and Workshop/Studio wiring."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import threading
import time
import zipfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from h3_app.config import RuntimeConfig
from h3_app.drama3 import sounds as drama3_sounds, speakers as drama3_speakers
from h3_app.drama3.serving.takes import takes_root
from h3_app.dubbing import projects as dubbing_projects, results as dub_results, session as dub_sessions
from h3_app.errors import H3Error
from h3_app.project_transfer import (
    TRANSFER_SCHEMA_VERSION,
    build_auth_headers,
    delete_local_project,
    export_project_archive,
    fetch_remote_projects,
    import_project_archive,
    import_remote_projects,
    list_local_projects,
    normalize_server_url,
    rename_local_project,
)
from h3_app.provenance import read_snapshot, snapshot_path, write_snapshot
from h3_app.remake.project import Projects as RemakeProjects
from h3_app.storyboard import StoryboardService, model as sb_model
from h3_app.workspace_store import WorkspaceStore
from h3_ui.drama3 import visuals as drama3_visuals
from h3_ui.drama3.controls import UPLOAD_SOUND, UPLOAD_VOICE
from h3_ui.project_transfer import install_project_transfer_routes


def _make_workspace(root: Path) -> tuple[RuntimeConfig, WorkspaceStore]:
    root = root.resolve()
    comfy = root / "ComfyUI"
    input_dir = root / "input"
    output_dir = root / "output"
    outputs_dir = root / "gradio_outputs"
    for directory in (comfy, input_dir, output_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    runtime = replace(
        RuntimeConfig.from_environment(
            Path(__file__).resolve().parents[1],
            {
                "COMFY_DIR": str(comfy),
                "MODELS_CONFIG": str(root / "h3_models.json"),
                "GRADIO_OUTPUT_DIR": str(outputs_dir),
            },
        ),
        input_root=input_dir,
        output_root=output_dir,
    )
    store = WorkspaceStore(outputs_dir / ".h3-workspace")
    return runtime, store


@contextmanager
def _using_workspace_env(runtime: RuntimeConfig):
    previous = os.environ.get("H3_WORKSPACE_DIR")
    os.environ["H3_WORKSPACE_DIR"] = str((runtime.outputs_dir / ".h3-workspace").resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("H3_WORKSPACE_DIR", None)
        else:
            os.environ["H3_WORKSPACE_DIR"] = previous


@contextmanager
def _serve_fastapi(app: FastAPI):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started:
        if time.monotonic() > deadline:
            server.should_exit = True
            raise RuntimeError("Timed out starting test uvicorn server")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_normalize_server_url_and_auth_headers():
    assert normalize_server_url("https://demo.modal.run/") == "https://demo.modal.run"
    assert normalize_server_url("https://demo.modal.run/soundstage/?name=x#top") == "https://demo.modal.run"
    assert normalize_server_url("http://127.0.0.1:7860/dubbing") == "http://127.0.0.1:7860"
    assert normalize_server_url("https://proxy.example.com/prism/storyboard/") == "https://proxy.example.com/prism"
    assert normalize_server_url("http://localhost:7860/remake/") == "http://localhost:7860"

    for invalid in ("", "localhost:7860", "ftp://example.com", "https://"):
        with pytest.raises(H3Error):
            normalize_server_url(invalid)

    assert build_auth_headers() == {}
    assert build_auth_headers(" k1 ", " s1 ") == {"Modal-Key": "k1", "Modal-Secret": "s1"}
    with pytest.raises(H3Error):
        build_auth_headers("k1", "")
    with pytest.raises(H3Error):
        build_auth_headers("", "s1")


def test_soundstage_roundtrip_with_mix_presets_stills_and_takes(tmp_path: Path):
    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")

    with _using_workspace_env(src_runtime):
        voice_ref = tmp_path / "alice.wav"
        voice_ref.write_bytes(b"RIFF-alice")
        saved_speaker = drama3_speakers.save_speaker(
            "Alice", voice_ref, "Hello there.", src_runtime, "Warm alto", "StoryCast"
        )
        sound_ref = tmp_path / "thunder.wav"
        sound_ref.write_bytes(b"RIFF-thunder")
        saved_sound = drama3_sounds.save_sound("Thunder", sound_ref, "Distant rumble", 4, src_runtime)

        clips_dir = src_runtime.output_dir / "h3" / ".soundstage" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        (clips_dir / "slot0.wav").write_bytes(b"slot0-audio")
        (clips_dir / "sfx0.wav").write_bytes(b"sfx0-audio")
        (clips_dir / "bgm.mp3").write_bytes(b"bgm-audio")

        h3_dir = src_runtime.output_dir / "h3"
        h3_dir.mkdir(parents=True, exist_ok=True)
        mix_path = h3_dir / "drama3_mix_001.mp3"
        mix_path.write_bytes(b"mix-mp3-bytes")
        take_key = "a" * 32
        write_snapshot(
            mix_path,
            {
                "family": "Drama 3",
                "settings": {
                    "title": "Chapter 1",
                    "chunks": [{"take": take_key, "speaker": 0, "name": "Narrator", "text": "Once."}],
                },
            },
        )
        drama3_visuals.write_plan(mix_path, "Editorial ink", 6, {"shots": [{"beat": "Opening"}], "script_cues": []})
        stills_folder = drama3_visuals.stills_dir(mix_path, src_runtime.output_dir)
        stills_folder.mkdir(parents=True, exist_ok=True)
        (stills_folder / "shot1.jpg").write_bytes(b"jpeg-bytes")
        (stills_folder / ".effect-masks").mkdir(parents=True, exist_ok=True)
        (stills_folder / ".effect-masks" / "mask1.png").write_bytes(b"mask-bytes")
        (stills_folder / drama3_visuals.STILLS_RECORD).write_text(
            json.dumps({"revision": "rev1", "source": mix_path.name, "shots": [{"image": "shot1.jpg"}]}),
            encoding="utf-8",
        )
        src_takes = takes_root(src_runtime)
        src_takes.mkdir(parents=True, exist_ok=True)
        (src_takes / f"{take_key}.wav").write_bytes(b"take-wav-bytes")

        payload = {
            "text": "[Narrator] Hello\n[Alice] Hi",
            "title": "Chapter 1",
            "shown_slots": 2,
            "shown_sound_slots": 2,
            "voice_0_preset": UPLOAD_VOICE,
            "voice_0_audio": "slot0.wav",
            "voice_1_preset": saved_speaker.key,
            "sound_0_preset": UPLOAD_SOUND,
            "sound_0_audio": "sfx0.wav",
            "sound_1_preset": saved_sound.name,
            "background_music": "bgm.mp3",
            "applied_music": {"background_music": "bgm.mp3", "music_enabled": True, "music_volume": 18},
        }
        src_store.save_soundstage_project("Episode 1", payload)
        src_store.mark_soundstage_generated("Episode 1", str(mix_path))

        archive = export_project_archive(
            "soundstage", "Episode 1", tmp_path / "episode1.zip", store=src_store, runtime=src_runtime
        )

    # Pre-seed a local speaker with the same name in target to confirm it is NOT overwritten.
    with _using_workspace_env(dst_runtime):
        existing_voice = tmp_path / "existing_alice.wav"
        existing_voice.write_bytes(b"EXISTING-DO-NOT-OVERWRITE")
        drama3_speakers.save_speaker(
            "Alice", existing_voice, "Keep me.", dst_runtime, "Existing local voice", "StoryCast"
        )
        result = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert result["status"] == "imported"
        assert result["target_name"] == "Episode 1"

        loaded_payload, loaded_result = dst_store.soundstage_project("Episode 1")
        assert loaded_payload["text"] == payload["text"]
        assert loaded_result is not None
        dst_mix = Path(loaded_result)
        assert dst_mix.is_file() and dst_mix.read_bytes() == b"mix-mp3-bytes"
        assert snapshot_path(dst_mix).is_file()
        assert drama3_visuals.read_plan(dst_mix)["style"] == "Editorial ink"
        dst_stills = drama3_visuals.read_stills(dst_mix, dst_runtime.output_dir)
        assert dst_stills is not None and dst_stills["shots"][0]["image"] == "shot1.jpg"
        assert (
            drama3_visuals.stills_dir(dst_mix, dst_runtime.output_dir) / ".effect-masks" / "mask1.png"
        ).read_bytes() == b"mask-bytes"
        assert (takes_root(dst_runtime) / f"{take_key}.wav").read_bytes() == b"take-wav-bytes"
        assert (dst_runtime.output_dir / "h3" / ".soundstage" / "clips" / "slot0.wav").read_bytes() == b"slot0-audio"
        assert (dst_runtime.output_dir / "h3" / ".soundstage" / "clips" / "bgm.mp3").read_bytes() == b"bgm-audio"
        assert drama3_sounds.find_sound("Thunder", dst_runtime).audio.read_bytes() == b"RIFF-thunder"
        assert drama3_speakers.load_speaker(saved_speaker.key, dst_runtime).audio.read_bytes() == b"EXISTING-DO-NOT-OVERWRITE"
        audio_records, audio_total = dst_store.catalog_page(
            kind="Audio", roots=(dst_runtime.output_dir, dst_runtime.outputs_dir)
        )
        assert audio_total == 1
        assert Path(audio_records[0]["path"]).resolve() == dst_mix.resolve()


def test_soundstage_draft_plan_and_conflict_modes(tmp_path: Path):
    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")

    with _using_workspace_env(src_runtime):
        src_store.save_soundstage_project("Story", {"text": "Draft line", "shown_slots": 1})
        plan_audio = src_runtime.output_dir / ".soundstage" / "plans" / (
            hashlib.sha256(b"Story").hexdigest() + ".wav"
        )
        plan_audio.parent.mkdir(parents=True, exist_ok=True)
        drama3_visuals.write_plan(plan_audio, "Clean line", 4, {"shots": [{"beat": "Draft shot"}], "script_cues": []})
        plan_stills = drama3_visuals.stills_dir(plan_audio, src_runtime.output_dir)
        plan_stills.mkdir(parents=True, exist_ok=True)
        (plan_stills / "draft.jpg").write_bytes(b"draft-still")
        (plan_stills / drama3_visuals.STILLS_RECORD).write_text(
            json.dumps({"revision": "r1", "source": plan_audio.name, "shots": [{"image": "draft.jpg"}]}),
            encoding="utf-8",
        )
        archive = export_project_archive(
            "soundstage", "Story", tmp_path / "story.zip", store=src_store, runtime=src_runtime
        )

    with _using_workspace_env(dst_runtime):
        # Pre-seed an existing generated project named "Story".
        old_mix = dst_runtime.output_dir / "h3" / "old.mp3"
        old_mix.parent.mkdir(parents=True, exist_ok=True)
        old_mix.write_bytes(b"old-mix")
        dst_store.save_soundstage_project("Story", {"text": "Old local"})
        dst_store.mark_soundstage_generated("Story", str(old_mix))

        skipped = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="skip")
        assert skipped["status"] == "skipped"
        assert dst_store.soundstage_project("Story")[0]["text"] == "Old local"

        renamed = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert renamed["status"] == "renamed"
        assert renamed["target_name"] == "Story (imported)"
        renamed_plan_audio = dst_runtime.output_dir / ".soundstage" / "plans" / (
            hashlib.sha256(b"Story (imported)").hexdigest() + ".wav"
        )
        assert drama3_visuals.read_plan(renamed_plan_audio)["style"] == "Clean line"
        assert drama3_visuals.read_stills(renamed_plan_audio, dst_runtime.output_dir) is not None

        overwritten = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="overwrite")
        assert overwritten["status"] == "overwritten"
        loaded_data, loaded_result = dst_store.soundstage_project("Story")
        assert loaded_data["text"] == "Draft line"
        assert loaded_result is None


def test_storyboard_roundtrip_preserves_media_takes_and_done_state(tmp_path: Path):
    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")
    src_service = StoryboardService(src_store, lambda: src_runtime)
    dst_service = StoryboardService(dst_store, lambda: dst_runtime)

    cast_pic = tmp_path / "hero.png"
    cast_pic.write_bytes(b"hero-png")
    frame_pic = tmp_path / "first.png"
    frame_pic.write_bytes(b"first-png")

    board_id = src_service.create(
        "owner-a",
        "Space Opera",
        sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        cast=[("Hero", cast_pic)],
    )
    board = src_service.load("owner-a", board_id)
    scene_id = board["scenes"][0]["id"]
    src_service.set_mode("owner-a", board_id, scene_id, sb_model.FRAMES)
    src_service.update_scene("owner-a", board_id, scene_id, prompt="Launch")
    src_service.set_scene_media("owner-a", board_id, scene_id, "first_image", frame_pic)
    board = src_service.load("owner-a", board_id)

    take_id = sb_model.new_id()
    prefix = src_service.take_prefix(board_id, scene_id, take_id)
    take_video = src_service.output_path(f"{prefix}.mp4")
    take_video.parent.mkdir(parents=True, exist_ok=True)
    take_video.write_bytes(b"take-video-bytes")
    write_snapshot(take_video, {"family": "MiniMax H3", "settings": {"prompt": "Launch"}})
    latent_rel = f"{prefix}.safetensors"
    src_service.output_path(latent_rel).write_bytes(b"latent-bytes")
    stage_rel = f"{prefix}_stage.safetensors"
    src_service.output_path(stage_rel).write_bytes(b"stage-bytes")

    inputs = sb_model.scene_inputs(board, 0)
    src_service.record_take(
        "owner-a",
        board_id,
        scene_id,
        take_id,
        video=str(take_video),
        latents={"latent": latent_rel, "stage_latent": stage_rel, "prefix_frames": 0},
        seed=42,
        inputs=inputs,
    )
    assert sb_model.scene_states(src_service.load("owner-a", board_id)) == [sb_model.DONE]

    archive = export_project_archive(
        "storyboard", board_id, tmp_path / "board.zip", store=src_store, runtime=src_runtime
    )

    imported = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
    assert imported["status"] == "imported"
    dst_board = dst_service.load("imported", imported["target_id"])
    assert dst_board["name"] == "Space Opera"
    assert Path(dst_board["cast"][0]["image"]).is_relative_to(dst_store.root)
    assert Path(dst_board["cast"][0]["image"]).read_bytes() == b"hero-png"
    assert Path(dst_board["scenes"][0]["first_image"]).is_relative_to(dst_runtime.output_dir)
    assert Path(dst_board["scenes"][0]["first_image"]).read_bytes() == b"first-png"

    dst_take = dst_board["takes"][take_id]
    assert Path(dst_take["video"]).is_absolute() and Path(dst_take["video"]).is_relative_to(dst_runtime.output_dir)
    assert Path(dst_take["video"]).read_bytes() == b"take-video-bytes"
    assert snapshot_path(dst_take["video"]).is_file()
    assert not Path(dst_take["latent"]).is_absolute()
    assert dst_service.output_path(dst_take["latent"]).read_bytes() == b"latent-bytes"
    assert dst_service.output_path(dst_take["stage_latent"]).read_bytes() == b"stage-bytes"
    assert sb_model.scene_states(dst_board) == [sb_model.DONE]

    # Rename conflict allocates a fresh board id and preserves the original.
    renamed = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
    assert renamed["status"] == "renamed"
    assert renamed["target_id"] != imported["target_id"]
    assert renamed["target_name"] == "Space Opera (imported)"


def test_dubbing_roundtrip_and_rename_conflict(tmp_path: Path):
    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")

    with _using_workspace_env(src_runtime):
        source_video = tmp_path / "clip.mp4"
        source_video.write_bytes(b"source-mp4")
        speaker_wav = tmp_path / "ref.wav"
        speaker_wav.write_bytes(b"speaker-wav")
        speaker = drama3_speakers.save_speaker("Host", speaker_wav, "Welcome", src_runtime)

        session = dub_sessions.create(src_runtime, source_video, video=True)
        dub_out_dir = src_runtime.outputs_dir / "dubbing"
        dub_out_dir.mkdir(parents=True, exist_ok=True)
        dub_mp3 = dub_out_dir / "clip_es.mp3"
        dub_mp4 = dub_out_dir / "clip_es.mp4"
        dub_srt = dub_out_dir / "clip_es.es.srt"
        dub_mp3.write_bytes(b"dub-mp3")
        dub_mp4.write_bytes(b"dub-mp4")
        dub_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")
        write_snapshot(dub_mp3, {"family": "Dub & Translate", "settings": {"session": session.id}})
        write_snapshot(dub_mp4, {"family": "Dub & Translate", "settings": {"session": session.id}})

        player_dir = dub_results.player_folder(dub_mp4)
        player_dir.mkdir(parents=True, exist_ok=True)
        (player_dir / "original.mp3").write_bytes(b"orig-mp3")
        dub_results.save_player(
            player_dir,
            dub_results.Player(
                video=dub_mp4,
                dub=dub_mp3,
                original=player_dir / "original.mp3",
                music=None,
                duration=2.0,
                lines=[{"source": "Hello", "translation": "Hola", "start": 0.0, "end": 1.0}],
                target_language="Spanish",
                source_language="English",
            ),
        )
        session.segments = [{"start": 0.0, "end": 1.0, "speaker": "Host", "source": "Hello", "translation": "Hola"}]
        session.data["pipeline"] = "Local: MOSS Transcribe + Diarize + Hy-MT"
        session.data["speaker_settings"] = {"Host": {"voice": speaker.key, "volume": 100.0}}
        session.data["last_output"] = str(dub_mp4)
        session.save()

        dubbing_projects.save(
            src_runtime,
            "Trailer Dub",
            session.id,
            source_video,
            {"target_language": "Spanish", "pipeline": session.data["pipeline"]},
        )
        archive = export_project_archive(
            "dubbing", "Trailer Dub", tmp_path / "dub.zip", store=src_store, runtime=src_runtime
        )

    with _using_workspace_env(dst_runtime):
        first = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert first["status"] == "imported"
        loaded_data, loaded_session = dubbing_projects.load(dst_runtime, "Trailer Dub")
        assert loaded_session.id != session.id
        assert loaded_session.path("source").read_bytes() == b"source-mp4"
        assert drama3_speakers.load_speaker(speaker.key, dst_runtime).audio.read_bytes() == b"speaker-wav"
        saved_result = dub_results.load_result(loaded_session.data["last_output"])
        assert saved_result is not None and saved_result.video.read_bytes() == b"dub-mp4"
        assert saved_result.player is not None and saved_result.player.dub.read_bytes() == b"dub-mp3"
        roots = (dst_runtime.output_dir, dst_runtime.outputs_dir)
        videos, video_total = dst_store.catalog_page(kind="Video", roots=roots)
        audios, audio_total = dst_store.catalog_page(kind="Audio", roots=roots)
        assert video_total == 1 and Path(videos[0]["path"]).resolve() == saved_result.video.resolve()
        assert audio_total == 1 and Path(audios[0]["path"]).resolve() == saved_result.audio.resolve()

        # Rename conflict creates a new project and avoids clobbering existing dub output files.
        second = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert second["status"] == "renamed"
        assert second["target_name"] == "Trailer Dub (imported)"
        _, renamed_session = dubbing_projects.load(dst_runtime, "Trailer Dub (imported)")
        assert Path(renamed_session.data["last_output"]).name != "clip_es.mp4"
        renamed_result = dub_results.load_result(renamed_session.data["last_output"])
        assert renamed_result is not None and renamed_result.player is not None
        videos, video_total = dst_store.catalog_page(kind="Video", roots=roots)
        audios, audio_total = dst_store.catalog_page(kind="Audio", roots=roots)
        assert video_total == 2
        assert audio_total == 2


def test_remake_roundtrip_and_conflict_modes(tmp_path: Path):
    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")

    src_projects = RemakeProjects(src_runtime.output_dir)
    src_video = tmp_path / "dance.mp4"
    src_video.write_bytes(b"dance-mp4")
    pic = tmp_path / "char.png"
    pic.write_bytes(b"char-png")

    key = "a1b2c3d4e5f607080910"
    project = src_projects.create(
        "owner-a",
        key,
        str(src_video),
        {"name": "Dance Clip", "start": 0.0, "frames": 81, "width": 832, "height": 480, "mix": 0},
        [{"render": [0, 81], "own": [0, 81], "hold": 0}],
    )
    stored_pic = Path(project["folder"]) / "pictures" / "char.png"
    stored_pic.write_bytes(b"char-png")
    take_file = Path(project["folder"]) / "w01_take.mp4"
    take_file.write_bytes(b"remake-take")
    write_snapshot(take_file, {"family": "Viggle Remake", "settings": {"mode": "Swap"}})
    src_projects.edit(
        project,
        lambda current: current["windows"][0].update(
            pictures=[str(stored_pic)], pins=[0], prompt="Spin", take=str(take_file), state="done"
        ),
    )

    archive = export_project_archive(
        "remake", key, tmp_path / "remake.zip", store=src_store, runtime=src_runtime
    )

    imported = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
    assert imported["status"] == "imported"
    dst_projects = RemakeProjects(dst_runtime.output_dir)
    loaded = dst_projects.load(None, key)
    assert loaded is not None
    assert Path(loaded["source"]).is_relative_to(Path(loaded["folder"]))
    assert Path(loaded["source"]).read_bytes() == b"dance-mp4"
    assert Path(loaded["windows"][0]["pictures"][0]).read_bytes() == b"char-png"
    assert Path(loaded["windows"][0]["take"]).read_bytes() == b"remake-take"
    assert snapshot_path(loaded["windows"][0]["take"]).is_file()

    renamed = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
    assert renamed["status"] == "renamed"
    assert renamed["target_id"] != key
    assert renamed["target_name"] == "Dance Clip (imported)"


@pytest.mark.parametrize("kind, metadata", [
    ("remake", {"key": "subfolder/key"}),
    ("dubbing", {"last_output_name": "movie.mp4", "last_output_stem": "other"}),
    ("dubbing", {"last_output_name": "subfolder/movie.mp4"}),
])
def test_import_rejects_invalid_project_identifiers(tmp_path, kind, metadata):
    runtime, store = _make_workspace(tmp_path / "target")
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1, "kind": kind, "id": "test"}))
        zf.writestr("project.json", json.dumps(metadata))
    with pytest.raises(H3Error, match="Invalid|Unsafe|Inconsistent"):
        import_project_archive(archive, store=store, runtime=runtime)


def test_archive_security_rejects_traversal_and_symlinks(tmp_path: Path):
    runtime, store = _make_workspace(tmp_path / "target")

    bad_zip = tmp_path / "traversal.zip"
    with zipfile.ZipFile(bad_zip, "w") as archive:
        archive.writestr("../escape.txt", b"bad")
    with pytest.raises(H3Error, match="Unsafe archive"):
        import_project_archive(bad_zip, store=store, runtime=runtime)

    symlink_zip = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink_zip, "w") as archive:
        info = zipfile.ZipInfo("clips/link.wav")
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, b"/etc/passwd")
    with pytest.raises(H3Error, match="Unsafe archive"):
        import_project_archive(symlink_zip, store=store, runtime=runtime)


def test_end_to_end_http_transfer_with_auth_and_workshop_commit(tmp_path: Path, monkeypatch):
    src_runtime, src_store = _make_workspace(tmp_path / "workshop")
    dst_runtime, dst_store = _make_workspace(tmp_path / "studio")

    with _using_workspace_env(src_runtime):
        src_store.save_soundstage_project("Remote Script", {"text": "[Host] Line one", "title": "Remote"})
        board_service = StoryboardService(src_store, lambda: src_runtime)
        board_id = board_service.create(
            "workshop-owner",
            "Remote Board",
            sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        )

        # Remote Workshop app protected by Modal Proxy Auth headers.
        remote_app = FastAPI()

        @remote_app.middleware("http")
        async def require_modal_auth(request: Request, call_next):
            if request.headers.get("Modal-Key") != "mk" or request.headers.get("Modal-Secret") != "ms":
                return JSONResponse({"detail": "Unauthorized"}, status_code=403)
            return await call_next(request)

        monkeypatch.setenv("H3_CPU_ONLY", "1")
        install_project_transfer_routes(remote_app, store=src_store, runtime_config=lambda: src_runtime)

        # Target Studio app with on_imported callback tracking commit.
        commit_mock = Mock()
        local_app = FastAPI()
        install_project_transfer_routes(
            local_app, store=dst_store, runtime_config=lambda: dst_runtime, on_imported=commit_mock
        )

        with _serve_fastapi(remote_app) as remote_url, _serve_fastapi(local_app) as local_url:
            with pytest.raises(H3Error, match="Authentication failed"):
                fetch_remote_projects(remote_url)

            listing = fetch_remote_projects(
                f"{remote_url}/soundstage/", headers=build_auth_headers("mk", "ms")
            )
            assert listing["app"] == "Prism Workshop"
            assert listing["schema_version"] == TRANSFER_SCHEMA_VERSION
            assert {(p["kind"], p["id"]) for p in listing["projects"]} == {
                ("soundstage", "Remote Script"),
                ("storyboard", board_id),
            }

            with _using_workspace_env(dst_runtime):
                resp = httpx.post(
                    f"{local_url}/api/projects/transfer/import",
                    json={
                        "url": f"{remote_url}/dubbing",
                        "items": [
                            {"kind": "soundstage", "id": "Remote Script"},
                            {"kind": "storyboard", "id": board_id},
                        ],
                        "conflict": "rename",
                        "modal_key": "mk",
                        "modal_secret": "ms",
                    },
                    timeout=30.0,
                )
                assert resp.status_code == 200
                results = resp.json()["results"]
                assert [r["status"] for r in results] == ["imported", "imported"]
                commit_mock.assert_called_once()
                assert dst_store.soundstage_project("Remote Script")[0]["title"] == "Remote"
                assert dst_store.storyboard(None, board_id)["name"] == "Remote Board"


def test_workshop_ui_wires_transfer_routes_and_volume_commit(tmp_path: Path, monkeypatch):
    import h3_workshop as workshop_module
    from h3_app import media_catalog
    from h3_app.jobs import JOBS
    from h3_ui import application as studio_app

    previous_catalog = media_catalog.get_media_catalog()
    previous_store = JOBS.store
    runtime, store = _make_workspace(tmp_path / "ws")
    volume = Mock()
    manager = workshop_module.ModelManager(tmp_path / "ws", volume)
    with monkeypatch.context() as m:
        m.setenv("H3_CPU_ONLY", "1")
        ui = workshop_module.build_ui(manager, (runtime, store))
    try:
        installers = getattr(ui, "h3_routes", ())
        transfer_installer = next(
            fn for fn in installers if getattr(fn, "func", None) is install_project_transfer_routes
        )
        assert transfer_installer.keywords["on_imported"] == volume.commit
        labels = {c.get("props", {}).get("label") for c in ui.get_config_file()["components"]}
        assert "Remote Studio or Workshop URL" in labels
        assert "Local project" in labels
    finally:
        ui.close()
        media_catalog._CATALOG = previous_catalog
        JOBS.store = previous_store
        studio_app.set_runtime_override(None)


def test_gradio_mounted_server_serves_transfer_routes_and_ui_callbacks(tmp_path: Path, monkeypatch):
    import gradio as gr
    from h3_app import media_catalog
    from h3_app.catalog import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, LTX25_WORKFLOWS, VIDEO_EXTENSIONS
    from h3_app.jobs import JOBS
    from h3_app.server import ServerConfig, build_server
    from h3_ui import application as studio_app
    from h3_ui.project_transfer import build_project_import_view

    src_runtime, src_store = _make_workspace(tmp_path / "studio_src")
    dst_runtime, dst_store = _make_workspace(tmp_path / "workshop_dst")

    previous_catalog = media_catalog.get_media_catalog()
    previous_store = JOBS.store
    try:
        with _using_workspace_env(src_runtime):
            src_store.save_soundstage_project("From Studio", {"text": "[Host] Hello from Studio", "title": "Studio"})

            JOBS.configure(src_store)
            with gr.Blocks() as remote_demo:
                gr.Markdown("Studio")
            remote_demo.h3_routes = (
                lambda app: install_project_transfer_routes(
                    app, store=src_store, runtime_config=lambda: src_runtime
                ),
            )
            monkeypatch.delenv("H3_CPU_ONLY", raising=False)
            remote_asgi = build_server(
                remote_demo,
                [str(src_runtime.output_dir), str(src_runtime.outputs_dir)],
                ServerConfig(
                    src_runtime.comfy_url,
                    src_runtime.output_dir,
                    src_runtime.outputs_dir,
                    LTX25_WORKFLOWS,
                    src_runtime.workflow_dir,
                    VIDEO_EXTENSIONS,
                    IMAGE_EXTENSIONS,
                    AUDIO_EXTENSIONS,
                    "",
                ),
            )

            with _serve_fastapi(remote_asgi) as remote_url:
                commit_mock = Mock()
                with gr.Blocks() as local_demo:
                    with gr.Column() as system_col:
                        build_project_import_view(
                            system_col,
                            store=dst_store,
                            runtime_config=lambda: dst_runtime,
                            on_imported=commit_mock,
                        )
                config = local_demo.get_config_file()
                def button_fn(label: str):
                    for dep in config["dependencies"]:
                        for target in dep.get("targets", []):
                            cid = target[0] if isinstance(target, (list, tuple)) else target
                            block = local_demo.blocks.get(cid)
                            if getattr(block, "value", None) == label:
                                return local_demo.fns[dep["id"]].fn
                    raise AssertionError(f"Missing button dependency for {label!r}")

                load_fn = button_fn("Load projects")
                select_all_fn = button_fn("Select all")
                import_fn = button_fn("Import selected projects")

                proj_update, choices, status = load_fn(remote_url, "", "")
                assert "Prism Studio" in status
                assert len(choices) == 1
                selected = select_all_fn(choices)
                assert len(selected) == 1

                with _using_workspace_env(dst_runtime):
                    import_status = import_fn(remote_url, "", "", selected, "rename")
                    assert "1 imported" in import_status
                    commit_mock.assert_called_once()
                    assert dst_store.soundstage_project("From Studio")[0]["title"] == "Studio"
                local_demo.close()
            remote_demo.close()
    finally:
        media_catalog._CATALOG = previous_catalog
        JOBS.store = previous_store
        studio_app.set_runtime_override(None)


def test_soundstage_transfer_preserves_sam_segments_across_drama3_and_legacy_manifest_keys(tmp_path: Path):
    from PIL import Image

    from h3_app.drama3.visuals import FPS, Shot
    from h3_ui.drama3 import effect_regions, effects_pass

    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")

    with _using_workspace_env(src_runtime):
        drama3_dir = src_runtime.output_dir / "drama3"
        drama3_dir.mkdir(parents=True, exist_ok=True)
        mix_path = drama3_dir / "sam_story_20261123.mp3"
        mix_path.write_bytes(b"drama3-mix-bytes")
        write_snapshot(
            mix_path,
            {"family": "Drama 3", "settings": {"title": "SAM Story", "duration": 4.0, "chunks": [], "sounds": []}},
        )

        drama3_visuals.write_plan(
            mix_path,
            drama3_visuals.STYLE_PRESETS[drama3_visuals.DEFAULT_STYLE],
            1,
            {
                "characters": {"elena": "Elena, a woman in a navy coat"},
                "shots": [
                    {
                        "id": "shot_1",
                        "start": 0,
                        "transition": "cut",
                        "beat": "Window",
                        "references": ["elena"],
                        "prompt": "Elena stands by the window.",
                        "effects": [],
                    }
                ],
                "script_cues": [],
            },
        )
        still_src = tmp_path / "shot1.png"
        Image.new("RGB", (64, 36), "teal").save(still_src)
        shots = [Shot(0, 4 * FPS, False, "Elena stands by the window.", "Window", id="shot_1")]
        drama3_visuals.keep_stills(mix_path, src_runtime.output_dir, shots, [still_src], {}, total_frames=4 * FPS)
        stills_folder = drama3_visuals.stills_dir(mix_path, src_runtime.output_dir)
        image_name = drama3_visuals.read_stills(mix_path, src_runtime.output_dir)["shots"][0]["image"]

        mask_dir = stills_folder / ".effect-masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_name = f"{image_name}.{'b' * 32}.region_1.png"
        Image.new("RGBA", (64, 36), (255, 255, 255, 200)).save(mask_dir / mask_name)
        legacy_regions = [
            {"id": "region_1", "label": "window", "mask": mask_name, "box": [0.1, 0.1, 0.8, 0.8]},
        ]
        (mask_dir / f"{image_name}.0123456789abcdef.json").write_text(
            json.dumps(legacy_regions), encoding="utf-8"
        )

        subjects = [{"key": "elena", "kind": "characters", "anchor": "Elena, a woman in a navy coat"}]
        rows = [
            {
                "shot_id": "shot_1",
                "index": 0,
                "image_file": image_name,
                "start": 0.0,
                "end": 4.0,
                "beat": "Window",
                "prompt": "Elena stands by the window.",
                "subjects": ["elena"],
                "effects": [],
            }
        ]
        expected_labels = effect_regions.concepts(rows[0], {s["key"]: s for s in subjects})
        canonical_manifest_name = effect_regions.manifest_path(stills_folder, image_name, expected_labels).name

        effect_stack = [{"preset": "rain", "mode": "inside", "region_id": "region_1"}]
        payload = {
            "text": "[Elena] Look at the storm.",
            "title": "SAM Story",
            "shown_slots": 1,
            "video_effect_stacks": {"shot_1": {"image": image_name, "effects": effect_stack}},
        }
        src_store.save_soundstage_project("SAM Story", payload)
        # Store a stale directory prefix to verify export resolves mix files moved into output/drama3.
        src_store.mark_soundstage_generated("SAM Story", str(tmp_path / "relocated" / "drama3" / mix_path.name))

        archive = export_project_archive(
            "soundstage", "SAM Story", tmp_path / "sam_story.zip", store=src_store, runtime=src_runtime
        )
        with zipfile.ZipFile(archive, "r") as zf:
            assert f"stills/.effect-masks/{canonical_manifest_name}" in zf.namelist()

    with _using_workspace_env(dst_runtime):
        result = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert result["status"] == "imported"
        loaded_payload, loaded_result = dst_store.soundstage_project("SAM Story")
        assert loaded_result is not None
        dst_mix = Path(loaded_result)
        assert dst_mix.is_file() and dst_mix.read_bytes() == b"drama3-mix-bytes"
        dst_stills_dir = drama3_visuals.stills_dir(dst_mix, dst_runtime.output_dir)
        assert (dst_stills_dir / ".effect-masks" / canonical_manifest_name).is_file()

        items = effects_pass.inventory(dst_mix, dst_runtime.output_dir, rows, subjects, loaded_payload)
        assert len(items) == 1
        assert items[0]["regions"] == legacy_regions
        assert items[0]["applied"] == effect_stack

        public = effects_pass.public_pass("SAM Story", items, [], loaded_payload)
        shot = public["shots"][0]
        assert shot["segmented"] is True
        assert [r["id"] for r in shot["regions"]] == ["region_1"]
        assert shot["effects"] == effect_stack

        # Pre-create a conflicting drama3 mix on target to ensure conflict="rename" avoids colliding stems/stills.
        dst_drama3 = dst_runtime.output_dir / "drama3"
        dst_drama3.mkdir(parents=True, exist_ok=True)
        (dst_drama3 / mix_path.name).write_bytes(b"existing-drama3-mix")
        dst_mix.unlink()
        dst_stills = drama3_visuals.stills_dir(dst_mix, dst_runtime.output_dir)
        assert dst_stills.is_dir()

        renamed = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert renamed["status"] == "renamed"
        _, renamed_result = dst_store.soundstage_project(renamed["target_name"])
        assert Path(renamed_result).name == "sam_story_20261123_imported.mp3"


def test_local_project_rename_and_delete_all_four_kinds(tmp_path: Path):
    from h3_app.project_transfer import _soundstage_plan_audio

    runtime, store = _make_workspace(tmp_path / "workspace")
    with _using_workspace_env(runtime):
        # 1. Soundstage (with draft plan & stills + duplicate guard)
        store.save_soundstage_project("SS One", {"text": "[Host] Hello", "title": "One"})
        store.save_soundstage_project("SS Two", {"text": "[Host] Two", "title": "Two"})
        old_plan_audio = _soundstage_plan_audio(runtime, "SS One")
        old_plan_audio.parent.mkdir(parents=True, exist_ok=True)
        drama3_visuals.write_plan(old_plan_audio, "Editorial ink", 2, {"shots": [], "script_cues": []})
        old_stills = drama3_visuals.stills_dir(old_plan_audio, runtime.output_dir)
        old_stills.mkdir(parents=True, exist_ok=True)
        (old_stills / "shot1.jpg").write_bytes(b"jpg-bytes")
        (old_stills / drama3_visuals.STILLS_RECORD).write_text(
            json.dumps({"revision": "rev1", "source": old_plan_audio.name, "shots": [{"image": "shot1.jpg"}]}),
            encoding="utf-8",
        )

        with pytest.raises(H3Error, match="already exists"):
            rename_local_project("soundstage", "SS One", "SS Two", store=store, runtime=runtime)
        with pytest.raises(H3Error, match="Enter a project name"):
            rename_local_project("soundstage", "SS One", "   ", store=store, runtime=runtime)

        ss_ren = rename_local_project("soundstage", "SS One", "SS Renamed", store=store, runtime=runtime)
        assert ss_ren["id"] == "SS Renamed"
        new_plan_audio = _soundstage_plan_audio(runtime, "SS Renamed")
        assert drama3_visuals.plan_path(new_plan_audio).is_file()
        assert not drama3_visuals.plan_path(old_plan_audio).exists()
        assert not old_stills.exists()
        stills_rec = json.loads(
            (drama3_visuals.stills_dir(new_plan_audio, runtime.output_dir) / drama3_visuals.STILLS_RECORD).read_text(
                encoding="utf-8"
            )
        )
        assert stills_rec["source"] == new_plan_audio.name

        delete_local_project("soundstage", "SS Renamed", store=store, runtime=runtime)
        with pytest.raises(ValueError):
            store.soundstage_project("SS Renamed")
        assert not drama3_visuals.plan_path(new_plan_audio).exists()
        assert not drama3_visuals.stills_dir(new_plan_audio, runtime.output_dir).exists()

        # 2. Storyboard
        board_service = StoryboardService(store, lambda: runtime)
        board_id = board_service.create(
            "owner-1",
            "Board One",
            sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        )
        cast_dir = store.root / "storyboards" / board_id / "cast"
        cast_dir.mkdir(parents=True, exist_ok=True)
        (cast_dir / "hero.png").write_bytes(b"png")

        sb_ren = rename_local_project("storyboard", board_id, "Board Renamed", store=store, runtime=runtime)
        assert sb_ren["name"] == "Board Renamed"
        assert store.storyboard(None, board_id)["name"] == "Board Renamed"

        delete_local_project("storyboard", board_id, store=store, runtime=runtime)
        assert not (store.root / "storyboards" / board_id).exists()
        with pytest.raises(ValueError):
            store.storyboard(None, board_id)

        # 3. Dubbing (with revision dir + duplicate guard)
        dub_src = tmp_path / "dub.wav"
        dub_src.write_bytes(b"RIFF-dub")
        sess = dubbing_projects.save(runtime, "Dub One", None, str(dub_src), {"target_language": "English"})
        dubbing_projects.save(runtime, "Dub Two", None, str(dub_src), {"target_language": "Japanese"})
        dub_path = dubbing_projects._path(runtime, "Dub One")
        rev_dir = dub_path.with_suffix("")
        rev_dir.mkdir(parents=True, exist_ok=True)
        (rev_dir / "rev.json").write_text("{}", encoding="utf-8")

        with pytest.raises(H3Error, match="already exists"):
            rename_local_project("dubbing", "Dub One", "Dub Two", store=store, runtime=runtime)

        dub_ren = rename_local_project("dubbing", "Dub One", "Dub Renamed", store=store, runtime=runtime)
        assert dub_ren["id"] == "Dub Renamed"
        assert not dub_path.exists()
        assert not rev_dir.exists()
        new_dub_path = dubbing_projects._path(runtime, "Dub Renamed")
        assert (new_dub_path.with_suffix("") / "rev.json").is_file()
        assert dubbing_projects.load(runtime, "Dub Renamed")[1].id == sess.id

        delete_local_project("dubbing", "Dub Renamed", store=store, runtime=runtime)
        assert "Dub Renamed" not in dubbing_projects.names(runtime)
        assert not new_dub_path.exists()
        assert not new_dub_path.with_suffix("").exists()

        # 4. Remake
        remake_src = tmp_path / "clip.mp4"
        remake_src.write_bytes(b"mp4-source")
        remake_projects = RemakeProjects(runtime.output_dir)
        created = remake_projects.create(
            "owner-1",
            "remake_key_1",
            str(remake_src),
            {"name": "Remake One"},
            [{"render": (0, 24)}],
        )
        rm_ren = rename_local_project("remake", "remake_key_1", "Remake Renamed", store=store, runtime=runtime)
        assert rm_ren["name"] == "Remake Renamed"
        assert remake_projects.load(None, "remake_key_1")["name"] == "Remake Renamed"

        delete_local_project("remake", "remake_key_1", store=store, runtime=runtime)
        assert remake_projects.load(None, "remake_key_1") is None
        assert not Path(created["folder"]).exists()


def test_local_project_manage_routes_and_ui_callbacks(tmp_path: Path):
    import gradio as gr
    from fastapi.testclient import TestClient

    from h3_ui.project_transfer import build_project_manage_view

    runtime, store = _make_workspace(tmp_path / "manage_ws")
    with _using_workspace_env(runtime):
        store.save_soundstage_project("Alpha Stage", {"text": "[Host] Hi", "title": "Alpha"})
        board_id = StoryboardService(store, lambda: runtime).create(
            "owner-1",
            "Beta Board",
            sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        )
        dub_src = tmp_path / "dub.wav"
        dub_src.write_bytes(b"RIFF-dub")
        dubbing_projects.save(runtime, "Gamma Dub", None, str(dub_src), {"target_language": "English"})

        changed_mock = Mock()
        app = FastAPI()
        install_project_transfer_routes(
            app, store=store, runtime_config=lambda: runtime, on_imported=changed_mock
        )
        client = TestClient(app)

        resp = client.post(
            "/api/projects/manage/rename",
            json={"kind": "storyboard", "id": board_id, "name": "Beta Board Renamed"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Beta Board Renamed"
        assert store.storyboard(None, board_id)["name"] == "Beta Board Renamed"
        assert changed_mock.call_count == 1

        resp = client.post(
            "/api/projects/manage/delete",
            json={"kind": "storyboard", "id": board_id},
        )
        assert resp.status_code == 200
        assert changed_mock.call_count == 2

        missing = client.post(
            "/api/projects/manage/delete",
            json={"kind": "storyboard", "id": board_id},
        )
        assert missing.status_code == 404

        with gr.Blocks() as demo:
            with gr.Column() as system_col:
                build_project_manage_view(
                    system_col,
                    store=store,
                    runtime_config=lambda: runtime,
                    on_changed=changed_mock,
                )
        config = demo.get_config_file()

        def callback_for(*, label: str | None = None, value: str | None = None):
            for dep in config["dependencies"]:
                for target in dep.get("targets", []):
                    cid = target[0] if isinstance(target, (list, tuple)) else target
                    block = demo.blocks.get(cid)
                    if label is not None and getattr(block, "label", None) == label:
                        return demo.fns[dep["id"]].fn
                    if value is not None and getattr(block, "value", None) == value:
                        return demo.fns[dep["id"]].fn
            raise AssertionError(f"Missing callback for label={label!r}, value={value!r}")

        filter_fn = callback_for(label="Project type")
        select_fn = callback_for(label="Local project")
        refresh_fn = callback_for(value="Refresh list")
        rename_fn = callback_for(value="Rename project")
        delete_fn = callback_for(value="Remove project")

        drop_u, dub_name, names_map, msg = filter_fn("dubbing", None)
        assert dub_name == "Gamma Dub"
        assert "Showing 1 project" in msg
        dub_key = drop_u["value"]
        assert select_fn(dub_key, names_map) == "Gamma Dub"

        drop_u, renamed_name, names_map, msg = rename_fn("dubbing", dub_key, "Gamma Dub v2", names_map)
        assert renamed_name == "Gamma Dub v2"
        assert 'Renamed Dubbing "Gamma Dub" to "Gamma Dub v2"' in msg
        assert changed_mock.call_count == 3

        drop_u, after_del_name, names_map, msg = delete_fn("dubbing", drop_u["value"], names_map)
        assert drop_u["value"] is None
        assert after_del_name == ""
        assert 'Removed Dubbing "Gamma Dub v2"' in msg
        assert changed_mock.call_count == 4

        drop_u, rem_name, names_map, msg = refresh_fn("all", None)
        assert rem_name == "Alpha Stage"
        assert len(list_local_projects(store, runtime)) == 1
        demo.close()


def test_dubbing_transfer_preserves_lipsync_plan_overrides_and_skips_caches(tmp_path: Path):
    from h3_app.dubbing.lip_sync import (
        PLAN_FILE,
        REPORT_FILE,
        WINDOW_CACHE_DIR,
        build_lip_sync_plan_payload,
        load_lip_sync_plan,
        load_manual_overrides,
        save_manual_overrides,
    )
    from h3_app.dubbing.lip_sync_match import (
        FaceTrack,
        Shot,
        ShotSpeakerAssignment,
        SourceScan,
        TrackObservation,
        video_fingerprint,
    )

    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")

    with _using_workspace_env(src_runtime):
        source_video = tmp_path / "talk.mp4"
        source_video.write_bytes(b"source-talk-mp4")
        session = dub_sessions.create(src_runtime, source_video, video=True)
        (session.root / "vocals.wav").write_bytes(b"RIFF-vocals-wav")
        (session.root / "takes").mkdir(parents=True, exist_ok=True)
        (session.root / "takes" / "0.wav").write_bytes(b"RIFF-take-wav")
        # Regenerable caches and transient intermediates must not bloat transfers.
        (session.root / "timed").mkdir(parents=True, exist_ok=True)
        (session.root / "timed" / "0.wav").write_bytes(b"RIFF-timed-cache")
        (session.root / "speech-16k.wav").write_bytes(b"RIFF-temp-16k")
        (session.root / "speech.wav").write_bytes(b"RIFF-temp-speech")
        (session.root / "mix.wav").write_bytes(b"RIFF-temp-mix")

        dub_out_dir = src_runtime.outputs_dir / "dubbing"
        dub_out_dir.mkdir(parents=True, exist_ok=True)
        dub_mp3 = dub_out_dir / "talk_zh.mp3"
        dub_mp4 = dub_out_dir / "talk_zh.mp4"
        dub_lipsync = dub_out_dir / "talk_zh.lipsync.mp4"
        dub_mp3.write_bytes(b"dub-mp3-bytes")
        dub_mp4.write_bytes(b"dub-mp4-video-bytes-for-fingerprint")
        dub_lipsync.write_bytes(b"dub-lipsync-video-bytes")
        for media_file in (dub_mp3, dub_mp4, dub_lipsync):
            write_snapshot(media_file, {"family": "Dub & Translate", "settings": {"session": session.id}})

        player_dir = dub_results.player_folder(dub_mp4)
        player_dir.mkdir(parents=True, exist_ok=True)
        (player_dir / "original.mp3").write_bytes(b"orig-mp3")
        lines = [{"speaker": "Host", "start": 0.0, "end": 1.2, "source": "Hello", "translation": "你好"}]
        spoken = {"0": [0.0, 1.2]}
        (player_dir / "alignment.json").write_text(json.dumps({"lines": lines, "spoken": spoken}), encoding="utf-8")
        (player_dir / "alignment.npz").write_bytes(b"npz-clips")
        dub_results.save_player(
            player_dir,
            dub_results.Player(
                video=dub_mp4,
                dub=dub_mp3,
                original=player_dir / "original.mp3",
                music=None,
                duration=2.0,
                lines=[{"source": "Hello", "translation": "你好", "start": 0.0, "end": 1.2}],
                target_language="Chinese",
                source_language="English",
            ),
        )
        track = FaceTrack(7, 0, 0.0, 1.2, (TrackObservation(0.6, (0.2, 0.2, 0.3, 0.4), 0.9),))
        scan = SourceScan("src-fp", 2.0, 640, 360, 24.0, (), (Shot(0, 0.0, 2.0, (track,)),), "yolo-mediapipe")
        assignments = {(0, "Host"): ShotSpeakerAssignment(0, "Host", "manual", 7, (0.35, 0.4), 0.6, "manual", 1.0)}
        plan = build_lip_sync_plan_payload(
            dub=dub_mp4, scan=scan, lines=lines, spoken=spoken, assignments=assignments, windows=[]
        )
        (player_dir / PLAN_FILE).write_text(json.dumps(plan), encoding="utf-8")
        save_manual_overrides(dub_mp4, {(0, "Host"): 7})
        (player_dir / REPORT_FILE).write_text(
            json.dumps({"output": dub_lipsync.name, "total_windows": 1, "backend": "yolo-mediapipe"}),
            encoding="utf-8",
        )
        (player_dir / WINDOW_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        (player_dir / WINDOW_CACHE_DIR / "win_cached.mp4").write_bytes(b"cached-window-mp4")

        session.segments = lines
        session.data["last_output"] = str(dub_mp4)
        session.save()
        dubbing_projects.save(src_runtime, "Talk Dub", session.id, source_video, {"target_language": "Chinese"})
        archive = export_project_archive("dubbing", "Talk Dub", tmp_path / "talk_dub.zip", store=src_store, runtime=src_runtime)

    with zipfile.ZipFile(archive, "r") as zf:
        names = set(zf.namelist())
        assert f"result/{dub_lipsync.name}" in names
        assert f"result_player/{PLAN_FILE}" in names
        assert f"result_player/{REPORT_FILE}" in names
        assert f"result_player/{WINDOW_CACHE_DIR}/win_cached.mp4" not in names
        assert "session/vocals.wav" in names
        assert "session/takes/0.wav" in names
        assert "session/timed/0.wav" not in names
        assert "session/speech-16k.wav" not in names
        assert "session/speech.wav" not in names
        assert "session/mix.wav" not in names
        assert zf.getinfo("session/vocals.wav").compress_type == zipfile.ZIP_STORED
        assert zf.getinfo("session/takes/0.wav").compress_type == zipfile.ZIP_STORED

    with _using_workspace_env(dst_runtime):
        first = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert first["status"] == "imported"
        _, imported_session = dubbing_projects.load(dst_runtime, "Talk Dub")
        imported_dub = Path(imported_session.data["last_output"])
        # Simulate mtime drift on destination filesystem; fingerprint still matches.
        orig_stat = imported_dub.stat()
        os.utime(imported_dub, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns + 5_000_000_000))
        loaded_plan = load_lip_sync_plan(imported_dub)
        assert loaded_plan is not None
        assert loaded_plan["video"] == imported_dub.name
        assert loaded_plan["video_fingerprint"] == video_fingerprint(imported_dub)
        assert loaded_plan["assignments"][0]["track_id"] == 7
        assert load_manual_overrides(imported_dub) == {(0, "Host"): 7}

        roots = (dst_runtime.output_dir, dst_runtime.outputs_dir)
        videos, video_total = dst_store.catalog_page(kind="Video", roots=roots)
        audios, audio_total = dst_store.catalog_page(kind="Audio", roots=roots)
        assert video_total == 2 and audio_total == 1
        assert {Path(row["path"]).name for row in videos} == {imported_dub.name, f"{imported_dub.stem}.lipsync.mp4"}

        # Conflict rename updates plan video target and report output filename.
        second = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert second["status"] == "renamed"
        _, renamed_session = dubbing_projects.load(dst_runtime, second["target_name"])
        renamed_dub = Path(renamed_session.data["last_output"])
        assert renamed_dub.name != imported_dub.name
        renamed_plan = load_lip_sync_plan(renamed_dub)
        assert renamed_plan is not None and renamed_plan["video"] == renamed_dub.name
        renamed_report = json.loads((dub_results.player_folder(renamed_dub) / REPORT_FILE).read_text(encoding="utf-8"))
        assert renamed_report["output"] == f"{renamed_dub.stem}.lipsync.mp4"

    with _using_workspace_env(src_runtime):
        dub_mp4.write_bytes(b"replacement video with a different fingerprint")
        export_project_archive("dubbing", "Talk Dub", archive, store=src_store, runtime=src_runtime)
    with _using_workspace_env(dst_runtime):
        stale_subtitle = imported_dub.with_suffix(".obsolete.srt")
        stale_subtitle.touch()
        cached_audio = imported_session.root / "timed" / "0.wav"
        cached_audio.parent.mkdir(exist_ok=True)
        cached_audio.write_bytes(b"stretched take")
        import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="overwrite")
        assert load_lip_sync_plan(imported_dub) is None
        assert cached_audio.read_bytes() == b"stretched take"
        assert not stale_subtitle.exists()


def test_storyboard_references_cast_and_media_scenes_stay_done_after_transfer(tmp_path: Path):
    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")
    src_service = StoryboardService(src_store, lambda: src_runtime)
    dst_service = StoryboardService(dst_store, lambda: dst_runtime)

    cast_pic = tmp_path / "hero.png"
    cast_pic.write_bytes(b"hero-cast-png")
    media_clip = tmp_path / "intro_clip.mp4"
    media_clip.write_bytes(b"intro-clip-mp4")

    board_id = src_service.create(
        "owner-a",
        "Cast & Media Board",
        sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        cast=[("Hero", cast_pic)],
    )
    board = src_service.load("owner-a", board_id)
    scene0_id = board["scenes"][0]["id"]
    # Scene 0 uses REFERENCES mode with Cast (records [name, filename] in take inputs).
    src_service.set_mode("owner-a", board_id, scene0_id, sb_model.REFERENCES)
    src_service.update_scene("owner-a", board_id, scene0_id, prompt="Hero walks in", cast=["Hero"])

    # Scene 1 is a VIDEO_FROM_MEDIA scene (records video basename in take inputs).
    stored_clip = src_service.media_dir(board_id) / "intro_clip.mp4"
    stored_clip.parent.mkdir(parents=True, exist_ok=True)
    stored_clip.write_bytes(b"intro-clip-mp4")
    media_scene = {
        **sb_model.new_media_scene("Intro Cut"),
        "video": str(stored_clip),
        "video_info": {"seconds": 4.0, "fps": 24.0, "width": 1280, "height": 720},
        "video_in": 0.0,
        "video_out": 4.0,
    }
    src_service.change("owner-a", board_id, lambda b: b["scenes"].append(media_scene))
    board = src_service.load("owner-a", board_id)
    scene1_id = board["scenes"][1]["id"]

    for idx, sc_id in enumerate((scene0_id, scene1_id)):
        take_id = sb_model.new_id()
        prefix = src_service.take_prefix(board_id, sc_id, take_id)
        take_video = src_service.output_path(f"{prefix}.mp4")
        take_video.parent.mkdir(parents=True, exist_ok=True)
        take_video.write_bytes(f"video-{idx}".encode())
        board = src_service.load("owner-a", board_id)
        src_service.record_take(
            "owner-a",
            board_id,
            sc_id,
            take_id,
            video=str(take_video),
            latents={"latent": None},
            seed=10 + idx,
            inputs=sb_model.scene_inputs(board, idx),
        )

    assert sb_model.scene_states(src_service.load("owner-a", board_id)) == [sb_model.DONE, sb_model.DONE]
    archive = export_project_archive(
        "storyboard", board_id, tmp_path / "cast_media_board.zip", store=src_store, runtime=src_runtime
    )
    imported = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
    dst_board = dst_service.load("imported", imported["target_id"])
    assert sb_model.scene_states(dst_board) == [sb_model.DONE, sb_model.DONE]

    def replace_inputs(board):
        board["takes"][board["scenes"][0]["in_use"]]["inputs"]["cast"][0][1] = "original.png"
        board["takes"][board["scenes"][1]["in_use"]]["inputs"]["video"] = "original.mp4"

    src_service.change("owner-a", board_id, replace_inputs)
    export_project_archive("storyboard", board_id, archive, store=src_store, runtime=src_runtime)
    import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="overwrite")
    assert sb_model.scene_states(dst_service.load("imported", board_id)) == [sb_model.OUTDATED, sb_model.OUTDATED]


def test_soundstage_clip_collision_isolation_illustrated_video_and_pre1980_mtime(tmp_path: Path):
    from h3_ui.drama3.stage_video import ILLUSTRATED_FAMILY, latest_export

    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")

    with _using_workspace_env(src_runtime):
        clips_dir = src_runtime.output_dir / "h3" / ".soundstage" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clips_dir / "shared.wav"
        clip_path.write_bytes(b"REMOTE-VOICE-BYTES")
        # Set pre-1980 mtime (epoch 1970) to ensure zip export never fails strict_timestamps.
        os.utime(clip_path, (0, 0))

        h3_dir = src_runtime.output_dir / "h3"
        h3_dir.mkdir(parents=True, exist_ok=True)
        mix_path = h3_dir / "ep_mix.mp3"
        mix_path.write_bytes(b"ep-mix-mp3")
        write_snapshot(mix_path, {"family": "Drama 3", "settings": {"title": "Ep", "chunks": []}})
        illus_video = h3_dir / "ep_mix_illustrated_1234abcd.mp4"
        illus_video.write_bytes(b"illustrated-mp4-bytes")
        write_snapshot(
            illus_video,
            {"family": ILLUSTRATED_FAMILY, "settings": {"source": mix_path.name, "subtitles": 100}},
        )

        src_store.save_soundstage_project(
            "Ep",
            {"text": "[Host] Hi", "voice_0_preset": UPLOAD_VOICE, "voice_0_audio": "shared.wav"},
        )
        src_store.mark_soundstage_generated("Ep", str(mix_path))
        archive = export_project_archive(
            "soundstage", "Ep", tmp_path / "ep.zip", store=src_store, runtime=src_runtime
        )

    with _using_workspace_env(dst_runtime):
        dst_clips = dst_runtime.output_dir / "h3" / ".soundstage" / "clips"
        dst_clips.mkdir(parents=True, exist_ok=True)
        (dst_clips / "shared.wav").write_bytes(b"LOCAL-OTHER-PROJECT-VOICE")
        dst_store.save_soundstage_project(
            "Other Local",
            {"text": "[Other] Keep my voice", "voice_0_preset": UPLOAD_VOICE, "voice_0_audio": "shared.wav"},
        )

        first = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert first["status"] == "imported"
        # Unrelated local project's clip is untouched; imported project got an isolated clip name.
        assert (dst_clips / "shared.wav").read_bytes() == b"LOCAL-OTHER-PROJECT-VOICE"
        ep_payload, ep_result = dst_store.soundstage_project("Ep")
        assert ep_payload["voice_0_audio"] == "shared_imported.wav"
        assert (dst_clips / "shared_imported.wav").read_bytes() == b"REMOTE-VOICE-BYTES"
        exported_vid = latest_export(Path(ep_result))
        assert exported_vid is not None and exported_vid.read_bytes() == b"illustrated-mp4-bytes"

        # Importing again as a copy deduplicates the matching clip and renames both mix and illustrated video.
        second = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
        assert second["status"] == "renamed"
        ep2_payload, ep2_result = dst_store.soundstage_project(second["target_name"])
        assert (dst_clips / ep2_payload["voice_0_audio"]).read_bytes() == b"REMOTE-VOICE-BYTES"
        renamed_vid = latest_export(Path(ep2_result))
        assert renamed_vid is not None and renamed_vid.name != exported_vid.name
        assert read_snapshot(renamed_vid)["settings"]["source"] == Path(ep2_result).name
        obsolete = exported_vid.with_name(f"{Path(ep_result).stem}_illustrated_obsolete.mp4")
        obsolete.write_bytes(b"older illustration")
        write_snapshot(obsolete, {"family": ILLUSTRATED_FAMILY})
        import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="overwrite")
        assert not obsolete.exists() and not snapshot_path(obsolete).exists()
        assert exported_vid.is_file() and renamed_vid.is_file()


def test_corrupt_archive_rollback_preserves_existing_projects_and_leaves_no_orphans(tmp_path: Path):
    src_runtime, src_store = _make_workspace(tmp_path / "source")
    dst_runtime, dst_store = _make_workspace(tmp_path / "target")

    def corrupt_zip_entry(src_zip: Path, dst_zip: Path, target_suffix: str) -> Path:
        raw = bytearray(src_zip.read_bytes())
        with zipfile.ZipFile(src_zip, "r") as zf:
            info = next(i for i in zf.infolist() if i.filename.endswith(target_suffix) and i.file_size >= 8)
            # Flip bytes in the member payload so CRC verification fails mid-extraction.
            local_sig_offset = info.header_offset
            name_len = int.from_bytes(raw[local_sig_offset + 26 : local_sig_offset + 28], "little")
            extra_len = int.from_bytes(raw[local_sig_offset + 28 : local_sig_offset + 30], "little")
            data_offset = local_sig_offset + 30 + name_len + extra_len
            raw[data_offset] ^= 0xFF
            raw[data_offset + 1] ^= 0xFF
        dst_zip.write_bytes(bytes(raw))
        return dst_zip

    # 1. Remake overwrite rollback on corrupt archive
    src_remakes = RemakeProjects(src_runtime.output_dir)
    dst_remakes = RemakeProjects(dst_runtime.output_dir)
    vid = tmp_path / "src.mp4"
    vid.write_bytes(b"valid-remake-source-bytes-1234567890")
    src_remakes.create("owner-a", "rk1", str(vid), {"name": "Remake Safe"}, [{"render": [0, 24]}])
    good_remake_zip = export_project_archive(
        "remake", "rk1", tmp_path / "remake_good.zip", store=src_store, runtime=src_runtime
    )
    bad_remake_zip = corrupt_zip_entry(good_remake_zip, tmp_path / "remake_bad.zip", ".mp4")

    local_vid = tmp_path / "local.mp4"
    local_vid.write_bytes(b"EXISTING-LOCAL-REMAKE-MUST-SURVIVE")
    dst_proj = dst_remakes.create("owner-b", "rk1", str(local_vid), {"name": "Remake Safe"}, [{"render": [0, 24]}])

    with pytest.raises(zipfile.BadZipFile):
        import_project_archive(bad_remake_zip, store=dst_store, runtime=dst_runtime, conflict="overwrite")

    surviving = dst_remakes.load(None, "rk1")
    assert surviving is not None
    assert Path(surviving["source"]).read_bytes() == b"EXISTING-LOCAL-REMAKE-MUST-SURVIVE"
    owner_parent = Path(dst_proj["folder"]).parent
    assert not [p for p in owner_parent.iterdir() if p.name.startswith((".importing-", ".replacing-"))]

    # 2. Dubbing failed import cleans up new session_root without leaking orphan folders
    with _using_workspace_env(src_runtime):
        dub_src = tmp_path / "talk.mp4"
        dub_src.write_bytes(b"dub-source-bytes-1234567890")
        sess = dub_sessions.create(src_runtime, dub_src, video=True)
        dub_out = src_runtime.outputs_dir / "dubbing"
        dub_out.mkdir(parents=True, exist_ok=True)
        dub_mp4 = dub_out / "talk_fr.mp4"
        dub_mp4.write_bytes(b"dub-result-video-bytes-1234567890")
        dub_mp3 = dub_out / "talk_fr.mp3"
        dub_mp3.write_bytes(b"dub-result-audio-bytes-1234567890")
        sess.data["last_output"] = str(dub_mp4)
        sess.save()
        dubbing_projects.save(src_runtime, "Safe Dub", sess.id, dub_src, {"target_language": "French"})
        good_dub_zip = export_project_archive(
            "dubbing", "Safe Dub", tmp_path / "dub_good.zip", store=src_store, runtime=src_runtime
        )
    bad_dub_zip = corrupt_zip_entry(good_dub_zip, tmp_path / "dub_bad.zip", "talk_fr.mp4")

    with _using_workspace_env(dst_runtime):
        sessions_before = set(dub_sessions.sessions_dir(dst_runtime).glob("*"))
        with pytest.raises(zipfile.BadZipFile):
            import_project_archive(bad_dub_zip, store=dst_store, runtime=dst_runtime, conflict="rename")
        sessions_after = set(dub_sessions.sessions_dir(dst_runtime).glob("*"))
        assert sessions_after == sessions_before
        assert "Safe Dub" not in dubbing_projects.names(dst_runtime)

        # Overwriting Dubbing when a local .lipsync.mp4 is obsolete deletes it and forgets its catalog row.
        import_project_archive(good_dub_zip, store=dst_store, runtime=dst_runtime, conflict="rename")
        dst_lipsync = dst_runtime.outputs_dir / "dubbing" / "talk_fr.lipsync.mp4"
        dst_lipsync.write_bytes(b"obsolete-lipsync-bytes")
        dst_store.index_asset(dst_lipsync, "video", {"_media": {}, "family": "Dubbing lip sync"})
        assert any(Path(r["path"]).resolve() == dst_lipsync.resolve() for r in dst_store.search_assets(kind="video"))
        import_project_archive(good_dub_zip, store=dst_store, runtime=dst_runtime, conflict="overwrite")
        assert not dst_lipsync.exists()
        assert not any(Path(r["path"]).resolve() == dst_lipsync.resolve() for r in dst_store.search_assets(kind="video"))

    # 3. Soundstage failed overwrite rolls back staged clips without clobbering existing local clips
    with _using_workspace_env(src_runtime):
        ss_clips = src_runtime.output_dir / "h3" / ".soundstage" / "clips"
        ss_clips.mkdir(parents=True, exist_ok=True)
        (ss_clips / "host.wav").write_bytes(b"NEW-REMOTE-CLIP-BYTES-1234567890")
        ss_h3 = src_runtime.output_dir / "h3"
        ss_h3.mkdir(parents=True, exist_ok=True)
        ss_mix = ss_h3 / "safe_stage.mp3"
        ss_mix.write_bytes(b"NEW-REMOTE-MIX-BYTES-1234567890")
        src_store.save_soundstage_project(
            "Safe Stage",
            {"text": "[Host] Remote", "voice_0_preset": UPLOAD_VOICE, "voice_0_audio": "host.wav"},
        )
        src_store.mark_soundstage_generated("Safe Stage", str(ss_mix))
        good_ss_zip = export_project_archive(
            "soundstage", "Safe Stage", tmp_path / "ss_good.zip", store=src_store, runtime=src_runtime
        )
    bad_ss_zip = corrupt_zip_entry(good_ss_zip, tmp_path / "ss_bad.zip", "safe_stage.mp3")

    with _using_workspace_env(dst_runtime):
        dst_clips = dst_runtime.output_dir / "h3" / ".soundstage" / "clips"
        dst_clips.mkdir(parents=True, exist_ok=True)
        (dst_clips / "host.wav").write_bytes(b"ORIGINAL-LOCAL-CLIP-MUST-SURVIVE")
        dst_store.save_soundstage_project(
            "Safe Stage",
            {"text": "[Host] Local", "voice_0_preset": UPLOAD_VOICE, "voice_0_audio": "host.wav"},
        )
        with pytest.raises(zipfile.BadZipFile):
            import_project_archive(bad_ss_zip, store=dst_store, runtime=dst_runtime, conflict="overwrite")
        assert (dst_clips / "host.wav").read_bytes() == b"ORIGINAL-LOCAL-CLIP-MUST-SURVIVE"
        assert dst_store.soundstage_project("Safe Stage")[0]["text"] == "[Host] Local"


def test_http_retry_range_resume_skip_fastpath_and_parallel_ordering(tmp_path: Path, monkeypatch):
    from fastapi.responses import PlainTextResponse, StreamingResponse

    from h3_app import project_transfer as pt_mod
    from h3_app.project_transfer import import_remote_projects

    monkeypatch.setattr(pt_mod, "_retry_sleep_seconds", lambda *_args, **_kwargs: 0.01)

    src_runtime, src_store = _make_workspace(tmp_path / "remote")
    dst_runtime, dst_store = _make_workspace(tmp_path / "local")

    with _using_workspace_env(src_runtime):
        for i in range(4):
            src_store.save_soundstage_project(f"Stage {i}", {"text": f"[Host] Line {i}", "title": f"T{i}"})
        sb_service = StoryboardService(src_store, lambda: src_runtime)
        sb_id = sb_service.create(
            "owner-r",
            "Existing Board Name",
            sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        )

    export_builds: list[tuple[str, str]] = []
    real_export = pt_mod.export_project_archive

    def counting_export(kind: str, project_id: str, destination: Path, **kwargs) -> Path:
        export_builds.append((kind, project_id))
        return real_export(kind, project_id, destination, **kwargs)

    monkeypatch.setattr("h3_ui.project_transfer.export_project_archive", counting_export)

    list_calls = 0
    export_requests: list[tuple[str, str, str | None]] = []
    dropped_once: set[str] = set()

    remote_app = FastAPI()

    @remote_app.middleware("http")
    async def inject_transient_faults(request: Request, call_next):
        nonlocal list_calls
        if request.url.path == "/api/projects/transfer/list":
            list_calls += 1
            if list_calls == 1:
                return PlainTextResponse("Bad Gateway", status_code=502)
        if request.url.path == "/api/projects/transfer/export":
            kind = request.query_params.get("kind", "")
            pid = request.query_params.get("id", "")
            rng = request.headers.get("range")
            export_requests.append((kind, pid, rng))
            if pid == "Stage 0" and pid not in dropped_once:
                dropped_once.add(pid)
                full_resp = await call_next(request)
                body = b"".join([chunk async for chunk in full_resp.body_iterator])
                half = max(1, len(body) // 2)
                headers = {
                    k: v
                    for k, v in full_resp.headers.items()
                    if k.lower() in {"content-length", "etag", "last-modified", "accept-ranges"}
                }

                async def truncated_stream():
                    import asyncio

                    yield body[:half]
                    await asyncio.sleep(0.01)
                    raise ConnectionResetError("Simulated mid-stream tunnel drop")

                return StreamingResponse(
                    truncated_stream(),
                    status_code=200,
                    headers=headers,
                    media_type="application/zip",
                )
            if pid == "Stage 1" and pid not in dropped_once:
                dropped_once.add(pid)
                return PlainTextResponse("Service Unavailable", status_code=503)
        return await call_next(request)

    install_project_transfer_routes(remote_app, store=src_store, runtime_config=lambda: src_runtime)

    with _serve_fastapi(remote_app) as remote_url, _using_workspace_env(dst_runtime):
        # 1. fetch_remote_projects recovers from initial HTTP 502.
        listing = fetch_remote_projects(remote_url)
        assert list_calls == 2
        assert len(listing["projects"]) == 5

        # 2. Pre-seed Stage 3 and Storyboard (with different local ID, matching name) to verify 0-download skip.
        dst_store.save_soundstage_project("Stage 3", {"text": "Local keep"})
        local_sb_service = StoryboardService(dst_store, lambda: dst_runtime)
        local_sb_service.create(
            "owner-l",
            "Existing Board Name",
            sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        )

        commit_mock = Mock()
        selections = [
            ("soundstage", "Stage 0", "Stage 0"),
            ("soundstage", "Stage 1", "Stage 1"),
            ("soundstage", "Stage 2", "Stage 2"),
            ("soundstage", "Stage 3", "Stage 3"),
            ("storyboard", sb_id, "Existing Board Name"),
        ]
        results = import_remote_projects(
            remote_url,
            selections,
            store=dst_store,
            runtime=dst_runtime,
            conflict="skip",
            on_imported=commit_mock,
        )
        # Exact selection order preserved despite parallel workers and retries.
        assert [(r["kind"], r["source_id"], r["status"]) for r in results] == [
            ("soundstage", "Stage 0", "imported"),
            ("soundstage", "Stage 1", "imported"),
            ("soundstage", "Stage 2", "imported"),
            ("soundstage", "Stage 3", "skipped"),
            ("storyboard", sb_id, "skipped"),
        ]
        commit_mock.assert_called_once()

        # Skipped items never hit /api/projects/transfer/export at all.
        requested_ids = {pid for _k, pid, _rng in export_requests}
        assert "Stage 3" not in requested_ids
        assert sb_id not in requested_ids

        # Stage 0 resumed via Range header, and _EXPORT_CACHE built Stage 0's archive only ONCE.
        stage0_ranges = [rng for _k, pid, rng in export_requests if pid == "Stage 0"]
        assert stage0_ranges[0] is None
        assert stage0_ranges[1] is not None and stage0_ranges[1].startswith("bytes=")
        assert export_builds.count(("soundstage", "Stage 0")) == 1

        # Editing Stage 0 invalidates the export cache stamp and builds a fresh archive.
        src_store.save_soundstage_project("Stage 0", {"text": "[Host] Updated line", "title": "T0-v2"})
        overwrite_res = import_remote_projects(
            remote_url,
            [("soundstage", "Stage 0")],
            store=dst_store,
            runtime=dst_runtime,
            conflict="overwrite",
        )
        assert overwrite_res[0]["status"] == "overwritten"
        assert dst_store.soundstage_project("Stage 0")[0]["title"] == "T0-v2"
        assert export_builds.count(("soundstage", "Stage 0")) == 2


def test_workshop_to_gpu_studio_and_back_incremental_sync_all_four_kinds(tmp_path: Path, monkeypatch):
    from h3_app.project_transfer import import_remote_projects

    monkeypatch.delenv("H3_WORKSPACE_DIR", raising=False)
    ws_runtime, ws_store = _make_workspace(tmp_path / "workshop")
    gpu_runtime, gpu_store = _make_workspace(tmp_path / "gpu_studio")

    heavy_cast = os.urandom(160_000)
    heavy_sb_take0 = os.urandom(220_000)
    heavy_sb_latent0 = os.urandom(180_000)
    heavy_dub_source = os.urandom(260_000)
    heavy_dub_vocals = os.urandom(200_000)
    heavy_dub_take0 = os.urandom(120_000)
    heavy_ss_clip = os.urandom(140_000)
    heavy_ss_take0 = os.urandom(130_000)
    heavy_ss_still1 = os.urandom(110_000)
    heavy_remake_src = os.urandom(240_000)
    heavy_remake_pic = os.urandom(90_000)

    # 1. Seed all 4 project kinds on Workshop with sizable binary assets.
    ws_sb = StoryboardService(ws_store, lambda: ws_runtime)
    cast_file = tmp_path / "hero_big.png"
    cast_file.write_bytes(heavy_cast)
    board_id = ws_sb.create(
        "ws-owner",
        "Big Storyboard",
        sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        cast=[("Hero", cast_file)],
    )
    ws_sb.add_scene("ws-owner", board_id)
    board = ws_sb.load("ws-owner", board_id)
    sc0_id = board["scenes"][0]["id"]
    sc1_id = board["scenes"][1]["id"]
    tk0_id = sb_model.new_id()
    tk0_prefix = ws_sb.take_prefix(board_id, sc0_id, tk0_id)
    tk0_video = ws_sb.output_path(f"{tk0_prefix}.mp4")
    tk0_video.parent.mkdir(parents=True, exist_ok=True)
    tk0_video.write_bytes(heavy_sb_take0)
    write_snapshot(tk0_video, {"family": "MiniMax H3", "settings": {"prompt": "Scene 0"}})
    tk0_latent_rel = f"{tk0_prefix}.safetensors"
    ws_sb.output_path(tk0_latent_rel).write_bytes(heavy_sb_latent0)
    ws_sb.record_take(
        "ws-owner",
        board_id,
        sc0_id,
        tk0_id,
        video=str(tk0_video),
        latents={"latent": tk0_latent_rel, "stage_latent": None, "prefix_frames": 0},
        seed=7,
        inputs=sb_model.scene_inputs(ws_sb.load("ws-owner", board_id), 0),
    )

    # Dubbing
    dub_src_file = tmp_path / "movie_big.mp4"
    dub_src_file.write_bytes(heavy_dub_source)
    ws_dub_sess = dub_sessions.create(ws_runtime, dub_src_file, video=True)
    (ws_dub_sess.root / "vocals.wav").write_bytes(heavy_dub_vocals)
    ws_dub_sess.set_file("vocals", ws_dub_sess.root / "vocals.wav")
    (ws_dub_sess.root / "takes").mkdir(parents=True, exist_ok=True)
    (ws_dub_sess.root / "takes" / "0.wav").write_bytes(heavy_dub_take0)
    (ws_dub_sess.root / "takes" / "1.wav").write_bytes(b"old-line1-take")
    ws_dub_sess.segments = [
        {"start": 0.0, "end": 1.5, "speaker": "A", "source": "One", "translation": "Uno"},
        {"start": 1.5, "end": 3.0, "speaker": "A", "source": "Two", "translation": "Dos"},
    ]
    ws_dub_sess.save()
    dubbing_projects.save(ws_runtime, "Big Dub", ws_dub_sess.id, dub_src_file, {"target_language": "Spanish"})

    # Soundstage
    ss_clips = ws_runtime.output_dir / "h3" / ".soundstage" / "clips"
    ss_clips.mkdir(parents=True, exist_ok=True)
    (ss_clips / "narrator.wav").write_bytes(heavy_ss_clip)
    ss_h3 = ws_runtime.output_dir / "h3"
    ss_h3.mkdir(parents=True, exist_ok=True)
    ss_mix = ss_h3 / "big_drama.mp3"
    ss_mix.write_bytes(b"initial-mix-bytes")
    take0_key = "1" * 32
    write_snapshot(
        ss_mix,
        {"family": "Drama 3", "settings": {"title": "Big Drama", "chunks": [{"take": take0_key}]}},
    )
    ss_takes = takes_root(ws_runtime)
    ss_takes.mkdir(parents=True, exist_ok=True)
    (ss_takes / f"{take0_key}.wav").write_bytes(heavy_ss_take0)
    ss_stills = drama3_visuals.stills_dir(ss_mix, ws_runtime.output_dir)
    ss_stills.mkdir(parents=True, exist_ok=True)
    (ss_stills / "shot1.jpg").write_bytes(heavy_ss_still1)
    (ss_stills / "obsolete_shot.jpg").write_bytes(b"remove-me-on-sync")
    (ss_stills / drama3_visuals.STILLS_RECORD).write_text(
        json.dumps(
            {
                "revision": "r1",
                "source": ss_mix.name,
                "shots": [{"image": "shot1.jpg"}, {"image": "obsolete_shot.jpg"}],
            }
        ),
        encoding="utf-8",
    )
    ws_store.save_soundstage_project(
        "Big Stage",
        {
            "text": "[Host] Part 1",
            "title": "Big Drama",
            "voice_0_preset": UPLOAD_VOICE,
            "voice_0_audio": "narrator.wav",
        },
    )
    ws_store.mark_soundstage_generated("Big Stage", str(ss_mix))

    # Remake
    ws_remakes = RemakeProjects(ws_runtime.output_dir)
    remake_vid = tmp_path / "remake_big.mp4"
    remake_vid.write_bytes(heavy_remake_src)
    remake_key = "rk_big_sync_0001"
    rm_proj = ws_remakes.create(
        "ws-owner",
        remake_key,
        str(remake_vid),
        {"name": "Big Remake", "start": 0.0, "frames": 162, "width": 832, "height": 480, "mix": 0},
        [{"render": [0, 81], "own": [0, 81], "hold": 0}, {"render": [81, 162], "own": [81, 162], "hold": 0}],
    )
    rm_pic = Path(rm_proj["folder"]) / "pictures" / "actor.png"
    rm_pic.write_bytes(heavy_remake_pic)
    rm_w0_old = Path(rm_proj["folder"]) / "w0_old.mp4"
    rm_w0_old.write_bytes(b"old-window0-take")
    ws_remakes.edit(
        rm_proj,
        lambda cur: cur["windows"][0].update(
            pictures=[str(rm_pic)], pins=[0], prompt="Walk", take=str(rm_w0_old), state="done"
        ),
    )

    ws_app = FastAPI()
    ws_commits = Mock()
    install_project_transfer_routes(ws_app, store=ws_store, runtime_config=lambda: ws_runtime, on_imported=ws_commits)

    gpu_app = FastAPI()
    delta_wire_bytes: list[tuple[str, int]] = []

    @gpu_app.middleware("http")
    async def measure_gpu_transfer_bytes(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path in {"/api/projects/transfer/delta", "/api/projects/transfer/export"}:
            chunks = [chunk async for chunk in resp.body_iterator]
            raw = b"".join(chunks)
            delta_wire_bytes.append((request.url.path, len(raw)))
            from starlette.responses import Response

            return Response(
                content=raw,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                media_type=resp.media_type,
            )
        return resp

    install_project_transfer_routes(gpu_app, store=gpu_store, runtime_config=lambda: gpu_runtime)

    all_selections = [
        ("storyboard", board_id, "Big Storyboard"),
        ("dubbing", "Big Dub", "Big Dub"),
        ("soundstage", "Big Stage", "Big Stage"),
        ("remake", remake_key, "Big Remake"),
    ]

    with _serve_fastapi(ws_app) as ws_url, _serve_fastapi(gpu_app) as gpu_url:
        # 2. Temp GPU Studio imports all 4 projects from Workshop initially.
        initial = import_remote_projects(
            ws_url, all_selections, store=gpu_store, runtime=gpu_runtime, conflict="rename"
        )
        assert [r["status"] for r in initial] == ["imported", "imported", "imported", "imported"]

        # 3. Syncing back to Workshop BEFORE any GPU edits detects 100% up_to_date statelesly (0 zip bytes, 0 commits).
        delta_wire_bytes.clear()
        noop_sync = import_remote_projects(
            gpu_url,
            all_selections,
            store=ws_store,
            runtime=ws_runtime,
            conflict="overwrite",
            on_imported=ws_commits,
        )
        assert [r["status"] for r in noop_sync] == ["up_to_date", "up_to_date", "up_to_date", "up_to_date"]
        assert ws_commits.call_count == 0
        assert all(path == "/api/projects/transfer/delta" and size < 1024 for path, size in delta_wire_bytes)

        # Record Workshop mtimes & session IDs of unchanged local files BEFORE partial GPU updates.
        ws_dub_before_id = dubbing_projects.load(ws_runtime, "Big Dub")[1].id
        ws_sessions_count_before = len(list(dub_sessions.sessions_dir(ws_runtime).iterdir()))
        ws_dub_src_mtime = (dub_sessions.sessions_dir(ws_runtime) / ws_dub_before_id / "source.mp4").stat().st_mtime_ns
        ws_sb_tk0_mtime = tk0_video.stat().st_mtime_ns
        ws_remake_src_path = Path(RemakeProjects(ws_runtime.output_dir).load(None, remake_key)["source"])
        ws_remake_src_mtime = ws_remake_src_path.stat().st_mtime_ns

        # 4. Perform small GPU updates on Temp GPU Studio across all 4 projects.
        gpu_sb = StoryboardService(gpu_store, lambda: gpu_runtime)
        tk1_id = sb_model.new_id()
        tk1_prefix = gpu_sb.take_prefix(board_id, sc1_id, tk1_id)
        tk1_video = gpu_sb.output_path(f"{tk1_prefix}.mp4")
        tk1_video.parent.mkdir(parents=True, exist_ok=True)
        tk1_video.write_bytes(b"gpu-generated-scene1-video")
        gpu_sb.record_take(
            "gpu-owner",
            board_id,
            sc1_id,
            tk1_id,
            video=str(tk1_video),
            latents={"latent": None},
            seed=99,
            inputs=sb_model.scene_inputs(gpu_sb.load("gpu-owner", board_id), 1),
        )

        # Dubbing: regenerate take 1 + output mp3/mp4 while keeping 260 KB source + 200 KB vocals + 120 KB take 0.
        _, gpu_dub_sess = dubbing_projects.load(gpu_runtime, "Big Dub")
        (gpu_dub_sess.root / "takes" / "1.wav").write_bytes(b"gpu-regenerated-line1-wav")
        gpu_dub_out = gpu_runtime.outputs_dir / "dubbing"
        gpu_dub_out.mkdir(parents=True, exist_ok=True)
        (gpu_dub_out / "big_es.mp3").write_bytes(b"gpu-dub-mp3")
        (gpu_dub_out / "big_es.mp4").write_bytes(b"gpu-dub-mp4")
        gpu_dub_sess.data["last_output"] = str(gpu_dub_out / "big_es.mp4")
        gpu_dub_sess.save()

        # Soundstage: add take1 + shot2.jpg, drop obsolete_shot.jpg, keep clip + take0 + shot1.jpg.
        take1_key = "2" * 32
        (takes_root(gpu_runtime) / f"{take1_key}.wav").write_bytes(b"gpu-new-take1-wav")
        _, gpu_ss_res = gpu_store.soundstage_project("Big Stage")
        gpu_mix = Path(gpu_ss_res)
        gpu_mix.write_bytes(b"gpu-updated-mix-mp3")
        write_snapshot(
            gpu_mix,
            {
                "family": "Drama 3",
                "settings": {"title": "Big Drama", "chunks": [{"take": take0_key}, {"take": take1_key}]},
            },
        )
        gpu_stills = drama3_visuals.stills_dir(gpu_mix, gpu_runtime.output_dir)
        (gpu_stills / "obsolete_shot.jpg").unlink()
        (gpu_stills / "shot2.jpg").write_bytes(b"gpu-new-shot2-jpg")
        (gpu_stills / drama3_visuals.STILLS_RECORD).write_text(
            json.dumps(
                {
                    "revision": "r2",
                    "source": gpu_mix.name,
                    "shots": [{"image": "shot1.jpg"}, {"image": "shot2.jpg"}],
                }
            ),
            encoding="utf-8",
        )
        gpu_store.save_soundstage_project(
            "Big Stage",
            {
                "text": "[Host] Part 1\n[Host] Part 2",
                "title": "Big Drama",
                "voice_0_preset": UPLOAD_VOICE,
                "voice_0_audio": "narrator.wav",
            },
        )
        gpu_store.mark_soundstage_generated("Big Stage", str(gpu_mix))

        # Remake: render window 1 and replace window 0 take (removing old w0 take).
        gpu_remakes = RemakeProjects(gpu_runtime.output_dir)
        gpu_rm = gpu_remakes.load(None, remake_key)
        Path(gpu_rm["windows"][0]["take"]).unlink()
        w0_new = Path(gpu_rm["folder"]) / "w0_gpu.mp4"
        w0_new.write_bytes(b"gpu-w0-rendered")
        w1_new = Path(gpu_rm["folder"]) / "w1_gpu.mp4"
        w1_new.write_bytes(b"gpu-w1-rendered")
        gpu_remakes.edit(
            gpu_rm,
            lambda cur: (
                cur["windows"][0].update(take=str(w0_new), state="done"),
                cur["windows"][1].update(take=str(w1_new), state="done"),
            ),
        )

        # 5. Workshop syncs back from GPU Studio with conflict="overwrite": only deltas transfer!
        delta_wire_bytes.clear()
        synced = import_remote_projects(
            gpu_url,
            all_selections,
            store=ws_store,
            runtime=ws_runtime,
            conflict="overwrite",
            on_imported=ws_commits,
        )
        assert [r["status"] for r in synced] == ["overwritten", "overwritten", "overwritten", "overwritten"]
        assert ws_commits.call_count == 1
        for res in synced:
            assert res["incremental"] is True
            assert res["reused_files"] >= 2
            assert "reused" in res["message"]

        # Unchanged assets totalled >1.6 MB; total delta wire transfer across ALL 4 projects is tiny (<50 KB)!
        total_wire = sum(size for _path, size in delta_wire_bytes)
        assert all(path == "/api/projects/transfer/delta" for path, _ in delta_wire_bytes)
        assert total_wire < 50_000

        # Verify unchanged local files stayed untouched in place (exact mtime preserved) & updates landed:
        ws_sb_loaded = ws_store.storyboard(None, board_id)
        assert tk1_id in ws_sb_loaded["takes"]
        assert Path(ws_sb_loaded["takes"][tk1_id]["video"]).read_bytes() == b"gpu-generated-scene1-video"
        assert tk0_video.stat().st_mtime_ns == ws_sb_tk0_mtime

        _, ws_dub_after = dubbing_projects.load(ws_runtime, "Big Dub")
        assert ws_dub_after.id == ws_dub_before_id
        assert len(list(dub_sessions.sessions_dir(ws_runtime).iterdir())) == ws_sessions_count_before
        assert (ws_dub_after.root / "source.mp4").stat().st_mtime_ns == ws_dub_src_mtime
        assert (ws_dub_after.root / "takes" / "0.wav").read_bytes() == heavy_dub_take0
        assert (ws_dub_after.root / "takes" / "1.wav").read_bytes() == b"gpu-regenerated-line1-wav"
        assert Path(ws_dub_after.data["last_output"]).read_bytes() == b"gpu-dub-mp4"

        _, ws_ss_res = ws_store.soundstage_project("Big Stage")
        ws_ss_mix = Path(ws_ss_res)
        assert ws_ss_mix.read_bytes() == b"gpu-updated-mix-mp3"
        assert (takes_root(ws_runtime) / f"{take1_key}.wav").read_bytes() == b"gpu-new-take1-wav"
        ws_ss_stills = drama3_visuals.stills_dir(ws_ss_mix, ws_runtime.output_dir)
        assert (ws_ss_stills / "shot1.jpg").read_bytes() == heavy_ss_still1
        assert (ws_ss_stills / "shot2.jpg").read_bytes() == b"gpu-new-shot2-jpg"
        assert not (ws_ss_stills / "obsolete_shot.jpg").exists()

        ws_rm_after = RemakeProjects(ws_runtime.output_dir).load(None, remake_key)
        assert Path(ws_rm_after["source"]).stat().st_mtime_ns == ws_remake_src_mtime
        assert Path(ws_rm_after["windows"][0]["take"]).read_bytes() == b"gpu-w0-rendered"
        assert Path(ws_rm_after["windows"][1]["take"]).read_bytes() == b"gpu-w1-rendered"
        assert not [p for p in (Path(ws_rm_after["folder"]) / "takes").iterdir() if "w0_old" in p.name]

        # 6. Running backup sync again immediately reports all 4 projects as up_to_date.
        delta_wire_bytes.clear()
        second_sync = import_remote_projects(
            gpu_url,
            all_selections,
            store=ws_store,
            runtime=ws_runtime,
            conflict="overwrite",
            on_imported=ws_commits,
        )
        assert [r["status"] for r in second_sync] == ["up_to_date", "up_to_date", "up_to_date", "up_to_date"]
        assert ws_commits.call_count == 1  # Unchanged: no extra volume commit!

        # 7. Local edit on Workshop is detected and cleanly reverted when overwriting from GPU Studio.
        ws_sb.update_scene("ws-owner", board_id, sc0_id, prompt="Accidental local edit")
        revert_sync = import_remote_projects(
            gpu_url,
            [("storyboard", board_id, "Big Storyboard")],
            store=ws_store,
            runtime=ws_runtime,
            conflict="overwrite",
            on_imported=ws_commits,
        )
        assert revert_sync[0]["status"] == "overwritten"
        assert revert_sync[0]["transferred_files"] == 0
        assert revert_sync[0]["reused_files"] >= 3
        assert ws_store.storyboard(None, board_id)["scenes"][0]["prompt"] != "Accidental local edit"


def test_delta_range_resume_legacy_server_fallback_and_missing_donor_recovery(tmp_path: Path, monkeypatch):
    from fastapi.responses import PlainTextResponse, StreamingResponse

    from h3_app import project_transfer as pt_mod
    from h3_app.project_transfer import import_remote_projects

    monkeypatch.setattr(pt_mod, "_retry_sleep_seconds", lambda *_args, **_kwargs: 0.01)

    src_runtime, src_store = _make_workspace(tmp_path / "src")
    dst_runtime, dst_store = _make_workspace(tmp_path / "dst")

    src_remakes = RemakeProjects(src_runtime.output_dir)
    vid = tmp_path / "source.mp4"
    vid.write_bytes(os.urandom(120_000))
    rkey = "rk_resilience_1"
    proj = src_remakes.create(
        "owner-s",
        rkey,
        str(vid),
        {"name": "Resilient Remake", "start": 0.0, "frames": 81, "width": 832, "height": 480, "mix": 0},
        [{"render": [0, 81], "own": [0, 81], "hold": 0}],
    )

    remote_app = FastAPI()
    legacy_mode = False
    from h3_ui import project_transfer as transfer_routes
    monkeypatch.setattr(transfer_routes, "_EXPORT_CACHE_MAX_BYTES", 1)
    drop_delta_once = True
    delta_ranges: list[str | None] = []
    resumed_responses = []
    export_hits = 0

    @remote_app.middleware("http")
    async def fault_middleware(request: Request, call_next):
        nonlocal drop_delta_once, export_hits
        if request.url.path == "/api/projects/transfer/export":
            export_hits += 1
        if request.url.path == "/api/projects/transfer/delta":
            if legacy_mode:
                # Simulate an older server that has no /delta route (no X-Prism-Transfer header).
                return PlainTextResponse('{"detail":"Not Found"}', status_code=404, media_type="application/json")
            delta_ranges.append(request.headers.get("range"))
            if drop_delta_once:
                drop_delta_once = False
                full_resp = await call_next(request)
                body = b"".join([chunk async for chunk in full_resp.body_iterator])
                half = max(1, len(body) // 2)
                headers = {
                    k: v
                    for k, v in full_resp.headers.items()
                    if k.lower() in {"content-length", "etag", "last-modified", "accept-ranges", "x-prism-transfer"}
                }

                async def truncated_delta():
                    import asyncio

                    yield body[:half]
                    await asyncio.sleep(0.01)
                    raise ConnectionResetError("Simulated delta mid-stream drop")

                return StreamingResponse(
                    truncated_delta(),
                    status_code=200,
                    headers=headers,
                    media_type="application/zip",
                )
        response = await call_next(request)
        if request.url.path.endswith("/delta") and request.headers.get("range"):
            resumed_responses.append(response.status_code)
        return response

    install_project_transfer_routes(remote_app, store=src_store, runtime_config=lambda: src_runtime)

    with _serve_fastapi(remote_app) as remote_url:
        # Initial import
        res0 = import_remote_projects(remote_url, [("remake", rkey)], store=dst_store, runtime=dst_runtime)
        assert res0[0]["status"] == "imported"
        assert export_hits == 1

        # Add a take on source and test mid-stream HTTP Range resume on /delta
        take_file = Path(proj["folder"]) / "w0_new.mp4"
        take_file.write_bytes(os.urandom(60_000))
        src_remakes.edit(proj, lambda cur: cur["windows"][0].update(take=str(take_file), state="done"))

        res1 = import_remote_projects(
            remote_url, [("remake", rkey)], store=dst_store, runtime=dst_runtime, conflict="overwrite"
        )
        assert res1[0]["status"] == "overwritten"
        assert res1[0]["incremental"] is True
        assert delta_ranges[0] is None
        assert delta_ranges[1] is not None and delta_ranges[1].startswith("bytes=")
        assert export_hits == 1  # Never fell back to full /export

        assert resumed_responses == [206]

        # Simulate legacy remote server (404 without X-Prism-Transfer) -> falls back to /export cleanly
        legacy_mode = True
        src_remakes.edit(proj, lambda cur: cur["windows"][0].update(prompt="Legacy server prompt"))
        dst_src_before = Path(RemakeProjects(dst_runtime.output_dir).load(None, rkey)["source"]).stat().st_mtime_ns
        res2 = import_remote_projects(
            remote_url, [("remake", rkey)], store=dst_store, runtime=dst_runtime, conflict="overwrite"
        )
        assert res2[0]["status"] == "overwritten"
        assert export_hits == 2
        dst_loaded = RemakeProjects(dst_runtime.output_dir).load(None, rkey)
        assert dst_loaded["windows"][0]["prompt"] == "Legacy server prompt"
        # Even on legacy full-zip fallback, unchanged local source.mp4 was preserved in place!
        assert Path(dst_loaded["source"]).stat().st_mtime_ns == dst_src_before

        # Missing project on upgraded server returns 404 WITH X-Prism-Transfer -> fails fast without hitting /export
        legacy_mode = False
        missing_res = import_remote_projects(
            remote_url, [("remake", "nonexistent_rk")], store=dst_store, runtime=dst_runtime, conflict="overwrite"
        )
        assert missing_res[0]["status"] == "failed"

        # Mid-flight donor mutation triggers automatic full-export fallback recovery
        orig_inspect = pt_mod.inspect_local_project_for_delta

        def poisoned_inspect(*args, **kwargs):
            state = orig_inspect(*args, **kwargs)
            assert state is not None
            # Point every known sha256 to a vanished path so _ArchiveSource raises _MissingReusedFileError
            state.files_by_sha256 = {sha: [tmp_path / "vanished.bin"] for sha in state.files_by_sha256}
            state.portable_digest = "0" * 64
            return state

        monkeypatch.setattr(pt_mod, "inspect_local_project_for_delta", poisoned_inspect)
        src_remakes.edit(proj, lambda cur: cur["windows"][0].update(prompt="Recovered via fallback"))
        res3 = import_remote_projects(
            remote_url, [("remake", rkey)], store=dst_store, runtime=dst_runtime, conflict="overwrite"
        )
        assert res3[0]["status"] == "overwritten"
        assert export_hits == 3
        assert RemakeProjects(dst_runtime.output_dir).load(None, rkey)["windows"][0]["prompt"] == "Recovered via fallback"


def test_storyboard_transfer_preserves_video_posters_for_ffmpegless_workshop(tmp_path: Path, monkeypatch):
    from io import BytesIO
    import shutil as std_shutil

    import gradio as gr
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from PIL import Image

    from h3_app import gallery_store
    from h3_app.catalog import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    from h3_app.project_transfer import import_remote_projects
    from h3_app.server import ServerConfig, build_server
    from h3_ui.project_transfer import install_project_transfer_routes

    def jpeg_bytes(color: tuple[int, int, int]) -> bytes:
        buf = BytesIO()
        Image.new("RGB", (64, 36), color).save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    src_runtime, src_store = _make_workspace(tmp_path / "studio")
    dst_runtime, dst_store = _make_workspace(tmp_path / "workshop")
    src_service = StoryboardService(src_store, lambda: src_runtime)
    dst_service = StoryboardService(dst_store, lambda: dst_runtime)

    board_id = src_service.create(
        "owner-a",
        "Fox Mystery",
        sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
    )
    board = src_service.load("owner-a", board_id)
    scene0_id = board["scenes"][0]["id"]
    src_service.set_mode("owner-a", board_id, scene0_id, sb_model.REFERENCES)
    src_service.update_scene("owner-a", board_id, scene0_id, prompt="Opening shot")

    ref_video_src = tmp_path / "ref_clip.mp4"
    ref_video_src.write_bytes(b"R" * 16_000)
    src_service.set_scene_media("owner-a", board_id, scene0_id, "ref_videos", [ref_video_src])
    stored_ref_video = Path(src_service.load("owner-a", board_id)["scenes"][0]["ref_videos"][0])

    # Scene 1: Media scene with a clip video.
    stored_clip = src_service.media_dir(board_id) / "intro_clip.mp4"
    stored_clip.parent.mkdir(parents=True, exist_ok=True)
    stored_clip.write_bytes(b"C" * 16_000)
    media_scene = {
        **sb_model.new_media_scene("Intro Clip"),
        "video": str(stored_clip),
        "video_info": {"seconds": 4.0, "fps": 24.0, "width": 1280, "height": 720},
        "video_in": 0.0,
        "video_out": 4.0,
    }
    src_service.change("owner-a", board_id, lambda b: b["scenes"].append(media_scene))

    board = src_service.load("owner-a", board_id)
    inputs0 = sb_model.scene_inputs(board, 0)
    tk0_id = sb_model.new_id()
    tk0_video = src_service.output_path(f"{src_service.take_prefix(board_id, scene0_id, tk0_id)}.mp4")
    tk0_video.parent.mkdir(parents=True, exist_ok=True)
    tk0_video.write_bytes(b"0" * 32_000)
    src_service.record_take(
        "owner-a", board_id, scene0_id, tk0_id,
        video=str(tk0_video), latents={"latent": None}, seed=11, inputs=inputs0,
    )

    tk1_id = sb_model.new_id()
    tk1_video = src_service.output_path(f"{src_service.take_prefix(board_id, scene0_id, tk1_id)}.mp4")
    tk1_video.write_bytes(b"1" * 32_000)
    src_service.record_take(
        "owner-a", board_id, scene0_id, tk1_id,
        video=str(tk1_video), latents={"latent": None}, seed=22, inputs=inputs0,
    )

    # Pre-cache posters for tk0, ref_video, and intro_clip on Studio; leave tk1 uncached to verify
    # export generates missing posters via gallery_thumbnail when FFmpeg is available on Studio.
    expected_posters = {
        tk0_video.resolve(): jpeg_bytes((200, 40, 40)),
        tk1_video.resolve(): jpeg_bytes((40, 200, 40)),
        stored_ref_video.resolve(): jpeg_bytes((40, 40, 200)),
        stored_clip.resolve(): jpeg_bytes((200, 200, 40)),
    }
    src_runtime.gallery_thumbnails_dir.mkdir(parents=True, exist_ok=True)
    for video_path in (tk0_video, stored_ref_video, stored_clip):
        poster = gallery_store.gallery_thumbnail_path(video_path, runtime=src_runtime)
        poster.write_bytes(expected_posters[video_path.resolve()])

    real_gallery_thumbnail = gallery_store.gallery_thumbnail

    def studio_gallery_thumbnail(video: Path, *, runtime):
        resolved = Path(video).resolve()
        if resolved in expected_posters and Path(runtime.gallery_thumbnails_dir).resolve() == src_runtime.gallery_thumbnails_dir.resolve():
            poster = gallery_store.gallery_thumbnail_path(resolved, runtime=runtime)
            poster.parent.mkdir(parents=True, exist_ok=True)
            poster.write_bytes(expected_posters[resolved])
            return poster
        return real_gallery_thumbnail(video, runtime=runtime)

    with monkeypatch.context() as studio_mp:
        studio_mp.setattr(gallery_store, "gallery_thumbnail", studio_gallery_thumbnail)
        archive = export_project_archive(
            "storyboard", board_id, tmp_path / "fox_mystery.zip", store=src_store, runtime=src_runtime
        )

    with zipfile.ZipFile(archive, "r") as zf:
        poster_entries = sorted(name for name in zf.namelist() if name.startswith("posters/"))
    assert len(poster_entries) == 4
    assert any(name.startswith("posters/takes/") and name.endswith(".jpg") for name in poster_entries)
    assert any(name.startswith("posters/media/") and name.endswith(".jpg") for name in poster_entries)

    # Import into a minimal Workshop without FFmpeg (matching modal_workshop.py).
    real_which = std_shutil.which
    monkeypatch.setattr(std_shutil, "which", lambda cmd, *a, **kw: None if cmd == "ffmpeg" else real_which(cmd, *a, **kw))

    with gr.Blocks() as demo:
        gr.Markdown()
    ws_server_cfg = ServerConfig(
        comfy_url="http://127.0.0.1:9",
        output_dir=dst_runtime.output_dir,
        outputs_dir=dst_runtime.outputs_dir,
        workflows={},
        workflow_dir=tmp_path,
        video_extensions=VIDEO_EXTENSIONS,
        image_extensions=IMAGE_EXTENSIONS,
        audio_extensions=AUDIO_EXTENSIONS,
        css="",
        thumbnail_dir=dst_runtime.gallery_thumbnails_dir,
    )
    ws_client = TestClient(build_server(demo, [str(dst_runtime.output_dir)], ws_server_cfg))

    def assert_all_posters_served(imported_board: dict[str, Any]) -> None:
        checks = [
            (imported_board["takes"][tk0_id]["video"], expected_posters[tk0_video.resolve()]),
            (imported_board["takes"][tk1_id]["video"], expected_posters[tk1_video.resolve()]),
            (imported_board["scenes"][0]["ref_videos"][0], expected_posters[stored_ref_video.resolve()]),
            (imported_board["scenes"][1]["video"], expected_posters[stored_clip.resolve()]),
        ]
        for video_str, expected_jpg in checks:
            resp = ws_client.get("/video-posters", params={"path": video_str})
            assert resp.status_code == 200, f"missing poster for {video_str}"
            assert resp.headers["content-type"] == "image/jpeg"
            assert resp.content == expected_jpg

    imported = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
    assert imported["status"] == "imported"
    dst_board = dst_service.load("imported", imported["target_id"])
    assert_all_posters_served(dst_board)

    # Importing a renamed copy allocates a new target_id and maps posters to the new video paths.
    renamed = import_project_archive(archive, store=dst_store, runtime=dst_runtime, conflict="rename")
    assert renamed["status"] == "renamed"
    assert renamed["target_id"] != imported["target_id"]
    renamed_board = dst_service.load("imported", renamed["target_id"])
    assert_all_posters_served(renamed_board)

    # Simulate an older Workshop import missing cached posters: incremental overwrite sync backfills
    # only the posters while reusing the local MP4s, then reports up_to_date on the next sync.
    for poster_file in dst_runtime.gallery_thumbnails_dir.glob("*.jpg"):
        poster_file.unlink()
    assert ws_client.get("/video-posters", params={"path": dst_board["takes"][tk0_id]["video"]}).status_code == 404

    studio_app = FastAPI()
    install_project_transfer_routes(studio_app, store=src_store, runtime_config=lambda: src_runtime)
    with _serve_fastapi(studio_app) as studio_url:
        backfill = import_remote_projects(
            studio_url,
            [("storyboard", board_id, "Fox Mystery")],
            store=dst_store,
            runtime=dst_runtime,
            conflict="overwrite",
        )
        assert backfill[0]["status"] == "overwritten"
        assert backfill[0]["incremental"] is True
        assert backfill[0]["reused_files"] >= 4
        assert_all_posters_served(dst_service.load("imported", imported["target_id"]))

        second = import_remote_projects(
            studio_url,
            [("storyboard", board_id, "Fox Mystery")],
            store=dst_store,
            runtime=dst_runtime,
            conflict="overwrite",
        )
        assert second[0]["status"] == "up_to_date"


def test_storyboard_takes_in_use_roundtrip_and_merge_back(tmp_path: Path):
    ws_runtime, ws_store = _make_workspace(tmp_path / "workshop")
    gpu_runtime, gpu_store = _make_workspace(tmp_path / "gpu")
    ws_sb = StoryboardService(ws_store, lambda: ws_runtime)
    gpu_sb = StoryboardService(gpu_store, lambda: gpu_runtime)

    board_id = ws_sb.create(
        "ws-owner",
        "In Use Board",
        sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
    )
    ws_sb.add_scene("ws-owner", board_id)
    board = ws_sb.load("ws-owner", board_id)
    sc0_id = board["scenes"][0]["id"]
    sc1_id = board["scenes"][1]["id"]
    ws_sb.update_scene("ws-owner", board_id, sc0_id, prompt="Scene zero")
    ws_sb.update_scene("ws-owner", board_id, sc1_id, prompt="Scene one", link=sb_model.CUT)

    ver_pic = tmp_path / "ver_only.png"
    ver_pic.write_bytes(b"version-only-pic")
    ws_sb.set_scene_media("ws-owner", board_id, sc0_id, "first_image", ver_pic)
    ws_sb.save_version("ws-owner", board_id, "Version 1")
    ws_sb.set_scene_media("ws-owner", board_id, sc0_id, "first_image", None)

    def make_take(service: StoryboardService, bid: str, sid: str, tid: str, video_bytes: bytes, idx: int) -> Path:
        prefix = service.take_prefix(bid, sid, tid)
        vid = service.output_path(f"{prefix}.mp4")
        vid.parent.mkdir(parents=True, exist_ok=True)
        vid.write_bytes(video_bytes)
        write_snapshot(vid, {"family": "MiniMax H3", "settings": {"seed": idx}})
        lat = f"{prefix}.safetensors"
        service.output_path(lat).write_bytes(f"latent-{tid}".encode())
        b = service.load("ws-owner", bid)
        s_idx = service.scene_index(b, sid)
        service.record_take(
            "ws-owner",
            bid,
            sid,
            tid,
            video=str(vid),
            latents={"latent": lat, "stage_latent": None, "prefix_frames": 0},
            seed=idx,
            inputs=sb_model.scene_inputs(b, s_idx),
        )
        return vid

    t0a, t0b, t1a, t1b = "t0a_old", "t0b_cur", "t1a_old", "t1b_cur"
    v0a = make_take(ws_sb, board_id, sc0_id, t0a, b"unused-take-0a", 1)
    v0b = make_take(ws_sb, board_id, sc0_id, t0b, b"active-take-0b", 2)
    ws_sb.use_take("ws-owner", board_id, sc0_id, t0b)
    v1a = make_take(ws_sb, board_id, sc1_id, t1a, b"unused-take-1a", 3)
    v1b = make_take(ws_sb, board_id, sc1_id, t1b, b"active-take-1b", 4)
    ws_sb.use_take("ws-owner", board_id, sc1_id, t1b)

    v0a_mtime = v0a.stat().st_mtime_ns
    v1a_mtime = v1a.stat().st_mtime_ns

    archive = export_project_archive(
        "storyboard",
        board_id,
        tmp_path / "in_use.zip",
        store=ws_store,
        runtime=ws_runtime,
        storyboard_takes="in_use",
    )
    with zipfile.ZipFile(archive, "r") as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["storyboard_takes"] == "in_use"
    assert any(v0b.name in n for n in names)
    assert any(v1b.name in n for n in names)
    assert not any(v0a.name in n for n in names)
    assert not any(v1a.name in n for n in names)
    assert not any(n.startswith("media/") for n in names)

    ws_app = FastAPI()
    install_project_transfer_routes(ws_app, store=ws_store, runtime_config=lambda: ws_runtime)
    gpu_app = FastAPI()
    install_project_transfer_routes(gpu_app, store=gpu_store, runtime_config=lambda: gpu_runtime)

    with _serve_fastapi(ws_app) as ws_url, _serve_fastapi(gpu_app) as gpu_url:
        res_gpu = import_remote_projects(
            ws_url,
            [("storyboard", board_id, "In Use Board")],
            store=gpu_store,
            runtime=gpu_runtime,
            conflict="rename",
            storyboard_takes="in_use",
        )
        assert res_gpu[0]["status"] == "imported"
        assert res_gpu[0]["storyboard_takes"] == "in_use"
        gpu_board = gpu_sb.load("imported", board_id)
        assert set(gpu_board["takes"]) == {t0b, t1b}
        assert gpu_board["scenes"][0]["takes"] == [t0b]
        assert gpu_board["scenes"][1]["takes"] == [t1b]
        assert gpu_board["versions"] == []
        assert sb_model.scene_states(gpu_board) == [sb_model.DONE, sb_model.DONE]

        # Generate a new alternate take on GPU for Scene 1 and merge back into Workshop.
        t1c = "t1c_gpu"
        make_take(gpu_sb, board_id, sc1_id, t1c, b"gpu-new-take-1c", 5)
        gpu_sb.use_take("imported", board_id, sc1_id, t1c)

        res_ws = import_remote_projects(
            gpu_url,
            [("storyboard", board_id, "In Use Board")],
            store=ws_store,
            runtime=ws_runtime,
            conflict="overwrite",
        )
        assert res_ws[0]["status"] == "overwritten"
        assert res_ws[0]["preserved_local_takes"] == 2
        merged = ws_sb.load("ws-owner", board_id)
        assert "_partial_takes" not in merged
        assert set(merged["takes"]) == {t0a, t0b, t1a, t1b, t1c}
        assert merged["scenes"][0]["takes"] == [t0a, t0b]
        assert merged["scenes"][1]["takes"] == [t1a, t1b, t1c]
        assert merged["scenes"][1]["in_use"] == t1c
        assert len(merged["versions"]) == 1
        assert v0a.is_file() and v0a.stat().st_mtime_ns == v0a_mtime
        assert v1a.is_file() and v1a.stat().st_mtime_ns == v1a_mtime


def test_storyboard_takes_none_workshop_gpu_roundtrip(tmp_path: Path):
    from h3_app.media_review import review_path
    from h3_ui.storyboard.payload import board_payload

    ws_runtime, ws_store = _make_workspace(tmp_path / "workshop")
    gpu_runtime, gpu_store = _make_workspace(tmp_path / "gpu")
    ws_sb = StoryboardService(ws_store, lambda: ws_runtime)
    gpu_sb = StoryboardService(gpu_store, lambda: gpu_runtime)

    cast_pic = tmp_path / "actor.png"
    cast_pic.write_bytes(b"actor-png")
    board_id = ws_sb.create(
        "ws-owner",
        "Remote Render Board",
        sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
        cast=[("Actor", cast_pic)],
    )
    for _ in range(4):
        ws_sb.add_scene("ws-owner", board_id)

    board = ws_sb.load("ws-owner", board_id)
    sc1_id, sc2_id, sc3_id, sc4_id, sc5_id = [s["id"] for s in board["scenes"]]
    ws_sb.update_scene("ws-owner", board_id, sc1_id, prompt="Shot 1", link=sb_model.CUT)
    ws_sb.update_scene("ws-owner", board_id, sc2_id, prompt="Shot 2 original", link=sb_model.CUT)
    ws_sb.update_scene("ws-owner", board_id, sc3_id, prompt="Shot 3", link=sb_model.CUT)
    ws_sb.update_scene("ws-owner", board_id, sc4_id, prompt="Shot 4 original", link=sb_model.FLOW)
    ws_sb.update_scene("ws-owner", board_id, sc5_id, prompt="Shot 5 new draft", link=sb_model.CUT)

    heavy_vid_payload = os.urandom(120_000)
    t3_latent_bytes = os.urandom(8_000)
    t3_stage_bytes = os.urandom(6_000)

    def seed_take(
        service: StoryboardService,
        sid: str,
        tid: str,
        *,
        video_bytes: bytes,
        latent_bytes: bytes | None = None,
        stage_bytes: bytes | None = None,
        seed: int = 1,
        add_review: bool = False,
    ) -> Path:
        prefix = service.take_prefix(board_id, sid, tid)
        vid = service.output_path(f"{prefix}.mp4")
        vid.parent.mkdir(parents=True, exist_ok=True)
        vid.write_bytes(video_bytes)
        write_snapshot(vid, {"family": "MiniMax H3", "settings": {"seed": seed}})
        if add_review:
            review_path(vid).write_text(json.dumps({"decision": {"score": 9}}), encoding="utf-8")
        lat_rel = None
        if latent_bytes is not None:
            lat_rel = f"{prefix}.safetensors"
            service.output_path(lat_rel).write_bytes(latent_bytes)
        stage_rel = None
        if stage_bytes is not None:
            stage_rel = f"{prefix}_stage.safetensors"
            service.output_path(stage_rel).write_bytes(stage_bytes)
        b = service.load("ws-owner", board_id)
        s_idx = service.scene_index(b, sid)
        service.record_take(
            "ws-owner",
            board_id,
            sid,
            tid,
            video=str(vid),
            latents={"latent": lat_rel, "stage_latent": stage_rel, "prefix_frames": 0},
            seed=seed,
            inputs=sb_model.scene_inputs(b, s_idx),
        )
        return vid

    t1a_vid = seed_take(ws_sb, sc1_id, "t1a_alt", video_bytes=heavy_vid_payload, latent_bytes=b"t1a-lat", seed=1)
    t1b_vid = seed_take(
        ws_sb, sc1_id, "t1b_use", video_bytes=heavy_vid_payload, latent_bytes=b"t1b-lat", seed=2, add_review=True
    )
    ws_sb.use_take("ws-owner", board_id, sc1_id, "t1b_use")
    t2a_vid = seed_take(ws_sb, sc2_id, "t2a_old", video_bytes=heavy_vid_payload, latent_bytes=b"t2a-lat", seed=3)
    t3a_vid = seed_take(
        ws_sb,
        sc3_id,
        "t3a_use",
        video_bytes=heavy_vid_payload,
        latent_bytes=t3_latent_bytes,
        stage_bytes=t3_stage_bytes,
        seed=4,
    )
    t4a_vid = seed_take(ws_sb, sc4_id, "t4a_old", video_bytes=heavy_vid_payload, latent_bytes=b"t4a-lat", seed=5)

    ws_sb.save_version("ws-owner", board_id, "Saved Cut")

    # User edits Shot 2 and Shot 4 prompts in Workshop, making them OUTDATED.
    ws_sb.update_scene("ws-owner", board_id, sc2_id, prompt="Shot 2 edited")
    ws_sb.update_scene("ws-owner", board_id, sc4_id, prompt="Shot 4 edited")
    assert sb_model.scene_states(ws_sb.load("ws-owner", board_id)) == [
        sb_model.DONE,
        sb_model.OUTDATED,
        sb_model.DONE,
        sb_model.OUTDATED,
        sb_model.DRAFT,
    ]

    t1a_mtime = t1a_vid.stat().st_mtime_ns
    t1b_mtime = t1b_vid.stat().st_mtime_ns
    t1b_rev_mtime = review_path(t1b_vid).stat().st_mtime_ns
    t3a_mtime = t3a_vid.stat().st_mtime_ns

    ws_app = FastAPI()
    ws_wire_bytes: list[tuple[str, int]] = []

    @ws_app.middleware("http")
    async def measure_ws_bytes(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path in {"/api/projects/transfer/export", "/api/projects/transfer/delta"}:
            chunks = [chunk async for chunk in resp.body_iterator]
            raw = b"".join(chunks)
            ws_wire_bytes.append((request.url.path, len(raw)))
            from starlette.responses import Response

            return Response(
                content=raw,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                media_type=resp.media_type,
            )
        return resp

    install_project_transfer_routes(ws_app, store=ws_store, runtime_config=lambda: ws_runtime)
    gpu_app = FastAPI()
    gpu_wire_bytes: list[tuple[str, int]] = []

    @gpu_app.middleware("http")
    async def measure_gpu_bytes(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path in {"/api/projects/transfer/export", "/api/projects/transfer/delta"}:
            chunks = [chunk async for chunk in resp.body_iterator]
            raw = b"".join(chunks)
            gpu_wire_bytes.append((request.url.path, len(raw)))
            from starlette.responses import Response

            return Response(
                content=raw,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                media_type=resp.media_type,
            )
        return resp

    install_project_transfer_routes(gpu_app, store=gpu_store, runtime_config=lambda: gpu_runtime)

    with _serve_fastapi(ws_app) as ws_url, _serve_fastapi(gpu_app) as gpu_url:
        # 1. Temp GPU Studio imports storyboard without takes (only Scene 3 continuation latents transfer).
        gpu_import = import_remote_projects(
            ws_url,
            [("storyboard", board_id, "Remote Render Board")],
            store=gpu_store,
            runtime=gpu_runtime,
            conflict="rename",
            storyboard_takes="none",
        )
        assert gpu_import[0]["status"] == "imported"
        assert gpu_import[0]["storyboard_takes"] == "none"
        # 5 scenes x 120KB = 600KB of MP4 videos skipped; only ~14KB of Scene 3 continuation latents sent.
        assert ws_wire_bytes and ws_wire_bytes[0][1] < 30_000

        gpu_board = gpu_sb.load("imported", board_id)
        assert sb_model.scene_states(gpu_board) == [
            sb_model.DONE,
            sb_model.OUTDATED,
            sb_model.DONE,
            sb_model.OUTDATED,
            sb_model.DRAFT,
        ]
        assert sb_model.pending_scenes(gpu_board) == [1, 3, 4]
        assert gpu_board["takes"]["t1b_use"]["omitted"] is True
        assert gpu_board["takes"]["t1b_use"]["video"] is None
        assert gpu_board["takes"]["t1b_use"]["latent"] is None
        assert gpu_board["takes"]["t3a_use"]["omitted"] is True
        assert gpu_board["takes"]["t3a_use"]["video"] is None
        assert gpu_sb.output_path(gpu_board["takes"]["t3a_use"]["latent"]).read_bytes() == t3_latent_bytes
        assert gpu_sb.output_path(gpu_board["takes"]["t3a_use"]["stage_latent"]).read_bytes() == t3_stage_bytes

        # UI payload & clips safeguard check
        ui_payload = board_payload(board_id, gpu_board)
        assert ui_payload["scenes"][0]["takes"][0]["omitted"] is True
        assert ui_payload["scenes"][0]["takes"][0]["missing"] is True
        assert ui_payload["scenes"][0]["takes"][0]["video"] is None
        with pytest.raises(H3Error, match="wasn't transferred"):
            gpu_sb.clips(gpu_board)

        # If user switches Scene 2 to flow from Scene 1 (whose latent wasn't transferred), scene_issues flags it.
        board_flow_check = copy.deepcopy(gpu_board)
        board_flow_check["scenes"][1]["link"] = sb_model.FLOW
        assert any("wasn't transferred" in issue for issue in sb_model.scene_issues(board_flow_check, 1))

        # Re-importing unchanged storyboard with overwrite + storyboard_takes="none" detects up_to_date.
        noop_gpu = import_remote_projects(
            ws_url,
            [("storyboard", board_id, "Remote Render Board")],
            store=gpu_store,
            runtime=gpu_runtime,
            conflict="overwrite",
            storyboard_takes="none",
        )
        assert noop_gpu[0]["status"] == "up_to_date"

        # 2. Temp GPU Studio generates new takes for outdated/draft scenes (Scenes 2, 4, 5).
        def gpu_generate(sid: str, tid: str, video_bytes: bytes, seed: int) -> None:
            prefix = gpu_sb.take_prefix(board_id, sid, tid)
            vid = gpu_sb.output_path(f"{prefix}.mp4")
            vid.parent.mkdir(parents=True, exist_ok=True)
            vid.write_bytes(video_bytes)
            lat_rel = f"{prefix}.safetensors"
            gpu_sb.output_path(lat_rel).write_bytes(f"gpu-lat-{tid}".encode())
            b = gpu_sb.load("imported", board_id)
            s_idx = gpu_sb.scene_index(b, sid)
            gpu_sb.record_take(
                "imported",
                board_id,
                sid,
                tid,
                video=str(vid),
                latents={"latent": lat_rel, "stage_latent": None, "prefix_frames": 0},
                seed=seed,
                inputs=sb_model.scene_inputs(b, s_idx),
            )

        gpu_generate(sc2_id, "t2_new", b"gpu-rendered-scene2", 20)
        gpu_generate(sc4_id, "t4_new", b"gpu-rendered-scene4", 40)
        gpu_generate(sc5_id, "t5_new", b"gpu-rendered-scene5", 50)
        assert sb_model.scene_states(gpu_sb.load("imported", board_id)) == [sb_model.DONE] * 5

        # Partial->partial overwrite preserves omitted_versions=True when target also has no versions.
        gpu2_runtime, gpu2_store = _make_workspace(tmp_path / "gpu2")
        gpu2_sb = StoryboardService(gpu2_store, lambda: gpu2_runtime)
        import_remote_projects(
            ws_url,
            [("storyboard", board_id, "Remote Render Board")],
            store=gpu2_store,
            runtime=gpu2_runtime,
            conflict="rename",
            storyboard_takes="none",
        )
        import_remote_projects(
            gpu_url,
            [("storyboard", board_id, "Remote Render Board")],
            store=gpu2_store,
            runtime=gpu2_runtime,
            conflict="overwrite",
        )
        gpu2_board = gpu2_sb.load("imported", board_id)
        assert gpu2_board["_partial_takes"]["omitted_versions"] is True

        # 3. Workshop imports back with conflict="overwrite": merges new takes & preserves all local takes!
        gpu_wire_bytes.clear()
        ws_sync = import_remote_projects(
            gpu_url,
            [("storyboard", board_id, "Remote Render Board")],
            store=ws_store,
            runtime=ws_runtime,
            conflict="overwrite",
        )
        assert ws_sync[0]["status"] == "overwritten"
        assert ws_sync[0]["preserved_local_takes"] == 5
        assert ") (" not in ws_sync[0]["message"]
        assert "kept 5 local takes" in ws_sync[0]["message"]
        assert gpu_wire_bytes and gpu_wire_bytes[0][0] == "/api/projects/transfer/delta"
        assert gpu_wire_bytes[0][1] < 40_000

        final_ws = ws_sb.load("ws-owner", board_id)
        assert "_partial_takes" not in final_ws
        assert sb_model.scene_states(final_ws) == [sb_model.DONE] * 5
        assert set(final_ws["takes"]) == {
            "t1a_alt",
            "t1b_use",
            "t2a_old",
            "t2_new",
            "t3a_use",
            "t4a_old",
            "t4_new",
            "t5_new",
        }
        assert final_ws["scenes"][0]["takes"] == ["t1a_alt", "t1b_use"]
        assert final_ws["scenes"][0]["in_use"] == "t1b_use"
        assert final_ws["scenes"][1]["takes"] == ["t2a_old", "t2_new"]
        assert final_ws["scenes"][1]["in_use"] == "t2_new"
        assert final_ws["scenes"][3]["takes"] == ["t4a_old", "t4_new"]
        assert final_ws["scenes"][3]["in_use"] == "t4_new"
        assert final_ws["scenes"][4]["takes"] == ["t5_new"]
        assert final_ws["scenes"][4]["in_use"] == "t5_new"
        assert len(final_ws["versions"]) == 1
        assert t1a_vid.is_file() and t1a_vid.stat().st_mtime_ns == t1a_mtime
        assert t1b_vid.is_file() and t1b_vid.stat().st_mtime_ns == t1b_mtime
        assert review_path(t1b_vid).is_file() and review_path(t1b_vid).stat().st_mtime_ns == t1b_rev_mtime
        assert t3a_vid.is_file() and t3a_vid.stat().st_mtime_ns == t3a_mtime
        assert Path(final_ws["takes"]["t2_new"]["video"]).read_bytes() == b"gpu-rendered-scene2"
        assert Path(final_ws["takes"]["t4_new"]["video"]).read_bytes() == b"gpu-rendered-scene4"
        assert Path(final_ws["takes"]["t5_new"]["video"]).read_bytes() == b"gpu-rendered-scene5"


def test_export_archive_cache_isolates_storyboard_takes_modes_and_validates(tmp_path: Path):
    src_runtime, src_store = _make_workspace(tmp_path / "src")
    src_sb = StoryboardService(src_store, lambda: src_runtime)
    board_id = src_sb.create(
        "owner-a",
        "Cache Modes",
        sb_model.make_profile(sb_model.request_defaults(), sb_model.request_defaults()),
    )
    board = src_sb.load("owner-a", board_id)
    sc0_id = board["scenes"][0]["id"]
    src_sb.update_scene("owner-a", board_id, sc0_id, prompt="Shot")
    for tid in ("tk_old", "tk_cur"):
        vid = src_sb.output_path(f"{src_sb.take_prefix(board_id, sc0_id, tid)}.mp4")
        vid.parent.mkdir(parents=True, exist_ok=True)
        vid.write_bytes(f"vid-{tid}".encode())
        b = src_sb.load("owner-a", board_id)
        src_sb.record_take(
            "owner-a", board_id, sc0_id, tid, video=str(vid), latents={"latent": None}, seed=1,
            inputs=sb_model.scene_inputs(b, 0),
        )
    src_sb.use_take("owner-a", board_id, sc0_id, "tk_cur")

    app = FastAPI()
    install_project_transfer_routes(app, store=src_store, runtime_config=lambda: src_runtime)
    with _serve_fastapi(app) as base_url:
        r_all = httpx.get(f"{base_url}/api/projects/transfer/export", params={"kind": "storyboard", "id": board_id})
        r_use = httpx.get(
            f"{base_url}/api/projects/transfer/export",
            params={"kind": "storyboard", "id": board_id, "storyboard_takes": "in_use"},
        )
        r_none = httpx.get(
            f"{base_url}/api/projects/transfer/export",
            params={"kind": "storyboard", "id": board_id, "storyboard_takes": "none"},
        )
        assert r_all.status_code == 200 and r_use.status_code == 200 and r_none.status_code == 200
        assert len(r_all.content) > len(r_use.content) > len(r_none.content)

        bad_export = httpx.get(
            f"{base_url}/api/projects/transfer/export",
            params={"kind": "storyboard", "id": board_id, "storyboard_takes": "invalid"},
        )
        assert bad_export.status_code == 400
        bad_delta = httpx.post(
            f"{base_url}/api/projects/transfer/delta",
            json={"kind": "storyboard", "id": board_id, "storyboard_takes": "invalid"},
        )
        assert bad_delta.status_code == 400
        bad_import = httpx.post(
            f"{base_url}/api/projects/transfer/import",
            json={"url": base_url, "items": [{"kind": "storyboard", "id": board_id}], "storyboard_takes": "invalid"},
        )
        assert bad_import.status_code == 400
