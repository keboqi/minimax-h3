"""Tests for cross-server project export, import, routes, and Workshop/Studio wiring."""

from __future__ import annotations

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
    export_project_archive,
    fetch_remote_projects,
    import_project_archive,
    normalize_server_url,
)
from h3_app.provenance import snapshot_path, write_snapshot
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
        assert any(
            c.get("props", {}).get("label") == "Remote Studio or Workshop URL"
            for c in ui.get_config_file()["components"]
        )
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
