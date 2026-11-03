"""The Storyboard tab on the real server stack, with takes faked so no GPU is needed."""

from pathlib import Path
import struct
import sys
import threading
import wave

import gradio as gr
from PIL import Image
from fastapi import Request
from starlette.routing import Mount
import uvicorn

from h3_app import storyboard as sb
from h3_app.config import RuntimeConfig
from h3_app.generation.results import GenerationUpdate
from h3_app.jobs import JOBS
from h3_app.server import ServerConfig, build_server
from h3_app.workspace_store import COOKIE, WorkspaceStore
from h3_ui.storyboard.handlers import StoryboardHandlers, StoryboardServices
from h3_ui.storyboard.tab import build_storyboard_tab

root = Path(sys.argv[2])
store = WorkspaceStore(root / "state")
JOBS.configure(store)
runtime = RuntimeConfig(script_dir=root, comfy_url="http://comfy", comfy_dir=root / "comfy",
                        models_config=root / "models.json", outputs_dir=root / "outputs")
runtime.output_dir.mkdir(parents=True, exist_ok=True)
service = sb.StoryboardService(store, lambda: runtime)
CONTINUE = threading.Event()
CONTINUE.set()


def fake_take(values, graph, outputs):
    yield GenerationUpdate(None, "Sampling")
    if not CONTINUE.wait(timeout=30):
        raise RuntimeError("The test never released the take.")
    video = runtime.output_dir / f"{graph.output_prefix}_00001_.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"mp4")
    latent = runtime.output_dir / f"{graph.latent_output_prefix}_00001_.safetensors"
    latent.write_bytes(struct.pack("<Q", 2) + b"{}")
    outputs["latent"] = latent.relative_to(runtime.output_dir).as_posix()
    yield GenerationUpdate(str(video), "Completed")


def interrupt(request, family):
    return JOBS.cancel(store.verify_owner(request.cookies.get(COOKIE)), family, None, None)


handlers = StoryboardHandlers(StoryboardServices(
    service=lambda: service, generate_take=fake_take, request_defaults=sb.request_defaults,
    publish=lambda path, metadata: None,
))
with gr.Blocks() as demo:
    root_column = gr.Column()
    build_storyboard_tab(root_column, demo, handlers, interrupt=interrupt)
demo.queue()
config = ServerConfig(
    comfy_url="http://127.0.0.1:9", output_dir=runtime.output_dir, outputs_dir=runtime.outputs_dir,
    workflows={}, workflow_dir=root, video_extensions=frozenset({".mp4"}), image_extensions=frozenset({".png"}),
    audio_extensions=frozenset({".wav"}), css=".h3-storyboard-frame { width: 100%; height: 900px; border: 0; }",
    thumbnail_dir=runtime.gallery_thumbnails_dir,
)
server = build_server(demo, [str(runtime.output_dir)], config)


def media(name, color):
    picture = root / f"{name}.png"
    Image.new("RGB", (1600, 1600), color).save(picture)
    voice = root / f"{name}.wav"
    with wave.open(str(voice), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000)
    return str(picture), str(voice)


@server.post("/test/seed")
def seed(request: Request):
    """Two storyboards of eight scenes that differ in every media field, like a real one."""
    owner = store.verify_owner(request.cookies.get(COOKIE))
    boards = []
    for number in range(2):
        profile = sb.make_profile(sb.request_defaults(), sb.request_defaults())
        scenes = [sb.new_scene(f"Scene {index + 1}", f"Board {number} prompt {index + 1}.", seconds=6)
                  for index in range(8)]
        storyboard_id = service.create(owner, f"Board {number}", profile, scenes=scenes)
        for index, scene in enumerate(scenes):
            picture, voice = media(f"b{number}s{index}", (index * 30, number * 90, 80))
            service.set_scene_media(owner, storyboard_id, scene["id"], "first_image", picture)
            service.set_scene_media(owner, storyboard_id, scene["id"], "ref_images", [picture])
            service.set_scene_media(owner, storyboard_id, scene["id"], "ref_audios", [voice])
            service.set_mode(owner, storyboard_id, scene["id"], sb.FRAMES if index % 2 else sb.REFERENCES)
        boards.append(storyboard_id)
    return {"boards": boards}


@server.post("/test/hold")
def hold():
    CONTINUE.clear()
    return {"held": True}


@server.post("/test/release")
def release():
    CONTINUE.set()
    return {"released": True}


@server.get("/test/ready")
def ready():
    return {"ready": True}


server.router.routes.sort(key=lambda route: isinstance(route, Mount))
uvicorn.run(server, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
