"""A small queued Gradio app with gated progress for browser transport checks."""

from pathlib import Path
import sys
import threading
import time

import gradio as gr
import uvicorn
from starlette.routing import Mount

from h3_app.server import ServerConfig, build_server


CONTINUE = threading.Event()


def generation(progress=gr.Progress(track_tqdm=False)):
    progress((1, 3), desc="Sampling fixture")
    yield "Sampling step 1 / 3"
    if not CONTINUE.wait(timeout=20):
        raise gr.Error("Browser did not receive intermediate progress")
    progress((2, 3), desc="Decoding fixture")
    yield "Decoding step 2 / 3"
    time.sleep(0.5)
    progress(1, desc="Complete")
    yield "Generation complete"


with gr.Blocks() as demo:
    run = gr.Button("Generate fixture")
    status = gr.Textbox(label="Generation progress")
    run.click(generation, outputs=status)
    demo.load(lambda: "Ready", outputs=status)
demo.queue()
root = Path(sys.argv[2])
config = ServerConfig(
    comfy_url="http://127.0.0.1:9", output_dir=root, outputs_dir=root,
    workflows={}, workflow_dir=root,
    video_extensions=frozenset(), image_extensions=frozenset(),
    audio_extensions=frozenset(), css="",
)
server = build_server(demo, [], config)


@server.post("/test/continue")
def continue_generation():
    CONTINUE.set()
    return {"released": True}


@server.get("/test/ready")
def ready():
    return {"ready": True}


server.router.routes.sort(key=lambda route: isinstance(route, Mount))
uvicorn.run(server, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
