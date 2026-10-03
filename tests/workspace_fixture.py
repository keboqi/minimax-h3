"""Local backend-free server used only by browser_workspace.py."""

import uvicorn
from starlette.routing import Mount
import sys
import time
import inspect
import subprocess
from PIL import Image
import gradio as gr
from h3_ui import application as app
from h3_app.jobs import CURRENT_JOB, JOBS, record_failure
from threading import Event

RELEASE_A = Event()


def fake_generation(batch_count, *args):
    from h3_app.contracts import GENERATION_FIELDS

    values = dict(zip(GENERATION_FIELDS, args, strict=True))
    job = CURRENT_JOB.get()
    job.variant_seeds[0] = 314159 if int(values["seed"]) < 0 else int(values["seed"])
    job.state = "running"
    job.stage = "Mock generation: " + values["prompt"]
    yield (*(gr.skip() for _ in range(11)), job.stage)
    if values["prompt"] == "Request A":
        while not RELEASE_A.wait(0.05):
            job.check()
    for _ in range(40):
        job.check()
        time.sleep(0.1)
    if "fail" in values["prompt"]:
        record_failure(RuntimeError("Fixture variant failed"))
    yield (*(gr.skip() for _ in range(11)), "Mock generation finished")


app.backend_status = lambda: "Connected browser test fixture"
app.generate_for_ui = fake_generation


def fake_engine(original, family):
    signature = inspect.signature(original)
    names = [
        name for name in signature.parameters if name not in {"request", "progress"}
    ]

    def run(*args, **kwargs):
        job = CURRENT_JOB.get()
        values = dict(zip(names, args))
        job.variant_seeds[0] = int(values.get("seed", 123))
        job.state = "running"
        job.stage = f"Fixture {family} generation"
        yield gr.skip(), job.stage
        time.sleep(0.3)
        yield gr.skip(), f"Fixture {family} completed"

    run.__signature__ = signature
    run.__name__ = original.__name__
    return run


for callback, family in (
    ("generate_ltx25", "ltx"),
    ("generate_music3", "music"),
    ("generate_qwen_image21", "qwen_image21"),
    ("generate_yue2", "yue2"),
):
    setattr(app, callback, fake_engine(getattr(app, callback), family))

app.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
for name, color in (("alpha.png", "red"), ("beta.png", "blue")):
    Image.new("RGB", (320, 256), color).save(app.OUTPUTS_DIR / name)
for name, rate, duration, color in (
    ("alpha.mp4", 24, 1, "red"),
    ("beta.mp4", 30, 2, "blue"),
):
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x240:r={rate}:d={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(app.OUTPUTS_DIR / name),
        ],
        check=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
app.enhance_h3_prompt = lambda *args: (
    args[0] + " Enhanced fixture.",
    "Enhancement finished.",
)
demo = app.build_ui().queue(default_concurrency_limit=1, max_size=8)
server = app.build_server(
    demo, [str(app.OUTPUT_DIR.resolve()), str(app.OUTPUTS_DIR.resolve())]
)


@server.get("/test/jobs")
def jobs():
    return [
        {
            "id": job.id,
            "owner": job.owner,
            "family": job.family,
            "state": job.state,
            "values": job.values(),
            "seeds": {str(i): seed for i, seed in job.variant_seeds.items()},
            "retry_of": job.retry_of,
        }
        for job in JOBS.records.values()
        if job.snapshot_json
    ]


@server.post("/test/queue/{limit}")
def queue_capacity(limit: int):
    demo._queue.max_size = limit
    return {"limit": limit}


@server.post("/test/release-a")
def release_a():
    RELEASE_A.set()
    return {"released": True}


server.router.routes.sort(key=lambda route: isinstance(route, Mount))
uvicorn.run(server, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
