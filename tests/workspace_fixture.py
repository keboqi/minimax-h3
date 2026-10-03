"""Local backend-free server used only by browser_workspace.py."""

import sys
import time
import gradio as gr
import gradio_app as app
from h3_app.jobs import CURRENT_JOB, JOBS, record_failure


def fake_generation(batch_count, *args):
    from h3_app.contracts import GENERATION_FIELDS

    values = dict(zip(GENERATION_FIELDS, args, strict=True))
    job = CURRENT_JOB.get()
    job.variant_seeds[0] = 314159 if int(values["seed"]) < 0 else int(values["seed"])
    job.state = "running"
    job.stage = "Mock generation: " + values["prompt"]
    yield (*(gr.skip() for _ in range(11)), job.stage)
    for _ in range(40):
        job.check()
        time.sleep(0.1)
    if "fail" in values["prompt"]:
        record_failure(RuntimeError("Fixture variant failed"))
    yield (*(gr.skip() for _ in range(11)), "Mock generation finished")


app.backend_status = lambda: "Connected browser test fixture"
app.generate_for_ui = fake_generation
app.enhance_h3_prompt = lambda *args: (
    args[0] + " Enhanced fixture.",
    "Enhancement finished.",
)
demo = app.build_ui().queue(default_concurrency_limit=1, max_size=8)
server, _, _ = demo.launch(
    server_name="127.0.0.1",
    server_port=int(sys.argv[1]),
    inbrowser=False,
    ssr_mode=False,
    css=demo.h3_css,
    prevent_thread_lock=True,
    theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate"),
)


@server.get("/test/jobs")
def jobs():
    return [
        {
            "id": job.id,
            "owner": job.owner,
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


while True:
    time.sleep(1)
