"""Run Fibo inside ComfyUI's owned queue without importing its Diffusers fork."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def run_worker(python, request_path, *, check_cancelled, progress):
    """Keep cancellation responsive during downloads, loading and sampling; reap the GPU child."""
    request_path = Path(request_path)
    log_path = request_path.with_suffix(".log")
    progress_path = request_path.with_suffix(".progress")
    with log_path.open("wb") as log:
        check_cancelled()
        process = subprocess.Popen(
            [str(python), str(Path(__file__).with_name("worker.py")), str(request_path)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        previous = 0
        try:
            while process.poll() is None:
                check_cancelled()
                try:
                    current = int(progress_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    # Absent until the first step; Windows may deny access mid-rename.
                    current = previous
                if current != previous:
                    progress(current)
                    previous = current
                time.sleep(0.1)
            check_cancelled()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode:
            with log_path.open("rb") as diagnostic:
                diagnostic.seek(max(0, log_path.stat().st_size - 65536))
                detail = diagnostic.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Fibo 2.0 worker failed:\n{detail}")


class H3Fibo2Generate:
    CATEGORY = "H3/image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_choice": (["Dev", "Edit Turbo"],),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
            },
            "optional": {**{f"image_{i}": ("IMAGE",) for i in range(1, 5)}, "mask_image": ("IMAGE",)},
        }

    def generate(self, model_choice, prompt, width, height, seed, steps, cfg, **images):
        import comfy.model_management as mm
        import folder_paths
        import numpy as np
        import torch
        from comfy.utils import ProgressBar
        from PIL import Image

        python = Path(os.environ.get("H3_FIBO2_PYTHON") or (
            Path(folder_paths.base_path) / ".fibo2-venv" /
            ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        ))
        if not python.is_file():
            raise RuntimeError("Fibo runtime is missing. Rerun setup_h3.py without --skip-env and restart ComfyUI.")
        with tempfile.TemporaryDirectory(prefix="h3-fibo2-", dir=folder_paths.get_temp_directory()) as temp:
            directory = Path(temp)
            references = []
            mask_path = None
            for name in [*(f"image_{i}" for i in range(1, 5)), "mask_image"]:
                tensor = images.get(name)
                if tensor is None:
                    continue
                if len(tensor) != 1:
                    raise ValueError("Fibo expects one image per reference slot.")
                path = directory / f"{name}.png"
                array = (tensor[0].detach().cpu().float().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                Image.fromarray(array).save(path)
                if name == "mask_image":
                    mask_path = str(path)
                else:
                    references.append(str(path))
            request = dict(model_choice=model_choice, prompt=prompt, width=width, height=height,
                           seed=seed, steps=steps, cfg=cfg, references=references, mask=mask_path,
                           cache_dir=str(Path(folder_paths.models_dir) / "fibo2"),
                           output=str(directory / "result.png"))
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            mm.throw_exception_if_processing_interrupted()
            mm.unload_all_models()
            mm.soft_empty_cache()
            progress = ProgressBar(steps)
            run_worker(python, request_path, check_cancelled=mm.throw_exception_if_processing_interrupted,
                       progress=lambda step: progress.update_absolute(step, steps))
            with Image.open(request["output"]) as result:
                array = np.array(result.convert("RGB"), dtype=np.float32) / 255.0
            progress.update_absolute(steps, steps)
            return (torch.from_numpy(array).unsqueeze(0),)
