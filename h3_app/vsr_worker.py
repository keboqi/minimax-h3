"""Isolated NVIDIA VFX worker: RGB8 frames on stdin/stdout, JSON on stderr.

Run by absolute filename so an independently configured GPU Python can use it
without installing this application's Gradio dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys


def transform_frames(width, height, out_width, out_height, settings):
    import numpy as np
    import torch
    from nvvfx import VideoSuperRes
    from importlib.metadata import version

    if not torch.cuda.is_available():
        raise RuntimeError("VSR requires a CUDA-enabled PyTorch environment.")
    device = int(settings.get("device", 0))
    torch.cuda.set_device(device)
    stream = torch.cuda.current_stream(device)
    effects = []

    def effect(quality, strength, w, h):
        sr = VideoSuperRes(
            quality=VideoSuperRes.QualityLevel[quality], strength=strength,
            device=device, image_encoding=VideoSuperRes.ImageEncoding.RGB8,
        )
        effects.append(sr)
        sr.input_width, sr.input_height = width, height
        sr.output_width, sr.output_height = w, h
        sr.load()
        return sr

    try:
        pre = None
        preprocessing = settings["preprocess"]
        if preprocessing != "None" and (out_width, out_height) != (width, height):
            pre = effect(preprocessing.upper() + "_HIGH", settings["preprocess_strength"], width, height)
        quality = settings["quality"].upper()
        if (out_width, out_height) == (width, height):
            quality = preprocessing.upper() + "_" + quality
        elif settings["profile"] == "High bitrate":
            quality = "HIGHBITRATE_" + quality
        strength = settings["preprocess_strength"] if (out_width, out_height) == (width, height) else settings["strength"]
        sr = effect(quality, strength, out_width, out_height)
        print("H3VFX " + json.dumps({
            "runtime": version("nvidia-vfx"), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device), "quality": quality,
        }), file=sys.stderr, flush=True)
        frame_size = width * height * 3
        frames = 0
        with torch.inference_mode():
            while True:
                raw = bytearray()
                while len(raw) < frame_size:
                    part = sys.stdin.buffer.read(frame_size - len(raw))
                    if not part:
                        break
                    raw.extend(part)
                if not raw:
                    break
                if len(raw) != frame_size:
                    raise RuntimeError("Truncated input RGB frame.")
                image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                gpu = torch.from_numpy(image.copy()).to(device=f"cuda:{device}", dtype=torch.float32)
                gpu = gpu.permute(2, 0, 1).contiguous().div_(255)
                for current in (pre, sr):
                    if current is None:
                        continue
                    stream.synchronize()
                    result = current.run(gpu, stream_ptr=int(stream.cuda_stream))
                    stream.synchronize()
                    gpu = torch.from_dlpack(result.image).clone()
                    stream.synchronize()
                output = gpu.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu()
                data = memoryview(output.numpy().tobytes())
                while data:
                    written = sys.stdout.buffer.write(data)
                    if not written:
                        raise RuntimeError("Output pipe stopped accepting frames.")
                    data = data[written:]
                sys.stdout.buffer.flush()
                frames += 1
                if frames % 5 == 0:
                    print(f"H3PROG {frames}", file=sys.stderr, flush=True)
        if not frames:
            raise RuntimeError("No input frames.")
    finally:
        stream.synchronize()
        for current in reversed(effects):
            current.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("width", "height", "out-width", "out-height"):
        parser.add_argument("--" + name, required=True, type=int)
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    if sys.platform == "win32":
        import msvcrt
        import os
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    transform_frames(args.width, args.height, args.out_width, args.out_height, json.loads(args.settings))


if __name__ == "__main__":
    main()
