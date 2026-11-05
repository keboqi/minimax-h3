`nodes.py` is vendored from https://huggingface.co/Viggle/Viggle-Remake at
21665fa7a1e038e0a5312fb15f37b18d0f06f117, `comfyui/ComfyUI-Viggle-Remake/nodes.py`, under Apache-2.0 (`LICENSE-CODE`).

Local change: `RemakeConditioning`'s `short_edge` maximum is 1472 instead of 736, so H3 latent upscale can rebuild
the conditioning at twice the canvas. Runs without latent upscale stay within the upstream range.

`hold_audio.py` is local. The model weights are not part of this directory; they stay under the MiniMax H3
Community License, as Viggle's NOTICE states.
