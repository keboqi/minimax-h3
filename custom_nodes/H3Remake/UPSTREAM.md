`nodes.py` is vendored from https://huggingface.co/Viggle/Viggle-Remake at
21665fa7a1e038e0a5312fb15f37b18d0f06f117, `comfyui/ComfyUI-Viggle-Remake/nodes.py`, under Apache-2.0 (`LICENSE-CODE`).

Local change: `RemakeConditioning` accepts `-` in `pins` to leave one reference unpinned, because a long Remake pins
the frame carried from the previous window while other pictures only set the look.

`hold_audio.py` is local. The model weights are not part of this directory; they stay under the MiniMax H3
Community License, as Viggle's NOTICE states.
