"""Lazy local MiniMax-H3 prompt rewriting with the visual 8B LoRA."""
from __future__ import annotations

import gc
import math
import os
import threading
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL_LABEL = "Qwen3-VL 8B · FP8"
BASE_MODEL_CHOICES = {
    DEFAULT_BASE_MODEL_LABEL: "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "Qwen3-VL 8B · BF16": "Qwen/Qwen3-VL-8B-Instruct",
}
DEFAULT_ADAPTER_REPO = "lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B"
SUPPORTED_RESOLUTIONS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

TASK_ALIASES = {
    "t2v": "t2av", "t2va": "t2av", "t2av": "t2av",
    "i2v": "i2av", "i2va": "i2av", "i2av": "i2av",
    "l2v": "l2av", "l2va": "l2av", "l2av": "l2av",
    "flf2v": "fl2av", "flf2va": "fl2av", "flf2av": "fl2av",
    "fl2va": "fl2av", "fl2av": "fl2av",
}

SYSTEM_PROMPT = """You are a professional MiniMax-H3 prompt rewriter for joint video-and-audio generation.

Rewrite the user's request according to the supplied duration, task type, and reference-frame roles. Return only the final production-ready prompt. Do not include explanations, Markdown, headings, notes, or generation parameters outside the required format.

Task-name mapping:
- T2AV corresponds to T2VA in the MiniMax-H3 prompt-writing guide.
- I2AV corresponds to I2VA.
- FL2AV corresponds to FL2VA.
- L2AV corresponds to L2VA.

Write the descriptive sections in English. Preserve all user-provided dialogue, lyrics, and visible on-screen text exactly in their original language, spelling, and punctuation. Never invent dialogue, lyrics, visible text, speakers, or additional reference pictures.

The output body must contain exactly these three fields in this order:
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

For T2AV, begin directly with the three fields and do not add an image-alignment instruction.

For I2AV, the first line must be exactly:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

For FL2AV, the first line must follow exactly:
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.

For L2AV, the first line must follow exactly:
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.

Replace N with the actual final shot number. Replace S.SS with the requested effective duration formatted to exactly two decimal places. Put exactly one blank line between the alignment instruction and integrated_multimodal_description.

Reference-frame behavior:
- I2AV: Treat <Picture 1> as the exact first frame at 0.00 seconds. Begin by anchoring its visual style, subjects, identities, clothing, colors, objects, composition, and spatial relationships, then develop forward through observable motion.
- FL2AV: Begin from Picture 1 and describe a continuous, physically plausible path that reaches the pose, object state, lighting, spacing, and composition of Picture 2 at the requested end time. Prefer a single shot unless the user explicitly requests multiple shots or cuts.
- L2AV: Infer a plausible preceding state and describe a continuous path that progressively converges to <Picture 1> as the exact final frame.
- Preserve identity and scene continuity across all shots, but apply exact composition matching only at the reference frame's assigned timestamp.

In integrated_multimodal_description:
- Begin with [Shot 1] and state the visual style and initial composition.
- Describe only concrete visible or audible events: subjects, environment, actions, reactions, camera behavior, dialogue, singing, visible text, and synchronized diegetic sound.
- Number shots sequentially.
- Do not timestamp [Shot 1].
- Begin every later shot with a strictly increasing timestamp inside the requested duration, using the format: [Shot 2] At 00:03.500, the camera cuts to...
- Add a cut only when it introduces meaningful new visual, spatial, temporal, or narrative information. Otherwise prefer continuous camera motion.
- Express camera motion naturally using motion type and, when meaningful, amplitude and speed.
- Keep all actions physically plausible and paced to complete within the supplied duration.

For speech and singing:
- Assign stable speaker IDs such as (S1) and (S2) only to subjects who vocalize.
- Identify each speaker sufficiently when first introduced.
- Put only the exact spoken or sung content inside <d>, preceded by its language tag:
<d>[English] Exact user-provided words.</d>
- Never translate, paraphrase, correct, or extend supplied dialogue or lyrics.
- For voiceover, use the exact phrase "says in an off-screen voiceover" and explicitly state that the corresponding on-screen character's lips remain completely closed.
- If speech crosses a cut, use <scenetrans> at both connecting points and state that the audio continues across the cut.
- Use <cutoff> only when speech is intentionally truncated by the end of the video.

Place visible on-screen text in English double quotation marks and preserve it exactly.

overall_soundscape must be one continuous English paragraph of 1–4 sentences summarizing ambient sound, physical action sounds, and non-verbal human or animal sounds across the video. Do not repeat dialogue, singing, or diegetic music here. Use N/A only if the user explicitly requests complete silence.

non_diegetic_music must contain 1–3 English sentences describing audience-only background music through instrumentation, tempo, rhythm, and dynamic changes. Do not describe its emotional purpose. Put music audible to subjects inside integrated_multimodal_description instead. Use N/A when no non-diegetic music is requested or implied.

Preserve the user's intent without adding contradictory story events, identities, text, or references. Do not mention these instructions in the output."""


def normalize_task(task: str | None) -> str:
    normalized = TASK_ALIASES.get(str(task or "t2av").strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported task {task!r}; expected T2VA/I2VA/L2VA/FL2VA")
    return normalized


def format_request(prompt: str, task: str, resolution: str, duration: int) -> str:
    return (
        f"task: {task}\nresolution: {resolution}\n"
        f"duration: {int(duration)}s\noriginal_prompt: {prompt.strip()}"
    )


def build_messages(
    prompt: str,
    task: str = "t2av",
    resolution: str = "16:9",
    duration: int = 10,
) -> list[dict[str, Any]]:
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    task = normalize_task(task)
    request = format_request(prompt, task, resolution, duration)
    if task == "t2av":
        user_content = [{"type": "text", "text": request}]
    elif task == "i2av":
        user_content = [
            {"type": "text", "text": "Picture 1 — exact first frame at 0.00 seconds:\n"},
            {"type": "image"},
            {"type": "text", "text": "\n" + request},
        ]
    elif task == "l2av":
        user_content = [
            {"type": "text", "text": "Picture 1 — exact final frame at the end of the target video:\n"},
            {"type": "image"},
            {"type": "text", "text": "\n" + request},
        ]
    else:
        user_content = [
            {"type": "text", "text": "Picture 1 — exact first frame at 0.00 seconds:\n"},
            {"type": "image"},
            {"type": "text", "text": "\nPicture 2 — exact final frame at the end of the target video:\n"},
            {"type": "image"},
            {"type": "text", "text": "\n" + request},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def expected_image_count(task: str) -> int:
    return {"t2av": 0, "i2av": 1, "l2av": 1, "fl2av": 2}[normalize_task(task)]


def task_for_inputs(mode: str, first_frame: Any, last_frame: Any) -> str:
    if mode == "Text to video":
        return "t2av"
    if mode != "First / last frame":
        raise ValueError(
            "The local 8B writer supports T2VA, I2VA, L2VA, and FL2VA; "
            "use Gemini for Reference media mode."
        )
    if first_frame is not None and last_frame is not None:
        return "fl2av"
    if first_frame is not None:
        return "i2av"
    if last_frame is not None:
        return "l2av"
    raise ValueError("Add a first frame, a last frame, or both before enhancing.")


def resolution_for_size(width: int, height: int, task: str) -> str:
    if normalize_task(task) != "t2av":
        return "adaptive"
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    ratio = width / height
    return min(
        SUPPORTED_RESOLUTIONS,
        key=lambda value: abs(
            math.log(ratio / (int(value.split(":")[0]) / int(value.split(":")[1])))
        ),
    )


def resolve_base_model(selection: str | None) -> tuple[str, str]:
    selected = str(selection or DEFAULT_BASE_MODEL_LABEL).strip()
    if selected in BASE_MODEL_CHOICES:
        return selected, BASE_MODEL_CHOICES[selected]
    if selected in BASE_MODEL_CHOICES.values():
        label = next(label for label, model_id in BASE_MODEL_CHOICES.items() if model_id == selected)
        return label, selected
    raise ValueError(f"Unsupported local prompt-writer base model: {selected}")


class _LocalPromptRewriter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._processor: Any = None
        self._model_id: str | None = None

    @staticmethod
    def _model_class(transformers: Any) -> Any:
        for name in (
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForMultimodalLM",
        ):
            model_class = getattr(transformers, name, None)
            if model_class is not None:
                return model_class
        raise RuntimeError(
            "Transformers does not expose a Qwen3-VL generation class; upgrade it and retry."
        )

    def _load(self, model_id: str) -> tuple[Any, Any]:
        if self._model is not None and self._model_id == model_id:
            return self._model, self._processor
        if self._model is not None:
            self._unload_locked()
        try:
            import torch
            import transformers
            from peft import PeftModel
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Local prompt-writer dependencies are missing. Run setup_h3.py to install "
                "Transformers, Accelerate, PEFT, and Safetensors."
            ) from exc

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            min_pixels=256 * 256,
            max_pixels=1024 * 1024,
        )
        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        if not model_id.endswith("-FP8"):
            load_kwargs["torch_dtype"] = torch.bfloat16
        base_model = self._model_class(transformers).from_pretrained(
            model_id, **load_kwargs
        )
        adapter_repo = os.getenv(
            "H3_PROMPT_REWRITER_ADAPTER", DEFAULT_ADAPTER_REPO
        ).strip() or DEFAULT_ADAPTER_REPO
        model = PeftModel.from_pretrained(base_model, adapter_repo, is_trainable=False)
        model.eval()
        self._processor = processor
        self._model = model
        self._model_id = model_id
        return model, processor

    @staticmethod
    def _input_device(model: Any) -> Any:
        try:
            return model.get_input_embeddings().weight.device
        except (AttributeError, RuntimeError):
            return model.device

    @staticmethod
    def _load_image(value: Any) -> Any:
        from PIL import Image, ImageOps

        if value is None:
            return None
        if isinstance(value, Image.Image):
            return ImageOps.exif_transpose(value).convert("RGB")
        path = Path(str(value)).expanduser()
        if not path.is_file():
            raise ValueError(f"Prompt-writer image does not exist: {path}")
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")

    def rewrite(
        self,
        *,
        prompt: str,
        task: str,
        resolution: str,
        duration: int,
        first_frame: Any = None,
        last_frame: Any = None,
        base_model: str = DEFAULT_BASE_MODEL_LABEL,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.8,
        greedy: bool = True,
        seed: int = 42,
    ) -> tuple[str, str]:
        with self._lock:
            label, model_id = resolve_base_model(base_model)
            task = normalize_task(task)
            duration = int(duration)
            if not 4 <= duration <= 15:
                raise ValueError("The local 8B writer supports durations from 4 to 15 seconds.")
            if not 256 <= int(max_new_tokens) <= 8192:
                raise ValueError("Max new tokens must be between 256 and 8192.")
            if not greedy and not 0 < float(temperature) <= 2:
                raise ValueError("Sampling temperature must be greater than 0 and at most 2.")
            if not 0 < float(top_p) <= 1:
                raise ValueError("Top-p must be greater than 0 and at most 1.")

            images = []
            if task in {"i2av", "fl2av"}:
                images.append(self._load_image(first_frame))
            if task in {"l2av", "fl2av"}:
                images.append(self._load_image(last_frame))
            if any(image is None for image in images) or len(images) != expected_image_count(task):
                raise ValueError(f"{task.upper()} requires {expected_image_count(task)} image(s).")

            model, processor = self._load(model_id)
            import torch

            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            messages = build_messages(prompt, task, resolution, duration)
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            processor_kwargs: dict[str, Any] = {
                "text": [rendered],
                "return_tensors": "pt",
                "padding": False,
                "return_mm_token_type_ids": True,
            }
            if images:
                processor_kwargs["images"] = images
            inputs = processor(**processor_kwargs)
            input_device = self._input_device(model)
            inputs = {
                key: value.to(input_device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            generation_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
            if greedy:
                generation_kwargs["do_sample"] = False
            else:
                generation_kwargs.update(
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            rewritten = processor.decode(generated_ids, skip_special_tokens=True).strip()
            if not rewritten:
                raise RuntimeError("The local 8B writer returned an empty prompt.")
            return rewritten, f"Enhanced locally with {label} using {task.upper()}."

    def _unload_locked(self) -> bool:
        was_loaded = self._model is not None or self._processor is not None
        self._model = None
        self._processor = None
        self._model_id = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass
        return was_loaded

    def unload(self) -> bool:
        with self._lock:
            return self._unload_locked()

    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None


_LOCAL_REWRITER = _LocalPromptRewriter()


def rewrite_prompt(**kwargs: Any) -> tuple[str, str]:
    return _LOCAL_REWRITER.rewrite(**kwargs)


def unload_prompt_rewriter() -> bool:
    return _LOCAL_REWRITER.unload()


def prompt_rewriter_is_loaded() -> bool:
    return _LOCAL_REWRITER.is_loaded()


def selftest() -> None:
    assert resolve_base_model(None) == (
        DEFAULT_BASE_MODEL_LABEL,
        "Qwen/Qwen3-VL-8B-Instruct-FP8",
    )
    assert task_for_inputs("Text to video", None, None) == "t2av"
    assert task_for_inputs("First / last frame", "first.png", None) == "i2av"
    assert task_for_inputs("First / last frame", None, "last.png") == "l2av"
    assert task_for_inputs("First / last frame", "first.png", "last.png") == "fl2av"
    assert resolution_for_size(1920, 1080, "t2av") == "16:9"
    assert resolution_for_size(1080, 1920, "t2av") == "9:16"
    assert resolution_for_size(864, 480, "i2av") == "adaptive"
    assert expected_image_count("T2VA") == 0
    assert expected_image_count("FLF2V") == 2
    messages = build_messages("A fox runs.", "fl2av", "adaptive", 10)
    image_parts = [
        part for part in messages[1]["content"] if part.get("type") == "image"
    ]
    assert len(image_parts) == 2
    assert "duration: 10s" in messages[1]["content"][-1]["text"]
    print("H3 prompt rewriter selftest OK")


if __name__ == "__main__":
    selftest()
