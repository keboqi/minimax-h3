#!/usr/bin/env python3
"""Shared dependency compatibility policy for MiniMax H3 deployments."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePath, PureWindowsPath
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import urlopen


TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
TORCHAUDIO_VERSION = "2.11.0"
TENSORRT_PACKAGE = "tensorrt-cu13>=11.2,<12"
NUMPY_VERSION = "1.26.4"
SCIPY_VERSION = "1.15.3"
# Keep the validated Hub 1.x stack; newer Transformers releases allow Hub 2.x
# but require a separate model-loading and kernels compatibility review.
HUGGINGFACE_HUB_REQUIREMENT = "huggingface-hub>=1.5,<2"
# Transformers' fine-grained FP8 loader requires this exact minor release.
KERNELS_VERSION = "0.16.0"
# Pinned LTXVideo uses torch.nn.functional.pad instead of Kornia's removed
# pyramid.pad export.
KORNIA_VERSION = "0.8.3"
KORNIA_RS_VERSION = "0.1.14"
# New LTX HDR nodes require these packages. Later releases require NumPy 2,
# so keep their versions compatible with the pinned NumPy 1.26/CUDA stack.
LTX_HDR_REQUIREMENTS = ("colour-science==0.4.6", "openimageio==3.0.12.0")
# Keep the ComfyUI source and its pinned comfy-kitchen dependency in lockstep.
# ComfyUI v0.40.0 fixes Qwen Image low-VRAM caching and adds direct single-latent
# H3 decoder metadata. Keep frontend and Kitchen aligned with its requirements.
H3_AUDIO_T8_REPO = "https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git"
H3_AUDIO_T8_REF = "6063fafbd9c3b85c5ff40aef435ae11b2844e558"

# ComfyUI v0.42.1: Qwen decode optimizations, TextGenerate cleanup and PDD heads. Keep this
# source paired with Kitchen 0.2.39 and Drama's new/old cache compatibility.
COMFY_REF = "63be2f9b82ad0dac88a7ceaa28a9cf3154125a18"
# Drama 3 speech runs on SGLang-Omni; this node pack supplies the Qwen3-ASR transcription nodes.
DRAMA3_REPO = "https://github.com/PedroMarinhoDev/ComfyUI-Drama3.git"
# CUDA expert kernels/graphs, plus compatibility with Core's new Qwen cache API.
DRAMA3_REF = "67a733a96890637dc27b1914974ad00c569405fb"
# Drama 3 backend, isolated from ComfyUI's Torch stack.
DRAMA3_SGLANG_TORCH_BACKEND = "cu130"
DRAMA3_SGLANG_PYTHON = "3.12"
# Parallel Drama 3 backend: SGLang-Omni main. Drama 3
# (FishOmniModel) landed after v0.1.8 (#2763, 2026-10-13); pin a main commit.
DRAMA3_SGLANG_OMNI_REPO = "https://github.com/sgl-project/sglang-omni.git"
DRAMA3_SGLANG_OMNI_REF = "caae7eb18d8c8d22643d4cf8e377226a3b1259e9"
DRAMA3_HF_REPO = "fishaudio/drama-3"
DRAMA3_HF_REVISION = "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
COMFY_KITCHEN_VERSION = "0.2.39"
COMFY_FRONTEND_VERSION = "1.55.18"
WSPROTO_VERSION = "1.2.0"
# Drama 3 speaker vocal effects; without it they fall back to cruder scipy versions.
PEDALBOARD_VERSION = "0.9.19"
GRADIO_VERSION = "6.30.0"
LTX25_WORKFLOW_FILENAMES = (
    "LTX-2.5_T2V_I2V_Single_Stage_Distilled.json",
    "LTX-2.5_T2V_I2V_Two_Stage_Distilled.json",
    "LTX-2.5_A2V_Two_Stage_Distilled.json",
    "LTX-2.5_T2A_Single_Stage_Distilled.json",
    "LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json",
    "LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json",
    "LTX-2.5_ICLoRA_Motion_Track_Distilled.json",
    "LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json",
    "LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json",
    "LTX-2.5_ICLoRA_Union_Control_Distilled.json",
    "LTX-2.5_V2V_TiledFusion_Native_4K_8K.json",
    "LTX-2.5_V2V_TiledFusion_Upscale.json",
)

ABI_CONSTRAINTS = (
    f"torch=={TORCH_VERSION}",
    f"torchvision=={TORCHVISION_VERSION}",
    f"torchaudio=={TORCHAUDIO_VERSION}",
    f"numpy=={NUMPY_VERSION}",
    f"scipy=={SCIPY_VERSION}",
)
INSTALL_CONSTRAINTS = (*ABI_CONSTRAINTS, HUGGINGFACE_HUB_REQUIREMENT)

PINNED_REQUIREMENTS = frozenset(
    {"torch", "torchvision", "torchaudio", "numpy", "scipy", "huggingface-hub"}
)


def sync_ltx25_workflows(source: Path, destination: Path) -> tuple[Path, ...]:
    """Install the exact official LTX 2.5 templates from the pinned node."""
    source = Path(source)
    destination = Path(destination)
    discovered = {path.name for path in source.glob("*.json")}
    expected = set(LTX25_WORKFLOW_FILENAMES)
    if discovered != expected:
        missing = sorted(expected - discovered)
        extra = sorted(discovered - expected)
        raise RuntimeError(
            "Official LTX 2.5 workflow set does not match the pinned release; "
            f"missing={missing}, extra={extra}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    installed = []
    for filename in LTX25_WORKFLOW_FILENAMES:
        target = destination / filename
        shutil.copy2(source / filename, target)
        installed.append(target)
    return tuple(installed)


def comfy_frontend_static_references(content: str) -> list[str]:
    """Return immutable browser assets referenced by a ComfyUI index."""
    candidates = re.findall(
        r"(?i)\b(?:href|src)\s*=\s*[\"']([^\"']+)",
        content,
    )
    references = []
    for candidate in candidates:
        parsed = urlsplit(candidate)
        if candidate.startswith(("//", "#")) or parsed.scheme or not parsed.path:
            continue
        normalized = parsed.path.lstrip("/").removeprefix("./")
        # user.css and api/userdata/user.css are optional, runtime-served files.
        # Hashed assets and the icon stylesheet must exist in the frontend wheel.
        if normalized.startswith("assets/") or normalized == (
            "materialdesignicons.min.css"
        ):
            references.append(candidate)
    return references


def comfy_frontend_package_is_ready() -> bool:
    """Validate assets using the same containment rule as aiohttp static."""
    try:
        if version("comfyui-frontend-package") != COMFY_FRONTEND_VERSION:
            return False
        import comfyui_frontend_package
    except (ImportError, PackageNotFoundError):
        return False

    root = Path(resources.files(comfyui_frontend_package) / "static")
    index = root / "index.html"
    if not index.is_file():
        return False
    try:
        content = index.read_text(encoding="utf-8")
    except OSError:
        return False

    references = comfy_frontend_static_references(content)
    try:
        resolved_root = root.resolve(strict=True)
        resolved_assets = [
            (root / urlsplit(reference).path.lstrip("/").removeprefix("./")).resolve(
                strict=True
            )
            for reference in references
        ]
    except (OSError, RuntimeError):
        return False
    return bool(resolved_assets) and all(
        asset.is_file() and asset.is_relative_to(resolved_root)
        for asset in resolved_assets
    )


def probe_comfy_frontend(base_url: str, timeout: float = 5) -> int:
    """Fetch a ComfyUI index and all immutable assets; return asset count."""
    index_url = base_url.rstrip("/") + "/"
    with urlopen(index_url, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(f"ComfyUI index returned HTTP {response.status}")
        content = response.read().decode(
            response.headers.get_content_charset() or "utf-8",
            errors="replace",
        )

    references = comfy_frontend_static_references(content)
    if not references:
        raise RuntimeError("ComfyUI index did not reference any static assets")
    for reference in references:
        asset_url = urljoin(index_url, reference)
        with urlopen(asset_url, timeout=timeout) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"ComfyUI asset returned HTTP {response.status}: {asset_url}"
                )
    return len(references)


def probe_comfy_workflow(base_url: str, timeout: float = 5) -> int:
    """Fetch one bundled nested workflow through ComfyUI's userdata route."""
    relative_path = f"workflows/LTX 2.5/{LTX25_WORKFLOW_FILENAMES[3]}"
    workflow_url = urljoin(
        base_url.rstrip("/") + "/",
        "api/userdata/" + quote(relative_path, safe=""),
    )
    with urlopen(workflow_url, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(
                f"ComfyUI workflow returned HTTP {response.status}: {workflow_url}"
            )
        content = response.read()
    if not content:
        raise RuntimeError(f"ComfyUI workflow was empty: {workflow_url}")
    return len(content)


def requirement_name(line: str) -> str | None:
    """Return a normalized package name for a requirement-file line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    # Keep options, includes, URLs, and editable requirements untouched. They
    # are not safely comparable to one of our pinned package names.
    if stripped.startswith(("-", "http://", "https://", "git+")):
        return None

    name = re.split(r"[<>=!~;\[\s]", stripped, maxsplit=1)[0]
    return name.lower().replace("_", "-") or None


def filter_pinned_requirements(
    lines: Iterable[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Remove protected packages and report the skipped entries."""
    filtered: list[str] = []
    skipped: list[tuple[str, str]] = []
    for line in lines:
        package = requirement_name(line)
        if package in PINNED_REQUIREMENTS:
            skipped.append((package, line.strip()))
        else:
            filtered.append(line)
    return filtered, skipped


DRAMA3_SGLANG_READY_MARKER = "h3-drama3-ready.txt"

DRAMA3_SGLANG_SMOKE = (
    "import importlib.metadata as m, torch; "
    "assert torch.version.cuda, 'SGLang-Omni installed CPU-only Torch'; "
    "print('sglang', m.version('sglang'), 'sglang-omni', m.version('sglang-omni'), 'cuda', torch.version.cuda)"
)


# Mirrors sglang-omni's [tool.uv] override-dependencies for Linux.
DRAMA3_SGLANG_OVERRIDES = "protobuf>=6.31.1,<7.0.0\n"


def drama3_sglang_fingerprint() -> str:
    """Identifies an installed SGLang-Omni environment built from these exact pins."""
    import hashlib

    pins = "|".join((DRAMA3_SGLANG_PYTHON, DRAMA3_SGLANG_OMNI_REPO, DRAMA3_SGLANG_OMNI_REF,
                     DRAMA3_SGLANG_TORCH_BACKEND, DRAMA3_SGLANG_OVERRIDES))
    return hashlib.sha256(pins.encode("utf-8")).hexdigest()


def drama3_sglang_install_commands(venv: str | Path, uv: str | list[str] = "uv") -> list[list[str]]:
    """Install steps for the SGLang-Omni Drama 3 environment (image build and first use)."""
    base = venv if isinstance(venv, PurePath) else Path(venv)
    python = str(base / "bin" / "python")
    overrides = str(base / "h3-overrides.txt")
    uv = [uv] if isinstance(uv, str) else list(uv)
    return [
        [*uv, "venv", "--python", DRAMA3_SGLANG_PYTHON, "--seed", str(base)],
        # The [fishaudio] extra's descript-audiotools pins protobuf<3.20, while SGLang
        # needs >=6.30. The project overrides it in [tool.uv], which uv ignores for a
        # git-URL install, so apply the same override explicitly.
        [python, "-c", f"open({overrides!r}, 'w').write({DRAMA3_SGLANG_OVERRIDES!r})"],
        # SGLang pins Torch 2.13; name the CUDA wheel because builders have no GPU.
        [*uv, "pip", "install", "--python", python, "--prerelease=allow",
         f"--torch-backend={DRAMA3_SGLANG_TORCH_BACKEND}", "--overrides", overrides,
         f"sglang-omni[fishaudio] @ git+{DRAMA3_SGLANG_OMNI_REPO}@{DRAMA3_SGLANG_OMNI_REF}"],
        [python, "-c", DRAMA3_SGLANG_SMOKE],
    ]


AUDIO_TOOLS_READY_MARKER = "h3-audio-tools-ready.txt"
AUDIO_TOOLS_PYTHON = "3.12"
STABLE_AUDIO_TOOLS_REPO = "https://github.com/Stability-AI/stable-audio-tools.git"
# Pinned: an unpinned main moved its imports without changing the fingerprint.
STABLE_AUDIO_TOOLS_REF = "3241adba4fc2a85cf5b29d9eb68d42f40a28e820"
# Community FlashAttention 2.8.3 builds for this exact Torch/CUDA/Python ABI;
# Dao-AILab ships none for cu130 + Torch 2.11. Both models fall back to SDPA without it.
_FLASH_ATTN_RELEASES = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download"
FLASH_ATTN_WHEELS = {
    "linux": (f"{_FLASH_ATTN_RELEASES}/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-linux_x86_64.whl",
              "173b0e1a5d6a0becb4ce11b755605c4b20289a6d17816a00cfd7a69078c3e612"),
    "windows": (f"{_FLASH_ATTN_RELEASES}/v0.9.6/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-win_amd64.whl",
                "c779503a38ec91f8fed4cfd8b3446c1d1724df03fad13dad3997dc5fbed5f5fd"),
}
# Qwen VoiceDesign and Stable Audio 3 Medium share one interpreter and one resident
# worker. qwen-tts pins Transformers 4.57.3 (Hub < 1), which also provides
# stable-audio-tools' T5Gemma encoder; keep both outside ComfyUI's environment.
AUDIO_TOOLS_REQUIREMENTS = (
    "qwen-tts==0.1.1",
    f"torch=={TORCH_VERSION}",
    f"torchaudio=={TORCHAUDIO_VERSION}",
    f"torchvision=={TORCHVISION_VERSION}",
    f"numpy=={NUMPY_VERSION}",
    f"scipy=={SCIPY_VERSION}",
    "huggingface-hub>=0.34,<1",
    "hf-xet>=1.1",
    # stable-audio-tools' runtime imports (it installs with --no-deps).
    "alias-free-torch",
    "dill",
    "einops",
    "einops-exts",
    "importlib-resources",
    "nnAudio",
    "PyWavelets",
    # Its models.lora package imports a Lightning checkpoint callback at model load.
    "pytorch-lightning==2.5.5",
    "safetensors",
    "sentencepiece",
    "setuptools<81",
    "soundfile",
    "soxr",
    "torchsde",
    "tqdm",
    "v-diffusion-pytorch",
    "vector-quantize-pytorch",
)
AUDIO_TOOLS_SMOKE = (
    "import importlib.metadata as m, inspect, torch, soundfile, hf_xet; "
    "from qwen_tts import Qwen3TTSModel; "
    # The worker's own imports: model loading pulls in diffusion, DiT and LoRA modules.
    "import stable_audio_tools.models.diffusion, stable_audio_tools.inference.generation; "
    "from stable_audio_tools.models.transformer import TransformerBlock; "
    "from transformers import T5GemmaEncoderModel; "
    "assert torch.version.cuda, 'Audio tools installed CPU-only Torch'; "
    "assert callable(Qwen3TTSModel.generate_voice_design); "
    "assert 'local_add_cond_dim' in inspect.signature(TransformerBlock.__init__).parameters, "
    "'stable-audio-tools is missing Stable Audio 3 local_add_cond_dim support'; "
    # Builders have no GPU, so check the wheel without loading its CUDA extension.
    "print('flash-attn', m.version('flash-attn'))"
)


def flash_attn_wheel(windows: bool) -> str:
    url, digest = FLASH_ATTN_WHEELS["windows" if windows else "linux"]
    return f"flash-attn @ {url}#sha256={digest}"


def audio_tools_fingerprint() -> str:
    """Match first-use installs and the prebuilt Modal audio tools environment."""
    import hashlib

    wheels = (part for wheel in FLASH_ATTN_WHEELS.values() for part in wheel)
    pins = "|".join((AUDIO_TOOLS_PYTHON, *AUDIO_TOOLS_REQUIREMENTS, STABLE_AUDIO_TOOLS_REPO,
                     STABLE_AUDIO_TOOLS_REF, *wheels, TORCH_INDEX, AUDIO_TOOLS_SMOKE))
    return hashlib.sha256(pins.encode("utf-8")).hexdigest()


def audio_tools_install_commands(venv: str | Path, uv: str | list[str] = "uv") -> list[list[str]]:
    """Install VoiceDesign, stable-audio-tools (--no-deps, pinned) and FlashAttention together."""
    base = venv if isinstance(venv, PurePath) else Path(venv)
    windows = isinstance(base, PureWindowsPath)
    python = str(base / ("Scripts/python.exe" if windows else "bin/python"))
    uv = [uv] if isinstance(uv, str) else list(uv)
    return [
        [*uv, "venv", "--python", AUDIO_TOOLS_PYTHON, "--seed", str(base)],
        # Use the index URL: older uv releases do not recognize --torch-backend=cu130.
        [*uv, "pip", "install", "--python", python, "--extra-index-url", TORCH_INDEX,
         "--index-strategy", "unsafe-best-match", *AUDIO_TOOLS_REQUIREMENTS],
        [*uv, "pip", "install", "--python", python, "--no-deps",
         f"git+{STABLE_AUDIO_TOOLS_REPO}@{STABLE_AUDIO_TOOLS_REF}", flash_attn_wheel(windows)],
        [python, "-c", AUDIO_TOOLS_SMOKE],
    ]


def selftest() -> None:
    import tempfile
    from unittest.mock import patch

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read() -> bytes:
            return b"{}"

    source = [
        "torch>=2.0",
        "NumPy==2.0; python_version >= '3.12'",
        "scipy[extra]~=1.14",
        "huggingface_hub>=2.0",
        "requests>=2.32",
        "-r optional.txt",
        "# torch is intentionally pinned elsewhere",
        "",
    ]
    filtered, skipped = filter_pinned_requirements(source)
    assert [package for package, _ in skipped] == [
        "torch",
        "numpy",
        "scipy",
        "huggingface-hub",
    ]
    assert filtered == source[4:]
    assert ABI_CONSTRAINTS == (
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
        "numpy==1.26.4",
        "scipy==1.15.3",
    )
    assert INSTALL_CONSTRAINTS == (*ABI_CONSTRAINTS, "huggingface-hub>=1.5,<2")
    assert KORNIA_VERSION == "0.8.3"
    assert KORNIA_RS_VERSION == "0.1.14"
    assert LTX_HDR_REQUIREMENTS == ("colour-science==0.4.6", "openimageio==3.0.12.0")
    assert KERNELS_VERSION == "0.16.0"
    assert COMFY_REF == "63be2f9b82ad0dac88a7ceaa28a9cf3154125a18"
    assert DRAMA3_REF == "67a733a96890637dc27b1914974ad00c569405fb"
    assert COMFY_KITCHEN_VERSION == "0.2.39"
    assert COMFY_FRONTEND_VERSION == "1.55.18"
    assert WSPROTO_VERSION == "1.2.0"
    assert len(LTX25_WORKFLOW_FILENAMES) == 12
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "source"
        destination = root / "destination"
        source.mkdir()
        for filename in LTX25_WORKFLOW_FILENAMES:
            (source / filename).write_text(filename, encoding="utf-8")
        installed = sync_ltx25_workflows(source, destination)
        assert tuple(path.name for path in installed) == LTX25_WORKFLOW_FILENAMES
        assert all(path.read_text(encoding="utf-8") == path.name for path in installed)
    assert comfy_frontend_static_references(
        '<link href="user.css"><link href="materialdesignicons.min.css">'
        '<script src="./assets/index-abc.js"></script>'
    ) == ["materialdesignicons.min.css", "./assets/index-abc.js"]
    with patch(f"{__name__}.urlopen", return_value=FakeResponse()) as request:
        assert probe_comfy_workflow("http://127.0.0.1:7860/comfyui/") == 2
        workflow_url = request.call_args.args[0]
        assert "/api/userdata/workflows%2FLTX%202.5%2F" in workflow_url
    print("h3_requirements selftest OK")


if __name__ == "__main__":
    selftest()
