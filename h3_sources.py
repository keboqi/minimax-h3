"""Pinned external source catalog shared by local and Modal provisioning."""

DLSSNR_VERSION = "v1.4.1"
DLSSNR_URL = f"https://github.com/DaniilSokolyuk/video2dlssnr/releases/download/{DLSSNR_VERSION}/video2dlssnr_release.zip"
DLSSNR_SHA256 = "cf1e01b3715b5ee75708744a5d5e6a4f2a96b5504834dd854b4c06388f230ccf"
WINE_COMPAT_VERSION = "wine-11.0-amd64-wow64"
WINE_COMPAT_URL = f"https://github.com/Kron4ek/Wine-Builds/releases/download/11.0/{WINE_COMPAT_VERSION}.tar.xz"
WINE_COMPAT_SHA256 = "39574efa1132c3ca0d5c77dd2eddbe4a49cca0d6cc2c290ff4924493a1c40314"

# RunningHub's MIT host/bridge implement the Linux DNR2 transport and NGX ABI.
RH_DLSS_REF = "71ab1060865ea781c7f4a8b7c0c79600f51b6617"
NVAPI_HEADLESS_REF = "f9fdf268e888d17ae9875a4f01908164ff05f740"
RH_DLSS_FILES = {
    "native/bin/dlss5nr_host.exe": "8d02590c9b21c2b4d101029d7f7c579ff5362322",
    "native/bin/dlss5nr_bridge.dll": "e9eb9e95db3f6b3cf03bfdb513b159f4fbdacf6f",
    "runtime/caller/nvngx.dll_comfy.dll": "4b73bce8e69f6a046fa24918d804d6c15a6fc80e",
}
DLSS_GRAPHICS = (
    ("dxvk-3.1", "https://github.com/doitsujin/dxvk/releases/download/v3.1/dxvk-3.1.tar.gz",
     "30f9cc326874be344285582275446968cfa4c069db31ce56df312d6644179154", "dxvk-3.1"),
    ("vkd3d-proton-3.0.1", "https://github.com/HansKristian-Work/vkd3d-proton/releases/download/v3.0.1/vkd3d-proton-3.0.1.tar.zst",
     "3cf2315522af5e43605ef6d3c41dad91387040bf97199934f3f7ab76caaa2f0c", "vkd3d-proton-3.0.1"),
    ("dxvk-nvapi-0.9.2", "https://github.com/jp7677/dxvk-nvapi/releases/download/v0.9.2/dxvk-nvapi-v0.9.2.tar.gz",
     "60c284223530d643c446c263f1e1a96c6de7b5ff21796219646da734d97a70d6", ""),
)
# Extract this Windows NGX core only; never execute/install the driver package.
NGX_WINDOWS_URL = "https://us.download.nvidia.com/Windows/616.56/616.56-desktop-win10-win11-64bit-international-dch-whql.exe"
NGX_WINDOWS_SHA256 = "da50178def40a17c40d7b4622003b2073efd20c9422d0bb707bc7e59e8cc4868"

COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
CONTROLNET_AUX_REF = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
CONTROLNET_AUX_REPO = "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
H3_LATENT_UPSCALER_NODE_REF = "d7c01b9011f2e8439493f6c02c29995a27df276f"
H3_LATENT_UPSCALER_NODE_REPO = (
    "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git"
)
KJNODES_REF = "57105374f47d0fbb49c9c3926fb981702e0a4b5c"
KJNODES_REPO = "https://github.com/kijai/ComfyUI-KJNodes.git"
LARRY_TURBO_REF = (
    "4274783a23afcfdbea3b4876cb79effd6c510785"  # v1.2.3+ audio/reference fixes
)
LARRY_TURBO_REPO = "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git"
LTXVIDEO_REF = "3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f"
LTXVIDEO_REPO = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
SAGE_WHEEL_NAME = "sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
SAGE_WHEEL_URL = "https://huggingface.co/JahJedi/sageattention-flashattn-blackwell-cu130-torch211-cp312/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
SLA_REF = "d58d006a4ea32c25c06499f2ff104f0852a045a6"
SLA_REPO = "https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes.git"
VEDA_REF = "8d7ac153ac215d636eaf1b5910b0536c3de9e6ed"
VEDA_REPO = "https://github.com/veda-sparse/Veda-on-ComfyUI.git"
SOL_REF = "930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf"
# The original Saganaki22 repository became unavailable. This public fork retains
# the exact reviewed v0.6.2 commit, so the node code and graph contract stay fixed.
SOL_REPO = "https://github.com/wildparky/ComfyUI-sol-attn.git"
SPECTRUM_REF = "5161f0457bc8c52535212d6783eee73f439e1537"  # v0.2.28
SPECTRUM_REPO = "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git"
SPECTRUM_QWEN_REF = "ddb5470f3a885fd912e2b01aa8e20f9768148e42"
SPECTRUM_QWEN_REPO = "https://github.com/xmarre/ComfyUI-Spectrum-Qwen-Proper.git"
VIDEO_DEPTH_REF = "a0db08e63d1ea571601c45cde4aaee0acdd0544d"
VIDEO_DEPTH_REPO = "https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git"
