"""Native RelayVSR upscaling with no input or output resizing."""

from typing import Any

from h3_app.catalog import RELAYVSR_OPTIONS
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_models import MODEL_SPECS


def relayvsr_model_keys(option: str) -> tuple[str, str]:
    try:
        scale = RELAYVSR_OPTIONS[option]
    except KeyError as exc:
        raise H3Error(f"Unknown RelayVSR method: {option}") from exc
    return f"relayvsr_{scale}x_dit", f"relayvsr_{scale}x_vae"


def required_relayvsr_nodes(*, image: bool = False) -> set[str]:
    processing = {"UNETLoader", "VAELoader", "RelayVSRConditioning", "KSampler", "VSRDecode"}
    if image:
        return processing | {"LoadImage", "SaveImage"}
    return processing | {"LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo"}


def build_relayvsr_graph(
    *, option: str, source: str, seed: int, image: bool = False,
    fps: float = 24.0, output_token: str = "0",
) -> dict[str, Any]:
    """Use the selected checkpoint's fixed scale on the original input pixels."""
    dit_key, vae_key = relayvsr_model_keys(option)
    graph = Graph()
    if image:
        loaded = graph.add("LoadImage", image=source)
        frames = Graph.out(loaded)
    else:
        loaded = graph.add("LoadVideo", file=source)
        components = graph.add("GetVideoComponents", video=Graph.out(loaded))
        frames = Graph.out(components)
    model = graph.add(
        "UNETLoader", unet_name=MODEL_SPECS[dit_key].local_name, weight_dtype="default",
    )
    vae = graph.add("VAELoader", vae_name=MODEL_SPECS[vae_key].local_name)
    conditioning = graph.add("RelayVSRConditioning", model=Graph.out(model), images=frames)
    sampled = graph.add(
        "KSampler", model=Graph.out(model), seed=int(seed), steps=1, cfg=1.0,
        sampler_name="euler", scheduler="simple", positive=Graph.out(conditioning),
        negative=Graph.out(conditioning), latent_image=Graph.out(conditioning, 1), denoise=1.0,
    )
    decoded = graph.add(
        "VSRDecode", vae=Graph.out(vae), samples=Graph.out(sampled), images=frames, detail=1.0,
    )
    if image:
        graph.add(
            "SaveImage", images=Graph.out(decoded),
            filename_prefix=f"h3/input_upscale/{output_token}_gallery",
        )
    else:
        video = graph.add(
            "CreateVideo", images=Graph.out(decoded), audio=Graph.out(components, 1),
            fps=float(fps), bit_depth=8,
        )
        graph.add(
            "SaveVideo", video=Graph.out(video), filename_prefix=f"relayvsr/upscale_{output_token}",
            format="auto", codec="auto",
        )
    return graph.nodes
