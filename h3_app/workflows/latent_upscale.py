"""Native H3 upscaler execution with explicit checkpoint precision."""

from h3_app.catalog import H3_LATENT_UPSCALER_LOADER_NODE, H3_LATENT_UPSCALER_NODE
from h3_app.errors import H3Error
from h3_app.graph import Graph


def add_h3_latent_upscale(graph, samples, *, model_name, precision, width, height):
    if precision not in {"bf16", "fp16", "fp32"}:
        raise H3Error("Unsupported H3 latent upscaler precision.")
    if width is None or height is None or min(width, height) < 32 or width % 32 or height % 32:
        raise H3Error("Native H3 latent upscaling requires target dimensions aligned to 32 pixels.")
    model = graph.add(H3_LATENT_UPSCALER_LOADER_NODE, model_name=model_name, precision=precision)
    return Graph.out(graph.add(
        H3_LATENT_UPSCALER_NODE, model=Graph.out(model), samples=samples,
        width=int(width), height=int(height),
    ))
