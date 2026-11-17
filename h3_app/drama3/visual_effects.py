"""Reusable page-playback effects; these never become part of an image prompt."""

from h3_app.errors import H3Error


EFFECT_PRESETS = {
    "push_in": {"label": "Slow push in", "kind": "camera", "description": "Gently approach the central subject."},
    "pull_out": {"label": "Slow pull out", "kind": "camera", "description": "Slowly reveal the surrounding scene."},
    "pan_left": {"label": "Pan left", "kind": "camera", "description": "Drift left across the image."},
    "pan_right": {"label": "Pan right", "kind": "camera", "description": "Drift right across the image."},
    "rain": {"label": "Rain", "kind": "overlay", "description": "Slanting rain across the frame."},
    "snow": {"label": "Snow", "kind": "overlay", "description": "Soft drifting snowflakes."},
    "fog": {"label": "Fog", "kind": "overlay", "description": "Slow translucent banks of mist."},
    "dust": {"label": "Floating dust", "kind": "overlay", "description": "Sparse warm dust motes in the light."},
    "embers": {"label": "Embers", "kind": "overlay", "description": "Warm sparks rising from below."},
    "lightning": {"label": "Lightning", "kind": "overlay", "description": "Occasional brief cool light pulses for a storm."},
    "flash": {"label": "Flash", "kind": "overlay", "description": "One brief soft white flash at the start of a reveal or impact."},
    "warm_glow": {"label": "Warm glow", "kind": "overlay", "description": "Subtle warm light breathing gently over the image."},
    "vignette": {"label": "Vignette", "kind": "overlay", "description": "Soft dark edges to focus attention on the center."},
}


def check_effects(value):
    """Old shots default to still; allow at most one camera move and one overlay."""
    if not isinstance(value, list) or len(value) > 2:
        raise H3Error("Shot effects must be a list with at most one camera move and one overlay.")
    kinds = set()
    for effect in value:
        if not isinstance(effect, str) or effect not in EFFECT_PRESETS:
            raise H3Error("That shot has an unknown visual effect.")
        kind = EFFECT_PRESETS[effect]["kind"]
        if kind in kinds:
            raise H3Error("Choose at most one camera move and one overlay per shot.")
        kinds.add(kind)
    return list(value)
