"""Reusable page-playback effects; these never become part of an image prompt."""

from h3_app.errors import H3Error


EFFECT_PRESETS = {
    "push_in": {"label": "Slow push in", "kind": "camera", "description": "Gently approach the central subject."},
    "pull_out": {"label": "Slow pull out", "kind": "camera", "description": "Slowly reveal the surrounding scene."},
    "pan_left": {"label": "Pan left", "kind": "camera", "description": "Drift left across the image."},
    "pan_right": {"label": "Pan right", "kind": "camera", "description": "Drift right across the image."},
    "pan_up": {"label": "Pan up", "kind": "camera", "description": "Tilt or rise upward across the frame."},
    "pan_down": {"label": "Pan down", "kind": "camera", "description": "Drift downward to reveal the scene below."},
    "subtle_sway": {"label": "Subtle sway", "kind": "camera", "description": "Gentle handheld drift for tense or intimate moments."},
    "impact_shake": {"label": "Impact shake", "kind": "camera", "description": "Brief damped camera shake at the start of the shot."},
    "rain": {"label": "Rain", "kind": "atmosphere", "description": "Layered slanting rain across the frame."},
    "snow": {"label": "Snow", "kind": "atmosphere", "description": "Soft drifting snowflakes with gentle depth."},
    "fog": {"label": "Fog", "kind": "atmosphere", "description": "Slow translucent banks of drifting mist."},
    "dust": {"label": "Floating dust", "kind": "atmosphere", "description": "Sparse warm dust motes drifting in the light."},
    "embers": {"label": "Embers", "kind": "atmosphere", "description": "Glowing warm sparks rising from below."},
    "sunbeams": {"label": "Sunbeams", "kind": "atmosphere", "description": "Diagonal shafts of light drifting gently from above."},
    "smoke": {"label": "Smoke", "kind": "atmosphere", "description": "Soft curling haze rising slowly through the frame."},
    "petals": {"label": "Petals", "kind": "atmosphere", "description": "Drifting blossoms or leaves tumbling in the breeze."},
    "lightning": {"label": "Lightning", "kind": "lighting", "description": "Occasional brief cool light pulses for a storm."},
    "flash": {"label": "Flash", "kind": "lighting", "description": "One brief soft white flash at the start of a reveal or impact."},
    "warm_glow": {"label": "Warm glow", "kind": "lighting", "description": "Soft warm radial light breathing gently over the image."},
    "cold_tint": {"label": "Cold moonlight", "kind": "lighting", "description": "Subtle cool blue atmospheric wash for night or eerie scenes."},
    "flicker": {"label": "Firelight flicker", "kind": "lighting", "description": "Gentle organic warmth flicker like a candle or hearth."},
    "vignette": {"label": "Vignette", "kind": "lighting", "description": "Soft dark edges to focus attention on the center."},
    "film_grain": {"label": "Film grain", "kind": "lighting", "description": "Subtle animated 35mm cinema grain."},
    "dream_bloom": {"label": "Dream bloom", "kind": "lighting", "description": "Soft ethereal halation for memories or surreal moments."},
    "scanlines": {"label": "Scanlines", "kind": "lighting", "description": "Faint monitor scanlines and cool phosphor glow."},
}


def check_effects(value):
    """Old shots default to still; allow at most one camera move, one atmosphere, and one lighting effect."""
    if not isinstance(value, list) or len(value) > 3:
        raise H3Error("Shot effects must be a list with at most one camera move, one atmosphere, and one lighting effect.")
    kinds = set()
    for effect in value:
        if not isinstance(effect, str) or effect not in EFFECT_PRESETS:
            raise H3Error("That shot has an unknown visual effect.")
        kind = EFFECT_PRESETS[effect]["kind"]
        if kind in kinds:
            raise H3Error("Choose at most one camera move, one atmosphere, and one lighting effect per shot.")
        kinds.add(kind)
    return list(value)
