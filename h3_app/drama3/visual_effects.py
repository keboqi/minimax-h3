"""Reusable page-playback effects; these never become part of an image prompt."""

from h3_app.errors import H3Error


EFFECT_PRESETS = {
    "push_in": {"label": "Slow push in", "kind": "camera", "description": "Gently approach the central subject with 2.5D depth parallax."},
    "pull_out": {"label": "Slow pull out", "kind": "camera", "description": "Slowly reveal the surrounding scene with 2.5D depth parallax."},
    "pan_left": {"label": "Pan left", "kind": "camera", "description": "Drift left across depth planes to reveal detail."},
    "pan_right": {"label": "Pan right", "kind": "camera", "description": "Drift right across depth planes to reveal detail."},
    "pan_up": {"label": "Pan up", "kind": "camera", "description": "Rise upward across depth planes to reveal height or sky."},
    "pan_down": {"label": "Pan down", "kind": "camera", "description": "Drift downward across depth planes to reveal the foreground."},
    "subtle_sway": {"label": "Subtle sway", "kind": "camera", "description": "Gentle handheld parallax drift for tense or intimate moments."},
    "impact_shake": {"label": "Impact shake", "kind": "camera", "description": "Brief damped camera shake at shot start or on sound cues."},
    "dutch_tilt": {"label": "Dutch tilt", "kind": "camera", "description": "Gentle rotational drift with depth push for unease or tension."},
    "rack_focus": {"label": "Rack focus", "kind": "camera", "description": "Optical focus pull between foreground and subject depth planes at shot start."},
    "rain": {"label": "Rain", "kind": "atmosphere", "description": "3D depth-layered slanting rain falling in front of and behind subjects."},
    "snow": {"label": "Snow", "kind": "atmosphere", "description": "Soft drifting snowflakes occluded naturally across depth planes."},
    "fog": {"label": "Fog", "kind": "atmosphere", "description": "Volumetric distance mist that thickens with scene depth."},
    "dust": {"label": "Floating dust", "kind": "atmosphere", "description": "Warm dust motes drifting across near and far depth layers."},
    "embers": {"label": "Embers", "kind": "atmosphere", "description": "Glowing sparks rising across foreground and midrange depth."},
    "sunbeams": {"label": "Sunbeams", "kind": "atmosphere", "description": "Depth-occluded shafts of light streaming from sky or high openings."},
    "smoke": {"label": "Smoke", "kind": "atmosphere", "description": "Curving volumetric haze settling across midrange and deeper space."},
    "petals": {"label": "Petals", "kind": "atmosphere", "description": "Tumbling blossoms or leaves drifting across foreground and midground."},
    "underwater": {"label": "Underwater", "kind": "atmosphere", "description": "Exponential depth tint, rising bubbles, and surface-angled caustics."},
    "fireflies": {"label": "Fireflies", "kind": "atmosphere", "description": "Bioluminescent motes drifting across separated depth planes at night."},
    "lightning": {"label": "Lightning", "kind": "lighting", "description": "Brief cool storm pulses illuminating sky openings and foreground rims."},
    "flash": {"label": "Flash", "kind": "lighting", "description": "One brief white light burst with distance falloff at reveal or impact."},
    "warm_glow": {"label": "Warm glow", "kind": "lighting", "description": "Soft warm 3D light anchored near the scene's warmest practical source."},
    "cold_tint": {"label": "Cold moonlight", "kind": "lighting", "description": "Cool blue moonlight wash deepening across shadows and distance."},
    "flicker": {"label": "Firelight flicker", "kind": "lighting", "description": "Organic warm firelight relighting sculpted surfaces around a flame source."},
    "vignette": {"label": "Vignette", "kind": "lighting", "description": "Soft dark edges to focus attention on the center."},
    "film_grain": {"label": "Film grain", "kind": "lighting", "description": "Subtle animated 35mm cinema grain weighted to midtones."},
    "dream_bloom": {"label": "Dream bloom", "kind": "lighting", "description": "Soft ethereal halation stronger over distant highlights."},
    "scanlines": {"label": "Scanlines", "kind": "lighting", "description": "Faint monitor scanlines and cool phosphor glow for screens."},
    "spotlight": {"label": "Spotlight", "kind": "lighting", "description": "Focused overhead light cone isolating the subject from dark surroundings."},
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
