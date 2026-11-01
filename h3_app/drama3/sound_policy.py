"""Shared semantic controls for generated sounds in both writer workflows."""

SOUND_ROLES = frozenset({"physical", "social", "group_reaction", "emotional", "comic", "transition"})
PERSPECTIVES = frozenset({"in_world", "subjective", "editorial"})
DURATION_CLASSES = {"accent": 1, "reaction": 3, "event": 6, "texture": 25}

# (loop, volume, duration) by sound kind: the Guided defaults, within the Single-pass ranges.
SOUND_DEFAULTS = {"ambience": (True, 0.4, 25), "music": (True, 0.2, 40),
                  "effect": (False, 1.0, 4), "stinger": (False, 0.4, 5)}
# Relative gain by kind and level; the mix first levels each sound to the speech.
LEVELS = ("low", "mid", "high")
LEVEL_VOLUMES = {"ambience": {"low": 0.2, "mid": 0.4, "high": 0.55},
                 "music": {"low": 0.12, "mid": 0.2, "high": 0.28},
                 "effect": {"low": 0.6, "mid": 1.0, "high": 1.3},
                 "stinger": {"low": 0.25, "mid": 0.4, "high": 0.55}}

# Scene transitions: the bed lead-in before the scene's first line, the previous scene's bed
# tail under this one, and the gap between them. None keeps the renderer's default. A bridge's
# outgoing tail (unducked once its own speech ends) is nearly gone when the new speech starts.
SCENE_GAP_SECONDS = 0.8
TRANSITIONS = {"cut": (None, None, SCENE_GAP_SECONDS),
               "establish": (2.5, None, SCENE_GAP_SECONDS),
               "bridge": (2.0, 2.5, 0.3)}
LEAD_IN_RANGE = (0.3, 4.0)
TAIL_RANGE = (0.3, 4.0)


def sound_volume(kind: str, level: str = "") -> float:
    """The relative volume for ``kind`` heard at ``level`` (the kind's default when unset)."""
    if level in LEVELS and kind in LEVEL_VOLUMES:
        return LEVEL_VOLUMES[kind][level]
    return SOUND_DEFAULTS.get(kind, (False, 1.0, 4))[1]


def event_budget(lines: int) -> int:
    """How many sound events one scene of ``lines`` utterances keeps: dense enough for action."""
    return min(32, max(6, (lines + 1) // 2))


def gain_envelope(value):
    """Bounded relative-time/gain points; malformed automation uses the default mix."""
    try:
        points = [tuple(float(n) for n in pair.split(":")) for pair in str(value).split(",")]
        if not 2 <= len(points) <= 8:
            return ()
        if any(len(p) != 2 or not all(0 <= n <= 1 for n in p) for p in points):
            return ()
        if any(a[0] >= b[0] for a, b in zip(points, points[1:])):
            return ()
        return tuple(points)
    except (ValueError, TypeError):
        return ()


def sound_perspective(role: str, kind: str, explicit: str = "") -> str:
    if explicit in PERSPECTIVES:
        return explicit
    if role in {"comic", "transition"} or kind in {"music", "stinger"}:
        return "editorial"
    if role == "emotional":
        return "subjective"
    return "in_world"
