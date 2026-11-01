"""Shared semantic controls for generated sounds in both writer workflows."""

SOUND_ROLES = frozenset({"physical", "social", "group_reaction", "emotional", "comic", "transition"})
PERSPECTIVES = frozenset({"in_world", "subjective", "editorial"})
DURATION_CLASSES = {"accent": 1, "reaction": 3, "texture": 25}


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
