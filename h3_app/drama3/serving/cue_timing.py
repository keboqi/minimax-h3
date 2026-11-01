"""Resolve semantic cues against measured speech, including CJK word tokenization."""

import re


def normalized(text: str) -> str:
    return "".join(re.findall(r"[^\W_]", text.casefold()))


def phrase_time(phrase: str, words, *, edge="end", occurrence=1):
    target = normalized(phrase)
    if not target:
        return None
    text, owners = "", []
    for word, start, end in words:
        part = normalized(word)
        text += part
        owners.extend([(start, end)] * len(part))
    cursor = -1
    for _ in range(max(1, min(occurrence, 100))):
        while True:
            cursor = text.find(target, cursor + 1)
            if cursor < 0:
                return None
            last = cursor + len(target) - 1
            # English substrings such as "he" in "the" are not word anchors.
            inside_start = target[0].isascii() and target[0].isalpha() and cursor > 0 and owners[cursor] == owners[cursor - 1]
            inside_end = target[-1].isascii() and target[-1].isalpha() and last + 1 < len(owners) and owners[last] == owners[last + 1]
            if not inside_start and not inside_end:
                break
    return owners[cursor][0] if edge == "start" else owners[cursor + len(target) - 1][1]


def resolve_anchor(phrase, words, lines, *, edge="end", occurrence=1):
    """Prefer word alignment, then the corresponding matching line; never invent a time."""
    at = phrase_time(phrase, words, edge=edge, occurrence=occurrence)
    if at is not None:
        return int(at)
    target = normalized(phrase)
    remaining = max(1, min(occurrence, 100))
    for text, start, end in lines:
        if target and target.isascii():
            tokens = re.findall(r"[^\W_]+", phrase)
            pattern = r"(?<!\w)" + r"\W*".join(re.escape(t) for t in tokens) + r"(?!\w)"
            count = len(re.findall(pattern, text, re.IGNORECASE))
        else:
            count = normalized(text).count(target) if target else 0
        if count >= remaining:
            return start if edge == "start" else end
        remaining -= count
    return None


def shifted_time(value, holds, *, end=False):
    """Map an original time through inserted holds; a preceding word ends before its hold."""
    return value + sum(duration for at, duration in holds if at < value or (at == value and not end))


def insert_holds(samples, rate, lines, words, texts, holds, consumed):
    """Insert holds into dry speech and update alignment using a piecewise time mapping."""
    import numpy as np

    flat_words = [word for line in words or [] for word in line]
    measured_lines = [(text, a, b) for text, (a, b) in zip(texts, lines)]
    insertions = []
    for hold in holds:
        phrase, seconds = hold[-2:]
        key = hold
        if key in consumed:
            continue
        at = resolve_anchor(phrase, flat_words, measured_lines)
        if at is None:
            continue
        consumed.add(key)
        insertions.append((max(0, min(at, len(samples))), int(seconds * rate)))
    insertions.sort()
    result = samples.copy()
    for at, extra in reversed(insertions):
        fade = min(int(.005 * rate), at, len(result) - at)
        if fade:
            result[at-fade:at] *= np.linspace(1, 0, fade)
            result[at:at+fade] *= np.linspace(0, 1, fade)
        result = np.concatenate((result[:at], np.zeros(extra, dtype=np.float32), result[at:]))
    bounds = [(shifted_time(a, insertions), shifted_time(b, insertions, end=True)) for a, b in lines]
    timed = None if words is None else [
        [(t, shifted_time(a, insertions), shifted_time(b, insertions, end=True)) for t, a, b in line]
        for line in words]
    return result, bounds, timed, insertions
