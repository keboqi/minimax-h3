"""Check a character's looks against their Cast voice: the gender and age each one states.

Only plain statements count ("an elderly man", "woman in her thirties", "16岁的少年"); a
description that says nothing about gender or age is never flagged.
"""

import re

_MALE = {"man", "men", "male", "boy", "boys", "he", "his", "him", "father", "grandfather", "husband",
         "gentleman", "king", "son", "brother", "uncle", "lad", "guy", "masculine"}
_FEMALE = {"woman", "women", "female", "girl", "girls", "she", "her", "hers", "mother", "grandmother", "wife",
           "lady", "queen", "daughter", "sister", "aunt", "feminine"}
# Not 他: it is part of 其他 ("other").
_ZH_MALE = ("男", "父亲", "爷爷", "先生", "少年")
_ZH_FEMALE = ("女", "她", "母亲", "奶奶", "夫人", "少女", "姑娘")

_NUMBERS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}
_DECADES = {"teens": 13, "twenties": 20, "thirties": 30, "forties": 40, "fifties": 50, "sixties": 60,
            "seventies": 70, "eighties": 80, "nineties": 90}
# Age words, most specific first; each gives a range of years.
_AGE_WORDS = [
    (r"\bmiddle[- ]aged\b", (40, 60)), (r"\b(elderly|senior|old (?:man|woman|lady)|grand(?:father|mother)|"
                                       r"grandpa|grandma)\b", (60, 100)),
    (r"\b(teen|teenage|teenager|adolescent)\b", (13, 19)), (r"\b(toddler|baby|infant)\b", (0, 3)),
    (r"\b(child|kid|little (?:boy|girl))\b", (3, 12)), (r"\b(boy|girl)\b", (5, 17)),
    (r"\byoung (?:man|woman|adult)\b", (18, 30)), (r"\byoung\b", (16, 35)),
    (r"老年|老人|老头|老太|老妇|老者|年迈|爷爷|奶奶", (60, 100)), (r"中年", (40, 60)), (r"青年|年轻", (18, 35)),
    (r"少年|少女", (13, 19)), (r"孩子|小孩|儿童|男孩|女孩", (3, 12)),
]


def gender(text):
    """``male``, ``female`` or None when the text states neither, or both."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    male = bool(words & _MALE) or any(mark in text for mark in _ZH_MALE)
    female = bool(words & _FEMALE) or any(mark in text for mark in _ZH_FEMALE)
    return "male" if male and not female else "female" if female and not male else None


def age(text):
    """The age range ``(low, high)`` in years the text states, or None."""
    lower = text.lower()
    match = re.search(r"\b(?:in (?:his|her|their|the)|aged) (\d{1,2})0s\b", lower) or re.search(r"(\d{1,2})0多岁|(\d{1,2})0来岁", text)
    if match:
        decade = int(next(group for group in match.groups() if group)) * 10
        return decade, decade + 9
    # "aged" alone is no age: it describes things too ("an aged brass locket"); "aged 30" is.
    match = re.search(r"\b(\d{1,2})[- ]?(?:years?[- ]old|yo\b)|\baged (?:about |around )?(\d{1,2})\b|(\d{1,2})\s*岁",
                      lower)
    if match:
        years = int(next(group for group in match.groups() if group))
        return years, years
    for word, decade in _DECADES.items():
        if re.search(rf"\b{word}\b", lower):
            return decade, decade + 9
    # A number word is an age only beside "years" or "aged"/"about" ("ten-gallon hat" is not).
    for word, years in _NUMBERS.items():
        if re.search(rf"\b(?:aged|age|about|around) {word}\b|\b{word}[- ]years?[- ]old\b", lower):
            return years, years
    for pattern, years in _AGE_WORDS:
        if re.search(pattern, lower if pattern.startswith("\\b") else text):
            return years
    return None


def _years(span):
    low, high = span
    return f"{low}" if low == high else f"{low}–{high}" if high < 100 else f"{low}+"


def conflicts(looks, voice):
    """What the looks contradict in the voice: a list of short notes, empty when they agree."""
    notes = []
    seen, heard = gender(looks), gender(voice)
    if seen and heard and seen != heard:
        notes.append(f"Looks say {seen}; the voice is {heard}.")
    seen, heard = age(looks), age(voice)
    # Neighbouring ranges (thirties and forties) agree: only a gap of more than five years counts.
    if seen and heard and max(seen[0] - heard[1], heard[0] - seen[1]) > 5:
        notes.append(f"Looks say about {_years(seen)}; the voice sounds {_years(heard)}.")
    return notes
