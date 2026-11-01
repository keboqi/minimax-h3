"""Apply validated content annotations without granting later stages rewrite access."""

from dataclasses import replace
import re

from h3_app.errors import H3Error
from .fidelity import fix_pattern
from .model import Utterance


def add_pronunciations(text: str, readings: dict[str, str]) -> str:
    if not readings:
        return text
    alternatives = [r'(?<![A-Za-z])' + re.escape(word) + r'(?![A-Za-z])'
                    if word.isascii() else re.escape(word)
                    for word in sorted(readings, key=len, reverse=True)]

    def annotate(match):
        if text[match.end():match.end() + 1] == '{' or text[:match.start()].rfind('{') > text[:match.start()].rfind('}'):
            return match.group()  # existing notes stay authoritative
        return f'{match.group()}{{{readings[match.group()]}}}'

    return re.sub('|'.join(alternatives), annotate, text)


def apply_content_annotations(utterances: list[Utterance], answer: dict) -> list[Utterance]:
    additions = {}
    for index, (after, speaker, text) in enumerate(answer.get('insertions', [])):
        additions.setdefault(after, []).append((index, speaker, text))
    readings = dict(answer.get('pronunciations', []))
    result = []
    for position, utterance in enumerate(utterances):
        text = utterance.text
        for uid, old, new in answer.get('fixes', []):
            if uid in utterance.sources:
                pattern = fix_pattern(old)
                if len(pattern.findall(text)) != 1:
                    raise H3Error('A spelling repair is ambiguous inside merged narration.')
                text = pattern.sub(lambda _match: new, text, count=1)
        result.append(replace(utterance, text=add_pronunciations(text, readings), id=f'u{len(result) + 1}'))
        if position + 1 < len(utterances) and utterances[position + 1].sources == utterance.sources:
            continue  # a joint quote finishes before its following insertion
        for uid in utterance.sources:
            for index, speaker, text in additions.pop(uid, []):
                result.append(Utterance(f'u{len(result) + 1}', speaker, add_pronunciations(text, readings),
                                        sources=(uid, f'addition_{index}')))
    return result
