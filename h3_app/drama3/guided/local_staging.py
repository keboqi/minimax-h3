"""Place semantic cues beside their speech, preserving continuous speaker passages."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable

from h3_app.drama3.phonemes import strip_notes
from h3_app.drama3.serving.cue_timing import normalized
from h3_app.drama3.serving.planner import spoken_text
from h3_app.drama3.serving.speaker_effects import extract_speaker_effects
from h3_app.drama3.serving.tree import parse_sound_tag
from h3_app.errors import H3Error


@dataclass
class _Part:
    opening: str = ''
    children: list['_Part'] = field(default_factory=list)
    raw: str = ''
    uid: str = ''
    passage: bool = False
    scope: bool = False


def _voice(raw):
    tagged = re.match(r'^(<\|speaker:\d+\|>)\s*(.*)$', raw)
    if tagged:
        return tagged.groups()
    if ': ' in raw:
        return raw.split(': ', 1)
    return 'Speaker 0', raw


def phrase_count(phrase: str, text: str) -> int:
    text = strip_notes(spoken_text(text))
    target = normalized(phrase)
    if not target:
        return 0
    if target.isascii():
        tokens = re.findall(r'[^\W_]+', phrase)
        pattern = r'(?<!\w)' + r'\W*'.join(re.escape(token) for token in tokens) + r'(?!\w)'
        return len(re.findall(pattern, text, re.I))
    return normalized(text).count(target)


def _leaves(part: _Part) -> list[_Part]:
    return [part] if part.uid else [leaf for child in part.children for leaf in _leaves(child)]


def _path(part: _Part, uid: str) -> list[_Part]:
    if part.uid == uid:
        return [part]
    for child in part.children:
        found = _path(child, uid)
        if found:
            return [part, *found]
    return []


def _replace(root: _Part, old: _Part, replacement: _Part) -> bool:
    for index, child in enumerate(root.children):
        if child is old:
            root.children[index] = replacement
            return True
        if _replace(child, old, replacement):
            return True
    return False


def _scope(root: _Part, first: str, last: str) -> _Part:
    left, right = _path(root, first), _path(root, last)
    if not left or not right:
        raise ValueError('A local cue references missing speech')
    # A passage is indivisible: its lines must still synthesize together.
    common = root
    for a, b in zip(left, right):
        if a is not b:
            break
        common = a
    if common.uid:
        common = left[-2]  # a singleton passage, or an existing simultaneous branch
    if common.passage or common.opening.casefold().startswith('<parallel'):
        selected = common
        leaves = _leaves(selected)
        wanted = {leaf.uid for leaf in leaves}
        # Reuse an existing wrapper for the identical speech range.
        for ancestor in reversed(left[:-1]):
            if ancestor.scope and {leaf.uid for leaf in _leaves(ancestor)} == wanted:
                return ancestor
        scope = _Part('<parallel>', [_Part('<sequential>', [selected])], scope=True)
        _replace(root, selected, scope)
        return scope

    # For a multi-turn music span, wrap only its contiguous sequence range.
    start = next(i for i, child in enumerate(common.children) if _path(child, first))
    end = next(i for i, child in enumerate(common.children) if _path(child, last))
    start, end = min(start, end), max(start, end)
    selected = common.children[start:end + 1]
    wanted = {leaf.uid for child in selected for leaf in _leaves(child)}
    for ancestor in reversed(left):
        if ancestor.scope and {leaf.uid for leaf in _leaves(ancestor)} == wanted:
            return ancestor
    scope = _Part('<parallel>', [_Part('<sequential>', selected)], scope=True)
    common.children[start:end + 1] = [scope]
    return scope


def _occurrence(scope: _Part, uid: str, phrase: str, occurrence: int) -> int:
    before = 0
    for leaf in _leaves(scope):
        if leaf.uid == uid:
            return before + occurrence
        before += phrase_count(phrase, _voice(leaf.raw)[1])
    raise ValueError('Missing cue target in local scope')


def _parse(items, lines=None):
    root = _Part('<sequential>')
    stack = [root]
    spoken = iter(lines) if lines is not None else None
    count = 0
    for raw in items:
        raw = raw.strip()
        if not raw.strip():
            continue
        container = re.fullmatch(r'<(/?)(parallel|sequential)\b[^<>]*>', raw, re.I)
        if container and container.group(1):
            kind = container.group(2).casefold()
            if len(stack) == 1 or not stack[-1].opening.casefold().startswith('<' + kind):
                raise H3Error('The script has mismatched staging container tags.')
            stack.pop()
        elif container:
            part = _Part(raw)
            stack[-1].children.append(part)
            stack.append(part)
        else:
            uid = ''
            if not raw.casefold().startswith('<sound') and not raw.startswith('<#'):
                count += 1
                uid = f'u{count}'
                if spoken is not None:
                    uid, expected = next(spoken)
                    if raw != expected:
                        raise ValueError('The staged timeline changed speech order')
            stack[-1].children.append(_Part(raw=raw, uid=uid))
    if len(stack) != 1:
        raise H3Error('The script has unclosed staging container tags.')
    return root


def _emit(part):
    if part.raw:
        return [part.raw]
    content = [line for child in part.children for line in _emit(child)]
    if not part.opening:
        return content
    closing = '</parallel>' if part.opening.casefold().startswith('<parallel') else '</sequential>'
    return [part.opening, *content, closing]


def local_staging(items: list[str], lines: list[tuple[str, str]],
                  cues: list[tuple[str, str, str, str, int, int, Callable[[int, int], str]]],
                  holds: list[tuple[str, str, float]]) -> list[str]:
    """Cues carry start/end unit IDs, phrases, line-local occurrences and a formatter.

    Only sequential speech runs with identical speaker/effects become passages.
    Existing reaction, interruption and sound-chain structure remains intact.
    """
    if not cues and not holds:
        return items
    root = _parse(items, lines)

    def passages(part):
        for child in part.children:
            passages(child)
        if not part.opening.casefold().startswith('<sequential'):
            # Parallel speech is simultaneous, so group its leaves separately.
            part.children = [_Part(children=[child], passage=True) if child.uid else child
                             for child in part.children]
            return
        grouped = []
        key = None
        for child in part.children:
            current = None
            if child.uid:
                speaker, body = _voice(child.raw)
                _, effects = extract_speaker_effects(body)
                current = (speaker, effects)
            if current is not None and key == current:
                grouped[-1].children.append(child)
            elif current is not None:
                grouped.append(_Part(children=[child], passage=True))
            else:
                grouped.append(child)
            key = current
        part.children = grouped

    passages(root)
    for first, last, phrase, end_phrase, occurrence, end_occurrence, format_sound in cues:
        scope = _scope(root, first, last)
        number = _occurrence(scope, first, phrase, occurrence)
        end_number = _occurrence(scope, last, end_phrase, end_occurrence) if end_phrase else 1
        scope.children.insert(len(scope.children) - 1, _Part(raw=format_sound(number, end_number)))
    for uid, phrase, seconds in holds:
        scope = _scope(root, uid, uid)
        hold = f'{phrase}~{seconds:g}'
        if ' holds="' in scope.opening:
            scope.opening = scope.opening[:-2] + ';' + hold + '">'
        else:
            scope.opening = scope.opening[:-1] + f' holds="{hold}">'

    return [line for child in root.children for line in _emit(child)]


def localize_existing_cues(script: str) -> str:
    """Narrow existing hoisted cues without changing assets, words or simultaneous branches."""
    root = _parse(script.splitlines())

    def visit(part):
        for child in part.children:
            visit(child)
        sequences = [child for child in part.children if child.opening.casefold().startswith('<sequential')]
        if not part.opening.casefold().startswith('<parallel') or len(sequences) != 1:
            return
        sequence = sequences[0]
        if any(child is not sequence and (child.opening or child.uid) for child in part.children):
            return  # multiple simultaneous speech branches must retain their structure
        leaves = _leaves(sequence)
        if not leaves:
            return

        def locate(phrase, occurrence):
            for index, leaf in enumerate(leaves):
                count = phrase_count(phrase, _voice(leaf.raw)[1])
                if count >= occurrence:
                    return index, leaf.uid, occurrence
                occurrence -= count
            return None

        def passage_key(leaf):
            name, body = _voice(leaf.raw)
            _, effects = extract_speaker_effects(body)
            return name, effects

        def ambiguous_overlap(node, phrase):
            if not phrase:
                return False
            if node.opening.casefold().startswith('<parallel'):
                matching_branches = sum(any(phrase_count(phrase, _voice(leaf.raw)[1])
                                            for leaf in _leaves(child)) for child in node.children)
                if matching_branches > 1:
                    return True
            return any(ambiguous_overlap(child, phrase) for child in node.children)

        cues, moved = [], []
        for child in part.children:
            sound = parse_sound_tag(child.raw) if child.raw.casefold().startswith('<sound') else None
            if sound is None or not sound.anchor:
                continue
            # Text order cannot reconstruct the measured order of concurrent voices.
            # Retain the original search scope when either phrase is ambiguous there.
            if ambiguous_overlap(sequence, sound.anchor) or ambiguous_overlap(sequence, sound.end_anchor):
                continue
            start = locate(sound.anchor, sound.occurrence)
            end = locate(sound.end_anchor, sound.end_occurrence) if sound.end_anchor else (
                (len(leaves) - 1, leaves[-1].uid, 1) if sound.loop else start)
            if start is None or end is None:
                continue  # preserve unresolved legacy cues for review, never guess a new target
            a, b = min(start[0], end[0]), max(start[0], end[0])
            while a > 0 and passage_key(leaves[a - 1]) == passage_key(leaves[a]):
                a -= 1
            while b + 1 < len(leaves) and passage_key(leaves[b + 1]) == passage_key(leaves[b]):
                b += 1
            if a == 0 and b == len(leaves) - 1:
                continue  # already the smallest useful speech span; remain idempotent

            def format_sound(number, end_number, raw=child.raw, has_end=bool(sound.end_anchor)):
                prefix, body = raw.split('>', 1)
                for key, value in [('occurrence', number), ('end_occurrence', end_number)]:
                    if key == 'end_occurrence' and not has_end:
                        continue
                    pattern = r'(?<!\w)' + key + r'\s*=\s*(?:"[^"]*"|\x27[^\x27]*\x27|[^\s>]+)'
                    if re.search(pattern, prefix):
                        prefix = re.sub(pattern, f'{key}="{value}"', prefix)
                    elif value != 1:
                        prefix += f' {key}="{value}"'
                return prefix + '>' + body

            cues.append((start[1], end[1], sound.anchor, sound.end_anchor, start[2], end[2], format_sound))
            moved.append(child)
        if cues:
            lines = [(leaf.uid, leaf.raw) for leaf in leaves]
            items = [line for child in sequence.children for line in _emit(child)]
            sequence.children = _parse(local_staging(items, lines, cues, []), lines).children
            part.children = [child for child in part.children if not any(child is old for old in moved)]

    visit(root)
    return '\n'.join(line for child in root.children for line in _emit(child))
