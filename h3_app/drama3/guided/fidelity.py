"""Ordered source checks shared by the two script writers."""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from h3_app.errors import H3Error
from h3_app.drama3.phonemes import strip_notes
from .segment import BRIEF, input_kind, segment_source, source_request
from .spoken_forms import spoken_forms

_TOKENS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_PERFORMANCE = re.compile(r"\((?:sighs|laughs|gasps|coughs|whispering)\)|[（](?:笑|叹气|低声|耳语)[）]", re.I)


_SPELLING_REPAIRS = {'teh': 'the', 'becuase': 'because', 'recieve': 'receive', 'thier': 'their',
                     '以经': '已经', '在见': '再见'}


def words(text: str, *, repairs: bool = False) -> list[str]:
    text = strip_notes(text)
    text = _PERFORMANCE.sub('', text)
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", '', text)
    if repairs:
        for old, new in _SPELLING_REPAIRS.items():
            pattern = r'\b' + re.escape(old) + r'\b' if old.isascii() else re.escape(old)
            text = re.sub(pattern, new, text)
    text = spoken_forms(text)
    return [token.casefold() for token in _TOKENS.findall(text)]


# Chinese characters commonly typed for one another that read the same in context (的/地/得 as
# the particle de), so swapping within a group never changes what is heard.
_HOMOPHONE_SLIPS = tuple(frozenset(group) for group in ("的地得", "在再", "以已", "象像", "做作", "青轻", "帐账"))
_HAN = re.compile(r"[\u3400-\u9fff]")


def _homophone_slips(old: list[str], new: list[str]) -> bool:
    """Whether ``new`` only swaps characters within a homophone slip group, one for one."""
    return (0 < len(old) == len(new) <= 2
            and all(a == b or any(a in group and b in group for group in _HOMOPHONE_SLIPS)
                    for a, b in zip(old, new)))


def _restores_name(before: list[str], inserted: list[str], after: list[str], names: set[str]) -> bool:
    """Whether ``inserted`` completes a cast name the source wrote short ("韩竟是" for "韩立竟是")."""
    if not inserted or len(inserted) > 2 or not all(_HAN.fullmatch(token) for token in inserted):
        return False
    added = "".join(inserted)
    for name in names:
        for cut in range(1, len(name)):
            head, rest = name[:cut], name[cut:]
            if (rest.startswith(added) and "".join(before[-cut:]) == head
                    and "".join(after[:len(rest) - len(added)]) == rest[len(added):]):
                return True
    return False


def fix_pattern(old: str) -> re.Pattern:
    """A repair target as a whole word, so "the the" never matches inside "the theory"."""
    return re.compile(r'(?<!\w)' + re.escape(old) + r'(?!\w)' if old.isascii() else re.escape(old))


def certain_correction(old: str, new: str) -> bool:
    """Small spelling repairs, never a changed number or an arbitrary replacement."""
    return old == new or _SPELLING_REPAIRS.get(old) == new


def _joint_quotes(units):
    """Only explicit joint attributions authorize performing a quotation more than once."""
    attribution = re.compile(
        r'\b(?:said|answered|replied|cried|shouted|called|chorused)\b[^.!?]{0,60}'
        r'\b(?:together|in unison|simultaneously)\b|'
        r'\b(?:together|in unison|simultaneously)\b[^.!?]{0,60}'
        r'\b(?:said|answered|replied|cried|shouted|called|chorused)\b|'
        r'(?:同时|齐声|异口同声|一同|一起)[^。！？]{0,12}(?:说|道|应|答|喊)', re.I)
    joint = set()
    for index, unit in enumerate(units):
        if unit.kind != 'quote':
            continue
        context = []
        if index + 1 < len(units) and units[index + 1].kind == 'narration':
            following = units[index + 1].text.strip()
            if not following.endswith((':', '：')):
                context.append(re.split(r'[.!?。！？]', following)[0])
        if index and units[index - 1].kind == 'narration':
            preceding = units[index - 1].text.strip()
            if preceding.endswith((':', '：')) or index < 2 or units[index - 2].kind != 'quote':
                sentences = [part for part in re.split(r'[.!?。！？]', preceding) if part.strip()]
                if sentences:
                    context.append(sentences[-1])
        if any(attribution.search(text) for text in context):
            joint.add(tuple(words(unit.text, repairs=True)))
    return joint


def _performed_turns(script, joint, sequential):
    from h3_app.drama3.serving.tree import ParallelNode, SpeechNode, iter_speech_nodes, parse_drama3_tree

    def collect(node):
        if isinstance(node, SpeechNode):
            return [(words(turn.text, repairs=True), turn.speaker) for turn in node.turns]
        if isinstance(node, ParallelNode):
            branches = [[turn for speech in iter_speech_nodes(child) for turn in speech.turns]
                        for child in node.children]
            branches = [branch for branch in branches if branch]
            if (len(branches) > 1 and all(len(branch) == 1 for branch in branches)
                    and len({branch[0].speaker for branch in branches}) == len(branches)):
                tokens = [words(branch[0].text, repairs=True) for branch in branches]
                if tuple(tokens[0]) in joint and all(value == tokens[0] for value in tokens):
                    return [(tokens[0], branches[0][0].speaker)]
        return [turn for child in getattr(node, 'children', ()) for turn in collect(child)]

    performed = collect(parse_drama3_tree(script))
    if not sequential:
        return performed
    result = []
    index = 0
    while index < len(performed):
        tokens, speaker = performed[index]
        result.append((tokens, speaker))
        index += 1
        seen = {speaker}
        while (tuple(tokens) in joint and index < len(performed) and performed[index][0] == tokens
               and performed[index][1] not in seen):
            seen.add(performed[index][1])
            index += 1
    return result


def validate_source_fidelity(source: str, script: str, *, single: bool = False,
                             expansion: bool = False, sequential_joint: bool = False) -> None:
    """Reject source loss, reordering and changed known labels, allowing narrow repairs."""
    source, directives = source_request(source)
    expansion = expansion or bool(directives)
    kind = input_kind(source)
    if kind == BRIEF:
        return
    units = segment_source(source, kind).units
    expected, labels, origins = [], [], []
    for unit in units:
        tokens = words(unit.text, repairs=True)
        label = re.fullmatch(r"Speaker (\d+)", unit.label, re.I)
        expected.extend(tokens)
        labels.extend([int(label.group(1)) if label else None] * len(tokens))
        origins.extend([unit] * len(tokens))
    actual, speakers, names = [], [], set()
    for tokens, speaker in _performed_turns(script, _joint_quotes(units), sequential_joint):
        match = re.fullmatch(r'Speaker (\d+)', speaker)
        actual.extend(tokens)
        speakers.extend([int(match.group(1)) if match else None] * len(tokens))
        if not match and len(speaker) >= 2 and _HAN.search(speaker) and speaker in source:
            names.add(speaker)
    if not expected:
        return
    for op, a, b, c, d in SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes():
        if op == 'equal':
            if not single and any(label is not None and label != speaker
                                  for label, speaker in zip(labels[a:b], speakers[c:d])):
                raise H3Error("The writer changed a supplied speaker assignment. Your source was kept.")
            continue
        if (op == 'delete' and b - a == 1 and a > 0 and expected[a] == expected[a - 1]
                and origins[a].id == origins[a - 1].id
                and re.search(r'\b' + re.escape(expected[a]) + r'\s+' + re.escape(expected[a]) + r'\b',
                              origins[a].text, re.I)):
            continue  # a repeated word, not a repeated turn or sentence
        if op == 'insert' and expansion:
            continue
        if op == 'replace' and _homophone_slips(expected[a:b], actual[c:d]):
            continue  # a same-sound character slip, such as 年青 for 年轻
        if op == 'insert' and _restores_name(expected[:a], actual[c:d], expected[a:], names):
            continue  # a character dropped from a cast name
        raise H3Error("The writer omitted, reordered or changed source words beyond a spelling repair. "
                      "Your source was kept; retry the enhancement.")
