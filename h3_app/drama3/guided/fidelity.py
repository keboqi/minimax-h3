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


SPEAKER_CHANGE = "The writer changed a supplied speaker assignment. Your source was kept."
WORDS_CHANGE = ("The writer omitted, reordered or changed source words beyond a spelling repair. "
                "Your source was kept; retry the enhancement.")


# Characters whose doubling is a real word (慢慢, 看看, 妈妈), so dropping a copy changes the source.
_REDUPLICATED = frozenset("慢渐常刚轻静悄默偷紧深微稍早久好快远纷频连处时人个天年家事样步"
                          "看想试说走听等问找谈聊尝笑跳摸拍敲"
                          "妈爸哥姐弟妹奶爷婆公叔伯姑舅姥宝星")


def _doubled_slip(expected: list[str], origins: list, first: int) -> bool:
    """Whether ``expected[first:first + 2]`` is one word typed twice ("the the", 三名名老者) at this spot."""
    token, unit = expected[first], origins[first]
    if not _HAN.fullmatch(token):
        return bool(re.search(r'\b' + re.escape(token) + r'\s+' + re.escape(token) + r'\b', unit.text, re.I))
    if token in _REDUPLICATED:
        return False
    if any(0 <= i and i + 1 < len(expected) and expected[i] == expected[i + 1] != token
           and origins[i].id == unit.id for i in (first - 2, first + 2)):
        return False  # one half of an AABB form (老老实实)
    # The pair must sit between the same neighbours in the source text, not elsewhere in the unit.
    left = [expected[first - 1]] if first > 0 and origins[first - 1].id == unit.id else []
    right = [expected[first + 2]] if first + 2 < len(expected) and origins[first + 2].id == unit.id else []
    pattern = (r'\W*'.join(re.escape(part) for part in left) + (r'\W*' if left else '') + re.escape(token * 2)
               + (r'\W*' if right else '') + r'\W*'.join(re.escape(part) for part in right))
    return bool(re.search(pattern, unit.text))


def _joined(tokens: list[str]) -> str:
    return "".join(tokens) if all(_HAN.fullmatch(token) for token in tokens) else " ".join(tokens)


def source_changes(source: str, script: str, *, single: bool = False, expansion: bool = False,
                   sequential_joint: bool = False, label: bool = False) -> list[tuple]:
    """Every change the source check does not accept, as ``(kind, source words, script words)``.

    ``kind`` is "speaker" for a supplied label performed by another voice, otherwise "words".
    With ``label`` each entry gets a fourth item: Gemini Decision's probability that it is a
    typo repair, or None (speaker changes, no Decision key, or no answer).
    """
    changes = _changes(source, script, single=single, expansion=expansion, sequential_joint=sequential_joint)
    if not label:
        return [change[:3] for change in changes]
    verdicts = repair_probabilities(changes) if changes else None
    return [(*change[:3], verdicts[index] if verdicts else None) for index, change in enumerate(changes)]


def _changes(source: str, script: str, *, single: bool, expansion: bool, sequential_joint: bool) -> list[tuple]:
    """``source_changes`` entries; a "words" entry adds (before, source, script, after) token lists."""
    source, directives = source_request(source)
    expansion = expansion or bool(directives)
    kind = input_kind(source)
    if kind == BRIEF:
        return []
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
    changes = []
    if not expected:
        return changes
    for op, a, b, c, d in SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes():
        if op == 'equal':
            if not single and any(label is not None and label != speaker
                                  for label, speaker in zip(labels[a:b], speakers[c:d])):
                changes.append(("speaker", _joined(expected[a:b][:12]), _joined(actual[c:d][:12])))
            continue
        # Either copy of a doubled word may be the one dropped.
        twin = None
        if op == 'delete' and b - a == 1:
            twin = (a - 1 if a > 0 and expected[a] == expected[a - 1]
                    else b if b < len(expected) and expected[a] == expected[b] else None)
        if (twin is not None and origins[a].id == origins[twin].id
                and _doubled_slip(expected, origins, min(a, twin))):
            continue  # a repeated word or character (三名名老者), not a repeated turn or sentence
        if op == 'insert' and expansion:
            continue
        if op == 'replace' and _homophone_slips(expected[a:b], actual[c:d]):
            continue  # a same-sound character slip, such as 年青 for 年轻
        if op == 'insert' and _restores_name(expected[:a], actual[c:d], expected[a:], names):
            continue  # a character dropped from a cast name
        context = _joined(expected[max(0, a - 4):a])
        changes.append(("words", context + "[" + _joined(expected[a:b][:12]) + "]",
                        context + "[" + _joined(actual[c:d][:12]) + "]",
                        (expected[max(0, a - 4):a], expected[a:b], actual[c:d], expected[b:b + 4])))
    return changes


# Changes small enough to judge as typo repairs; anything larger is always rejected.
MAX_JUDGED_CHANGES = 8
MAX_JUDGED_TOKENS = 3

_JUDGE_REQUEST = """A script writer copied an author's text into a script and must keep the words exactly.
The only allowed edits are obvious typo repairs: a misspelling, a wrong Chinese character that sounds the same,
an accidentally doubled word or character, or a dropped character restored in a name.
For each change below, answer true only if it is clearly such a repair and what is heard and meant stays the same.
A changed, added or dropped word that alters meaning, tone, tense, number or negation is false. If in doubt, false.
Reply with only JSON: {{"ok": [true or false, ...]}} in change order.
Changes (the edited words are in brackets):
{items}"""


# Numbers reach the check spelled out (spoken_forms), so changed amounts are caught by their words.
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
    "million billion point percent half first second third".split()
) | frozenset("零一二两三四五六七八九十百千万亿半")


def _judgeable(change: tuple) -> bool:
    if change[0] != "words":
        return False
    _before, old, new, _after = change[3]
    return (len(old) <= MAX_JUDGED_TOKENS and len(new) <= MAX_JUDGED_TOKENS
            and not any(token in _NUMBER_WORDS or any(char.isdigit() for char in token) for token in old + new))


def _change_lines(changes: list[tuple]) -> list[tuple[str, str]]:
    """Each "words" change as (source, script) context lines with the edited words in brackets."""
    lines = []
    for _kind, _old, _new, (before, old, new, after) in changes:
        source = " ".join(filter(None, (_joined(before), f"[{_joined(old)}]", _joined(after))))
        script = " ".join(filter(None, (_joined(before), f"[{_joined(new)}]", _joined(after))))
        lines.append((source, script))
    return lines


_DECISION_REPAIR = (
    "A script writer copied an author's text and may only fix obvious typos: a misspelling, a wrong "
    "Chinese character that sounds the same, an accidentally doubled word or character, or a dropped "
    "character restored in a name. Is this change such a repair, with what is heard and meant unchanged?\n"
    "source: {source}\nscript: {script}")
# A Decision verdict counts as a typo repair at or above this probability.
REPAIR_THRESHOLD = 0.85
# With Decision, more and larger changes can be judged in one request than Gemma's JSON reply allows.
MAX_DECISION_CHANGES = 60


def _decision_key() -> str | None:
    from h3_app.llm import decisions, gemma

    # The writer's own Gemini key, when it set one for its Gemma checks, else GEMINI_API_KEY.
    return decisions.key_or_none(gemma.current_key())


def repair_probabilities(changes: list[tuple]) -> list[float | None] | None:
    """Gemini Decision's probability that each "words" change is a typo repair, in one request.

    Speaker changes and anything Decision did not answer are None. Returns None when no
    Decision key is set or the request failed, so callers keep their other checks.
    """
    from h3_app.llm import decisions

    key = _decision_key()
    words = [index for index, change in enumerate(changes) if change[0] == "words" and len(change) > 3]
    if key is None or not words:
        return None
    lines = _change_lines([changes[index] for index in words])
    questions = {f"change_{index}": decisions.yes_no(_DECISION_REPAIR.format(source=source, script=script))
                 for index, (source, script) in zip(words, lines)}
    try:
        answers = decisions.decide(questions, decisions.text(
            "Each question shows one edit an AI script writer made to an author's text."), key=key, timeout=30)
    except H3Error:
        return None
    return [decisions.yes_probability(answers.get(f"change_{index}")) for index in range(len(changes))]


def _judged_repairs(changes: list[tuple]) -> bool:
    """Whether every remaining change is judged a typo repair; False when it cannot be answered.

    Gemini Decision judges when a key is set, else Gemma. Numbers and long edits are never judged.
    """
    if not changes or not all(map(_judgeable, changes)):
        return False
    if len(changes) <= MAX_DECISION_CHANGES:
        verdicts = repair_probabilities(changes)
        if verdicts is not None:
            return all(p is not None and p >= REPAIR_THRESHOLD for p in verdicts)
    if len(changes) > MAX_JUDGED_CHANGES:
        return False
    from h3_app.llm import gemma

    lines = [f"{number}. source: {source}\n   script: {script}"
             for number, (source, script) in enumerate(_change_lines(changes), 1)]
    answer = gemma.ask_json(_JUDGE_REQUEST.format(items="\n".join(lines)), max_output_tokens=16 * len(changes) + 32)
    verdicts = answer.get("ok") if answer else None
    return isinstance(verdicts, list) and len(verdicts) == len(changes) and all(v is True for v in verdicts)


def validate_source_fidelity(source: str, script: str, *, single: bool = False,
                             expansion: bool = False, sequential_joint: bool = False) -> list[tuple[str, str, str]]:
    """Reject source loss, reordering and changed known labels, allowing narrow repairs.

    Small leftover word changes judged typo repairs (Gemini Decision, else Gemma) are accepted and returned,
    so the caller can list them; with no verdict the script is rejected as before.
    """
    changes = _changes(source, script, single=single, expansion=expansion, sequential_joint=sequential_joint)
    if not changes:
        return []
    if _judged_repairs(changes):
        return [change[:3] for change in changes]
    raise H3Error(SPEAKER_CHANGE if changes[0][0] == "speaker" else WORDS_CHANGE)


def _listed(changes: list[tuple[str, str, str]], limit: int) -> str:
    shown = "; ".join(f"{old} → {new}" if kind == "words" else f"speaker changed at {old}"
                      for kind, old, new in changes[:limit])
    more = len(changes) - limit
    return f"{shown}{f'; and {more} more' if more > 0 else ''}"


def accepted_repairs_note(changes: list[tuple[str, str, str]], limit: int = 4) -> str:
    """A status note listing small repairs accepted as typo fixes under the strict source check."""
    if not changes:
        return ""
    return (f"Source check accepted {len(changes)} small typo repair{'s' if len(changes) != 1 else ''} "
            f"({_listed(changes, limit)}).")


def source_change_note(changes: list[tuple], limit: int = 4) -> str:
    """A status note listing changes kept because the source check was set to warn.

    Entries labelled by ``source_changes(label=True)`` are sorted: changes that may alter the
    meaning come first and are named as such; likely typo fixes are counted after them.
    """
    if not changes:
        return ""
    labelled = [change for change in changes if len(change) > 3 and change[3] is not None]
    if not labelled:
        plain = [change[:3] for change in changes]
        return (f"Source check off: kept {len(changes)} change{'s' if len(changes) != 1 else ''} to your words "
                f"({_listed(plain, limit)}). Review them in the script.")
    typos = [change for change in labelled if change[3] >= REPAIR_THRESHOLD]
    risky = sorted((change for change in changes if change not in typos),
                   key=lambda change: change[3] if len(change) > 3 and change[3] is not None else -1.0)
    note = f"Source check off: kept {len(changes)} change{'s' if len(changes) != 1 else ''} to your words."
    if risky:
        note += (f" {len(risky)} may change the meaning ({_listed([change[:3] for change in risky], limit)});"
                 " review them in the script.")
    if typos:
        note += (f" {len(typos)} look{'s' if len(typos) == 1 else ''} like typo fixes"
                 f" ({_listed([change[:3] for change in typos], 2)}).")
    return note
