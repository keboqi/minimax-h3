"""Bounded scene windows and metadata shared by acting and sound stages."""

from dataclasses import dataclass, replace

from .budget import Budget
from .model import Scene, Staging, Utterance

SMART_LINES = 16
SMART_CHARS = 5000


def windows(items, text, limit=SMART_LINES, *, characters=SMART_CHARS, weight=None):
    """Keep source identities intact; one indivisible unit may exceed the text budget."""
    batch, size, work = [], 0, 0
    for item in items:
        length = len(text(item))
        cost = weight(item) if weight else 1
        if batch and (work + cost > limit or size + length > characters):
            yield batch
            batch, size, work = [], 0, 0
        batch.append(item)
        size += length
        work += cost
    if batch:
        yield batch


def scene_batches(scenes, by_id, limit, budget: Budget | None = None):
    def weight(uid):
        line = by_id[uid]
        return 1 + min(2, len(line.text) // 200) + bool(line.effects or line.kind in ("reaction", "interrupt"))

    characters = budget.characters if budget else SMART_CHARS
    pieces = [replace(scene, utterances=tuple(chunk)) for scene in scenes
              for chunk in windows(scene.utterances, lambda uid: by_id[uid].text, limit,
                                   characters=characters, weight=weight if budget else None)]
    if budget is None:
        return [[piece] for piece in pieces]
    batches, current, size, work = [], [], 0, 0
    for piece in pieces:
        count = sum(weight(uid) for uid in piece.utterances)
        length = sum(len(by_id[uid].text) for uid in piece.utterances)
        if current and (work + count > limit or size + length > characters
                        or any(s.id == piece.id for s in current)):
            batches.append(current)
            current, size, work = [], 0, 0
        current.append(piece)
        size += length
        work += count
    if current:
        batches.append(current)
    return batches


@dataclass(frozen=True)
class StageBatch:
    scenes: tuple[Scene, ...]
    line_ids: frozenset[str]
    speaker_names: frozenset[str]
    neighbors: tuple[Utterance, ...]
    continued: tuple[str, ...]

    @property
    def scene_lines(self) -> dict[str, set[str]]:
        return {scene.id: set(scene.utterances) for scene in self.scenes}


def prepare_batches(batches, scenes, utterances, *, neighbors=0) -> list[StageBatch]:
    """Compute immutable request metadata once, before any concurrent stage starts."""
    full_scenes = {scene.id: scene for scene in scenes}
    order = {u.id: index for index, u in enumerate(utterances)}
    result = []
    for batch in batches:
        ids = frozenset(uid for scene in batch for uid in scene.utterances)
        first, last = min(order[uid] for uid in ids), max(order[uid] for uid in ids)
        adjacent = tuple(utterances[max(0, first - neighbors):first] + utterances[last + 1:last + neighbors + 1])
        result.append(StageBatch(tuple(batch), ids,
            frozenset(utterances[order[uid]].speaker for uid in ids), adjacent,
            tuple(s.id for s in batch if s.utterances[0] != full_scenes[s.id].utterances[0])))
    return result


def continuation_staging(staging, continued, notes):
    """Later windows cannot become scene masters, even if the first sound request failed."""
    result = dict(staging)
    for sid in continued:
        if sid not in result:
            continue
        plan = result[sid]
        if plan.ambience or plan.music or plan.room or plan.transition:
            notes.append(f"{sid}: ignored scene-wide sound settings from a continuation window.")
        result[sid] = replace(plan, ambience="", music="", room="", transition="",
                              ambience_level="", music_level="")
    return result


def merge_staging(previous: Staging, incoming: Staging, notes=None, *, scene_id="") -> Staging:
    """Append local events; keep the first established scene-wide speaker positions."""
    positions = dict(previous.positions)
    for name, value in incoming.positions:
        if name in positions and positions[name] != value:
            if notes is not None:
                notes.append(f"{scene_id}: kept {name}'s initial scene position; later windows cannot move earlier speech.")
        else:
            positions[name] = value
    return replace(previous, events=previous.events + incoming.events, pauses=previous.pauses + incoming.pauses,
                   positions=tuple(positions.items()))
