"""Dub & Translate's interactive panels as Gradio custom HTML components.

Each renders from its value in the browser (texts arrive HTML-escaped from
:mod:`.presenters`) and reports what the viewer did by changing the value and
firing a named event, which Python handles like any other event:

- choice cards fire ``choose`` with the picked key;
- the line board fires ``focus`` (a line clicked here or on its timeline),
  ``pick`` (lines ticked for the line actions) and ``dub`` (a line's Dub switch,
  named by ``changed``); its filter stays in the browser, and a value marked
  ``keepFocus`` keeps the line the viewer focused last;
- speaker cards fire ``save`` with the changed card's index in ``changed``;
- earlier sessions fire ``open`` with the chosen row's index in ``chosen``.
"""

from __future__ import annotations

from html import escape
import json
from typing import Sequence

import gradio as gr

from .presenters import empty_board

_CHOICE_JS = """
element.addEventListener('change', (event) => {
  if (!event.target.matches('input[type=radio]')) return;
  props.value = event.target.value;
  trigger('choose');
});
"""


def choice_cards(options: Sequence[tuple[str, str, str]], value: str, *, label: str, name: str,
                 elem_classes: Sequence[str] = ()) -> gr.HTML:
    """Radio choices as cards, each ``(key, title, detail)``; ``name`` groups the radios on the page."""
    cards = "".join(
        f'<label class="h3-dub-choice${{value === {json.dumps(key)} ? " on" : ""}}">'
        f'<input type="radio" name="{escape(name, quote=True)}" value="{escape(key, quote=True)}" '
        f'${{value === {json.dumps(key)} ? "checked" : ""}}>'
        f"<span><strong>{escape(title)}</strong><span>{escape(detail)}</span></span></label>"
        for key, title, detail in options)
    template = (f'<fieldset class="h3-dub-choices"><legend>{escape(label)}</legend>'
                f'<div class="h3-dub-choice-grid">{cards}</div></fieldset>')
    return gr.HTML(value, html_template=template, js_on_load=_CHOICE_JS, apply_default_css=False,
                   elem_classes=["h3-dub-choice-panel", *elem_classes])


BOARD_TEMPLATE = r"""${(() => {
  const v = value || {};
  const lines = v.lines || [];
  if (!lines.length) {
    return `<p class="h3-dub-board-empty">Analyze a source, or reopen an earlier session, to see its lines here.</p>`;
  }
  const speakers = v.speakers || [];
  const picked = new Set(v.picked || []);
  const filter = v.filter || 'all';
  const needs = (l) => l.flags.includes('long') || l.flags.includes('empty');
  const tests = { all: () => true, attention: needs, empty: (l) => l.flags.includes('empty') };
  const shown = (l) => filter.startsWith('speaker:') ? l.speaker === Number(filter.slice(8)) : (tests[filter] || tests.all)(l);
  const count = (test) => lines.filter(test).length;
  const chips = [['all', 'All', lines.length], ['attention', 'Needs attention', count(needs)],
    ['empty', 'Not translated', count(tests.empty)],
    ...(speakers.length > 1 ? speakers.map((s) => ['speaker:' + s.k, s.label, s.count]) : [])]
    .map(([key, label, n]) => `<button type="button" data-filter="${key}" aria-pressed="${key === filter}">${label} <span>${n}</span></button>`).join('');
  const span = Math.max(v.duration || 0, ...lines.map((l) => l.e)) || 1;
  const stamp = (t) => { const s = Math.round(t); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => `<span>${stamp(span * f)}</span>`).join('');
  const lanes = speakers.map((s) => `<div class="h3-dub-lane"><span class="h3-dub-lane-name tone-${s.tone}">${s.label}</span><div class="h3-dub-lane-track">${lines.map((l, i) => l.speaker !== s.k ? '' :
    `<button type="button" class="h3-dub-seg tone-${l.tone}${i === v.focus ? ' focus' : ''}${needs(l) ? ' warn' : ''}${l.dub ? '' : ' off'}" data-seg="${i}" style="left:${(l.s / span * 100).toFixed(3)}%;width:${Math.max(0.2, (l.e - l.s) / span * 100).toFixed(3)}%" aria-label="Line ${i + 1}, ${l.time}" title="Line ${i + 1}"></button>`).join('')}</div></div>`).join('');
  const flag = (l) => l.flags.includes('off') ? `<span class="h3-dub-flag">Original voice</span>`
    : l.flags.includes('empty') ? `<span class="h3-dub-flag bad">Not translated</span>`
    : l.flags.includes('long') ? `<span class="h3-dub-flag warn">May be too long</span>` : '';
  const rows = lines.map((l, i) => !shown(l) ? '' : `<div class="h3-dub-row${i === v.focus ? ' focus' : ''}${picked.has(i) ? ' picked' : ''}${l.dub ? '' : ' off'}" data-row="${i}">
<input type="checkbox" data-pick="${i}" ${picked.has(i) ? 'checked' : ''} aria-label="Pick line ${i + 1}">
<button type="button" class="h3-dub-row-main" data-open="${i}" aria-label="Open line ${i + 1}">
<span class="h3-dub-row-meta"><span class="h3-dub-row-number">${i + 1}</span><span class="h3-dub-row-time">${l.time}</span><span class="h3-dub-speaker tone-${l.tone}">${l.label}</span>${flag(l)}</span>
<span class="h3-dub-row-text"><span class="h3-dub-row-source">${l.source}</span><span class="h3-dub-row-translation${l.translation ? '' : ' none'}">${l.translation || 'No translation yet'}</span></span>
</button>
<input type="checkbox" role="switch" class="h3-dub-switch" data-dub="${i}" ${l.dub ? 'checked' : ''} aria-label="Dub line ${i + 1}" title="Dub this line">
</div>`).join('');
  const bar = picked.size ? `<div class="h3-dub-picked" role="status"><span><strong>${picked.size} picked</strong>: Preview, New take and Translate again use them.</span><button type="button" data-clear>Clear</button></div>` : '';
  return `<div class="h3-dub-board">
<div class="h3-dub-board-head"><p>${v.summary}</p><div class="h3-dub-filters" role="group" aria-label="Show">${chips}</div></div>
<div class="h3-dub-timeline" aria-label="Timeline">${lanes}<div class="h3-dub-ticks" aria-hidden="true">${ticks}</div></div>
${bar}
<div class="h3-dub-rows" role="list">${rows || '<p class="h3-dub-board-empty">No line matches this filter.</p>'}</div>
</div>`;
})()}"""

_BOARD_JS = """
// The line the viewer focused last. A value from a save says keepFocus: it was built from the
// board as it was when the save started, and the viewer may have clicked another line since.
let viewerFocus = props.value ? props.value.focus : null;
const update = (patch, event) => {
  if ('focus' in patch) viewerFocus = patch.focus;
  props.value = { ...props.value, ...patch, keepFocus: false };
  if (event) trigger(event);
};
element.addEventListener('click', (event) => {
  const target = event.target;
  const filter = target.closest('[data-filter]');
  if (filter) { update({ filter: filter.dataset.filter }); return; }
  if (target.closest('[data-clear]')) { update({ picked: [] }, 'pick'); return; }
  const line = target.closest('[data-seg], [data-open]');
  if (line) update({ focus: Number(line.dataset.seg ?? line.dataset.open) }, 'focus');
});
element.addEventListener('change', (event) => {
  const target = event.target;
  if (target.matches('[data-pick]')) {
    const picked = new Set(props.value.picked || []);
    const number = Number(target.dataset.pick);
    if (target.checked) picked.add(number); else picked.delete(number);
    update({ picked: [...picked].sort((a, b) => a - b) }, 'pick');
  } else if (target.matches('[data-dub]')) {
    const number = Number(target.dataset.dub);
    const lines = props.value.lines.map((line, index) => index === number ? { ...line, dub: target.checked } : line);
    update({ lines, changed: { line: number, dub: target.checked } }, 'dub');
  }
});
let shownFocus = null;
watch('value', () => {
  const value = props.value || {};
  if (value.keepFocus && value.focus !== viewerFocus) {
    props.value = { ...value, focus: viewerFocus, keepFocus: false };
    return;
  }
  viewerFocus = value.focus ?? null;
  requestAnimationFrame(scrollToFocus);
});
// Keep the focused line in view when it changes, from the timeline or the inspector.
const scrollToFocus = () => {
  const focus = props.value && props.value.focus;
  if (focus === shownFocus) return;
  shownFocus = focus;
  const row = element.querySelector(`[data-row="${focus}"]`), list = element.querySelector('.h3-dub-rows');
  if (!row || !list) return;
  const top = row.offsetTop - list.offsetTop;
  if (top < list.scrollTop || top + row.offsetHeight > list.scrollTop + list.clientHeight) {
    list.scrollTo({ top: Math.max(0, top - list.clientHeight / 3), behavior: 'smooth' });
  }
};
"""


def line_board() -> gr.HTML:
    return gr.HTML(empty_board(), html_template=BOARD_TEMPLATE, js_on_load=_BOARD_JS, apply_default_css=False,
                   elem_classes=["h3-dub-board-panel"])


SPEAKERS_TEMPLATE = r"""${(() => {
  const v = value || {};
  const cards = (v.speakers || []).map((s, i) => `<div class="h3-dub-speaker-card">
<div class="h3-dub-speaker-head"><span class="h3-dub-avatar tone-${s.tone}" aria-hidden="true">${s.initials}</span><span><strong>${s.label}</strong><span>${s.detail}</span></span></div>
<label>Voice <select data-voice="${i}">${(v.voices || []).map(([label, key]) => `<option value="${key}" ${key === s.voice ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
<label><span class="h3-dub-volume-label">Volume <span data-volume-text="${i}">${Math.round(s.volume)}%</span></span><input type="range" min="0" max="200" step="5" value="${s.volume}" data-volume="${i}" aria-label="${s.label} volume"></label>
</div>`).join('');
  return `<div class="h3-dub-speaker-cards">${cards}</div>`;
})()}"""

_SPEAKERS_JS = """
element.addEventListener('input', (event) => {
  const range = event.target.closest('[data-volume]');
  const text = range && element.querySelector(`[data-volume-text="${range.dataset.volume}"]`);
  if (text) text.textContent = `${range.value}%`;
});
element.addEventListener('change', (event) => {
  const target = event.target;
  const index = Number(target.dataset.voice ?? target.dataset.volume);
  if (Number.isNaN(index)) return;
  const speakers = props.value.speakers.map((speaker, number) => number !== index ? speaker
    : target.dataset.voice !== undefined ? { ...speaker, voice: target.value } : { ...speaker, volume: Number(target.value) });
  props.value = { ...props.value, speakers, changed: index };
  trigger('save');
});
"""


def speaker_cards() -> gr.HTML:
    return gr.HTML({"speakers": [], "voices": [], "changed": None}, html_template=SPEAKERS_TEMPLATE,
                   js_on_load=_SPEAKERS_JS, apply_default_css=False, elem_classes=["h3-dub-speakers-panel"])


RECENT_TEMPLATE = r"""${(() => {
  const sessions = (value && value.sessions) || [];
  if (!sessions.length) return `<p class="h3-dub-board-empty">No analyzed sessions yet. Sessions are kept for 14 days.</p>`;
  return `<ul class="h3-dub-recent">${sessions.map((s, i) => `<li><button type="button" data-session="${i}"><span><strong>${s.name}</strong><span>${s.detail}</span></span><span class="h3-dub-recent-when">${s.when}</span></button></li>`).join('')}</ul>`;
})()}"""

_RECENT_JS = """
element.addEventListener('click', (event) => {
  const row = event.target.closest('[data-session]');
  if (!row) return;
  props.value = { ...props.value, chosen: Number(row.dataset.session) };
  trigger('open');
});
"""


def recent_sessions() -> gr.HTML:
    return gr.HTML({"sessions": [], "chosen": None}, html_template=RECENT_TEMPLATE, js_on_load=_RECENT_JS,
                   apply_default_css=False, elem_classes=["h3-dub-recent-panel"])
