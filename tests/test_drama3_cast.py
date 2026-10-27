"""Exercise the shipped cast grid controller against a small stand-in for the page."""

import ast
from pathlib import Path
import unittest

from tests.node_js import run_js


def cast_controller():
    # Read the browser code without importing Gradio or starting the application.
    source = Path(__file__).resolve().parents[1] / "h3_ui" / "drama3" / "view.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DRAMA3_CAST_JS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("Missing cast controller")


# Slots as Gradio renders them: a title in the accordion header, and .hidden while not shown.
# Each step changes the page, runs one poll, and reports the grid's cards and the shown slot.
PAGE = """(titles, shown, steps) => {
    let tick, click;
    globalThis.setInterval = fn => { tick = fn; };
    globalThis.window = {};
    const classes = (names = []) => {
        const set = new Set(names);
        return { contains: n => set.has(n), toggle(n, on) { on ? set.add(n) : set.delete(n); }, set };
    };
    const slots = titles.map((title, index) => {
        const header = { classList: classes(["open"]), clicks: 0,
                         click() { this.clicks += 1; this.classList.toggle("open", !this.classList.contains("open")); } };
        return { id: `h3-drama3-voice-${index}`, title, shown: index < shown, header, classList: classes(),
                 closest(sel) { return sel === ".hidden" && !this.shown ? this : null; },
                 querySelector(sel) { return sel === "button.label-wrap" ? header : { textContent: this.title }; } };
    });
    const panel = { querySelectorAll: () => slots };
    const grid = { dataset: {}, offsetParent: {}, innerHTML: "",
                   closest: sel => sel === ".h3-drama3-voices" ? panel : sel === ".h3-drama3-cast" ? grid : null };
    globalThis.document = {
        querySelectorAll: () => [grid],
        getElementById: id => slots.find(slot => slot.id === id),
        addEventListener: (type, fn) => { click = fn; },
    };
    (""" + "{controller}" + """)();
    const report = () => ({
        cards: [...grid.innerHTML.matchAll(/tone-(\\w+)" data-slot="([^"]+)" data-state="(\\w+)" aria-pressed="(\\w+)">.*?cast-name">([^<]*)</g)]
            .map(([, tone, slot, state, pressed, name]) => [name, tone, state, pressed === "true"]),
        visible: slots.filter(slot => slot.shown && !slot.classList.contains("h3-drama3-cast-off")).map(slot => slot.id),
        opened: slots.map(slot => slot.header.clicks),
    });
    return steps.map(([action, value]) => {
        if (action === "show") slots[value].shown = true;
        if (action === "hide") slots[value].shown = false;
        if (action === "fold") slots[value].header.classList.toggle("open", false);
        if (action === "title") slots[value[0]].title = value[1];
        if (action === "pick") {
            click({ target: { closest: () => ({ dataset: { slot: slots[value].id }, closest: () => grid }) } });
        } else {
            tick();
        }
        return report();
    });
}"""


class CastGridTests(unittest.TestCase):
    TITLES = ["Speaker 0 · narrator · saved voice", "Speaker 1 · Mara", "Speaker 2 · Tobias · voice ready",
              "Speaker 3"]

    def run_page(self, steps, shown=3):
        return run_js(PAGE.replace("{controller}", cast_controller()), self.TITLES, shown, steps)

    def test_cards_follow_slot_titles_and_the_first_speaker_is_picked(self):
        [first] = self.run_page([["poll", None]])
        self.assertEqual(first["cards"], [
            ["narrator", "narrator", "ready", True],
            ["Mara", "0", "missing", False],
            ["Tobias", "1", "ready", False],
        ])
        self.assertEqual(first["visible"], ["h3-drama3-voice-0"])

    def test_picking_a_card_shows_its_slot_and_opens_it_when_folded(self):
        *_, picked, again = self.run_page([["poll", None], ["fold", 1], ["pick", 1], ["fold", 1], ["pick", 1]])
        self.assertEqual(picked["visible"], ["h3-drama3-voice-1"])
        self.assertEqual([card[3] for card in picked["cards"]], [False, True, False])
        self.assertEqual(picked["opened"][1], 1)
        self.assertEqual(again["opened"][1], 2)  # picking the shown speaker again unfolds it

    def test_added_speaker_is_picked_and_a_removed_pick_falls_back_to_the_last(self):
        _, added, removed = self.run_page([["poll", None], ["show", 3], ["hide", 3]])
        self.assertEqual(added["visible"], ["h3-drama3-voice-3"])
        self.assertEqual(added["cards"][3], ["Unnamed", "none", "missing", True])
        self.assertEqual(removed["visible"], ["h3-drama3-voice-2"])

    def test_writer_filling_the_cast_keeps_the_pick_and_updates_cards(self):
        _, filled = self.run_page([["poll", None], ["title", [1, "Speaker 1 · Mara · voice ready"]]])
        self.assertEqual(filled["cards"][1], ["Mara", "0", "ready", False])
        self.assertEqual(filled["visible"], ["h3-drama3-voice-0"])


if __name__ == "__main__":
    unittest.main()
