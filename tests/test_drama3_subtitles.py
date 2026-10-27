"""Exercise the shipped subtitle controller with parallel and sequential timings."""

import ast
from pathlib import Path
import unittest

from tests.node_js import run_js


def subtitle_controller():
    # Read the browser code without importing Gradio or starting the application.
    source = Path(__file__).resolve().parents[1] / "h3_ui" / "drama3" / "view.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DRAMA3_SUBTITLES_JS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("Missing subtitle controller")


class SubtitlePlaybackTests(unittest.TestCase):
    def playback(self, timings, times):
        return run_js("""(timings, times) => {
            let tick;
            globalThis.setInterval = fn => { tick = fn; };
            globalThis.window = {};
            class Media {
                get src() { return this._src; }
                set src(value) { this._src = value; }
                play() {}
            }
            globalThis.HTMLMediaElement = Media;
            const lines = timings.map(([start, end], index) => ({
                index, dataset: { start: String(start), end: String(end) },
                classList: { toggle(name, value) { this[name] = value; } },
                offsetTop: index * 20, offsetHeight: 20,
            }));
            const box = {
                dataset: { media: "result", duration: "10" }, offsetParent: {},
                querySelectorAll: sel => sel === "li" ? lines : [],
                querySelector: () => ({ clientHeight: 100, scrollTo() {} }),
            };
            globalThis.document = {
                querySelectorAll: () => [box], addEventListener() {},
            };
            (""" + subtitle_controller() + """)();
            const player = new Media();
            player.src = "result.mp3";
            player.paused = false;
            return times.map(time => {
                player.currentTime = time;
                tick();
                return lines.filter(line => line.classList.active).map(line => line.index);
            });
        }""", timings, times)

    def test_parallel_traversal_order_does_not_override_later_speech(self):
        self.assertEqual(self.playback([[0, 2], [5, 7], [0, 1]], [5.5]), [[1]])

    def test_all_currently_playing_lines_are_highlighted(self):
        self.assertEqual(self.playback([[0, 8], [2, 3]], [2.5, 4]), [[0, 1], [0]])

    def test_gaps_keep_the_latest_started_line_in_time_order(self):
        self.assertEqual(self.playback([[0, 2], [5, 7], [0, 1]], [3, 8]), [[2], [1]])

    def test_playing_follows_every_frame_until_paused(self):
        frames = run_js("""() => {
            const queued = [];
            globalThis.requestAnimationFrame = fn => queued.push(fn);
            globalThis.setInterval = () => {};
            globalThis.window = {};
            class Media { set src(value) {} play() { this.paused = false; } }
            globalThis.HTMLMediaElement = Media;
            let syncs = 0;
            globalThis.document = { querySelectorAll: () => { syncs += 1; return []; }, addEventListener() {} };
            (""" + subtitle_controller() + """)();
            const player = new Media();
            player.play();
            player.play();  // a second play while following starts no second loop
            const counts = [];
            for (let i = 0; i < 3 && queued.length; i++) {
                if (i === 2) player.paused = true;
                queued.shift()();
                counts.push([syncs, queued.length]);
            }
            return counts;
        }""")
        self.assertEqual(frames, [[1, 1], [2, 1], [3, 0]])

    def test_timeline_segments_and_playhead_follow_the_player(self):
        states = run_js("""() => {
            let tick;
            globalThis.setInterval = fn => { tick = fn; };
            globalThis.window = {};
            class Media { set src(value) { this._src = value; } get src() { return this._src; } play() {} }
            globalThis.HTMLMediaElement = Media;
            const toggles = () => ({ toggle(name, value) { this[name] = value; } });
            const lines = [[0, 2], [3, 5]].map(([start, end], index) => ({
                dataset: { start: String(start), end: String(end) }, classList: toggles(),
                offsetTop: index * 20, offsetHeight: 20,
            }));
            const segments = lines.map((_, index) => ({ dataset: { line: String(index) }, classList: toggles() }));
            // A sound has no subtitle line; it lights while it plays.
            segments.push({ dataset: { start: "0.5", end: "2" }, classList: toggles() });
            const head = { hidden: true, style: {} };
            const box = {
                dataset: { media: "result", duration: "10" }, offsetParent: {},
                querySelectorAll: sel => sel === "li" ? lines : segments,
                querySelector: sel => sel === "ol" ? { clientHeight: 100, scrollTo() {} } : head,
            };
            globalThis.document = { querySelectorAll: () => [box], addEventListener() {} };
            (""" + subtitle_controller() + """)();
            const player = new Media();
            player.src = "result.mp3";
            return [1, 4].map(time => {
                player.currentTime = time;
                tick();
                return [segments.map(segment => Boolean(segment.classList.active)), head.hidden, head.style.left];
            });
        }""")
        self.assertEqual(states, [[[True, False, True], False, "10%"], [[False, True, False], False, "40%"]])

    def test_seeking_backwards_recomputes_the_active_lines(self):
        self.assertEqual(self.playback([[1, 2], [4, 5]], [0, 1.5, 4.5, 1.5, 0]),
                         [[], [0], [1], [0], []])


if __name__ == "__main__":
    unittest.main()
