"""Shared fullscreen controls for both modules and inline players."""

from pathlib import Path

FULLSCREEN_SCRIPT = Path(__file__).parent / "drama3" / "page" / "fullscreen.js"
BIND_FULLSCREEN_JS = FULLSCREEN_SCRIPT.read_text(encoding="utf-8").removeprefix("export ")
FULLSCREEN_BUTTON = '<button type="button" data-fullscreen hidden>Fullscreen</button>'

PLAYER_FULLSCREEN_CSS = """
[data-dub-fullscreen]:fullscreen, [data-fullscreen-pair]:fullscreen {
  display: flex; flex-direction: column; gap: 8px; box-sizing: border-box;
  padding: 12px; border: 0; border-radius: 0;
  background: var(--h3-surface, var(--surface, #151515));
}
[data-dub-fullscreen]:fullscreen > .h3-dub-player-screen {
  flex: 1; min-height: 0; aspect-ratio: auto; background: #000;
}
[data-dub-fullscreen]:fullscreen > .h3-dub-player-screen video,
[data-stage-fullscreen]:fullscreen video {
  width: 100%; height: 100%; max-height: none; object-fit: contain;
}
[data-dub-fullscreen]:fullscreen > :is(.cue-tools, [data-cues], .h3-dub-take-warning) { display: none; }
[data-stage-fullscreen]:fullscreen { background: #000; border-radius: 0; }
[data-stage-fullscreen] > [data-fullscreen] {
  position: absolute; top: 8px; right: 8px; z-index: 1; padding: 6px 10px;
  border: 1px solid #bbb; border-radius: 6px; background: #181818; color: #fff; cursor: pointer;
}
[data-stage-fullscreen] > [data-fullscreen-status] {
  position: absolute; top: 44px; right: 8px; padding: 6px 10px;
  border-radius: 6px; background: #181818; color: #fff;
}
[data-fullscreen-pair]:fullscreen > [data-compare-pictures] {
  flex: 1; min-height: 0; display: grid; grid-template: minmax(0, 1fr) / repeat(2, minmax(0, 1fr));
}
[data-fullscreen-pair]:fullscreen figure {
  display: grid; grid-template-rows: auto minmax(0, 1fr); min-width: 0; min-height: 0;
}
[data-fullscreen-pair]:fullscreen video {
  width: 100%; height: 100%; max-height: none; object-fit: contain; background: #000;
}
[data-fullscreen-pair]:fullscreen [data-compare-controls] { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
[data-fullscreen-pair]:fullscreen [data-compare-controls] label:first-of-type { flex: 1; min-width: 120px; }
@media (orientation: portrait) {
  [data-fullscreen-pair]:fullscreen > [data-compare-pictures] {
    grid-template: repeat(2, minmax(0, 1fr)) / minmax(0, 1fr);
  }
}
"""
