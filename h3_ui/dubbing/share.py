"""The view-only page behind a dub's share link: the result player alone, for any screen size.

These routes are public to anyone holding a link, so they serve only that dub's own tracks by
name, never a server path, and the page leaves out the editing notes the app's player shows.
"""

from __future__ import annotations

from html import escape

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse

from h3_app.dubbing import shares
from h3_app.dubbing.results import TRACKS

from .player import PLAYER_JS, player_html

ROUTE = "/dub-share"
_HEADERS = {
    "Cache-Control": "private, no-cache",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Robots-Tag": "noindex, nofollow",
}
_POLICY = ("default-src 'none'; media-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
           "base-uri 'none'; form-action 'none'")

PAGE_CSS = """
:root { color-scheme: light dark; --h3-bg: #f6f7f6; --h3-surface: #ffffff; --h3-muted-surface: #eef1ef;
  --h3-text: #17211d; --h3-muted: #5d6b64; --h3-border: #d5dcd8; --h3-accent: #0f766e; --h3-on-accent: #ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --h3-bg: #0e1412; --h3-surface: #17201c; --h3-muted-surface: #131b18; --h3-text: #e8eeeb;
    --h3-muted: #9fb0a8; --h3-border: #2c3833; --h3-accent: #2dd4bf; --h3-on-accent: #06201c; }
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--h3-bg); color: var(--h3-text);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", sans-serif; }
main { display: flex; flex-direction: column; gap: 10px; height: 100vh; height: 100dvh; max-width: 1500px; margin: 0 auto;
  padding: max(12px, env(safe-area-inset-top)) max(16px, env(safe-area-inset-right))
    max(12px, env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left)); }
header { flex: none; min-width: 0; }
h1 { margin: 0; overflow: hidden; font-size: clamp(16px, 2.2vw, 22px); font-weight: 650; line-height: 1.3;
  text-overflow: ellipsis; white-space: nowrap; }
header p { margin: 0; color: var(--h3-muted); font-size: 13px; }

/* Stacked: phones and portrait tablets. The lines take what the picture and controls leave. */
.h3-dub-player { flex: 1; min-height: 0; display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr);
  grid-template-rows: auto auto auto auto minmax(0, 1fr); }
.h3-dub-player-screen { grid-row: 1; position: relative; overflow: hidden; border-radius: 12px; background: #000; }
.h3-dub-player-screen video { display: block; width: 100%; max-height: 42vh; max-height: 42dvh; object-fit: contain;
  cursor: pointer; }
.h3-dub-player-poster { display: grid; place-items: center; min-height: 96px; height: 100%; color: #a9bdb3;
  font-size: 13px; letter-spacing: .08em; text-transform: uppercase; }
.h3-dub-player-caption { position: absolute; left: 3%; right: 3%; bottom: 5%; display: flex; flex-direction: column;
  align-items: center; gap: 3px; pointer-events: none; text-align: center; }
.h3-dub-player-caption > span { max-width: 100%; padding: 2px 8px; border-radius: 6px; background: rgba(0, 0, 0, .68);
  color: #ffffff; font-size: clamp(14px, 2.4vw, 22px); line-height: 1.35; white-space: pre-line; overflow-wrap: anywhere; }
.h3-dub-player-caption > span:empty { display: none; }
.h3-dub-player-caption [data-caption-source] { color: #d7e3de; font-size: clamp(12px, 1.9vw, 17px); }

.h3-dub-player-transport { grid-row: 2; display: flex; align-items: center; gap: 10px; min-width: 0; }
.h3-dub-player-transport button { flex: none; min-width: 76px; min-height: 44px; border: 0; border-radius: 10px;
  background: var(--h3-accent); color: var(--h3-on-accent); font: inherit; font-weight: 650; cursor: pointer; }
.h3-dub-player-transport input { flex: 1; min-width: 0; height: 44px; margin: 0; accent-color: var(--h3-accent); }
.h3-dub-player-time { flex: none; color: var(--h3-muted); font-size: 12px; font-variant-numeric: tabular-nums; }

.h3-dub-player-options { grid-row: 3; display: flex; flex-wrap: wrap; align-items: center; gap: 0 14px; min-width: 0;
  font-size: 14px; }
.h3-dub-player-options fieldset { display: flex; flex-wrap: wrap; gap: 0 12px; min-width: 0; margin: 0; padding: 0;
  border: 0; }
.h3-dub-player-options legend { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
.h3-dub-player-options label { display: inline-flex; align-items: center; gap: 6px; min-height: 40px; cursor: pointer; }
.h3-dub-player-options input { accent-color: var(--h3-accent); }
.h3-dub-player-select select { min-height: 36px; max-width: 46vw; padding: 2px 8px; border: 1px solid var(--h3-border);
  border-radius: 8px; background: var(--h3-surface); color: var(--h3-text); font-size: 16px; }
.h3-dub-player-note { grid-row: 4; margin: 0; color: var(--h3-muted); font-size: 12px; }
.h3-dub-player-note:empty { display: none; }

.h3-dub-player-cues { grid-row: 5; min-height: 0; overflow-y: auto; margin: 0; padding: 6px; list-style: none;
  border: 1px solid var(--h3-border); border-radius: 12px; background: var(--h3-muted-surface);
  overscroll-behavior: contain; -webkit-overflow-scrolling: touch; }
.h3-dub-player-cues li { display: flex; gap: 10px; padding: 8px 10px; border-radius: 8px; color: var(--h3-muted);
  cursor: pointer; line-height: 1.45; }
.h3-dub-player-cues li.past { opacity: .7; }
.h3-dub-player-cues li.active { color: var(--h3-text); background: var(--h3-surface);
  box-shadow: inset 3px 0 var(--h3-accent); }
.h3-dub-cue-time { flex: none; padding-top: 1px; font-size: 12px; font-variant-numeric: tabular-nums; }
.h3-dub-cue-speaker { flex: none; max-width: 25%; overflow: hidden; padding-top: 1px; color: var(--h3-accent);
  font-size: 12px; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
.h3-dub-cue-text { display: flex; flex-direction: column; min-width: 0; overflow-wrap: anywhere; }
.h3-dub-cue-source { font-size: 12.5px; opacity: .8; }
.h3-dub-player[data-subtitles="source"] .h3-dub-cue-translation,
.h3-dub-player[data-subtitles="translation"] .h3-dub-cue-source { display: none; }

.h3-word { border-radius: 3px; transition: color 90ms ease, background 90ms ease; }
li.active .h3-word:not(.spoken) { color: var(--h3-muted); }
.h3-word.current { color: var(--h3-accent); background: color-mix(in srgb, var(--h3-accent) 16%, transparent); }
.h3-dub-player-caption .h3-word:not(.spoken) { color: rgba(255, 255, 255, .6); }
.h3-dub-player-caption .h3-word.current { color: #ffd76a; background: none; }

/* Side by side: desktops, landscape tablets and landscape phones. */
@media (min-width: 900px), (orientation: landscape) and (min-width: 560px) {
  .h3-dub-player { gap: 8px 20px; grid-template-columns: minmax(0, 1.8fr) minmax(260px, 1fr);
    grid-template-rows: minmax(0, 1fr) auto auto auto; }
  .h3-dub-player-screen { grid-area: 1 / 1; }
  .h3-dub-player-screen video { height: 100%; max-height: none; }
  .h3-dub-player-transport { grid-area: 2 / 1; }
  .h3-dub-player-options { grid-area: 3 / 1; }
  .h3-dub-player-note { grid-area: 4 / 1; }
  .h3-dub-player-cues { grid-area: 1 / 2 / 5 / 3; }
}
@media (max-height: 480px) {
  header { display: none; }
  .h3-dub-player { gap: 2px 14px; }
  .h3-dub-player-options label { min-height: 34px; }
  .h3-dub-player-options select { min-height: 30px; }
}
"""

GONE_PAGE = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
             "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
             "<meta name=\"robots\" content=\"noindex\"><title>Link unavailable</title><style>"
             "body{margin:0;display:grid;place-items:center;min-height:100vh;padding:16px;box-sizing:border-box;"
             "font:16px system-ui,sans-serif;text-align:center}</style></head><body>"
             "<p>This link is no longer available.</p></body></html>")


def share_path(token: str) -> str:
    return f"{ROUTE}/{token}"


def page_html(share: shares.Share, token: str) -> str:
    """One self-contained page; its media comes only from this link's own track routes."""
    player = share.player
    urls = {getattr(player, track): f"{share_path(token)}/{track}" for track in TRACKS if getattr(player, track)}
    title = escape(share.title or "Dubbed video")
    language = f"<p>Dubbed into {escape(player.target_language)}</p>" if player.target_language else ""
    body = player_html(player, key="share", url=urls.__getitem__, viewer=True)
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
            "<meta name=\"robots\" content=\"noindex, nofollow\"><meta name=\"referrer\" content=\"no-referrer\">"
            f"<title>{title}</title><style>{PAGE_CSS}</style></head><body>"
            f"<main><header><h1>{title}</h1>{language}</header>{body}</main>"
            f"<script>{{ const element = document.body;{PLAYER_JS}}}</script></body></html>")


def install_share_routes(app: FastAPI, *, runtime) -> None:
    """Anyone with a link may watch it; nothing here reads or changes anything else."""

    @app.api_route(ROUTE + "/{token}", methods=["GET", "HEAD"], include_in_schema=False)
    def share_page(token: str) -> Response:
        share = shares.load(runtime, token)
        if share is None:
            return HTMLResponse(GONE_PAGE, status_code=404, headers=_HEADERS)
        return HTMLResponse(page_html(share, token), headers={**_HEADERS, "Content-Security-Policy": _POLICY})

    @app.api_route(ROUTE + "/{token}/{track}", methods=["GET", "HEAD"], include_in_schema=False)
    def share_track(token: str, track: str) -> FileResponse:
        share = shares.load(runtime, token)
        path = getattr(share.player, track, None) if share is not None and track in TRACKS else None
        if path is None:
            raise HTTPException(status_code=404)
        return FileResponse(path, headers=_HEADERS)
