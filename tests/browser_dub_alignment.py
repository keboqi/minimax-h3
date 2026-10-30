"""Repaired word spans must bind to playback when Gradio retains the player root."""

import os
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

from h3_app.dubbing.pipeline import Player
from h3_ui.dubbing.player import PLAYER_JS, player_html


def main():
    cue = {"start": 0, "end": 1, "shown_start": 0, "shown_end": 1,
           "speaker": "speaker1", "source": "Hello", "translation": "Salut"}
    player = Player(None, Path("dub.mp3"), Path("original.mp3"), None, 1, [cue], word_timing=True)
    plain = player_html(player, key="test")
    cue["source_words"] = [{"text": "Hello", "start": .1, "end": .8}]
    cue["translation_words"] = [{"text": "Salut", "start": .1, "end": .8}]
    repaired = player_html(player, key="test")
    executable = os.getenv("H3_BROWSER_EXECUTABLE")
    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if not executable and chrome.exists():
        executable = str(chrome)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({"executable_path": executable} if executable else {}))
        try:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content('<div id="host">' + plain + '</div>')
            page.evaluate("code => new Function('element', code)(document.querySelector('#host'))", PLAYER_JS)
            expect(page.locator(".h3-dub-player")).to_have_attribute("data-bound", "true")
            page.evaluate("html => { window.oldRoot = document.querySelector('.h3-dub-player'); "
                          "const next = new DOMParser().parseFromString(html, 'text/html').body.firstElementChild; "
                          "oldRoot.innerHTML = next.innerHTML; oldRoot.dataset.revision = next.dataset.revision; }", repaired)
            page.wait_for_function("document.querySelector('.h3-dub-player') === window.oldRoot")
            # A timeupdate on the patched media must light the repaired words.
            page.evaluate("() => { const voice = document.querySelector('[data-voice]'); "
                          "voice.currentTime = .3; voice.dispatchEvent(new Event('timeupdate')); }")
            expect(page.locator("[data-cues] .h3-word.current")).to_have_count(2)
            expect(page.locator("[data-caption-translation] .h3-word.current")).to_have_count(1)
            assert not errors, errors
            print("PASS: repaired words highlight on replay without regenerating audio")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
