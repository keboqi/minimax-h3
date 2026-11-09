"""Media's Download from a URL and its saved cookies, bound to the dubbing handlers that run them."""

from __future__ import annotations

from types import SimpleNamespace

_QUIET = {"queue": False, "api_name": False, "show_progress": "hidden"}


def bind_media_downloads(controls: SimpleNamespace, *, on_downloaded, actions) -> None:
    """Download from a URL and manage saved cookies inside the Media library.

    ``on_downloaded(event)`` chains the library refresh after a finished download.
    """
    controls.info.click(actions.video_info, inputs=controls.url, outputs=controls.status,
                        queue=True, api_name=False, concurrency_id="h3-download", concurrency_limit=2)
    downloaded = controls.download.click(
        actions.download_video, inputs=[controls.url, controls.quality],
        outputs=[controls.status, controls.path], api_name=False, show_progress="minimal",
        concurrency_id="h3-download", concurrency_limit=2,
    )
    # Signal the yt-dlp thread first: cancelling the event alone would leave it downloading.
    controls.stop.click(actions.stop_downloads, outputs=controls.status, queue=False, api_name=False,
                        show_progress="hidden").then(fn=None, cancels=[downloaded], queue=False, api_name=False)
    on_downloaded(downloaded)
    controls.cookies.expand(actions.cookie_sites, outputs=controls.sites, **_QUIET)
    controls.cookies.expand(actions.cookie_site_choices, outputs=controls.saved_site, **_QUIET)
    for event in (
        controls.import_curl.click(actions.import_curl_cookies, inputs=[controls.curl, controls.curl_domain],
                                   outputs=[controls.cookie_status, controls.sites, controls.curl], **_QUIET),
        controls.import_file.click(actions.import_cookie_file, inputs=[controls.cookie_file, controls.file_domain],
                                   outputs=[controls.cookie_status, controls.sites], **_QUIET),
        controls.delete.click(actions.delete_cookies, inputs=controls.saved_site,
                              outputs=[controls.cookie_status, controls.sites], **_QUIET),
    ):
        event.then(actions.cookie_site_choices, outputs=controls.saved_site, **_QUIET)
