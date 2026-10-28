"""Shared Create-tab navigation for the browser acceptance tests."""

from playwright.sync_api import expect


def choose_engine(page, task, engine):
    """Pick a task, then an engine, and wait until the engine choice holds.

    A task switch replaces the engine choices and restores that task's
    remembered engine in its own response; a pick made before that response
    lands is replaced by it, so the pick repeats until it holds.
    """
    posts = []

    def record(request):
        posts.append(request)

    page.on("request", record)
    try:
        page.locator(".h3-task-picker").get_by_label(task, exact=True).check()
        page.wait_for_timeout(100)
        for request in posts:
            if request.method == "POST":
                request.response()
    finally:
        page.remove_listener("request", record)
    page.wait_for_timeout(100)
    pill = engine_pill(page, engine)
    for _ in range(5):
        pill.check()
        page.wait_for_timeout(300)
        if pill.is_checked():
            return
    expect(pill).to_be_checked()


def engine_pill(page, engine):
    """The Engine row's choice for ``engine``, by its visible name."""
    return page.locator(".h3-engine-bar").get_by_label(engine, exact=True)


def open_create(page):
    """Return to Create: the top bar's current task stands in for the hidden Create tab."""
    page.locator(".h3-task-picker label:has(input:checked)").click()
    expect(page.locator('#h3-main-tabs [role="tab"][data-tab-id="create"]')).to_have_attribute(
        "aria-selected", "true"
    )
