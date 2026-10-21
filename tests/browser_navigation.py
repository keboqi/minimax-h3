"""Shared Create-tab navigation for the browser acceptance tests."""

from playwright.sync_api import expect


def choose_engine(page, task, engine):
    """Pick a task, then an engine, and wait until the engine choice holds.

    A task switch replaces the engine choices and restores that task's
    remembered engine in its own response. Opening the Engine list before that
    response lands lets it replace an immediate pick, and Gradio's open option
    list throws when its choices shrink underneath it.
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
    dropdown = page.get_by_label("Engine", exact=True)
    for _ in range(5):
        dropdown.click()
        page.get_by_role("option", name=engine, exact=True).click()
        page.wait_for_timeout(300)
        if dropdown.input_value() == engine:
            return
    expect(dropdown).to_have_value(engine)
