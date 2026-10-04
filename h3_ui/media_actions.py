"""Selection-specific deletion confirmation at the Media presentation boundary."""

from dataclasses import dataclass
from pathlib import Path
import gradio as gr
from h3_app.jobs import JOBS


@dataclass(frozen=True)
class DeletionIntent:
    mode: str
    selected: str | None
    action: str
    paths: tuple[str, ...]
    fingerprints: tuple[tuple[int, int, int, int], ...]


def file_fingerprints(paths):
    """Require the same file identities and contents at confirmation time."""
    return tuple(
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        for info in (Path(path).stat() for path in paths)
    )


def bind_safe_deletion(
    view,
    *,
    list_paths,
    delete,
    empty,
    mutation_outputs,
    sync_more,
    refresh_inputs=None,
    refresh_outputs=None,
):
    def prepare(mode, selected, action):
        paths = tuple(str(path) for path in list_paths(mode))
        if action == "selected":
            if not selected or str(selected) not in paths:
                raise gr.Error("Select an item in this library first.")
            paths = (str(selected),)
        count = len(paths)
        if not count:
            raise gr.Error("This library is empty.")
        # Native Markdown displays user-visible names, never raw HTML.
        names = ", ".join(Path(path).name for path in paths[:3])
        remainder = f" and {count - 3} more" if count > 3 else ""
        return (
            f"Permanently delete **{count} {mode.lower()} item(s)** from the generated library? "
            f"{names}{remainder}. Settings sidecars and previews are removed too.",
            gr.update(visible=True),
            DeletionIntent(mode, selected, action, paths, file_fingerprints(paths)),
        )

    for button, action in ((view.delete, "selected"), (view.empty, "library")):
        button.click(
            lambda mode, selected, action=action: prepare(mode, selected, action),
            inputs=[view.mode, view.selected],
            outputs=[
                view.deletion_message,
                view.deletion_confirmation,
                view.deletion_intent,
            ],
            queue=False,
            api_name=False,
        )

    def confirm(intent, mode, selected):
        if intent is None or intent.mode != mode or intent.selected != selected:
            raise gr.Error("The selection changed. Review the deletion again.")
        paths = tuple(str(path) for path in list_paths(mode))
        current = (
            (str(selected),)
            if intent.action == "selected" and selected in paths
            else paths
        )
        if current != intent.paths:
            raise gr.Error("The library changed. Review the deletion again.")
        try:
            unchanged = file_fingerprints(current) == intent.fingerprints
        except OSError:
            unchanged = False
        if not unchanged:
            raise gr.Error("The media changed. Review the deletion again.")
        if any(JOBS.references_path(path) for path in intent.paths):
            raise gr.Error(
                "A queued or active job uses this media. Cancel or finish that job first."
            )
        result = (
            delete(mode, selected, True)
            if intent.action == "selected"
            else empty(mode, selected, True)
        )
        return (*result, gr.update(visible=False), None)

    confirmed = view.deletion_confirm.click(
        confirm,
        inputs=[view.deletion_intent, view.mode, view.selected],
        outputs=[*mutation_outputs, view.deletion_confirmation, view.deletion_intent],
        queue=False,
        api_name=False,
    )
    refreshed = confirmed.success(
        sync_more,
        inputs=refresh_inputs or view.mode,
        outputs=refresh_outputs or [view.shown, view.show_more],
        queue=False,
        api_name=False,
        show_progress="hidden",
    )
    view.deletion_cancel.click(
        lambda: (gr.update(visible=False), None),
        outputs=[view.deletion_confirmation, view.deletion_intent],
        queue=False,
        api_name=False,
    )
    return refreshed
