"""One submission lifecycle and deadline across websocket, polling, and history."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import websocket

from .comfy import ComfyClient
from .config import RuntimeConfig
from .errors import H3Error
from .jobs import Job, JobCoordinator, scoped_graph
from .progress import ProgressUpdate, no_progress
from .status import graph_class_types, node_stage


@dataclass
class Submission:
    prompt_id: str
    graph: dict
    client: ComfyClient
    deadline: float
    timeout: float
    poll_seconds: float
    output_token: str = ""
    queued_at: float = 0
    socket: Any = None
    notice: str | None = None
    job: Job | None = None
    check_cancelled: Callable[[], None] = no_progress
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    _completed_history: dict | None = None

    def _get(self, path):
        self.check_cancelled()
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise H3Error(f"Generation timed out after {self.timeout:.0f} seconds")
        return self.client.get(path, timeout=min(self.client.timeout, remaining))

    def pause(self):
        delay = max(0, min(self.poll_seconds, self.deadline - self.clock()))
        if self.job:
            self.job.cancelled.wait(delay)
        else:
            self.sleeper(delay)
        self.check_cancelled()

    def progress(self):
        try:
            updates = (
                self._stream(self.socket, self.prompt_id, self.graph, self.clock())
                if self.socket is not None
                else self._poll(self.prompt_id, self.graph)
            )
            for update in updates:
                yield ProgressUpdate(*update)
        finally:
            self.close()

    def history(self):
        if self._completed_history is None:
            self._completed_history = self._wait_history(self.prompt_id)
            if self.job and self.job.prompt_id == self.prompt_id:
                self.job.prompt_id = None
        return self._completed_history

    def close(self):
        socket, self.socket = self.socket, None
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass

    def queue_position(self, prompt_id: str) -> tuple[str, int | None]:
        """Return ComfyUI's actual queue state and one-based waiting position."""
        try:
            payload = self._get("/queue").json()
            for item in payload.get("queue_running", []):
                if len(item) > 1 and str(item[1]) == prompt_id:
                    return "running", None
            for position, item in enumerate(payload.get("queue_pending", []), start=1):
                if len(item) > 1 and str(item[1]) == prompt_id:
                    return "queued", position
        except Exception:
            pass
        return "unknown", None

    def _stream(
        self,
        ws: websocket.WebSocket,
        prompt_id: str,
        graph: dict[str, Any],
        started: float,
    ) -> Iterable[tuple[str, int, int, int | None, int | None]]:
        """Yield live node and sampler progress from ComfyUI's websocket."""
        total_nodes = len(graph)
        workflow_classes = graph_class_types(graph)
        completed: set[str] = set()
        current_node: str | None = None
        deadline = self.deadline

        while self.clock() < deadline:
            self.check_cancelled()
            try:
                ws.settimeout(
                    max(0.001, min(self.poll_seconds, 1.0, deadline - self.clock()))
                )
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                history = self._get(f"/history/{prompt_id}").json().get(prompt_id)
                if history:
                    status = history.get("status", {})
                    if status.get("status_str") == "error":
                        raise H3Error(
                            f"ComfyUI execution failed: {status.get('messages', [])}"
                        )
                    if status.get("completed") or history.get("outputs"):
                        return
                state, position = self.queue_position(prompt_id)
                if state == "queued":
                    yield (
                        f"Waiting in queue (position {position})",
                        len(completed),
                        total_nodes,
                        None,
                        None,
                    )
                elif state == "running" and current_node is None:
                    yield "Starting workflow", len(completed), total_nodes, None, None
                continue
            except websocket.WebSocketException:
                yield from self._poll(prompt_id, graph)
                return

            # Binary messages are latent previews; progress metadata arrives as JSON.
            if not isinstance(raw, str):
                continue
            if not raw:
                yield from self._poll(prompt_id, graph)
                return
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event_type = str(message.get("type", ""))
            data = message.get("data", {})
            event_prompt_id = data.get("prompt_id")
            if event_prompt_id is not None and str(event_prompt_id) != prompt_id:
                continue

            if event_type == "execution_error":
                node = str(data.get("node_type") or data.get("node_id") or "workflow")
                error = (
                    data.get("exception_message")
                    or data.get("exception_type")
                    or "unknown error"
                )
                raise H3Error(f"ComfyUI failed in {node}: {error}")
            if event_type == "execution_interrupted":
                raise H3Error("Generation interrupted.")
            if event_type in {"execution_success", "execution_complete"}:
                return
            if event_type == "execution_cached":
                completed.update(str(node) for node in data.get("nodes", []))
                yield (
                    "Restoring cached workflow results",
                    len(completed),
                    total_nodes,
                    None,
                    None,
                )
                continue
            if event_type == "executed":
                node_id = str(data.get("node", ""))
                if node_id:
                    completed.add(node_id)
                continue
            if event_type == "executing":
                node = data.get("node")
                if node is None:
                    return
                if current_node and current_node != str(node):
                    completed.add(current_node)
                current_node = str(node)
                class_type = graph.get(current_node, {}).get("class_type", "Processing")
                yield (
                    node_stage(class_type, workflow_classes),
                    len(completed),
                    total_nodes,
                    None,
                    None,
                )
                continue
            if event_type == "progress":
                node_id = str(data.get("node") or current_node or "")
                class_type = graph.get(node_id, {}).get("class_type", "Processing")
                value = int(data.get("value", 0))
                maximum = int(data.get("max", 0))
                yield (
                    node_stage(class_type, workflow_classes),
                    len(completed),
                    total_nodes,
                    value,
                    maximum,
                )
        raise H3Error(f"Generation timed out after {self.timeout:.0f} seconds")

    def _poll(
        self,
        prompt_id: str,
        graph: dict[str, Any],
    ) -> Iterable[tuple[str, int, int, int | None, int | None]]:
        """Compatibility fallback for ComfyUI deployments without `/ws`."""
        deadline = self.deadline
        while self.clock() < deadline:
            self.check_cancelled()
            payload = self._get(f"/history/{prompt_id}").json()
            item = payload.get(prompt_id)
            if item:
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise H3Error(
                        f"ComfyUI execution failed: {status.get('messages', [])}"
                    )
                if status.get("completed") or item.get("outputs"):
                    return
            state, position = self.queue_position(prompt_id)
            if state == "queued":
                stage = f"Waiting in queue (position {position})"
            elif state == "running":
                stage = "Running workflow (live step events unavailable)"
            else:
                stage = "Waiting for ComfyUI"
            yield stage, 0, len(graph), None, None
            self.pause()
        raise H3Error(f"Generation timed out after {self.timeout:.0f} seconds")

    def _wait_history(self, prompt_id: str) -> dict[str, Any]:
        deadline = self.deadline
        while self.clock() < deadline:
            self.check_cancelled()
            payload = self._get(f"/history/{prompt_id}").json()
            item = payload.get(prompt_id)
            if item:
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise H3Error(
                        f"ComfyUI execution failed: {status.get('messages', [])}"
                    )
                if status.get("completed") or item.get("outputs"):
                    return item
            self.pause()
        raise H3Error(f"Generation timed out after {self.timeout:.0f} seconds")


class PromptId(str):
    """String-compatible public ID carrying its explicit execution context."""

    def __new__(cls, submission: Submission):
        value = str.__new__(cls, submission.prompt_id)
        value.submission = submission
        return value

    @property
    def notice(self):
        return self.submission.notice


@dataclass(frozen=True)
class ExecutionRunner:
    config: RuntimeConfig
    client: ComfyClient
    coordinator: JobCoordinator
    connect: Callable = websocket.create_connection
    clock: Callable[[], float] = time.monotonic

    def submit(self, graph: dict, client_id: str, job: Job | None = None) -> PromptId:
        check = job.check if job else no_progress
        check()
        deadline = self.clock() + self.config.generation_timeout
        socket = None
        notice = None
        try:
            socket = self.connect(
                self.client.websocket_url(client_id),
                timeout=max(
                    0.001,
                    min(
                        self.config.request_timeout, 10, self.config.generation_timeout
                    ),
                ),
            )
        except Exception as exc:
            notice = f"Live node/step events unavailable; using queue polling ({type(exc).__name__})."
        try:
            with self.coordinator.lock:
                check()
                token = uuid.uuid4().hex
                scoped_graph(graph, token)
                if job:
                    job.output_token = token
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise H3Error("Generation timed out during connection setup")
                payload = self.client.post(
                    "/prompt",
                    json={"prompt": graph, "client_id": client_id},
                    timeout=min(self.client.timeout, remaining),
                ).json()
                if "prompt_id" not in payload:
                    raise H3Error(json.dumps(payload, indent=2))
                prompt_id = str(payload["prompt_id"])
                if job:
                    job.prompt_id = prompt_id
            submission = Submission(
                prompt_id,
                graph,
                self.client,
                deadline,
                self.config.generation_timeout,
                self.config.poll_seconds,
                token,
                time.time(),
                socket,
                notice,
                job,
                check,
                self.clock,
            )
            if job:
                job.submissions.append(submission)
            return PromptId(submission)
        except BaseException:
            if socket is not None:
                try:
                    socket.close()
                except Exception:
                    pass
            raise
