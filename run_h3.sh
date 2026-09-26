#!/usr/bin/env bash
set -Eeuo pipefail

# MiniMax H3 fixed-default launcher.
# No environment variables or command-line options are required or read.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/h3"
COMFY_DIR="$INSTALL_DIR/ComfyUI"
MODELS_CONFIG="$INSTALL_DIR/h3_models.json"

PYTHON_BIN="python3"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

COMFY_HOST="127.0.0.1"
COMFY_PORT="8188"
COMFY_URL="http://127.0.0.1:8188"
COMFYUI_MEMORY_MODE="dynamic"
COMFY_MEMORY_ARGS=()

GRADIO_SERVER_NAME="0.0.0.0"
GRADIO_SERVER_PORT="7860"
GRADIO_OUTPUT_DIR="$INSTALL_DIR/gradio_outputs"

SERVER_ATTENTION_BACKEND="sol"
SERVER_DENSE_ATTENTION_BACKEND="comfy-kitchen"
COMFY_ATTENTION_ARGS=(--use-ck-attention)

COMFY_PID=""
GRADIO_PID=""

log() {
  printf '[h3-run] %s\n' "$*"
}

die() {
  printf '[h3-run error] %s\n' "$*" >&2
  exit 1
}

ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
    return
  fi

  command -v apt-get >/dev/null 2>&1 || die \
    "FFmpeg is missing and apt-get is unavailable; install ffmpeg and ffprobe manually"

  local -a apt_command=(apt-get)
  if (( EUID != 0 )); then
    command -v sudo >/dev/null 2>&1 || die \
      "FFmpeg is missing; install it as root or install sudo"
    apt_command=(sudo apt-get)
  fi

  log "FFmpeg is missing; installing the ffmpeg system package"
  "${apt_command[@]}" update
  "${apt_command[@]}" install -y ffmpeg

  command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg installation did not provide ffmpeg"
  command -v ffprobe >/dev/null 2>&1 || die "ffmpeg installation did not provide ffprobe"
}

environment_is_current() {
  [[ -d "$COMFY_DIR/.git" ]] || return 1

  local installed_ref expected_ref installed_kitchen installed_wsproto
  local installed_swiftvr_ref expected_swiftvr_ref
  local expected_kitchen expected_frontend expected_wsproto
  installed_ref="$(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null)" || return 1
  installed_swiftvr_ref="$(
    git -C "$INSTALL_DIR/SwiftVR" rev-parse HEAD 2>/dev/null
  )" || return 1
  read -r expected_ref expected_swiftvr_ref expected_kitchen expected_frontend expected_wsproto < <(
    "$PYTHON_BIN" - "$SCRIPT_DIR" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from h3_requirements import (
    COMFY_FRONTEND_VERSION,
    COMFY_KITCHEN_VERSION,
    COMFY_REF,
    WSPROTO_VERSION,
    SWIFTVR_REF,
)

print(
    COMFY_REF,
    SWIFTVR_REF,
    COMFY_KITCHEN_VERSION,
    COMFY_FRONTEND_VERSION,
    WSPROTO_VERSION,
)
PY
  ) || return 1
  installed_kitchen="$(
    "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("comfy-kitchen"))
except PackageNotFoundError:
    raise SystemExit(1)
PY
  )" || return 1
  installed_wsproto="$(
    "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("wsproto"))
except PackageNotFoundError:
    raise SystemExit(1)
PY
  )" || return 1

  [[ "$installed_ref" == "$expected_ref" ]] || return 1
  [[ "$installed_swiftvr_ref" == "$expected_swiftvr_ref" ]] || return 1
  [[ "$installed_kitchen" == "$expected_kitchen" ]] || return 1
  [[ "$installed_wsproto" == "$expected_wsproto" ]] || return 1
  "$PYTHON_BIN" - "$SCRIPT_DIR" <<'PY' >/dev/null || return 1
import sys

sys.path.insert(0, sys.argv[1])
from importlib.metadata import PackageNotFoundError, version
from h3_requirements import GRADIO_VERSION, comfy_frontend_package_is_ready

try:
    if version("gradio") != GRADIO_VERSION:
        raise SystemExit(1)
except PackageNotFoundError:
    raise SystemExit(1)

raise SystemExit(0 if comfy_frontend_package_is_ready() else 1)
PY
  "$PYTHON_BIN" - "$INSTALL_DIR/SwiftVR" <<'PY' >/dev/null 2>&1 || return 1
import sys

sys.path.insert(0, sys.argv[1])
import decord
import diffusers
import swiftvr
from transformers import CLIPTokenizer
PY
}

free_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -n tcp "$port" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null) || true
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  elif command -v ss >/dev/null 2>&1 && command -v grep >/dev/null 2>&1; then
    local pids
    pids=$(ss -lptn "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2) || true
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  local pids=()
  [[ -n "$GRADIO_PID" ]] && pids+=("$GRADIO_PID")
  [[ -n "$COMFY_PID" ]] && pids+=("$COMFY_PID")

  if [[ ${#pids[@]} -gt 0 ]]; then
    log "Stopping Gradio and ComfyUI"
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done

    local deadline=$((SECONDS + 3))
    for pid in "${pids[@]}"; do
      while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do
        sleep 0.1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
  fi

  free_port "$GRADIO_SERVER_PORT"
  free_port "$COMFY_PORT"

  exit "$status"
}
trap cleanup EXIT INT TERM

ensure_repo_integrity() {
  local needs_restore=0
  if [[ ! -d "$SCRIPT_DIR/h3_ui" || ! -f "$SCRIPT_DIR/h3_ui/application.py" ]]; then
    needs_restore=1
  fi
  if [[ ! -f "$SCRIPT_DIR/gradio_app.py" ]]; then
    needs_restore=1
  fi
  if [[ ! -f "$SCRIPT_DIR/custom_nodes/H3Acceleration/__init__.py" ]]; then
    needs_restore=1
  fi

  if (( needs_restore )) && [[ -d "$SCRIPT_DIR/minimax-h3/h3_ui" ]]; then
    log "Found nested repository at $SCRIPT_DIR/minimax-h3; moving files to root"
    cp -rn "$SCRIPT_DIR/minimax-h3"/* "$SCRIPT_DIR"/ 2>/dev/null || true
    if [[ -d "$SCRIPT_DIR/h3_ui" && -f "$SCRIPT_DIR/h3_ui/application.py" ]]; then
      needs_restore=0
    fi
  fi

  if (( needs_restore )); then
    log "Restoring missing repository modules (h3_ui, custom_nodes, gradio_app)"
    if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git -C "$SCRIPT_DIR" checkout HEAD -- . 2>/dev/null || true
    fi

    if [[ ! -d "$SCRIPT_DIR/h3_ui" || ! -f "$SCRIPT_DIR/h3_ui/application.py" ]]; then
      log "Fetching tracked repository modules from GitHub"
      if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$SCRIPT_DIR" fetch --depth 1 https://github.com/keboqi/minimax-h3.git HEAD 2>/dev/null || true
        git -C "$SCRIPT_DIR" checkout FETCH_HEAD -- . 2>/dev/null || true
      else
        local tmp_sync
        tmp_sync="$(mktemp -d 2>/dev/null || echo "/tmp/h3_sync_$$")"
        git clone --depth 1 https://github.com/keboqi/minimax-h3.git "$tmp_sync" 2>/dev/null || true
        if [[ -d "$tmp_sync/h3_ui" ]]; then
          cp -rn "$tmp_sync"/* "$SCRIPT_DIR"/ 2>/dev/null || true
          rm -rf "$tmp_sync" 2>/dev/null || true
        fi
      fi
    fi
  fi

  if [[ ! -f "$SCRIPT_DIR/gradio_app.py" ]]; then
    cat <<'EOF' > "$SCRIPT_DIR/gradio_app.py"
#!/usr/bin/env python3
"""Launch H3, or expose its temporary compatibility API to existing callers."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_NESTED = _ROOT / "minimax-h3"
if not (_ROOT / "h3_ui").is_dir() and (_NESTED / "h3_ui").is_dir() and str(_NESTED) not in sys.path:
    sys.path.insert(0, str(_NESTED))

try:
    from h3_ui import application
except ModuleNotFoundError:
    try:
        import subprocess
        print("[h3-run] h3_ui missing; restoring repository from GitHub...", flush=True)
        subprocess.run(
            ["git", "-C", str(_ROOT), "fetch", "--depth", "1", "https://github.com/keboqi/minimax-h3.git", "HEAD"],
            check=True, timeout=60, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(_ROOT), "checkout", "FETCH_HEAD", "--", "."],
            check=True, timeout=60, capture_output=True,
        )
    except Exception as exc:
        print(f"[h3-run] automatic Git recovery failed: {exc}", flush=True)
    from h3_ui import application

if __name__ == "__main__":
    application.main()
else:
    sys.modules[__name__] = application
EOF
    chmod +x "$SCRIPT_DIR/gradio_app.py" 2>/dev/null || true
  fi
}

ensure_ffmpeg
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python3 is required"
[[ -f "$SCRIPT_DIR/setup_h3.py" ]] || die "Missing setup_h3.py"
ensure_repo_integrity

if [[ ! -f "$COMFY_DIR/main.py" || ! -f "$MODELS_CONFIG" ]]; then
  log "Installation or models are missing; running automatic setup"
  "$PYTHON_BIN" "$SCRIPT_DIR/setup_h3.py" --install-dir "$INSTALL_DIR"
elif ! environment_is_current; then
  log "ComfyUI/Gradio environment or frontend assets are stale; refreshing the environment"
  "$PYTHON_BIN" "$SCRIPT_DIR/setup_h3.py" --install-dir "$INSTALL_DIR"
else
  log "ComfyUI environment is current; checking Hugging Face model versions"
  "$PYTHON_BIN" "$SCRIPT_DIR/setup_h3.py" \
    --install-dir "$INSTALL_DIR" \
    --skip-env
fi

if ! "$PYTHON_BIN" -c 'import websocket' >/dev/null 2>&1; then
  log "Installing live generation progress client"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON_BIN" --upgrade "websocket-client>=1.8"
  else
    "$PYTHON_BIN" -m pip install --upgrade "websocket-client>=1.8"
  fi
fi

[[ -f "$COMFY_DIR/main.py" ]] || die "ComfyUI installation failed"
[[ -f "$MODELS_CONFIG" ]] || die "Model setup failed"
log "Dense/fallback attention: Comfy Kitchen"

free_port "$COMFY_PORT"
free_port "$GRADIO_SERVER_PORT"

log "Starting ComfyUI at $COMFY_URL"
log "Memory profile: $COMFYUI_MEMORY_MODE"
(
  cd "$COMFY_DIR"
  exec "$PYTHON_BIN" -u main.py \
    --listen "$COMFY_HOST" \
    --port "$COMFY_PORT" \
    --fast fp16_accumulation \
    "${COMFY_MEMORY_ARGS[@]}" \
    "${COMFY_ATTENTION_ARGS[@]}" \
    --enable-cors-header "*"
) &
COMFY_PID=$!

log "Waiting for ComfyUI"
"$PYTHON_BIN" - "$COMFY_URL" "$COMFY_PID" <<'PY'
import os
import sys
import time
import urllib.request

url = sys.argv[1].rstrip("/") + "/system_stats"
pid = int(sys.argv[2])
deadline = time.monotonic() + 1200
last_error = None

while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"ComfyUI exited during startup: {exc}")

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print("[h3-run] ComfyUI API is ready", flush=True)
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc

    time.sleep(2)

raise SystemExit(f"Timed out waiting for ComfyUI: {last_error}")
PY

export COMFY_DIR="$COMFY_DIR"
export COMFY_URL="$COMFY_URL"
export MODELS_CONFIG="$MODELS_CONFIG"
export GRADIO_OUTPUT_DIR="$GRADIO_OUTPUT_DIR"
export SERVER_ATTENTION_BACKEND="$SERVER_ATTENTION_BACKEND"
export SERVER_DENSE_ATTENTION_BACKEND="$SERVER_DENSE_ATTENTION_BACKEND"
export SERVER_MEMORY_PROFILE="$COMFYUI_MEMORY_MODE"
export GRADIO_SERVER_NAME="$GRADIO_SERVER_NAME"
export GRADIO_SERVER_PORT="$GRADIO_SERVER_PORT"
export AUTO_START_COMFYUI="0"

log "Starting Gradio"
ensure_repo_integrity
if [[ -f "$SCRIPT_DIR/gradio_app.py" ]]; then
  PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -u "$SCRIPT_DIR/gradio_app.py" &
else
  PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -u -m h3_ui.application &
fi
GRADIO_PID=$!

log "Waiting for the main Gradio UI"
"$PYTHON_BIN" - "http://127.0.0.1:$GRADIO_SERVER_PORT/" "$GRADIO_PID" <<'PY'
import os
import sys
import time
import urllib.request

url = sys.argv[1]
pid = int(sys.argv[2])
deadline = time.monotonic() + 600
last_error = None
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"Gradio exited during startup: {exc}")
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print("[h3-run] Main Gradio UI is ready", flush=True)
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(2)
raise SystemExit(f"Timed out waiting for the main Gradio UI: {last_error}")
PY

log "MiniMax H3 is ready at http://127.0.0.1:$GRADIO_SERVER_PORT"
log "Checking the /comfyui proxy"
if ! "$PYTHON_BIN" - "http://127.0.0.1:$GRADIO_SERVER_PORT/comfyui/" "$GRADIO_PID" "$SCRIPT_DIR" <<'PY'
import os
import sys
import time

sys.path.insert(0, sys.argv[3])
from h3_requirements import probe_comfy_frontend, probe_comfy_workflow

url = sys.argv[1]
pid = int(sys.argv[2])
deadline = time.monotonic() + 30
last_error = None
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"Gradio exited during startup: {exc}")
    try:
        count = probe_comfy_frontend(url, timeout=2)
        workflow_size = probe_comfy_workflow(url, timeout=2)
        print(
            f"[h3-run] /comfyui proxy served {count} frontend assets and "
            f"a {workflow_size}-byte nested workflow",
            flush=True,
        )
        raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(2)
raise SystemExit(f"Timed out waiting for /comfyui proxy: {last_error}")
PY
then
  log "WARNING: Main UI is running, but /comfyui asset validation failed"
fi

log "Press Ctrl+C to stop"

set +e
wait -n "$COMFY_PID" "$GRADIO_PID"
status=$?
set -e

if ! kill -0 "$COMFY_PID" 2>/dev/null; then
  die "ComfyUI exited unexpectedly"
fi

if ! kill -0 "$GRADIO_PID" 2>/dev/null; then
  die "Gradio exited unexpectedly"
fi

exit "$status"
