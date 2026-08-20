#!/usr/bin/env bash
set -Eeuo pipefail

# MiniMax H3 fixed-default launcher.
# No environment variables or command-line options are required or read.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/h3"
COMFY_DIR="$INSTALL_DIR/ComfyUI"
MODELS_CONFIG="$INSTALL_DIR/h3_models.json"

PYTHON_BIN="python3"

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
  local expected_ref expected_kitchen expected_frontend expected_wsproto
  installed_ref="$(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null)" || return 1
  read -r expected_ref expected_kitchen expected_frontend expected_wsproto < <(
    "$PYTHON_BIN" - "$SCRIPT_DIR" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from h3_requirements import (
    COMFY_FRONTEND_VERSION,
    COMFY_KITCHEN_VERSION,
    COMFY_REF,
    WSPROTO_VERSION,
)

print(
    COMFY_REF,
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
  [[ "$installed_kitchen" == "$expected_kitchen" ]] || return 1
  [[ "$installed_wsproto" == "$expected_wsproto" ]] || return 1
  "$PYTHON_BIN" - "$SCRIPT_DIR" <<'PY' >/dev/null || return 1
import sys

sys.path.insert(0, sys.argv[1])
from h3_requirements import comfy_frontend_package_is_ready

raise SystemExit(0 if comfy_frontend_package_is_ready() else 1)
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

ensure_ffmpeg
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python3 is required"
[[ -f "$SCRIPT_DIR/setup_h3.py" ]] || die "Missing setup_h3.py"
[[ -f "$SCRIPT_DIR/gradio_app.py" ]] || die "Missing gradio_app.py"

if [[ ! -f "$COMFY_DIR/main.py" || ! -f "$MODELS_CONFIG" ]]; then
  log "Installation or models are missing; running automatic setup"
  "$PYTHON_BIN" "$SCRIPT_DIR/setup_h3.py" --install-dir "$INSTALL_DIR"
elif ! environment_is_current; then
  log "ComfyUI environment or frontend assets are stale; refreshing the environment"
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
"$PYTHON_BIN" -u "$SCRIPT_DIR/gradio_app.py" &
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
