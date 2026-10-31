#!/usr/bin/env bash
set -Eeuo pipefail

# MiniMax H3 fixed-default launcher.
#
#   bash run_h3.sh [--prepare GROUPS | --no-prepare]
#
# Default models download in the background from the start, alongside setup,
# ComfyUI and Gradio.
# GROUPS is a comma-separated list of h3 (MiniMax H3), drama3 (Drama 3) and
# qwen_image (Qwen Image 2.1), or all / none; H3_PREPARE_MODELS sets the same
# list. The default is h3. Models left out still download on first use.

usage() {
  sed -n '6,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

PREPARE_MODELS="${H3_PREPARE_MODELS-h3}"
while (( $# )); do
  case "$1" in
    --prepare) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; PREPARE_MODELS="$2"; shift 2 ;;
    --prepare=*) PREPARE_MODELS="${1#*=}"; shift ;;
    --no-prepare) PREPARE_MODELS="none"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf '[h3-run error] Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
# Reject a typo now rather than in the background after setup.
for group in ${PREPARE_MODELS//,/ }; do
  case "${group,,}" in
    h3|minimax_h3|minimax-h3|drama3|drama|qwen_image|qwen-image|qwen|all|none|off|0|false|no) ;;
    *) printf '[h3-run error] Unknown model group: %s (use h3, drama3, qwen_image, all or none)\n' "$group" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/h3"
COMFY_DIR="$INSTALL_DIR/ComfyUI"
MODELS_CONFIG="$INSTALL_DIR/h3_models.json"

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
elif [[ -x "/content/venv_h3/bin/python3" ]]; then
  PYTHON_BIN="/content/venv_h3/bin/python3"
  export PATH="/content/venv_h3/bin:$PATH"
  export VIRTUAL_ENV="/content/venv_h3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  printf '[h3-run error] python3 is required\n' >&2
  exit 1
fi
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
PREPARE_PIDS=()

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

# Report up front whether Drama 3's SGLang server can sleep (CUDA checkpoint)
# or will stop/restart. The host owns the driver, so this only informs.
report_sglang_sleep() {
  local mode="${H3_DRAMA3_SGLANG_SLEEP:-auto}"
  if [[ "${mode,,}" == "off" ]]; then
    log "SGLang sleep: off (H3_DRAMA3_SGLANG_SLEEP=off); Drama 3 SGLang will stop/restart"
    return
  fi
  if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    log "SGLang sleep: unavailable (requires Linux x86_64); Drama 3 SGLang will stop/restart"
    return
  fi
  local versions
  versions="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)" || versions=""
  if [[ -z "$versions" ]]; then
    log "SGLang sleep: unavailable (nvidia-smi not found); Drama 3 SGLang will stop/restart"
    return
  fi
  local version transport="CUDA IPC"
  while read -r version; do
    if (( ${version%%.*} < 580 )); then
      log "SGLang sleep: unavailable (NVIDIA driver $version; 580+ needed for CUDA checkpointing); Drama 3 SGLang will stop/restart"
      return
    fi
    # 580-609 cannot checkpoint CUDA IPC, so Omni's stages use host SHM instead.
    if (( ${version%%.*} < 610 )); then
      transport="host SHM"
    fi
  done <<< "$versions"
  log "SGLang sleep: available (NVIDIA driver ${versions//$'\n'/, }; Omni stages use $transport)"
}

environment_is_current() {
  [[ -d "$COMFY_DIR/.git" ]] || return 1

  local installed_ref expected_ref installed_kitchen installed_wsproto
  local expected_kitchen expected_frontend expected_wsproto
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
from importlib.metadata import PackageNotFoundError, version
from h3_requirements import GRADIO_VERSION, comfy_frontend_package_is_ready

try:
    if version("gradio") != GRADIO_VERSION:
        raise SystemExit(1)
except PackageNotFoundError:
    raise SystemExit(1)

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
  local prepare_pid
  for prepare_pid in "${PREPARE_PIDS[@]}"; do
    if kill -0 "$prepare_pid" 2>/dev/null; then
      # Stop its uv/pip children too; an interrupted download resumes next start.
      pkill -TERM -P "$prepare_pid" 2>/dev/null || true
      pids+=("$prepare_pid")
    fi
  done

  if [[ ${#pids[@]} -gt 0 ]]; then
    log "Stopping Gradio, ComfyUI and model preparation"
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
  local -a required_paths=(
    gradio_app.py
    h3_runtime.py
    h3_sources.py
    h3_ui/__init__.py
    h3_ui/application.py
    h3_app/__init__.py
    custom_nodes/H3Acceleration/__init__.py
  )
  local -a missing_paths=()
  local path
  for path in "${required_paths[@]}"; do
    [[ -f "$SCRIPT_DIR/$path" ]] || missing_paths+=("$path")
  done

  if (( ${#missing_paths[@]} )) && [[ -d "$SCRIPT_DIR/minimax-h3/h3_ui" ]]; then
    log "Found nested repository at $SCRIPT_DIR/minimax-h3; moving files to root"
    cp -rn "$SCRIPT_DIR/minimax-h3"/* "$SCRIPT_DIR"/ 2>/dev/null || true
  fi

  missing_paths=()
  for path in "${required_paths[@]}"; do
    [[ -f "$SCRIPT_DIR/$path" ]] || missing_paths+=("$path")
  done
  if (( ${#missing_paths[@]} )); then
    log "Restoring missing repository modules (h3_ui, custom_nodes, gradio_app)"
    if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git -C "$SCRIPT_DIR" checkout HEAD -- "${missing_paths[@]}" 2>/dev/null || true
    fi

    missing_paths=()
    for path in "${required_paths[@]}"; do
      [[ -f "$SCRIPT_DIR/$path" ]] || missing_paths+=("$path")
    done
    if (( ${#missing_paths[@]} )); then
      log "Fetching tracked repository modules from GitHub"
      if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if git -C "$SCRIPT_DIR" fetch --depth 1 https://github.com/keboqi/minimax-h3.git HEAD 2>/dev/null; then
          git -C "$SCRIPT_DIR" checkout FETCH_HEAD -- "${missing_paths[@]}" 2>/dev/null || true
        fi
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
    if (_ROOT / "h3_ui" / "application.py").is_file():
        raise
    try:
        import subprocess
        print("[h3-run] h3_ui missing; restoring repository from GitHub...", flush=True)
        subprocess.run(
            ["git", "-C", str(_ROOT), "fetch", "--depth", "1", "https://github.com/keboqi/minimax-h3.git", "HEAD"],
            check=True, timeout=60, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(_ROOT), "checkout", "FETCH_HEAD", "--", "h3_ui/application.py"],
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

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python3 is required"
[[ -f "$SCRIPT_DIR/setup_h3.py" ]] || die "Missing setup_h3.py"
ensure_repo_integrity
[[ -f "$SCRIPT_DIR/h3_runtime.py" ]] || die "Missing h3_runtime.py; update the repository"
PYTHON_BIN="$("$PYTHON_BIN" "$SCRIPT_DIR/h3_runtime.py" --install-dir "$INSTALL_DIR")"
# Keep setup, pip subprocesses, ComfyUI and Gradio on the same interpreter.
if [[ "$PYTHON_BIN" == */.venv/bin/python3 ]]; then
  export VIRTUAL_ENV="$(dirname -- "$(dirname -- "$PYTHON_BIN")")"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
fi
log "Runtime Python: $PYTHON_BIN"
ensure_ffmpeg
report_sglang_sleep

# start_prepare LABEL FLAG PYTHON...: run h3_prepare.py in the background (FLAG: --files-only,
# --skip-files or empty), yielding CPU and disk to setup and the starting servers; downloads
# still run at full speed when the machine is idle.
start_prepare() {
  local label="$1" flag="$2"
  shift 2
  local priority=() flags=()
  [[ -n "$flag" ]] && flags+=("$flag")
  command -v nice >/dev/null 2>&1 && priority+=(nice -n 10)
  command -v ionice >/dev/null 2>&1 && ionice -c 3 true 2>/dev/null && priority+=(ionice -c 3)
  log "Background model preparation: $label"
  "${priority[@]}" "$@" -u "$SCRIPT_DIR/h3_prepare.py" --install-dir "$INSTALL_DIR" \
    --groups "$PREPARE_MODELS" "${flags[@]}" &
  PREPARE_PIDS+=("$!")
}

# The model files need only huggingface_hub, so they start now, before setup, from an
# isolated uv environment on the runtime interpreter (setup is still installing into the
# app's). Without uv they start after setup, with the rest.
PREPARE_FILES_EARLY=0
case "${PREPARE_MODELS,,}" in
  ""|0|none|off|false|no) ;;
  *)
    if command -v uv >/dev/null 2>&1; then
      start_prepare "$PREPARE_MODELS model files, alongside setup" --files-only \
        uv run --quiet --isolated --no-project --python "$PYTHON_BIN" \
        --with "huggingface-hub>=1.5,<2" --with "hf-xet>=1.1" python
      PREPARE_FILES_EARLY=1
    fi
    ;;
esac

# Every 10 s until the downloads end and ComfyUI answers: network, disk writes, dirty pages,
# CPU and stall pressure, to show what startup waits on. H3_STARTUP_STATS=0 turns it off.
if [[ "${H3_STARTUP_STATS:-1}" != "0" && -f "$SCRIPT_DIR/h3_startup_stats.py" ]]; then
  stats_watch=()
  for prepare_pid in "${PREPARE_PIDS[@]}"; do stats_watch+=(--watch-pid "$prepare_pid"); done
  "$PYTHON_BIN" "$SCRIPT_DIR/h3_startup_stats.py" --until-url "$COMFY_URL/system_stats" "${stats_watch[@]}" &
fi

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

# The rest of the selected defaults (environments and checkpoints, or everything when the
# files could not start early) prepare while the servers start; failures only warn.
case "${PREPARE_MODELS,,}" in
  ""|0|none|off|false|no)
    log "Background model preparation: off"
    ;;
  *)
    if (( PREPARE_FILES_EARLY )); then
      if [[ ",${PREPARE_MODELS,,}," =~ ,(drama3|drama|all), ]]; then
        start_prepare "$PREPARE_MODELS environments and checkpoints" --skip-files "$PYTHON_BIN"
      fi
    else
      start_prepare "$PREPARE_MODELS" "" "$PYTHON_BIN"
    fi
    ;;
esac

free_port "$COMFY_PORT"
free_port "$GRADIO_SERVER_PORT"

log "Setup finished after ${SECONDS}s; starting ComfyUI at $COMFY_URL"
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

# Gradio starts beside ComfyUI rather than after it: the UI does not need the API to
# open (its status refreshes on page load), and jobs wait for ComfyUI themselves.
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

log "MiniMax H3 is ready at http://127.0.0.1:$GRADIO_SERVER_PORT (${SECONDS}s after launch)"

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

log "ComfyUI ready after ${SECONDS}s"

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
