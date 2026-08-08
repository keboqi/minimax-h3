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
SERVER_DENSE_ATTENTION_BACKEND="pytorch"
COMFY_ATTENTION_ARGS=()

COMFY_PID=""
GRADIO_PID=""

log() {
  printf '[h3-run] %s\n' "$*"
}

die() {
  printf '[h3-run error] %s\n' "$*" >&2
  exit 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if [[ -n "$GRADIO_PID" ]] && kill -0 "$GRADIO_PID" 2>/dev/null; then
    log "Stopping Gradio"
    kill "$GRADIO_PID" 2>/dev/null || true
  fi

  if [[ -n "$COMFY_PID" ]] && kill -0 "$COMFY_PID" 2>/dev/null; then
    log "Stopping ComfyUI"
    kill "$COMFY_PID" 2>/dev/null || true
  fi

  [[ -n "$GRADIO_PID" ]] && wait "$GRADIO_PID" 2>/dev/null || true
  [[ -n "$COMFY_PID" ]] && wait "$COMFY_PID" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python3 is required"
[[ -f "$SCRIPT_DIR/setup_h3.py" ]] || die "Missing setup_h3.py"
[[ -f "$SCRIPT_DIR/gradio_app.py" ]] || die "Missing gradio_app.py"

if [[ ! -f "$COMFY_DIR/main.py" || ! -f "$MODELS_CONFIG" ]]; then
  log "Installation or models are missing; running automatic setup"
  "$PYTHON_BIN" "$SCRIPT_DIR/setup_h3.py" --install-dir "$INSTALL_DIR"
else
  log "Checking Hugging Face model versions"
  "$PYTHON_BIN" "$SCRIPT_DIR/setup_h3.py"     --install-dir "$INSTALL_DIR"     --skip-env
fi

[[ -f "$COMFY_DIR/main.py" ]] || die "ComfyUI installation failed"
[[ -f "$MODELS_CONFIG" ]] || die "Model setup failed"
[[ -f "$SCRIPT_DIR/h3_attention.py" ]] || die "Missing h3_attention.py"

log "Probing native SageAttention dense backend"
if "$PYTHON_BIN" "$SCRIPT_DIR/h3_attention.py" --probe; then
  SERVER_DENSE_ATTENTION_BACKEND="sage"
  COMFY_ATTENTION_ARGS=(--use-sage-attention)
  log "Dense/fallback attention: SageAttention"
else
  SERVER_DENSE_ATTENTION_BACKEND="pytorch"
  COMFY_ATTENTION_ARGS=()
  log "Dense/fallback attention: PyTorch"
fi

log "Starting ComfyUI at $COMFY_URL"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  if [[ "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] && (( GPU_MEMORY_MIB >= 65536 )); then
    COMFYUI_MEMORY_MODE="gpu-only"
    COMFY_MEMORY_ARGS=(--gpu-only)
  fi
fi
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
                print("[h3-run] ComfyUI is ready", flush=True)
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

log "MiniMax H3 is ready at http://127.0.0.1:$GRADIO_SERVER_PORT"
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
