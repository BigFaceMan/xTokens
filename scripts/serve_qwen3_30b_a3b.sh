#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

MODEL_PATH=${XTOKENS_HF_MODEL:-/model}
SERVED_MODEL_NAME=${XTOKENS_SERVED_MODEL_NAME:-qwen3-30b-a3b}
HOST=${XTOKENS_HOST:-127.0.0.1}
PORT=${XTOKENS_PORT:-8991}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
PID_FILE=${XTOKENS_PID_FILE:-"$PROJECT_ROOT/.run/serve_qwen3_30b_a3b.pid"}

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

stop_server() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "xTokens server is not running (PID file not found: $PID_FILE)"
    return
  fi

  local pid
  pid=$(<"$PID_FILE")
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid PID file: $PID_FILE" >&2
    exit 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Removed stale PID file: $PID_FILE"
    return
  fi

  local command
  command=$(ps -p "$pid" -o command=)
  if [[ "$command" != *"x_tokens"* ]]; then
    echo "Refusing to stop PID $pid: it is not an xTokens process" >&2
    exit 1
  fi

  kill -TERM "$pid"
  for _ in {1..30}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Stopped xTokens server (PID $pid)"
      return
    fi
    sleep 1
  done

  echo "Timed out waiting for xTokens server (PID $pid) to stop" >&2
  exit 1
}

if [[ ${1:-} == "stop" ]]; then
  if [[ $# -ne 1 ]]; then
    echo "Usage: $0 stop" >&2
    exit 1
  fi
  stop_server
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Project virtual environment is unavailable: $PYTHON_BIN" >&2
  echo "Create it with: uv sync --extra hf" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import torch, transformers' >/dev/null 2>&1; then
  echo "HF dependencies are unavailable in $PROJECT_ROOT/.venv" >&2
  echo "Install them with: uv sync --extra hf" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Qwen model directory does not exist: $MODEL_PATH" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$(dirname -- "$PID_FILE")"
if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(<"$PID_FILE")
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "xTokens server is already running (PID $existing_pid)" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi
server_pid=""
cleanup() {
  rm -f "$PID_FILE"
}
stop_child() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  exit 0
}
trap cleanup EXIT
trap stop_child INT TERM

env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m x_tokens \
    --engine inproc \
    --model "$SERVED_MODEL_NAME" \
    --hf-model "$MODEL_PATH" \
    --hf-local-files-only \
    --hf-dtype bfloat16 \
    --hf-max-num-seqs "${XTOKENS_MAX_NUM_SEQS:-4}" \
    --host "$HOST" \
    --port "$PORT" \
    "$@" &
server_pid=$!
printf '%s\n' "$server_pid" > "$PID_FILE"
wait "$server_pid"
