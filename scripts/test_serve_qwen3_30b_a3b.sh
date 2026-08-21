#!/usr/bin/env bash

set -euo pipefail

HOST=${XTOKENS_HOST:-127.0.0.1}
PORT=${XTOKENS_PORT:-8991}
MODEL=${XTOKENS_SERVED_MODEL_NAME:-qwen3-30b-a3b}
PROMPT=${XTOKENS_TEST_PROMPT:-"Explain continuous batching in one sentence."}
BASE_URL="http://${HOST}:${PORT}"
PYTHON_BIN=${PYTHON_BIN:-python3}

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python is required to encode the JSON request body" >&2
  exit 1
fi

echo "Checking ${BASE_URL}/ready ..."
curl --fail --silent --show-error "${BASE_URL}/ready"
printf '\n\n'

echo "Sending non-streaming completion request ..."
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d "$(printf '%s' "$PROMPT" | "$PYTHON_BIN" -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "prompt": sys.stdin.read()}))' "$MODEL")" \
  "${BASE_URL}/v1/completions"
printf '\n\n'

echo "Sending streaming chat request ..."
curl --fail --silent --show-error --no-buffer \
  -H 'Content-Type: application/json' \
  -d "$(printf '%s' "$PROMPT" | "$PYTHON_BIN" -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": sys.stdin.read()}], "stream": True}))' "$MODEL")" \
  "${BASE_URL}/v1/chat/completions"
printf '\n'
