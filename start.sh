#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/start_options.sh
if qwen_configure_startup "$@"; then
  :
else
  result=$?
  [[ $result -eq 10 ]] && exit 0
  exit "$result"
fi
./start_backend.sh &
backend_pid=$!
./start_frontend.sh &
frontend_pid=$!
trap 'kill "$backend_pid" "$frontend_pid" 2>/dev/null || true' EXIT INT TERM
echo "Frontend: http://127.0.0.1:${QWEN_FRONTEND_PORT:-5173}"
echo "API docs: http://127.0.0.1:${QWEN_API_PORT:-8000}/docs"
wait "$backend_pid" "$frontend_pid"
