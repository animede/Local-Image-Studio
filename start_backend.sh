#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/start_options.sh
if [[ "${QWEN_OPTIONS_CONFIGURED:-0}" != "1" ]]; then
  if qwen_configure_startup "$@"; then
    :
  else
    result=$?
    [[ $result -eq 10 ]] && exit 0
    exit "$result"
  fi
fi
if [[ ! -x .venv/bin/python ]]; then
  echo ".venv がありません。先に python3 -m venv .venv を実行してください。" >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn backend.app.main:app --host "${QWEN_API_HOST:-127.0.0.1}" --port "${QWEN_API_PORT:-8000}"
