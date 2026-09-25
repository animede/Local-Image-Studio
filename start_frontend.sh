#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/frontend"
exec ../.venv/bin/python -m http.server "${QWEN_FRONTEND_PORT:-5173}" --bind "${QWEN_FRONTEND_HOST:-127.0.0.1}"

