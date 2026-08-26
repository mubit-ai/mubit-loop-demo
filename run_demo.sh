#!/usr/bin/env bash
# Run the feedback-loop demo. Reads .env if present; environment
# variables that are already set win.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

: "${MUBIT_ENDPOINT:?set MUBIT_ENDPOINT (e.g. http://127.0.0.1:3970)}"
: "${MUBIT_API_KEY:?set MUBIT_API_KEY}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

exec "$PY" demo.py "$@"
