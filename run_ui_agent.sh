#!/usr/bin/env bash
# Serve the agent-managed-memory UI on http://127.0.0.1:7875.
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

PORT="${PORT:-7875}"
exec .venv/bin/python -m uvicorn server_agent:app --host 127.0.0.1 --port "$PORT"
