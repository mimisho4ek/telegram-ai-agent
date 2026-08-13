#!/usr/bin/env bash
# Bot MCP Server launcher
# Reads TELEGRAM_BOT_TOKEN from the application environment file.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
APP_ROOT="${APP_ROOT:-${PROJECT_DIR:-$SCRIPT_DIR/../..}}"
ENV_FILE="${ENV_FILE:-$APP_ROOT/.env}"

if [ -f "$ENV_FILE" ]; then
  eval "$(/usr/bin/python3 - "$ENV_FILE" <<'PY'
import json
import re
import shlex
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    key, separator, raw = line.partition("=")
    if separator and key.strip() == "TELEGRAM_BOT_TOKEN":
        raw = raw.strip()
        if raw.startswith(('"', "'")):
            quote = raw[0]
            escaped = False
            for index, char in enumerate(raw[1:], start=1):
                if char == quote and not escaped:
                    suffix = raw[index + 1:].lstrip()
                    if not suffix or suffix.startswith("#"):
                        raw = raw[:index + 1]
                    break
                escaped = char == "\\" and not escaped
                if char != "\\":
                    escaped = False
        else:
            raw = re.split(r"\s+#", raw, maxsplit=1)[0].rstrip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] == "'" else raw
        if value:
            print(f"export TELEGRAM_BOT_TOKEN={shlex.quote(str(value))}")
        break
PY
)"
fi

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
  echo "Error: TELEGRAM_BOT_TOKEN not set in $ENV_FILE" >&2
  exit 1
fi

export BOT_TOKEN="$TELEGRAM_BOT_TOKEN"
exec "$APP_ROOT/.venv/bin/python" "$SCRIPT_DIR/server.py"
