#!/usr/bin/env bash
# Stop hook: the agent's "done" only counts with a green suite.
#
# The PostToolUse hook only sees Edit/Write under src/. A Bash command
# (a `cp` into data/raw/, a script that rewrites a file) never triggers it.
# This runs the whole suite, including the raw-data hash check, whenever
# the agent tries to finish. Exit 2 blocks the stop and sends stderr back.
set -uo pipefail

payload=$(cat)
active=$(printf '%s' "$payload" | python3 -c \
  'import json,sys; print(json.load(sys.stdin).get("stop_hook_active", False))')

# Already sent back once by this hook: let it stop, so it cannot loop.
[ "$active" = "True" ] && exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
if ! out=$(python -m pytest tests/ -q 2>&1); then
  echo "Not done: make test is red." >&2
  echo "$out" | grep -E "^(FAILED|ERROR)|passed|failed" | tail -15 >&2
  exit 2
fi
exit 0
