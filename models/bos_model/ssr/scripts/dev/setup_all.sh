#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/setup_bos_model.py"
USERS_FILE="$SCRIPT_DIR/users.txt"

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

# Read DATA_PATH from the first line
if [[ ! -s "$USERS_FILE" ]]; then
  echo "ERROR: $USERS_FILE is missing or empty."
  exit 1
fi
DATA_PATH="$(head -n 1 "$USERS_FILE")"
if [[ -z "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH (first line of $USERS_FILE) is empty."
  exit 1
fi
echo ">>> DATA_PATH: $DATA_PATH"

# Iterate absolute user paths from line 2 onwards
tail -n +2 "$USERS_FILE" | while IFS= read -r WORK_HOME || [[ -n "${WORK_HOME:-}" ]]; do
  [[ -z "$WORK_HOME" ]] && continue

  ts="$(date +%Y%m%d-%H%M%S)"
  log="$LOGDIR/$(basename "$WORK_HOME")-$ts.log"

  echo ">>> Running setup for $WORK_HOME (log: $log)"
  python3 "$PY" \
    --work-home "$WORK_HOME" \
    --data-path "$DATA_PATH" \
    --no-prompt \
    2>&1 | tee "$log"
done
