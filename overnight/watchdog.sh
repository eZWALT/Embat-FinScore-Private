#!/usr/bin/env bash
# Restart search_16h.py until the saved wall-clock deadline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p overnight/runs
LOG="$ROOT/overnight/runs/search_16h.log"
WDLOG="$ROOT/overnight/runs/watchdog.log"
DEADLINE="$ROOT/overnight/runs/deadline.txt"

echo "$(date -Is) watchdog start pid=$$" >>"$WDLOG"
while true; do
  if [[ -f "$DEADLINE" ]]; then
    dl="${DEADLINE}"
    wall="$(tr -d '[:space:]' <"$dl")"
    now="$(date +%s)"
    if awk -v w="$wall" -v n="$now" 'BEGIN { exit !(w <= n) }'; then
      echo "$(date -Is) 16h window ended" >>"$WDLOG"
      exit 0
    fi
  fi
  if ! pgrep -f "python3 -u overnight/search_16h.py" >/dev/null 2>&1; then
    echo "$(date -Is) start/restart search_16h.py" >>"$WDLOG"
    nohup python3 -u overnight/search_16h.py --hours 16 >>"$LOG" 2>&1 &
  fi
  sleep 60
done
