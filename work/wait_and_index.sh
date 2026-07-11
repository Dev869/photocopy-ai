#!/bin/bash
# Wait for LR bulk export to finish (no new .xmp for 15 min), then build the full index.
set -e
cd "$(dirname "$0")"
EXPORT="$HOME/Pictures/agent-export"
LOG=full_index.log

prev=-1; stable=0
while :; do
  n=$(find "$EXPORT" -name "*.xmp" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ] && [ "$n" -eq "$prev" ]; then stable=$((stable+1)); else stable=0; fi
  [ "$stable" -ge 15 ] && break
  prev=$n
  sleep 60
done

{
  echo "=== export stable at $n xmp files, $(date) ==="
  .venv/bin/python load_export.py "$EXPORT"
  .venv/bin/python proxies.py
  .venv/bin/python embed.py
  .venv/bin/python predict.py --eval
  echo "=== done $(date) ==="
} > "$LOG" 2>&1
