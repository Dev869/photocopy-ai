#!/bin/bash
# Wait for the rejects JPEG export to finish (no new files for 10 min), then train the culler.
set -e
cd "$(dirname "$0")"
EXPORT="$HOME/Pictures/agent-export-rejected-only"
prev=-1; stable=0
while :; do
  n=$(find "$EXPORT" -type f \( -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ] && [ "$n" -eq "$prev" ]; then stable=$((stable+1)); else stable=0; fi
  [ "$stable" -ge 10 ] && break
  prev=$n
  sleep 60
done
{
  echo "=== rejects export stable at $n files, $(date) ==="
  .venv/bin/python cull_train.py "$EXPORT"
  echo "=== done $(date) ==="
} > cull_train.log 2>&1
