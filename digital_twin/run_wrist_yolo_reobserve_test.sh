#!/usr/bin/env bash
# Fixed safe wrist-camera scan; detection only, never grasps.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
IPC_DIR="${IPC_DIR:-$HERE/runs/yolo_ipc}"
RUN="${1:-$HERE/runs/wrist_yolo_reobserve_$(date +%Y%m%d_%H%M%S)}"
if [ "$#" -gt 0 ]; then
    shift
fi

if [ ! -x "$ISAAC/python.sh" ]; then
    echo "Isaac launcher missing: $ISAAC/python.sh" >&2
    exit 1
fi
mkdir -p "$RUN" "$IPC_DIR"
RUN="$(cd "$RUN" && pwd)"
IPC_DIR="$(cd "$IPC_DIR" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Use an empty output directory: $RUN" >&2
    exit 1
fi

set +e
"$ISAAC/python.sh" "$HERE/wrist_yolo_reobserve_test.py" \
    --out "$RUN" \
    --ipc-dir "$IPC_DIR" \
    "$@"
STATUS=$?
set -e

REPORT="$RUN/wrist_yolo_reobserve_report.json"
if [ ! -s "$REPORT" ]; then
    echo "Wrist re-observation report was not created: $REPORT" >&2
    exit 2
fi

echo "Report: $REPORT"
echo "Images: $RUN/wrist_rgb_scan_*.png"
exit "$STATUS"
