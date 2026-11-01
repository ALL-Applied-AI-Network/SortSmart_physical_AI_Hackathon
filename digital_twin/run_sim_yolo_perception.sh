#!/usr/bin/env bash
# Isaac table camera -> external YOLO worker -> calibrated battery/bowl pose.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
IPC_DIR="${IPC_DIR:-$HERE/runs/yolo_ipc}"
RUN="${1:-$HERE/runs/sim_yolo_perception_$(date +%Y%m%d_%H%M%S)}"
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
    echo "Use an empty run directory: $RUN" >&2
    exit 1
fi

"$ISAAC/python.sh" "$HERE/sim_yolo_perception_test.py" \
    --out "$RUN" \
    --ipc-dir "$IPC_DIR" \
    "$@"

REPORT="$RUN/sim_yolo_perception_report.json"
if [ ! -s "$REPORT" ]; then
    echo "Simulation did not create its report: $REPORT" >&2
    exit 2
fi
echo "Report: $REPORT"
echo "Overlay: $RUN/sim_yolo_overlay.png"

