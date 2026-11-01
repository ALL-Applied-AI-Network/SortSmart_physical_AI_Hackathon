#!/usr/bin/env bash
# Sort all five complex-scene batteries and record one continuous side video.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${1:-$HERE/runs/complex_clutter_multi_$(date +%Y%m%d_%H%M%S)}"

ISAAC="${ISAAC:-/isaac-sim}" \
IPC_DIR="${IPC_DIR:-$HERE/runs/yolo_ipc}" \
"$HERE/run_battery_yolo_multi_pick_place.sh" "$RUN" \
    --max-batteries 5 \
    --video-camera side
