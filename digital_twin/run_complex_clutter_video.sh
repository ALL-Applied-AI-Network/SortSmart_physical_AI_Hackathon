#!/usr/bin/env bash
# Record one complete external-YOLO pick/place in a five-battery clutter scene.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${1:-$HERE/runs/complex_clutter_video_$(date +%Y%m%d_%H%M%S)}"

ISAAC="${ISAAC:-/isaac-sim}" \
IPC_DIR="${IPC_DIR:-$HERE/runs/yolo_ipc}" \
"$HERE/run_battery_yolo_pick_place.sh" "$RUN" \
    --complex-clutter-demo \
    --video-camera side
