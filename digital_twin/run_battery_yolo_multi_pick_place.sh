#!/usr/bin/env bash
# Repeated external-YOLO pick/place run with one continuous MP4.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
IPC_DIR="${IPC_DIR:-$HERE/runs/yolo_ipc}"
RUN="${1:-$HERE/runs/battery_yolo_multi_pick_$(date +%Y%m%d_%H%M%S)}"
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
"$ISAAC/python.sh" "$HERE/battery_yolo_multi_pick_place.py" \
    --out "$RUN" \
    --ipc-dir "$IPC_DIR" \
    "$@"
SIM_STATUS=$?
set -e

REPORT="$RUN/battery_yolo_multi_pick_place_report.json"
if [ ! -s "$REPORT" ]; then
    echo "Multi-pick report was not created: $REPORT" >&2
    exit 2
fi

if compgen -G "$RUN/frames/*.png" >/dev/null; then
    ENCODER_PY="${ENCODER_PY:-$HERE/.venv/bin/python}"
    VIDEO="$RUN/battery_yolo_multi_pick_place.mp4"
    if [ -x "$ENCODER_PY" ] && \
      "$ENCODER_PY" -c 'import cv2, imageio_ffmpeg' >/dev/null 2>&1; then
        "$ENCODER_PY" "$HERE/encode_video.py" \
            --frames "$RUN/frames" \
            --out "$VIDEO" \
            --fps 30
        echo "Video: $VIDEO"
    elif command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -loglevel warning -framerate 30 \
            -i "$RUN/frames/f%05d.png" \
            -c:v libx264 -pix_fmt yuv420p \
            "$VIDEO"
        echo "Video: $VIDEO"
    else
        echo "Video encoder unavailable; frames: $RUN/frames" >&2
    fi
fi

echo "Report: $REPORT"
exit "$SIM_STATUS"

