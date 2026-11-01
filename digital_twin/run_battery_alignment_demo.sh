#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"

if [ ! -x "$ISAAC/python.sh" ]; then
    echo "Isaac launcher missing: $ISAAC/python.sh" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    RUN="$1"
    shift
else
    RUN="$HERE/runs/battery_alignment_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"

if [ -n "$(ls -A "$RUN")" ]; then
    echo "Use an empty output directory: $RUN" >&2
    exit 1
fi

# Run simulation, but preserve frames even if placement validation fails.
set +e

"$ISAAC/python.sh" \
    "$HERE/battery_alignment_demo.py" \
    --out "$RUN" \
    "$@"

SIM_STATUS=$?

set -e

# Select a Python environment capable of encoding MP4.
ENCODER=""

if [ -x "$HERE/.venv/bin/python" ]; then
    ENCODER="$HERE/.venv/bin/python"
elif "$ISAAC/python.sh" \
    -c 'import cv2, imageio_ffmpeg' \
    >/dev/null 2>&1
then
    ENCODER="$ISAAC/python.sh"
fi

# Encode frames even if one of the placements failed.
if compgen -G "$RUN/frames/*.png" >/dev/null
then
    if [ -n "$ENCODER" ]; then
        "$ENCODER" \
            "$HERE/encode_video.py" \
            --frames "$RUN/frames" \
            --out "$RUN/battery_alignment.mp4" \
            --fps 30

        echo "Video: $RUN/battery_alignment.mp4"
    else
        echo "Frames are ready in: $RUN/frames"
        echo
        echo "Encode outside the Isaac container with:"
        echo
        echo ".venv/bin/python encode_video.py \\"
        echo "  --frames '$RUN/frames' \\"
        echo "  --out '$RUN/battery_alignment.mp4' \\"
        echo "  --fps 30"
    fi
fi

echo "Report: $RUN/battery_alignment_report.json"

exit "$SIM_STATUS"