#!/usr/bin/env bash
# Replay the six successful calibrated-2D AAA cases and create one annotated MP4.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
VIDEO_PYTHON="${VIDEO_PYTHON:-$HERE/.venv/bin/python}"
ROOT="${1:-$HERE/runs/battery_2d_video_$(date +%Y%m%d_%H%M%S)}"

if [ ! -x "$ISAAC/python.sh" ]; then
    echo "Isaac launcher missing: $ISAAC/python.sh" >&2
    exit 1
fi
if [ ! -x "$VIDEO_PYTHON" ]; then
    echo "Video Python missing: $VIDEO_PYTHON" >&2
    echo "Create the project venv and install requirements.txt, or set VIDEO_PYTHON." >&2
    exit 1
fi
if ! "$VIDEO_PYTHON" -c 'import cv2, imageio_ffmpeg, numpy' >/dev/null 2>&1; then
    echo "Video Python needs cv2, imageio_ffmpeg and numpy: $VIDEO_PYTHON" >&2
    echo 'Install them with: .venv/bin/pip install -r requirements.txt' >&2
    exit 1
fi

mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
if [ -n "$(ls -A "$ROOT")" ]; then
    echo "Use an empty video directory: $ROOT" >&2
    exit 1
fi

CASES=(
  "01 0.32 -0.10   0"
  "02 0.36 -0.08  30"
  "03 0.40 -0.04  60"
  "04 0.32  0.06  90"
  "05 0.36  0.10 120"
  "06 0.40  0.06 150"
)

for spec in "${CASES[@]}"; do
    read -r trial x y yaw <<<"$spec"
    echo "[video sweep] trial=$trial x=$x y=$y yaw=$yaw"
    ISAAC="$ISAAC" HOST_PYTHON=/nonexistent \
      "$HERE/run_battery_grasp.sh" "$ROOT/trial_$trial" \
        --use-2d-vision --camera table --video-motion-only \
        --x-m "$x" --y-m "$y" --yaw-deg "$yaw"
done

"$ISAAC/python.sh" "$HERE/aggregate_battery_trials.py" \
  --root "$ROOT" --expected "${#CASES[@]}"

"$VIDEO_PYTHON" "$HERE/encode_battery_2d_sweep_video.py" \
  --root "$ROOT" \
  --out "$ROOT/battery_2d_sweep.mp4" \
  --fps 30 --width 960 --height 540

echo "Video: $ROOT/battery_2d_sweep.mp4"
echo "Report: $ROOT/battery_2d_sweep_report.json"
