#!/usr/bin/env bash
# AAA-only calibrated 2D vision sweep over six table positions/orientations.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
ROOT="${1:-$HERE/runs/battery_2d_sweep_$(date +%Y%m%d_%H%M%S)}"
if [ ! -x "$ISAAC/python.sh" ]; then
    echo "Isaac launcher missing: $ISAAC/python.sh" >&2
    exit 1
fi
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
if [ -n "$(ls -A "$ROOT")" ]; then
    echo "Use an empty sweep directory: $ROOT" >&2
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
    echo "[sweep] trial=$trial x=$x y=$y yaw=$yaw"
    ISAAC="$ISAAC" HOST_PYTHON=/nonexistent \
      "$HERE/run_battery_grasp.sh" "$ROOT/trial_$trial" \
        --use-2d-vision --camera table --no-video-frames \
        --x-m "$x" --y-m "$y" --yaw-deg "$yaw"
done

"$ISAAC/python.sh" "$HERE/aggregate_battery_trials.py" \
  --root "$ROOT" --expected "${#CASES[@]}"

echo "AAA 2D sweep complete: $ROOT/battery_2d_sweep_report.json"
