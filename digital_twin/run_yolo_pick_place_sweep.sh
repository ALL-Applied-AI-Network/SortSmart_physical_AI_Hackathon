#!/usr/bin/env bash
# Reproducible 12-scene full pick/place validation matrix for v08.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
IPC_DIR="${IPC_DIR:-$HERE/runs/yolo_ipc}"
ROOT="${1:-$HERE/runs/pick_place_sweep_$(date +%Y%m%d_%H%M%S)}"

if [ ! -x "$ISAAC/python.sh" ]; then
    echo "Isaac launcher missing: $ISAAC/python.sh" >&2
    exit 1
fi
mkdir -p "$ROOT" "$IPC_DIR"
ROOT="$(cd "$ROOT" && pwd)"
IPC_DIR="$(cd "$IPC_DIR" && pwd)"
if [ -n "$(ls -A "$ROOT")" ]; then
    echo "Use an empty sweep directory: $ROOT" >&2
    exit 1
fi

# Seed 20261031: 6 trials per brand and 3 trials per visual roll.  Position and
# yaw endpoints are host-IK validated.  The awkward Amazon roll-90 scene is
# deliberately retained as trial 01.
CASES=(
  "01 0.34 -0.12   7.0 amazon       90"
  "02 0.39 -0.10  19.0 seven_eleven  0"
  "03 0.44 -0.08  38.0 amazon      180"
  "04 0.32 -0.05  57.0 seven_eleven 90"
  "05 0.37 -0.03  76.0 amazon      270"
  "06 0.43  0.00  94.0 seven_eleven 180"
  "07 0.33  0.03 113.0 amazon        0"
  "08 0.39  0.05 132.0 seven_eleven 270"
  "09 0.45  0.07 151.0 amazon       90"
  "10 0.35 -0.09 169.0 seven_eleven  0"
  "11 0.42 -0.04  27.5 amazon      270"
  "12 0.37  0.02 142.3 seven_eleven 180"
)

cat >"$ROOT/scene_manifest.csv" <<'CSV'
trial,x_m,y_m,yaw_deg,brand,visual_roll_deg
CSV

process_failures=0
for spec in "${CASES[@]}"; do
    read -r trial x y yaw brand roll <<<"$spec"
    echo "$trial,$x,$y,$yaw,$brand,$roll" >>"$ROOT/scene_manifest.csv"
    echo "[pick-place-sweep] trial=$trial brand=$brand roll=$roll x=$x y=$y yaw=$yaw"
    if ! ISAAC="$ISAAC" IPC_DIR="$IPC_DIR" \
      "$HERE/run_battery_yolo_pick_place.sh" "$ROOT/trial_$trial" \
        --x-m "$x" --y-m "$y" --yaw-deg "$yaw" \
        --battery-brand "$brand" --battery-visual-roll-deg "$roll" \
        --no-video-frames; then
        echo "[pick-place-sweep] trial=$trial failed; continuing" >&2
        process_failures=$((process_failures + 1))
    fi
done

set +e
python3 "$HERE/aggregate_yolo_pick_place_trials.py" \
  --root "$ROOT" --expected "${#CASES[@]}" --min-success-rate 0.90
aggregate_status=$?
set -e

echo "Sweep report: $ROOT/pick_place_sweep_report.json"
echo "Sweep CSV:    $ROOT/pick_place_sweep_trials.csv"
echo "Manifest:     $ROOT/scene_manifest.csv"
echo "Trial process failures: $process_failures"
exit "$aggregate_status"
