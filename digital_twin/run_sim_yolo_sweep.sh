#!/usr/bin/env bash
# External-YOLO table-camera sweep over arbitrary AAA positions and yaw angles.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
IPC_DIR="${IPC_DIR:-$HERE/runs/yolo_ipc}"
ROOT="${1:-$HERE/runs/sim_yolo_sweep_$(date +%Y%m%d_%H%M%S)}"

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

# Deliberately not restricted to multiples of 30 degrees.
CASES=(
  "01 0.34 -0.12   7.0 amazon"
  "02 0.39 -0.10  19.0 seven_eleven"
  "03 0.44 -0.08  38.0 amazon"
  "04 0.32 -0.05  57.0 seven_eleven"
  "05 0.37 -0.03  76.0 amazon"
  "06 0.43  0.00  94.0 seven_eleven"
  "07 0.33  0.03 113.0 amazon"
  "08 0.39  0.05 132.0 seven_eleven"
  "09 0.45  0.07 151.0 amazon"
  "10 0.35 -0.09 169.0 seven_eleven"
  "11 0.42 -0.04  27.5 amazon"
  "12 0.37  0.02 142.3 seven_eleven"
)

trial_failures=0
for spec in "${CASES[@]}"; do
    read -r trial x y yaw brand <<<"$spec"
    echo "[sim-yolo-sweep] trial=$trial brand=$brand x=$x y=$y yaw=$yaw"
    if ! ISAAC="$ISAAC" IPC_DIR="$IPC_DIR" \
      "$HERE/run_sim_yolo_perception.sh" "$ROOT/trial_$trial" \
        --x-m "$x" --y-m "$y" --yaw-deg "$yaw" \
        --battery-brand "$brand"; then
        echo "[sim-yolo-sweep] trial=$trial did not complete; continuing" >&2
        trial_failures=$((trial_failures + 1))
    fi
done

set +e
"$ISAAC/python.sh" "$HERE/aggregate_sim_yolo_trials.py" \
  --root "$ROOT" --expected "${#CASES[@]}"
aggregate_status=$?
set -e

echo "Sweep report: $ROOT/sim_yolo_sweep_report.json"
echo "Sweep CSV:    $ROOT/sim_yolo_sweep_trials.csv"
echo "Trial process failures: $trial_failures"
exit "$aggregate_status"
