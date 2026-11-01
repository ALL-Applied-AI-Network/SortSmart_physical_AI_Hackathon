#!/usr/bin/env bash
# Isolated AAA mechanics trial: build -> grasp -> lift -> hold -> optional video.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaacsim}"
HOST_PYTHON="${HOST_PYTHON:-$HERE/.venv/bin/python}"
if [ ! -x "$ISAAC/python.sh" ]; then
    echo "Isaac launcher missing: $ISAAC/python.sh" >&2
    exit 1
fi
RUN="${1:-$HERE/runs/battery_grasp_$(date +%Y%m%d_%H%M%S)}"
shift $(( $# > 0 ? 1 : 0 ))
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Use an empty run directory: $RUN" >&2
    exit 1
fi
"$ISAAC/python.sh" "$HERE/battery_grasp_test.py" --out "$RUN" "$@"
REPORT="$RUN/battery_grasp_report.json"
# Isaac can occasionally return exit code zero after a Python exception during
# shutdown.  A missing report is therefore a failed run, whatever the process
# exit code said.
if [ ! -s "$REPORT" ]; then
    echo "Battery test failed: report was not created: $REPORT" >&2
    exit 2
fi
if compgen -G "$RUN/frames/*.png" >/dev/null && [ -x "$HOST_PYTHON" ]; then
    "$HOST_PYTHON" "$HERE/encode_video.py" \
        --frames "$RUN/frames" --out "$RUN/battery_grasp.mp4" --fps 30
fi
echo "Review $REPORT and optional battery_grasp.mp4"
