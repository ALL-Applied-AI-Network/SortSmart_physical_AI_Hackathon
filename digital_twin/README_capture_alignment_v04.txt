Battery pick/place capture-alignment patch v04

Purpose
-------
Make the first planning frame use the exact same Isaac Sim timing as the
previously successful sim_yolo_perception_test.py run:

1. Park-arm action is issued once.
2. Battery remains kinematic for 90 rendered physics steps.
3. Battery is released and settles for 60 rendered physics steps.
4. The table-camera frame is read immediately, without an extra physics step.

The YOLO thresholds remain unchanged (battery 0.60, drop zone 0.30).
The existing five-frame fallback remains enabled after the aligned first frame.

Install from /workspace/fanuc_er4ia_sim
---------------------------------------
cp battery_yolo_pick_place.py battery_yolo_pick_place_before_capture_alignment.py
tar -xzf /workspace/battery_pick_place_capture_alignment_v04.tar.gz
/isaac-sim/python.sh -m py_compile battery_yolo_pick_place.py

Validation command
------------------
ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_battery_yolo_pick_place.sh runs/pick_place_247_05 \
  --x-m 0.34 \
  --y-m -0.12 \
  --yaw-deg 7.0 \
  --battery-brand seven_eleven \
  --battery-visual-roll-deg 0 \
  --no-video-frames

Inspect
-------
cat runs/pick_place_247_05/perception_attempts.json
cat runs/pick_place_247_05/battery_yolo_pick_place_report.json

Expected behavior
-----------------
If the aligned first frame restores the prior >=0.60 battery confidence, the
burst stops at attempt 1 and arm motion may begin.  If it does not, all five
attempts are evaluated and the original safe-abstention behavior is preserved.
