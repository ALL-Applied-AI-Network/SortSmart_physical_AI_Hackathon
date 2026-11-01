Wrist YOLO re-observation diagnostic v06

Purpose
-------
Validate active close-range vision before connecting it to grasp execution.
When the overhead/table view misses the battery, the arm visits four fixed
high-clearance observation poses.  The scan grid is predefined and does not
use simulator object truth or a weak detection to guide motion.

Safety properties
-----------------
- Tool tip stays at 100 mm above the table during scanning.
- Jaws remain fully open.
- The script never descends for grasping and never closes the jaws.
- YOLO battery threshold remains 0.60.
- An accepted mask must also pass calibrated AAA length/width/workspace checks.
- Simulator truth is read only after scanning, for evaluation.

Install from /workspace/fanuc_er4ia_sim inside the Isaac container
------------------------------------------------------------------
tar -xzf /workspace/wrist_yolo_reobserve_patch_v06.tar.gz
chmod +x run_wrist_yolo_reobserve_test.sh

/isaac-sim/python.sh -m py_compile \
  wrist_scan_planning.py \
  wrist_yolo_reobserve_test.py

Run the failed Amazon roll-90 scene
-----------------------------------
ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_wrist_yolo_reobserve_test.sh runs/wrist_reobserve_amazon_roll90_01 \
  --x-m 0.34 \
  --y-m -0.12 \
  --yaw-deg 7.0 \
  --battery-brand amazon \
  --battery-visual-roll-deg 90

Inspect
-------
cat runs/wrist_reobserve_amazon_roll90_01/wrist_yolo_reobserve_report.json

The report must show all of the following before integration with grasping:
  "task_complete": true
  "simulator_pose_used_for_control": false
  "observation_motion_started": true
  "grasp_motion_started": false
  "jaws_closed": false

Images are saved as:
  runs/wrist_reobserve_amazon_roll90_01/wrist_rgb_scan_*.png
  runs/wrist_reobserve_amazon_roll90_01/wrist_overlay_scan_*.png

This patch is intentionally diagnostic-only.  A successful result will be
integrated into battery_yolo_pick_place.py in the next patch.
