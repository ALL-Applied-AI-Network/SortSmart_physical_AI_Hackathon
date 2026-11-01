# Active wrist fallback for full YOLO pick-place (v07)

This patch integrates the validated wrist-camera re-observation scan into the
full battery pick-and-place program.

## Install

```bash
cd /workspace/fanuc_er4ia_sim
cp battery_yolo_pick_place.py battery_yolo_pick_place_before_wrist_fallback_v07.py
tar -xzf /workspace/active_wrist_pick_place_patch_v07.tar.gz
/isaac-sim/python.sh -m py_compile battery_yolo_pick_place.py wrist_scan_planning.py
```

Keep the existing external YOLO worker running with the established settings:
`imgsz=1024`, battery confidence `0.60`, and drop-zone confidence `0.30`.

## Run the difficult Amazon roll-90 scene

```bash
cd /workspace/fanuc_er4ia_sim
ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_battery_yolo_pick_place.sh runs/pick_place_amazon_roll90_wrist_01 \
  --x-m 0.34 \
  --y-m -0.12 \
  --yaw-deg 7.0 \
  --battery-brand amazon \
  --battery-visual-roll-deg 90 \
  --no-video-frames
```

The normal table-camera burst runs first. If it finds the bowl but misses the
battery, the robot visits the fixed high-clearance wrist scan poses with open
jaws. Grasping starts only after an external YOLO mask passes the unchanged
confidence threshold and calibrated AAA geometry validation.

## Inspect

```bash
cat runs/pick_place_amazon_roll90_wrist_01/perception_attempts.json
cat runs/pick_place_amazon_roll90_wrist_01/planning_perception.json
cat runs/pick_place_amazon_roll90_wrist_01/battery_yolo_pick_place_report.json
```

The expected success report contains:

- `task_complete: true`
- `simulator_pose_used_for_control: false`
- `observation_motion_started: true`
- `grasp_motion_started: true`
- `pickup_verified: true`
- `carried_to_detected_bowl: true`
- `inside_bowl: true`

If neither camera validates a battery, the program reports
`safely_abstained: true`, `grasp_motion_started: false`, and `jaws_closed:
false`.

Use `--disable-wrist-fallback` only when reproducing the original table-only
behavior.
