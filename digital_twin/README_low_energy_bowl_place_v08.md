# Low-energy bowl placement (v08)

This patch fixes an end-of-cycle failure in which vision, grasp and carry all
succeeded, but the AAA cell was released about 82 mm above the bowl floor and
occasionally bounced out.

The new sequence keeps the validated wrist-camera fallback, lowers the held
battery along the detected bowl centre to 1 mm above the internal floor, opens
the jaws more slowly, waits, and then retreats.  Carry verification is recorded
at the high pose over the bowl so the new low placement is not falsely scored
as a carry failure.

## Install

```bash
cd /workspace/fanuc_er4ia_sim
cp battery_yolo_pick_place.py battery_yolo_pick_place_before_v08.py
cp battery_sort_planning.py battery_sort_planning_before_v08.py
cp battery_drop_bowl.py battery_drop_bowl_before_v08.py
tar -xzf /workspace/low_energy_bowl_place_patch_v08.tar.gz
/isaac-sim/python.sh -m py_compile \
  battery_yolo_pick_place.py battery_sort_planning.py \
  battery_drop_bowl.py wrist_scan_planning.py
```

## Re-run the same difficult scene

Keep the external YOLO worker running, then execute:

```bash
cd /workspace/fanuc_er4ia_sim
ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_battery_yolo_pick_place.sh runs/pick_place_amazon_roll90_low_place_01 \
  --x-m 0.34 \
  --y-m -0.12 \
  --yaw-deg 7.0 \
  --battery-brand amazon \
  --battery-visual-roll-deg 90 \
  --no-video-frames
```

## Inspect

```bash
cat runs/pick_place_amazon_roll90_low_place_01/battery_yolo_pick_place_report.json
```

Expected key fields:

```text
task_complete: true
observation_motion_started: true
grasp_motion_started: true
pickup_verified: true
release_strategy: low_energy_place_on_bowl_floor
carried_to_detected_bowl: true
inside_bowl: true
```
