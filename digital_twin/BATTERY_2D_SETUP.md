# AAA battery grasp from a calibrated 2D camera

This patch keeps the successful mechanics baseline and adds a second control
mode that matches a flat-table FANUC iRVision 2D deployment.

## Files

- `battery_2d_perception.py`: simulator-independent mask-to-robot-XY/yaw code.
- `battery_grasp_test.py`: adds `--use-2d-vision`; simulator poses are then
  evaluation-only and cannot supply the commanded grasp.
- `run_battery_grasp.sh`: refuses to report success when Isaac exits without
  creating the JSON report.
- `test_battery_2d_perception.py`: host-only geometry regression.

Copy the four code files into the project root. The perception and test modules
are new; `battery_grasp_test.py` and `run_battery_grasp.sh` replace their prior
versions.

## Run in the Isaac Sim container

```bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim ./run_battery_grasp.sh runs/battery_2d_01 \
  --use-2d-vision --camera table
```

Review:

- `runs/battery_2d_01/battery_perception_overlay.png`
- `runs/battery_2d_01/battery_perception.json`
- `runs/battery_2d_01/battery_grasp_report.json`

Require both:

```json
"simulator_pose_used_for_control": false,
"task_complete": true
```

The current synthetic battery is detected with an orange HSV mask. This is a
deliberately replaceable simulation baseline, not the production detector for
the 24/7 Life wrapper. A real detector should output a binary body mask and call
the unchanged `estimate_pose_from_mask(...)` function. Camera intrinsics,
extrinsics, table height and FANUC TCP/UTOOL calibration must be re-measured on
the physical cell before commanding the real arm.
