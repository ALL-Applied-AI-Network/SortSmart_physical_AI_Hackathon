# RGB pile-aware multi-battery placement (v13)

This patch prevents the fifth AAA battery from being forced into an already
occupied floor slot.  It is deliberately designed around sensing and motion
logic that can also be implemented on the physical FANUC setup:

- calibrated table-camera RGB masks determine battery and bowl XY;
- a calibrated side-camera RGB mask estimates the pile-top height;
- no Isaac depth image, simulator object pose or contact oracle is used for
  control;
- the first four batteries use 18 mm-spaced floor cells;
- the fifth battery is aligned crosswise and released 3 mm above the estimated
  first layer;
- if side RGB cannot provide a valid height, a higher count-based fallback is
  used instead of descending blindly;
- simulation physics runs at 120 Hz while video remains 30 fps, and the fifth
  descent is slowed to reduce contact energy.

The simulator still reads object poses after commands are fixed, but only for
evaluation fields in the report.  Those values never enter the control plan.

## Install

```bash
cd /workspace/fanuc_er4ia_sim

tar -xzf /workspace/complex_clutter_rgb_pile_patch_v13.tar.gz
chmod +x \
  run_battery_yolo_multi_pick_place.sh \
  run_complex_clutter_multi_video.sh

/isaac-sim/python.sh -m py_compile \
  complex_battery_scene.py \
  multi_battery_planning.py \
  side_rgb_pile_height.py \
  battery_yolo_multi_pick_place.py
```

The package includes the v11 procedural-clutter repair and the v12 continuous
five-battery workflow, so it may be installed directly over the current
project.

## Run

Keep the external YOLO IPC worker running in terminal A.  In the Isaac
container, use a new empty run directory:

```bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_complex_clutter_multi_video.sh \
  runs/complex_clutter_rgb_pile_v13_01 \
  2>&1 | tee complex_clutter_rgb_pile_v13.log
```

Main outputs:

```text
runs/complex_clutter_rgb_pile_v13_01/battery_yolo_multi_pick_place_report.json
runs/complex_clutter_rgb_pile_v13_01/battery_yolo_multi_pick_place.mp4
runs/complex_clutter_rgb_pile_v13_01/cycle_05/side_pile_rgb.png
runs/complex_clutter_rgb_pile_v13_01/cycle_05/side_pile_overlay.png
runs/complex_clutter_rgb_pile_v13_01/cycle_05/side_pile_height.json
```

## Check the safety and completion gates

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path(
    "runs/complex_clutter_rgb_pile_v13_01/"
    "battery_yolo_multi_pick_place_report.json"
)
r = json.loads(p.read_text())
print("status =", r.get("status"))
print("task_complete =", r.get("task_complete"))
print("completed_pick_count =", r.get("completed_pick_count"))
print("all_quality_gates_passed =", r.get("all_quality_gates_passed"))
print(
    "all_still_inside =",
    r.get("evaluation_only", {}).get(
        "all_picked_batteries_still_inside_bowl"
    ),
)
fifth = r.get("cycles", [None] * 5)[4]
print("fifth_layer =", fifth.get("placement_layer"))
print("fifth_axis =", fifth.get("release_battery_axis"))
print(
    "bottom_clearance_m =",
    fifth.get("commanded_battery_bottom_clearance_m"),
)
print(
    "pile_sensor =",
    (fifth.get("pile_height_observation") or {}).get("sensor_mode"),
)
print(
    "simulator_depth_used =",
    (fifth.get("pile_height_observation") or {}).get(
        "simulator_depth_used"
    ),
)
PY

ls -lh \
  runs/complex_clutter_rgb_pile_v13_01/battery_yolo_multi_pick_place.mp4
```

Expected values are five completed picks, all quality gates true, all five
batteries still inside the bowl, fifth layer `1`, fifth axis `y`, bottom
clearance at least `0.0025`, and `simulator_depth_used = False`.

## Before transferring to the physical robot

Recalibrate both real cameras after any camera or table movement.  In
particular, verify the table plane and table-camera extrinsics with a measured
target, then verify the side-camera vertical reprojection at several known
heights.  Table RGB plus a calibrated plane is sufficient for XY, while side
RGB supplies the extra height cue needed for a crowded bowl.  Hardware depth,
gripper current or force sensing may be added as independent safety checks if
available, but this v13 controller does not require them.

Do not copy the simulator's metric camera transforms directly to the physical
cell.  The physical deployment must also retain conservative speed, clearance,
workspace, and emergency-stop limits.
