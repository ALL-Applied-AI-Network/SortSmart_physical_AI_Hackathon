# Complex clutter multi-battery sorting video (v12)

This patch extends the validated complex-clutter scene from one pick/place to
one continuous five-battery sorting run.

For every battery, the simulator:

1. returns the empty gripper to the fixed capture posture;
2. captures a fresh table-camera image and calls the external YOLO worker;
3. rejects detections already inside the bowl exclusion region;
4. optionally runs the existing fixed safe wrist-camera fallback;
5. plans the grasp entirely from the selected mask and calibrated geometry;
6. rotates the held battery to a common bowl orientation;
7. places it into one of five non-overlapping, bowl-relative slots;
8. checks pickup, carry and placement before continuing.

The run stops safely if a required perception or motion gate fails.  After the
fifth placement, a final inventory gate verifies that all five batteries are
still inside the bowl.  The complete sequence is recorded into one side-view
MP4 with cycle/phase labels.

## Install

```bash
cd /workspace/fanuc_er4ia_sim

tar -xzf /workspace/complex_clutter_multi_pick_patch_v12.tar.gz
chmod +x \
  run_battery_yolo_multi_pick_place.sh \
  run_complex_clutter_multi_video.sh

/isaac-sim/python.sh -m py_compile \
  complex_battery_scene.py \
  multi_battery_planning.py \
  battery_yolo_multi_pick_place.py
```

This package includes the v11 procedural-clutter fix, so it can be installed
directly over v10 or v11.

## Run

Keep the external YOLO IPC worker running in its own terminal. Then use a new,
empty output directory:

```bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_complex_clutter_multi_video.sh \
  runs/complex_clutter_multi_v12_01 \
  2>&1 | tee complex_clutter_multi_v12.log
```

The run is approximately one minute long. Output files:

```text
runs/complex_clutter_multi_v12_01/battery_yolo_multi_pick_place_report.json
runs/complex_clutter_multi_v12_01/battery_yolo_multi_pick_place.mp4
```

Each `cycle_01` ... `cycle_05` directory contains its table/wrist images,
overlays, raw YOLO responses and perception-attempt summary.

## Check the result

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path(
    "runs/complex_clutter_multi_v12_01/"
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
for cycle in r.get("cycles", []):
    print(
        "cycle", cycle.get("cycle"),
        "complete=", cycle.get("task_complete"),
        "confidence=",
        (cycle.get("perception", {}).get("selected_battery") or {}).get(
            "confidence"
        ),
    )
PY

ls -lh \
  runs/complex_clutter_multi_v12_01/battery_yolo_multi_pick_place.mp4
```

Expected final values are `task_complete = True`,
`completed_pick_count = 5`, `all_quality_gates_passed = True`, and
`all_still_inside = True`.

Host-side validation for this patch covers syntax, shell launchers, five-slot
geometry, bowl re-selection exclusion, and every interpolated IK endpoint for
all five complex-scene batteries. Isaac/PhysX execution must still be run in
the Spark container.
