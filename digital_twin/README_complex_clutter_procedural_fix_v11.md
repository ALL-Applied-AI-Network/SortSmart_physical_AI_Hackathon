# Complex clutter procedural fix (v11)

This patch fixes the v10 complex-clutter run that shut Isaac down before it
could create `battery_yolo_pick_place_report.json`.

The v10 scene reused legacy referenced cube/vial USD assets.  The failing log
ended in USD edit-target warnings under `/World/SortVial1/Visual/Looks`, which
is consistent with those referenced assets introducing a metrics adjustment
layer outside the anonymous stage's local layer stack.  v11 replaces every
non-battery clutter item with a static procedural USD primitive.  It does not
change the YOLO model, confidence gates, camera calibration, grasp plan, or the
validated low-energy placement sequence.

The scene still contains:

- five mixed Amazon / 24-7 Life AAA batteries;
- six coloured cubes;
- two green vial-like cylinders;
- two cap-like cylinders;
- three dark screw-like parts;
- one ivory collection bowl.

## Install over v10

```bash
cd /workspace/fanuc_er4ia_sim
cp complex_battery_scene.py complex_battery_scene_before_v11.py
tar -xzf /workspace/complex_clutter_procedural_fix_v11.tar.gz
chmod +x run_complex_clutter_video.sh run_battery_yolo_pick_place.sh

/isaac-sim/python.sh -m py_compile \
  cell.py complex_battery_scene.py battery_yolo_pick_place.py
```

## Record a fresh run

Keep the external YOLO worker running in its own terminal.  Use a new, empty
output directory:

```bash
cd /workspace/fanuc_er4ia_sim
ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_complex_clutter_video.sh runs/complex_clutter_video_v11_01 \
  2>&1 | tee complex_clutter_v11.log
```

Successful execution must create both:

```text
runs/complex_clutter_video_v11_01/battery_yolo_pick_place_report.json
runs/complex_clutter_video_v11_01/battery_yolo_pick_place.mp4
```

Check the core result:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("runs/complex_clutter_video_v11_01/battery_yolo_pick_place_report.json")
r = json.loads(p.read_text())
for key in ("task_complete", "safely_abstained", "arm_motion_started"):
    print(key, "=", r.get(key))
print("pickup_verified =", r.get("grasp", {}).get("pickup_verified"))
print("inside_bowl =", r.get("placement", {}).get("inside_bowl"))
PY
```

If the run still fails before a report is written, preserve the complete log:

```bash
tail -n 200 complex_clutter_v11.log
grep -n -E "Traceback|Error|Exception|Coding Error" complex_clutter_v11.log | tail -n 80
```

Host-side validation performed for this patch: Python compilation, shell
syntax checks, and 15 focused unit tests.  Isaac/PhysX execution must be
verified in the Spark container because Isaac Sim is unavailable in the patch
build environment.
