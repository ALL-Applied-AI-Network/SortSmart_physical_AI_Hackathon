# Full YOLO pick/place validation sweep (v09)

This patch adds a reproducible 12-scene validation matrix on top of the v08
low-energy bowl-placement flow.

Coverage:

- 6 Amazon and 6 7-Eleven AAA trials
- 12 validated table positions and non-cardinal yaw angles
- visual wrapper roll 0, 90, 180 and 270 degrees, 3 trials each
- table-camera burst plus fixed safe wrist-camera fallback
- complete grasp, carry and low-energy bowl placement

The sweep passes only when all reports are present, every trial remains
external-vision controlled, both brands are present, and at least 90% of the
12 trials succeed (therefore at least 11/12).

## Install

```bash
cd /workspace/fanuc_er4ia_sim
tar -xzf /workspace/pick_place_validation_sweep_patch_v09.tar.gz
chmod +x run_yolo_pick_place_sweep.sh
python3 -m py_compile aggregate_yolo_pick_place_trials.py
/isaac-sim/python.sh -m py_compile \
  battery_yolo_pick_place.py battery_sort_planning.py \
  battery_drop_bowl.py wrist_scan_planning.py
```

## Run

Keep the existing external YOLO worker running in terminal A. In the Isaac
container terminal run:

```bash
cd /workspace/fanuc_er4ia_sim
ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_yolo_pick_place_sweep.sh runs/pick_place_sweep_v09_01
```

Trials run sequentially and video capture is disabled to reduce runtime and
storage. A failed trial does not stop the remaining trials.

## Inspect

```bash
cat runs/pick_place_sweep_v09_01/pick_place_sweep_report.json
column -s, -t < runs/pick_place_sweep_v09_01/pick_place_sweep_trials.csv | less -S
```

The main fields are `successes`, `success_rate`, `brand_breakdown`,
`visual_roll_breakdown`, `table_camera_selections`,
`wrist_camera_selections`, `failed_trials`, and
`all_quality_gates_passed`.

An exit status of 2 means the sweep completed but one or more quality gates
failed; inspect `failed_trials` and the corresponding `trial_XX` report.
