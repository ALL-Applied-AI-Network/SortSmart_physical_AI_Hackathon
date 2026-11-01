# Complex clutter video demo (v10)

This patch adds a deterministic physical-camera-inspired digital-twin scene:

- five mixed Amazon / 24-7 Life AAA batteries at different poses and wrapper rolls;
- six coloured cubes;
- two green industrial vials;
- two cap-like parts;
- three dark screws;
- the validated ivory collection bowl.

External YOLO selects the highest-confidence battery that also passes calibrated
AAA geometry checks. The robot completes one wrist-fallback-capable pick, carry
and low-energy bowl placement while the other four batteries remain as
multi-target clutter. A later stage can repeat the loop until all five cells
are sorted.

## Install

```bash
cd /workspace/fanuc_er4ia_sim
cp cell.py cell_before_complex_v10.py
cp battery_yolo_pick_place.py battery_yolo_pick_place_before_complex_v10.py
tar -xzf /workspace/complex_clutter_video_patch_v10.tar.gz
chmod +x run_complex_clutter_video.sh run_battery_yolo_pick_place.sh
/isaac-sim/python.sh -m py_compile \
  cell.py complex_battery_scene.py battery_yolo_pick_place.py
```

## Record the video

Keep the external YOLO worker running, then execute:

```bash
cd /workspace/fanuc_er4ia_sim
ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_complex_clutter_video.sh runs/complex_clutter_video_v10_01
```

The side-view MP4 is written to:

```text
runs/complex_clutter_video_v10_01/battery_yolo_pick_place.mp4
```

The report is:

```text
runs/complex_clutter_video_v10_01/battery_yolo_pick_place_report.json
```

If the automatic encoder is unavailable, the PNG frames remain in the
`frames/` directory. The launcher now falls back to the system `ffmpeg` binary
when the project Python video environment is absent.
