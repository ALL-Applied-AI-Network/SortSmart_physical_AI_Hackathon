# AAA robot annotation batch

Classes:

- `0 aaa_battery`: Amazon and 7/11 AAA batteries share one class. Annotate each visible battery separately.
- `1 battery_drop_zone`: annotate the usable inner opening of the white/gray collection bowl, not the exterior shell.

Rules:

1. Use visible-instance polygons. For occlusion or image clipping, label only the visible region.
2. A battery inside the bowl remains class 0. Motion logic must suppress picks whose center lies inside class 1.
3. Do not label screws, colored blocks, gripper parts, shadows, or reflections.
4. Keep an empty `.txt` file for a true negative image with neither class visible.
5. All frames from one episode stay in one split to prevent temporal leakage.
6. YOLO segmentation rows use normalized coordinates: `class x1 y1 x2 y2 ...`.

Source policy:

- Old recording: episodes 3-9 at approximately z=-220 mm.
- New recording: episodes 1-4 at z=-245 mm.
- Fault, dry-run, high-only, duplicate, and moving frames are excluded.
