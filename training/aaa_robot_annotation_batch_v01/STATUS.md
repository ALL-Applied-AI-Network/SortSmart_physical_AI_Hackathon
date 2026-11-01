# Batch status

Completed:

- validated both LeRobot archives;
- selected usable episodes from both recording sessions;
- extracted 68 non-duplicate frames at stable robot poses;
- kept every episode inside a single train/validation/test split;
- recorded source, camera, frame index, timestamp, and TCP pose in `manifest.csv`;
- defined the two segmentation classes and annotation rules;
- generated contact sheets for visual QA.

Pending:

- pixel-level polygons for `aaa_battery` and `battery_drop_zone`;
- overlay QA after annotation;
- integration with phone images;
- YOLO26s-seg training after labels pass review.

The `labels/` directories are intentionally empty until reviewed polygons are available. Do not start final training from this package yet.
