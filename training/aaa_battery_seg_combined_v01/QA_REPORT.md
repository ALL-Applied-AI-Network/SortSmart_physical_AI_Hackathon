# Dataset QA report

- All 124 images have matching YOLO segmentation label files.
- All class IDs are valid (`0` or `1`).
- All normalized polygon coordinates lie in `[0, 1]`.
- No exact image duplicate crosses train, validation, or test splits.
- Phone annotations contain 154 reviewed AAA-battery polygons and 17 verified negative images.
- Robot annotations contain 272 AAA-battery polygons and 62 battery-drop-zone polygons.
- Phone images are used only for training; robot episode splits remain unchanged.
