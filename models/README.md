# AAA battery segmentation model

`aaa_yolo26s_best.pt` is the final Ultralytics YOLO instance-segmentation checkpoint used by the digital-twin and physical-cell code.

The repository stores the checkpoint under `parts/` because the connected publishing path has a small per-file transfer limit. Reconstruct it before running inference:

```bash
./models/assemble_checkpoint.sh
```

The script concatenates the numbered parts and verifies the exact hash below.

## Provenance

- Target class: `aaa_battery`
- Sources: phone photographs plus table, wrist and side robot-camera images
- Labels: manually reviewed instance polygons, including clutter-only negative images
- Intended use: AAA battery segmentation for calibrated robotic localization and grasp planning
- File size: 23,386,909 bytes
- SHA-256: `486c8599d8558b7826b9361f19b60bb67d3daf911f1d518436f2726c6bd01643`

## Limitations

The checkpoint is not a general battery or waste classifier. AA cells, other chemistries, crushed/occluded batteries and unseen industrial domains require additional representative data and evaluation. Model confidence is a detection score, not a calibrated guarantee of correctness and not a brand probability. Motion code must retain independent geometry, depth, workspace and safety checks.

