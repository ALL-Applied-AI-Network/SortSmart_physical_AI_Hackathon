#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from dataset_utils import materialize_absolute_dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--source", default="dataset/images/val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    data_yaml = materialize_absolute_dataset_yaml(args.data)
    model = YOLO(args.weights)
    metrics = model.val(data=data_yaml, device=args.device, imgsz=640)
    model.predict(
        source=args.source,
        device=args.device,
        conf=args.conf,
        imgsz=640,
        save=True,
        project="runs/aaa_seg",
        name="validation_overlays",
    )
    summary = {
        "weights": str(args.weights),
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
    }
    Path("runs/aaa_seg").mkdir(parents=True, exist_ok=True)
    Path("runs/aaa_seg/evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

