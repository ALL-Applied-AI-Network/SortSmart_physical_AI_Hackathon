#!/usr/bin/env python3
"""Two-stage transfer learning for the closed-set AAA battery segmenter."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from dataset_utils import materialize_absolute_dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=70)
    parser.add_argument("--project", default="runs/aaa_seg")
    args = parser.parse_args()

    data_yaml = materialize_absolute_dataset_yaml(args.data)
    model = YOLO(args.model)
    stage1 = model.train(
        data=data_yaml,
        epochs=args.stage1_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="stage1_frozen",
        freeze=10,
        patience=15,
        seed=42,
        deterministic=True,
        amp=True,
        degrees=180.0,
        translate=0.10,
        scale=0.30,
        fliplr=0.5,
        flipud=0.5,
        perspective=0.0,
        close_mosaic=5,
    )
    stage1_best = Path(stage1.save_dir) / "weights" / "best.pt"
    if not stage1_best.exists():
        raise SystemExit(f"Stage-1 checkpoint not found: {stage1_best}")

    model = YOLO(stage1_best)
    stage2 = model.train(
        data=data_yaml,
        epochs=args.stage2_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="stage2_unfrozen",
        freeze=0,
        patience=20,
        seed=42,
        deterministic=True,
        amp=True,
        degrees=180.0,
        translate=0.10,
        scale=0.30,
        fliplr=0.5,
        flipud=0.5,
        perspective=0.0,
        close_mosaic=10,
    )
    print("best checkpoint:", Path(stage2.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()

