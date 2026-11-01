#!/usr/bin/env python3
"""Train and test YOLO26s instance segmentation for battery sorting."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo26s-seg.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/aaa_yolo26s_seg"))
    parser.add_argument("--name", default="train_v01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    if not data.exists():
        raise SystemExit(f"Dataset YAML not found: {data}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        patience=30,
        amp=True,
        seed=42,
        deterministic=True,
        project=str(args.project),
        name=args.name,
        plots=True,
        degrees=12.0,
        translate=0.08,
        scale=0.25,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    best = Path(model.trainer.best)
    print(f"BEST_WEIGHTS={best}")
    YOLO(str(best)).val(data=str(data), split="test", imgsz=args.imgsz, device=args.device, plots=True)


if __name__ == "__main__":
    main()
