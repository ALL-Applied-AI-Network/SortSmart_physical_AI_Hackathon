#!/usr/bin/env python3
"""Run the zero-shot baseline and write draft YOLO segmentation labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLOE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("raw_images"))
    parser.add_argument("--labels", type=Path, default=Path("draft_labels"))
    parser.add_argument("--overlays", type=Path, default=Path("qa/pseudo_overlays"))
    parser.add_argument("--model", default="yoloe-26n-seg.pt")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    prompts = args.prompt or ["AAA battery"]
    image_paths = sorted(args.images.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No JPG images found in {args.images}")

    for directory in (args.labels, args.overlays):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    model = YOLOE(args.model)
    model.set_classes(prompts)

    summary = {
        "model": args.model,
        "prompts": prompts,
        "confidence_threshold": args.conf,
        "image_count": len(image_paths),
        "images_with_detections": 0,
        "predicted_instances": 0,
        "records": [],
    }

    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=args.conf,
        iou=args.iou,
        imgsz=640,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for result in results:
        source_path = Path(result.path)
        label_path = args.labels / f"{source_path.stem}.txt"
        lines: list[str] = []
        confidences: list[float] = []
        if result.masks is not None and result.boxes is not None:
            polygons = result.masks.xyn
            box_confidences = result.boxes.conf.detach().cpu().tolist()
            for polygon, confidence in zip(polygons, box_confidences):
                points = np.asarray(polygon, dtype=float)
                if points.ndim != 2 or points.shape[0] < 3:
                    continue
                coordinates = " ".join(f"{value:.6f}" for value in points.reshape(-1))
                # All prompt variants map to the single closed-set class aaa_battery.
                lines.append(f"0 {coordinates}")
                confidences.append(float(confidence))

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            summary["images_with_detections"] += 1
            summary["predicted_instances"] += len(lines)

        # result.plot() is BGR; Pillow expects RGB.
        overlay = result.plot()
        Image.fromarray(overlay[..., ::-1]).save(args.overlays / source_path.name, quality=90)
        summary["records"].append(
            {
                "image": source_path.name,
                "instances": len(lines),
                "confidences": confidences,
                "requires_human_review": True,
            }
        )

    Path("qa").mkdir(exist_ok=True)
    Path("qa/zero_shot_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    print("Draft labels are NOT ground truth. Review every overlay before training.")


if __name__ == "__main__":
    main()

