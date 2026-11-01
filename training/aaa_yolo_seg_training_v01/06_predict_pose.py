#!/usr/bin/env python3
"""Infer battery masks and estimate image-space center/yaw from each mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def polygon_pose(points: np.ndarray) -> tuple[list[float], float]:
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    yaw_deg = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    return [float(center[0]), float(center[1])], yaw_deg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(args.weights)
    output = []
    for result in model.predict(
        source=args.source,
        device=args.device,
        conf=args.conf,
        imgsz=640,
        stream=True,
        save=True,
        project="runs/aaa_seg",
        name="pose_overlays",
    ):
        detections = []
        if result.masks is not None and result.boxes is not None:
            for polygon, confidence in zip(
                result.masks.xy, result.boxes.conf.detach().cpu().tolist()
            ):
                points = np.asarray(polygon, dtype=float)
                if points.shape[0] < 3:
                    continue
                center, yaw_deg = polygon_pose(points)
                detections.append(
                    {
                        "confidence": float(confidence),
                        "center_pixel": center,
                        "battery_yaw_deg": yaw_deg,
                    }
                )
        output.append({"image": str(result.path), "detections": detections})

    Path("runs/aaa_seg/pose_predictions.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("wrote runs/aaa_seg/pose_predictions.json")


if __name__ == "__main__":
    main()

