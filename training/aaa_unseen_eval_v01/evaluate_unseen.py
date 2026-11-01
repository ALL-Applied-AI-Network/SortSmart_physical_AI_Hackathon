#!/usr/bin/env python3
"""Run YOLO segmentation and extract image-space battery grasp geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


BATTERY_CLASS = 0
DROP_ZONE_CLASS = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("best.pt"))
    parser.add_argument("--source", type=Path, default=Path("sanity_images"))
    parser.add_argument("--output", type=Path, default=Path("runs/unseen_eval_v01"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--battery-conf", type=float, default=0.60)
    parser.add_argument("--drop-conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def normalized_long_axis(rect: tuple) -> tuple[float, float, float, float, float]:
    (cx, cy), (width, height), angle = rect
    if width < height:
        angle += 90.0
    long_px = max(float(width), float(height))
    short_px = min(float(width), float(height))
    yaw = ((float(angle) + 90.0) % 180.0) - 90.0
    aspect = long_px / max(short_px, 1e-6)
    return float(cx), float(cy), yaw, long_px, aspect


def polygon_geometry(polygon: np.ndarray) -> dict[str, float | list[float]]:
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx, cy = map(float, polygon.mean(axis=0))
    _, _, yaw, long_px, aspect = normalized_long_axis(cv2.minAreaRect(contour))
    short_px = long_px / max(aspect, 1e-6)
    return {
        "center_px": [cx, cy],
        "yaw_image_deg_clockwise": yaw,
        "long_axis_px": long_px,
        "short_axis_px": short_px,
        "aspect_ratio": aspect,
        "mask_area_px2": area,
    }


def center_inside_polygon(center: list[float], polygon: np.ndarray) -> bool:
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(center[0]), float(center[1])), False) >= 0


def draw_detection(image: np.ndarray, record: dict, polygon: np.ndarray) -> None:
    class_id = int(record["class_id"])
    pick_candidate = bool(record.get("pick_candidate", False))
    if class_id == DROP_ZONE_CLASS:
        color = (255, 220, 0)  # BGR: cyan
    elif pick_candidate:
        color = (40, 210, 40)  # green
    else:
        color = (0, 150, 255)  # orange

    contour = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
    overlay = image.copy()
    cv2.fillPoly(overlay, [contour], color)
    cv2.addWeighted(overlay, 0.22, image, 0.78, 0.0, dst=image)
    cv2.polylines(image, [contour], True, color, 2, cv2.LINE_AA)

    cx, cy = record["center_px"]
    center = (int(round(cx)), int(round(cy)))
    cv2.circle(image, center, 4, color, -1, cv2.LINE_AA)

    if class_id == BATTERY_CLASS:
        yaw = math.radians(float(record["yaw_image_deg_clockwise"]))
        radius = 0.5 * float(record["long_axis_px"])
        dx, dy = radius * math.cos(yaw), radius * math.sin(yaw)
        p1 = (int(round(cx - dx)), int(round(cy - dy)))
        p2 = (int(round(cx + dx)), int(round(cy + dy)))
        cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)

    label = f'{record["class_name"]} {record["confidence"]:.2f}'
    if class_id == BATTERY_CLASS:
        label += f' yaw={record["yaw_image_deg_clockwise"]:.1f}'
        if not pick_candidate:
            label += f' [{record["pick_reason"]}]'
    x, y = contour.reshape(-1, 2).min(axis=0)
    cv2.putText(
        image,
        label,
        (int(x), max(22, int(y) - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def make_contact_sheet(images: list[Path], output: Path, columns: int = 4) -> None:
    if not images:
        return
    tile_width, tile_height, label_height = 480, 360, 34
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(images):
        with Image.open(path) as source:
            source = source.convert("RGB")
            source.thumbnail((tile_width, tile_height))
            x = (index % columns) * tile_width + (tile_width - source.width) // 2
            y0 = (index // columns) * (tile_height + label_height)
            y = y0 + (tile_height - source.height) // 2
            sheet.paste(source, (x, y))
        draw.text((index % columns * tile_width + 5, y0 + tile_height + 5), path.name, fill="black")
    sheet.save(output, quality=90)


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    overlay_dir = output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    if not 0.0 <= args.battery_conf <= 1.0 or not 0.0 <= args.drop_conf <= 1.0:
        raise SystemExit("Confidence thresholds must be in [0, 1]")

    model = YOLO(str(model_path))
    inference_conf = min(args.battery_conf, args.drop_conf)
    predictions = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=inference_conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        verbose=False,
        save=False,
    )

    image_records: list[dict] = []
    overlay_paths: list[Path] = []
    totals = {
        "images": 0,
        "battery_detections": 0,
        "drop_zone_detections": 0,
        "pick_candidates": 0,
        "batteries_inside_drop_zone": 0,
    }

    for image_index, result in enumerate(predictions):
        image = result.orig_img.copy()
        height, width = image.shape[:2]
        raw: list[tuple[dict, np.ndarray]] = []

        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            polygons = result.masks.xy
            for class_id, confidence, polygon in zip(classes, confidences, polygons):
                polygon = np.asarray(polygon, dtype=np.float32)
                if len(polygon) < 3:
                    continue
                geometry = polygon_geometry(polygon)
                class_name = str(result.names.get(int(class_id), class_id))
                record = {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    **geometry,
                    "polygon_px": polygon.round(2).tolist(),
                }
                raw.append((record, polygon))

        accepted_drop_polygons = [
            polygon
            for record, polygon in raw
            if record["class_id"] == DROP_ZONE_CLASS and record["confidence"] >= args.drop_conf
        ]

        records: list[dict] = []
        for record, polygon in raw:
            class_id = int(record["class_id"])
            if class_id == BATTERY_CLASS:
                inside = any(
                    center_inside_polygon(record["center_px"], drop_polygon)
                    for drop_polygon in accepted_drop_polygons
                )
                record["inside_drop_zone"] = inside
                if record["confidence"] < args.battery_conf:
                    reason = "low_confidence"
                elif inside:
                    reason = "inside_drop_zone"
                else:
                    reason = "candidate"
                record["pick_reason"] = reason
                record["pick_candidate"] = reason == "candidate"
                totals["battery_detections"] += 1
                totals["pick_candidates"] += int(record["pick_candidate"])
                totals["batteries_inside_drop_zone"] += int(inside)
            elif class_id == DROP_ZONE_CLASS:
                record["accepted"] = record["confidence"] >= args.drop_conf
                totals["drop_zone_detections"] += 1
            records.append(record)
            draw_detection(image, record, polygon)

        source_path = Path(str(result.path))
        suffix = source_path.suffix if source_path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
        output_name = f"{source_path.stem}_{image_index:04d}_pred{suffix}"
        overlay_path = overlay_dir / output_name
        cv2.imwrite(str(overlay_path), image)
        overlay_paths.append(overlay_path)
        image_records.append(
            {
                "source": str(source_path),
                "width": width,
                "height": height,
                "detections": records,
            }
        )
        totals["images"] += 1

    report = {
        "model": str(model_path),
        "source": str(source),
        "settings": {
            "imgsz": args.imgsz,
            "device": args.device,
            "battery_conf": args.battery_conf,
            "drop_conf": args.drop_conf,
            "iou": args.iou,
        },
        "totals": totals,
        "images": image_records,
        "coordinate_convention": {
            "center_px": "image coordinates, origin at top-left, x right, y down",
            "yaw_image_deg_clockwise": "undirected long-axis angle in [-90, 90); positive appears clockwise",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "unseen_eval_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    make_contact_sheet(overlay_paths, output / "overlay_contact_sheet.jpg")
    print(json.dumps(totals, indent=2))
    print(f"Report: {output / 'unseen_eval_report.json'}")
    print(f"Contact sheet: {output / 'overlay_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
