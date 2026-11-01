#!/usr/bin/env python3
"""Create YOLO segmentation draft labels using YOLOE visual prompts.

Both reference boxes use class 0, so Amazon and 7/11 batteries are treated as
the same downstream class: aaa_battery.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("raw_images"))
    parser.add_argument("--labels", type=Path, default=Path("draft_labels_visual"))
    parser.add_argument(
        "--overlays", type=Path, default=Path("qa/visual_prompt_overlays")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("visual_prompt_reference.png")
    )
    parser.add_argument("--model", default="yoloe-26s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="0 processes every image; a positive value selects a reproducible sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-boxes",
        type=float,
        nargs=8,
        metavar=("X1A", "Y1A", "X2A", "Y2A", "X1B", "Y1B", "X2B", "Y2B"),
        default=(580, 368, 830, 442, 573, 456, 855, 540),
        help="Two xyxy boxes in the supplied 1280x960 reference image.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    paths = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"No images found in {directory}")
    return paths


def write_yolo_segments(path: Path, result) -> int:
    lines: list[str] = []
    if result.masks is not None:
        for polygon in result.masks.xyn:
            polygon = np.asarray(polygon, dtype=np.float32)
            if polygon.ndim != 2 or polygon.shape[0] < 3:
                continue
            coords = " ".join(f"{float(value):.6f}" for value in polygon.reshape(-1))
            lines.append(f"0 {coords}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def make_contact_sheets(overlay_paths: list[Path], output_dir: Path) -> list[str]:
    per_page, columns = 30, 5
    cell_w, image_h, caption_h = 320, 230, 34
    saved: list[str] = []
    for page_index, start in enumerate(range(0, len(overlay_paths), per_page), start=1):
        page_paths = overlay_paths[start : start + per_page]
        rows = (len(page_paths) + columns - 1) // columns
        canvas = np.full((rows * (image_h + caption_h), columns * cell_w, 3), 245, np.uint8)
        for index, image_path in enumerate(page_paths):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            scale = min(cell_w / image.shape[1], image_h / image.shape[0])
            resized = cv2.resize(
                image,
                (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            row, column = divmod(index, columns)
            x0 = column * cell_w + (cell_w - resized.shape[1]) // 2
            y0 = row * (image_h + caption_h) + (image_h - resized.shape[0]) // 2
            canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
            caption = image_path.stem[:43]
            cv2.putText(
                canvas,
                caption,
                (column * cell_w + 5, row * (image_h + caption_h) + image_h + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        output_path = output_dir / f"visual_contact_sheet_{page_index:03d}.jpg"
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(str(output_path))
    return saved


def main() -> None:
    args = parse_args()
    if not args.reference.is_file():
        raise SystemExit(f"Reference image not found: {args.reference}")

    image_paths = list_images(args.images)
    if args.sample_size > 0 and args.sample_size < len(image_paths):
        image_paths = sorted(random.Random(args.seed).sample(image_paths, args.sample_size))

    args.labels.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    boxes = np.asarray(args.reference_boxes, dtype=np.float32).reshape(2, 4)
    visual_prompts = {
        "bboxes": boxes,
        "cls": np.zeros(2, dtype=np.int32),
    }

    model = YOLOE(args.model)
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        refer_image=str(args.reference),
        visual_prompts=visual_prompts,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        retina_masks=True,
        stream=True,
        verbose=False,
    )

    records: list[dict] = []
    overlay_paths: list[Path] = []
    total_instances = 0
    images_with_detections = 0

    for image_path, result in zip(image_paths, predictions, strict=True):
        label_path = args.labels / f"{image_path.stem}.txt"
        count = write_yolo_segments(label_path, result)
        total_instances += count
        images_with_detections += int(count > 0)

        overlay_path = args.overlays / f"{image_path.stem}.jpg"
        cv2.imwrite(str(overlay_path), result.plot())
        overlay_paths.append(overlay_path)

        confidences = []
        if result.boxes is not None and result.boxes.conf is not None:
            confidences = [float(value) for value in result.boxes.conf.cpu().tolist()]
        records.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "overlay": str(overlay_path),
                "instances": count,
                "confidences": confidences,
            }
        )

    contact_sheets = make_contact_sheets(overlay_paths, args.overlays.parent)
    report = {
        "mode": "YOLOE visual prompt",
        "model": args.model,
        "reference": str(args.reference),
        "reference_boxes_xyxy": boxes.tolist(),
        "class_names": ["aaa_battery"],
        "confidence_threshold": args.conf,
        "imgsz": args.imgsz,
        "image_count": len(image_paths),
        "images_with_detections": images_with_detections,
        "predicted_instances": total_instances,
        "contact_sheets": contact_sheets,
        "records": records,
    }
    report_path = args.overlays.parent / "visual_prompt_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Processed images: {len(image_paths)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Predicted instances: {total_instances}")
    print(f"Draft labels: {args.labels}")
    print(f"Overlays: {args.overlays}")
    print(f"Report: {report_path}")
    for path in contact_sheets:
        print(f"Contact sheet: {path}")


if __name__ == "__main__":
    main()
