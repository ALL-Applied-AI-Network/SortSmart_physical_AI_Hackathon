#!/usr/bin/env python3
"""Validate YOLO polygon labels and render review overlays."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_label(path: Path) -> tuple[list[list[tuple[float, float]]], list[str]]:
    polygons: list[list[tuple[float, float]]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if fields[0] != "0":
            errors.append(f"line {line_number}: class must be 0")
            continue
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric coordinate")
            continue
        if len(values) < 6 or len(values) % 2:
            errors.append(f"line {line_number}: polygon needs at least 3 points")
            continue
        if any(value < 0 or value > 1 for value in values):
            errors.append(f"line {line_number}: coordinate outside [0, 1]")
            continue
        polygons.append(list(zip(values[0::2], values[1::2])))
    return polygons, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("raw_images"))
    parser.add_argument("--labels", type=Path, default=Path("draft_labels"))
    parser.add_argument("--output", type=Path, default=Path("qa/review_overlays"))
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    report = {"images": 0, "instances": 0, "negative_images": 0, "errors": []}
    overlay_paths: list[Path] = []
    for image_path in sorted(args.images.glob("*.jpg")):
        report["images"] += 1
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            report["errors"].append({"image": image_path.name, "error": "missing label file"})
            continue
        polygons, errors = parse_label(label_path)
        for error in errors:
            report["errors"].append({"image": image_path.name, "error": error})
        if not polygons:
            report["negative_images"] += 1
        report["instances"] += len(polygons)

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for polygon in polygons:
            points = [(x * image.width, y * image.height) for x, y in polygon]
            draw.polygon(points, fill=(0, 255, 0, 70), outline=(0, 255, 0, 255), width=2)
        overlay_path = args.output / image_path.name
        image.save(overlay_path, quality=90)
        overlay_paths.append(overlay_path)

    columns, thumb_w, thumb_h, label_h = 5, 256, 192, 22
    rows = math.ceil(len(overlay_paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(overlay_paths):
        with Image.open(path) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (thumb_w, thumb_h))
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(thumb, (x, y))
        draw.text((x + 3, y + thumb_h + 3), path.stem[:36], fill="black")
    sheet.save(args.output.parent / "review_contact_sheet.jpg", quality=88)

    (args.output.parent / "label_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit("Label validation failed. Fix errors before building the dataset.")


if __name__ == "__main__":
    main()

