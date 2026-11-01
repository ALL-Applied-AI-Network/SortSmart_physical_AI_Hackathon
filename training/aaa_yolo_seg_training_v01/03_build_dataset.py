#!/usr/bin/env python3
"""Build leakage-aware train/validation splits from reviewed labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROBOT_TRAIN_EPISODES = {"episode_000", "episode_001", "episode_002"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("raw_images"))
    parser.add_argument("--labels", type=Path, default=Path("draft_labels"))
    parser.add_argument("--manifest", type=Path, default=Path("source_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    phone_groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["source_type"] == "phone":
            phone_groups.setdefault(row["source_group"], []).append(row)

    split_by_filename: dict[str, str] = {}
    for group_rows in phone_groups.values():
        group_rows.sort(key=lambda row: row["filename"])
        validation_count = max(1, round(len(group_rows) * 0.20))
        for row in group_rows[:-validation_count]:
            split_by_filename[row["filename"]] = "train"
        for row in group_rows[-validation_count:]:
            split_by_filename[row["filename"]] = "val"

    for row in rows:
        if row["source_type"] == "robot_table":
            split_by_filename[row["filename"]] = (
                "train" if row["episode_id"] in ROBOT_TRAIN_EPISODES else "val"
            )

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    split_manifest = []
    for row in rows:
        filename = row["filename"]
        image_path = args.images / filename
        label_path = args.labels / f"{Path(filename).stem}.txt"
        if not image_path.exists() or not label_path.exists():
            raise SystemExit(f"Missing image or reviewed label for {filename}")
        split = split_by_filename[filename]
        shutil.copy2(image_path, args.output / "images" / split / filename)
        shutil.copy2(label_path, args.output / "labels" / split / label_path.name)
        split_manifest.append({**row, "split": split})

    with (args.output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_manifest[0]))
        writer.writeheader()
        writer.writerows(split_manifest)

    counts = {
        split: sum(row["split"] == split for row in split_manifest)
        for split in ("train", "val")
    }
    (args.output / "split_report.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "robot_train_episodes": sorted(ROBOT_TRAIN_EPISODES),
                "robot_validation_episodes": sorted(
                    {
                        row["episode_id"]
                        for row in rows
                        if row["source_type"] == "robot_table"
                        and row["episode_id"] not in ROBOT_TRAIN_EPISODES
                    }
                ),
                "future_test_set": "new physical robot scenes; never use for training",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(counts)


if __name__ == "__main__":
    main()

