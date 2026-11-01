#!/usr/bin/env python3
"""Create a flat, traceable image pool from the supplied phone and robot data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import zipfile
from pathlib import Path


def safe_name(prefix: str, original: str) -> str:
    stem = Path(original).stem.replace(" ", "_")
    return f"{prefix}__{stem}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-primary-zip", type=Path, required=True)
    parser.add_argument("--phone-additional-zip", type=Path, required=True)
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("raw_images"))
    parser.add_argument("--manifest", type=Path, default=Path("source_manifest.csv"))
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    records: list[dict] = []
    seen_hashes: set[str] = set()

    for source_group, archive in (
        ("phone_primary", args.phone_primary_zip),
        ("phone_additional", args.phone_additional_zip),
    ):
        with zipfile.ZipFile(archive) as handle:
            members = sorted(
                name
                for name in handle.namelist()
                if name.lower().endswith((".jpg", ".jpeg"))
                and not name.startswith("__MACOSX/")
            )
            for member in members:
                payload = handle.read(member)
                digest = hashlib.sha256(payload).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                filename = safe_name(source_group, member)
                (args.output / filename).write_bytes(payload)
                records.append(
                    {
                        "filename": filename,
                        "source_type": "phone",
                        "source_group": source_group,
                        "episode_id": "",
                        "original_path": member,
                        "sha256": digest,
                    }
                )

    table_images = sorted(args.robot_root.glob("images/episode_*/table/*.jpg"))
    for path in table_images:
        episode_id = path.parents[1].name
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        filename = safe_name(f"robot_table__{episode_id}", path.name)
        (args.output / filename).write_bytes(payload)
        records.append(
            {
                "filename": filename,
                "source_type": "robot_table",
                "source_group": "robot_table",
                "episode_id": episode_id,
                "original_path": str(path),
                "sha256": digest,
            }
        )

    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    counts: dict[str, int] = {}
    for record in records:
        counts[record["source_group"]] = counts.get(record["source_group"], 0) + 1
    print({"total_images": len(records), "counts": counts})


if __name__ == "__main__":
    main()

