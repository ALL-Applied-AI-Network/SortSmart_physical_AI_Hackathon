#!/usr/bin/env python3
"""Extract a compact, episode-safe annotation batch from two LeRobot recordings."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = {
    "old": {
        "episodes": range(3, 10),
        "target_z": -220.0,
        "splits": {3: "train", 4: "train", 5: "train", 6: "train", 7: "train", 8: "val", 9: "test"},
    },
    "new": {
        "episodes": range(1, 5),
        "target_z": -245.0,
        "splits": {1: "train", 2: "train", 3: "train", 4: "val"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-zip", type=Path, required=True)
    parser.add_argument("--new-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unzip_dataset(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    roots = [p.parent for p in destination.rglob("manipbench.json") if "__MACOSX" not in p.parts]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one dataset root in {archive}, found {roots}")
    return roots[0]


def stable_low_picks(parquet: Path, target_z: float) -> tuple[pd.DataFrame, list[int]]:
    frame = pd.read_parquet(parquet)
    poses = np.asarray(frame["observation.tcp_pose"].tolist(), dtype=float)
    low = np.isclose(poses[:, 2], target_z, atol=0.25)
    stable = low.copy()
    stable[1:] &= np.linalg.norm(poses[1:, :3] - poses[:-1, :3], axis=1) < 0.05
    indices = np.flatnonzero(stable)
    runs: list[tuple[int, int]] = []
    if len(indices):
        start = previous = int(indices[0])
        for value in map(int, indices[1:]):
            if value != previous + 1:
                if previous - start + 1 >= 3:
                    runs.append((start, previous))
                start = value
            previous = value
        if previous - start + 1 >= 3:
            runs.append((start, previous))
    return frame, [(start + end) // 2 for start, end in runs]


def extract_frame(video: Path, frame_index: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-vsync",
        "vfr",
        "-frames:v",
        "1",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    archives = {"old": args.old_zip.resolve(), "new": args.new_zip.resolve()}
    with tempfile.TemporaryDirectory(prefix="aaa_robot_batch_") as tmp:
        temp = Path(tmp)
        roots = {name: unzip_dataset(path, temp / name) for name, path in archives.items()}
        for source_name, config in SOURCES.items():
            root = roots[source_name]
            for episode in config["episodes"]:
                parquet = root / "data" / "chunk-000" / f"file-{episode:03d}.parquet"
                frame, wrist_picks = stable_low_picks(parquet, float(config["target_z"]))
                if not wrist_picks:
                    raise RuntimeError(f"No stable low frames: {source_name} episode {episode}")

                split = config["splits"][episode]
                selections = [("wrist", value) for value in wrist_picks]
                scene_frame = wrist_picks[len(wrist_picks) // 2]
                selections.extend([("table", scene_frame), ("side", scene_frame)])

                for camera, frame_index in selections:
                    video = (
                        root
                        / "videos"
                        / f"observation.images.{camera}"
                        / "chunk-000"
                        / f"file-{episode:03d}.mp4"
                    )
                    name = f"{source_name}_e{episode:02d}_{camera}_f{frame_index:04d}.png"
                    image_path = output / "images" / split / name
                    extract_frame(video, frame_index, image_path)

                    pose = np.asarray(frame.iloc[frame_index]["observation.tcp_pose"], dtype=float)
                    rows.append(
                        {
                            "image": str(image_path.relative_to(output)),
                            "split": split,
                            "source": source_name,
                            "episode": episode,
                            "camera": camera,
                            "frame_index": frame_index,
                            "timestamp_s": float(frame.iloc[frame_index]["timestamp"]),
                            "tcp_x_mm": float(pose[0]),
                            "tcp_y_mm": float(pose[1]),
                            "tcp_z_mm": float(pose[2]),
                            "label_status": "pending",
                        }
                    )

    for split in ("train", "val", "test"):
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "qa").mkdir()

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, object] = {
        "images": len(rows),
        "by_split": {},
        "by_camera": {},
        "by_source": {},
        "classes": {"0": "aaa_battery", "1": "battery_drop_zone"},
    }
    for key, field in (("by_split", "split"), ("by_camera", "camera"), ("by_source", "source")):
        values: dict[str, int] = {}
        for row in rows:
            value = str(row[field])
            values[value] = values.get(value, 0) + 1
        counts[key] = values
    (output / "batch_report.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")

    guide = """# AAA robot annotation batch

Classes:

- `0 aaa_battery`: Amazon and 7/11 AAA batteries share one class. Annotate each visible battery separately.
- `1 battery_drop_zone`: annotate the usable inner opening of the white/gray collection bowl, not the exterior shell.

Rules:

1. Use visible-instance polygons. For occlusion or image clipping, label only the visible region.
2. A battery inside the bowl remains class 0. Motion logic must suppress picks whose center lies inside class 1.
3. Do not label screws, colored blocks, gripper parts, shadows, or reflections.
4. Keep an empty `.txt` file for a true negative image with neither class visible.
5. All frames from one episode stay in one split to prevent temporal leakage.
6. YOLO segmentation rows use normalized coordinates: `class x1 y1 x2 y2 ...`.

Source policy:

- Old recording: episodes 3-9 at approximately z=-220 mm.
- New recording: episodes 1-4 at z=-245 mm.
- Fault, dry-run, high-only, duplicate, and moving frames are excluded.
"""
    (output / "ANNOTATION_GUIDE.md").write_text(guide, encoding="utf-8")

    dataset_yaml = """path: .
train: images/train
val: images/val
test: images/test
names:
  0: aaa_battery
  1: battery_drop_zone
"""
    (output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    shutil.copy2(Path(__file__).resolve(), output / "prepare_robot_annotation_batch.py")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
