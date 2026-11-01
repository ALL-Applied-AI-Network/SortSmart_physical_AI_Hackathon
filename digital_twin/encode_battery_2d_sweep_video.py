"""Combine successful calibrated-2D AAA trials into one annotated MP4.

This is a host-side utility.  Run it with the project venv, not as an Isaac
simulation script.  Each trial contributes its perception overlay followed by
the recorded approach, side grasp, lift, and hold trajectory.
"""

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


PHASES = (
    (36, "approach"),
    (24, "descend"),
    (18, "close"),
    (15, "dwell"),
    (30, "lift"),
    (30, "hold"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="sweep output directory")
    parser.add_argument("--out", default=None, help="output MP4 path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--intro-seconds", type=float, default=1.25)
    parser.add_argument("--summary-seconds", type=float, default=2.0)
    return parser.parse_args()


def require_even_positive(name, value):
    if value <= 0 or value % 2:
        raise SystemExit(f"{name} must be a positive even integer")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fit_frame(image, width, height):
    if image is None:
        raise ValueError("cannot resize an empty image")
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(
        image,
        (max(1, round(source_w * scale)), max(1, round(source_h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def put_lines(image, lines, *, top=18, scale=0.62, background=True):
    line_height = max(22, round(34 * scale))
    if background:
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (image.shape[1], top + line_height * len(lines) + 10),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)
    for index, line in enumerate(lines):
        y = top + line_height * (index + 1)
        cv2.putText(
            image,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return image


def phase_for_frame(index):
    cursor = 0
    for count, name in PHASES:
        cursor += count
        if index < cursor:
            return name
    return "hold"


def yaw_error_deg(report):
    alignment = report.get("perception", {}).get("axis_alignment_abs_dot")
    if alignment is None:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, float(alignment)))))


def numbered_frames(trial_dir):
    entries = []
    for filename in glob.glob(str(trial_dir / "frames" / "*.png")):
        match = re.search(r"(\d+)$", Path(filename).stem)
        number = int(match.group(1)) if match else len(entries)
        entries.append((number, filename))
    return [filename for _, filename in sorted(entries)]


def main():
    args = parse_args()
    require_even_positive("--width", args.width)
    require_even_positive("--height", args.height)
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    root = Path(args.root).resolve()
    sweep_path = root / "battery_2d_sweep_report.json"
    if not sweep_path.is_file():
        raise SystemExit(f"missing sweep report: {sweep_path}")
    sweep = load_json(sweep_path)
    if not sweep.get("all_trials_successful"):
        raise SystemExit("refusing to label the video successful: sweep report failed")

    trial_dirs = sorted(path for path in root.glob("trial_*") if path.is_dir())
    expected = int(sweep.get("expected_trials", 0))
    if not trial_dirs or len(trial_dirs) != expected:
        raise SystemExit(
            f"expected {expected} trial directories, found {len(trial_dirs)}"
        )

    output = Path(args.out).resolve() if args.out else root / "battery_2d_sweep.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (args.width, args.height),
        fps=args.fps,
        quality=8,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)
    frames_written = 0

    def write_bgr(image):
        nonlocal frames_written
        writer.send(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes())
        frames_written += 1

    try:
        for trial_index, trial_dir in enumerate(trial_dirs, start=1):
            report_path = trial_dir / "battery_grasp_report.json"
            overlay_path = trial_dir / "battery_perception_overlay.png"
            if not report_path.is_file() or not overlay_path.is_file():
                raise SystemExit(f"trial media/report missing: {trial_dir}")
            report = load_json(report_path)
            if not report.get("task_complete"):
                raise SystemExit(f"trial is not successful: {trial_dir.name}")
            if report.get("control_source") != "calibrated_2d_rgb":
                raise SystemExit(f"trial was not controlled by calibrated 2D RGB: {trial_dir.name}")

            motion_files = numbered_frames(trial_dir)
            if not motion_files:
                raise SystemExit(f"no trajectory PNGs: {trial_dir / 'frames'}")

            battery = report["battery"]
            perception = report.get("perception") or {}
            requested_xy = battery["requested_xy_m"]
            confidence = float(perception.get("confidence", float("nan")))
            center_error_mm = float(perception.get("center_error_m", float("nan"))) * 1000
            yaw_error = yaw_error_deg(report)
            heading = (
                f"Trial {trial_index}/{len(trial_dirs)} | "
                f"x={requested_xy[0]:.2f} m  y={requested_xy[1]:.2f} m  "
                f"yaw={battery['requested_yaw_deg']:.0f} deg"
            )
            metrics = (
                f"2D vision | confidence={confidence:.2f}  "
                f"center error={center_error_mm:.2f} mm  yaw error={yaw_error:.2f} deg"
            )

            overlay = fit_frame(cv2.imread(str(overlay_path)), args.width, args.height)
            put_lines(overlay, [heading, metrics, "Detection -> side grasp -> lift -> hold"])
            for _ in range(max(1, round(args.intro_seconds * args.fps))):
                write_bgr(overlay)

            for frame_index, frame_path in enumerate(motion_files):
                frame = fit_frame(cv2.imread(frame_path), args.width, args.height)
                phase = phase_for_frame(frame_index)
                put_lines(frame, [heading, f"Phase: {phase} | vision-controlled grasp"], scale=0.58)
                write_bgr(frame)

        summary = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        center = sweep.get("center_error_mm", {})
        yaw = sweep.get("yaw_error_deg", {})
        drift = sweep.get("xy_drift_mm", {})
        lines = [
            "AAA calibrated 2D vision sweep complete",
            f"Successful grasps: {sweep.get('successes')}/{sweep.get('expected_trials')}",
            f"Mean center error: {center.get('mean', float('nan')):.2f} mm",
            f"Mean yaw error: {yaw.get('mean', float('nan')):.2f} deg",
            f"Maximum XY drift after lift: {drift.get('max', float('nan')):.2f} mm",
            "Simulation result; physical FANUC validation remains required.",
        ]
        put_lines(summary, lines, top=80, scale=0.72, background=False)
        for _ in range(max(1, round(args.summary_seconds * args.fps))):
            write_bgr(summary)
    finally:
        writer.close()

    size_mb = output.stat().st_size / 1_000_000
    print(
        f"wrote {output} ({frames_written} frames, "
        f"{frames_written / args.fps:.1f} s, {size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
