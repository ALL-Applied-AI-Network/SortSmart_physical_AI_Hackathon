"""Aggregate AAA 2D-vision grasp trials into JSON and CSV metrics."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def _percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    index = fraction * (len(values) - 1)
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def _stats(values, scale=1.0):
    values = [float(value) * scale for value in values]
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def summarize(reports, expected_trials=None):
    rows = []
    for path, report in reports:
        perception = report.get("perception") or {}
        dot = perception.get("axis_alignment_abs_dot")
        yaw_error = None
        if dot is not None:
            yaw_error = math.degrees(math.acos(max(0.0, min(1.0, abs(float(dot))))))
        battery = report.get("battery") or {}
        rows.append(
            {
                "trial": path.parent.name,
                "requested_x_m": (battery.get("requested_xy_m") or [None, None])[0],
                "requested_y_m": (battery.get("requested_xy_m") or [None, None])[1],
                "requested_yaw_deg": battery.get("requested_yaw_deg"),
                "confidence": perception.get("confidence"),
                "center_error_mm": (
                    None
                    if perception.get("center_error_m") is None
                    else 1000.0 * float(perception["center_error_m"])
                ),
                "yaw_error_deg": yaw_error,
                "lift_mm": 1000.0 * float(report.get("lift_delta_m", 0.0)),
                "xy_drift_mm": 1000.0 * float(report.get("xy_drift_m", 0.0)),
                "vision_control": bool(
                    report.get("control_source") == "calibrated_2d_rgb"
                    and report.get("simulator_pose_used_for_control") is False
                ),
                "success": bool(report.get("task_complete")),
            }
        )

    successes = sum(row["success"] for row in rows)
    vision_valid = sum(row["vision_control"] for row in rows)
    denominator = expected_trials if expected_trials is not None else len(rows)
    complete = len(rows) == denominator
    result = {
        "task": "aaa_battery_calibrated_2d_sweep",
        "expected_trials": denominator,
        "reports_found": len(rows),
        "all_reports_present": complete,
        "vision_control_trials": vision_valid,
        "successes": successes,
        "success_rate": successes / denominator if denominator else 0.0,
        "all_trials_successful": bool(complete and successes == denominator),
        "center_error_mm": _stats(
            [row["center_error_mm"] for row in rows if row["center_error_mm"] is not None]
        ),
        "yaw_error_deg": _stats(
            [row["yaw_error_deg"] for row in rows if row["yaw_error_deg"] is not None]
        ),
        "confidence": _stats(
            [row["confidence"] for row in rows if row["confidence"] is not None]
        ),
        "lift_mm": _stats([row["lift_mm"] for row in rows]),
        "xy_drift_mm": _stats([row["xy_drift_mm"] for row in rows]),
        "trials": rows,
    }
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    paths = sorted(root.glob("trial_*/battery_grasp_report.json"))
    reports = [(path, json.loads(path.read_text())) for path in paths]
    summary, rows = summarize(reports, expected_trials=args.expected)

    json_path = root / "battery_2d_sweep_report.json"
    csv_path = root / "battery_2d_sweep_trials.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        fieldnames = list(rows[0]) if rows else ["trial", "success"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[sweep] success={summary['successes']}/{summary['expected_trials']} "
        f"reports={summary['reports_found']} report={json_path}",
        flush=True,
    )
    if not summary["all_trials_successful"] or summary["vision_control_trials"] != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
