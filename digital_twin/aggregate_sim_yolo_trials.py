"""Aggregate randomized Isaac/external-YOLO perception trials."""

from __future__ import annotations

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


def _stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def summarize(reports, expected_trials, args):
    rows = []
    for path, report in reports:
        scene = report.get("requested_scene") or {}
        battery = report.get("selected_battery") or {}
        zone = report.get("selected_drop_zone") or {}
        evaluation = report.get("evaluation") or {}
        ipc = report.get("ipc") or {}
        rows.append(
            {
                "trial": path.parent.name,
                "requested_x_m": scene.get("battery_x_m"),
                "requested_y_m": scene.get("battery_y_m"),
                "requested_yaw_deg": scene.get("battery_yaw_deg"),
                "battery_brand": scene.get("battery_brand"),
                "battery_visual_roll_deg": scene.get("battery_visual_roll_deg"),
                "battery_detected": bool(battery),
                "drop_zone_detected": bool(zone and zone.get("center_xy_m") is not None),
                "task_complete": bool(report.get("task_complete")),
                "vision_only_control": bool(
                    report.get("control_source")
                    == "external_yolo_instance_masks_plus_calibrated_geometry"
                    and report.get("simulator_pose_used_for_control") is False
                ),
                "battery_confidence": battery.get("confidence"),
                "drop_zone_confidence": zone.get("confidence"),
                "battery_center_error_mm": evaluation.get("battery_center_error_mm"),
                "battery_yaw_error_deg": evaluation.get("battery_yaw_error_deg"),
                "bowl_center_error_mm": evaluation.get("bowl_center_error_mm"),
                "drop_zone_method": zone.get("method"),
                "fitted_bowl_radius_mm": (
                    None
                    if zone.get("fitted_radius_m") is None
                    else 1000.0 * float(zone["fitted_radius_m"])
                ),
                "bowl_fit_rmse_mm": zone.get("fit_rmse_mm"),
                "inference_ms": (
                    None
                    if ipc.get("inference_seconds") is None
                    else 1000.0 * float(ipc["inference_seconds"])
                ),
            }
        )

    success_count = sum(row["task_complete"] for row in rows)
    control_count = sum(row["vision_only_control"] for row in rows)
    success_rate = success_count / expected_trials if expected_trials else 0.0
    battery_error = _stats(row["battery_center_error_mm"] for row in rows)
    yaw_error = _stats(row["battery_yaw_error_deg"] for row in rows)
    bowl_error = _stats(row["bowl_center_error_mm"] for row in rows)
    all_reports = len(rows) == expected_trials
    brand_breakdown = {}
    for brand in sorted({row["battery_brand"] for row in rows if row["battery_brand"]}):
        brand_rows = [row for row in rows if row["battery_brand"] == brand]
        brand_breakdown[brand] = {
            "trials": len(brand_rows),
            "successes": sum(row["task_complete"] for row in brand_rows),
            "detections": sum(row["battery_detected"] for row in brand_rows),
            "confidence": _stats(row["battery_confidence"] for row in brand_rows),
        }

    gates = {
        "all_reports_present": all_reports,
        "vision_only_control_all_trials": bool(rows and control_count == len(rows)),
        "success_rate_at_least": {
            "threshold": args.min_success_rate,
            "actual": success_rate,
            "passed": success_rate >= args.min_success_rate,
        },
        "battery_center_p95_mm_at_most": {
            "threshold": args.max_battery_p95_mm,
            "actual": battery_error["p95"],
            "passed": battery_error["p95"] is not None
            and battery_error["p95"] <= args.max_battery_p95_mm,
        },
        "battery_yaw_p95_deg_at_most": {
            "threshold": args.max_yaw_p95_deg,
            "actual": yaw_error["p95"],
            "passed": yaw_error["p95"] is not None
            and yaw_error["p95"] <= args.max_yaw_p95_deg,
        },
        "bowl_center_p95_mm_at_most": {
            "threshold": args.max_bowl_p95_mm,
            "actual": bowl_error["p95"],
            "passed": bowl_error["p95"] is not None
            and bowl_error["p95"] <= args.max_bowl_p95_mm,
        },
    }
    gate_values = [
        gates["all_reports_present"],
        gates["vision_only_control_all_trials"],
        gates["success_rate_at_least"]["passed"],
        gates["battery_center_p95_mm_at_most"]["passed"],
        gates["battery_yaw_p95_deg_at_most"]["passed"],
        gates["bowl_center_p95_mm_at_most"]["passed"],
    ]

    summary = {
        "task": "aaa_external_yolo_random_pose_sweep",
        "expected_trials": expected_trials,
        "reports_found": len(rows),
        "successes": success_count,
        "success_rate": success_rate,
        "battery_detections": sum(row["battery_detected"] for row in rows),
        "drop_zone_detections": sum(row["drop_zone_detected"] for row in rows),
        "vision_only_control_trials": control_count,
        "brand_breakdown": brand_breakdown,
        "battery_center_error_mm": battery_error,
        "battery_yaw_error_deg": yaw_error,
        "bowl_center_error_mm": bowl_error,
        "battery_confidence": _stats(row["battery_confidence"] for row in rows),
        "drop_zone_confidence": _stats(row["drop_zone_confidence"] for row in rows),
        "inference_ms": _stats(row["inference_ms"] for row in rows),
        "quality_gates": gates,
        "all_quality_gates_passed": all(gate_values),
        "trials": rows,
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--max-battery-p95-mm", type=float, default=5.0)
    parser.add_argument("--max-yaw-p95-deg", type=float, default=5.0)
    parser.add_argument("--max-bowl-p95-mm", type=float, default=10.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = sorted(root.glob("trial_*/sim_yolo_perception_report.json"))
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    summary, rows = summarize(reports, args.expected, args)

    json_path = root / "sim_yolo_sweep_report.json"
    csv_path = root / "sim_yolo_sweep_trials.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["trial", "task_complete"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[sim-yolo-sweep] success={summary['successes']}/{args.expected} "
        f"quality_gates={summary['all_quality_gates_passed']} report={json_path}",
        flush=True,
    )
    if not summary["all_quality_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
