"""Aggregate full external-YOLO AAA pick/place validation trials."""

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


def _xy_error_mm(estimate, truth):
    if estimate is None or truth is None or len(estimate) < 2 or len(truth) < 2:
        return None
    return 1000.0 * math.hypot(
        float(estimate[0]) - float(truth[0]),
        float(estimate[1]) - float(truth[1]),
    )


def _axis_error_deg(estimate, truth):
    if estimate is None or truth is None or len(estimate) < 2 or len(truth) < 2:
        return None
    en = math.hypot(float(estimate[0]), float(estimate[1]))
    tn = math.hypot(float(truth[0]), float(truth[1]))
    if en < 1e-12 or tn < 1e-12:
        return None
    dot = abs(
        (float(estimate[0]) * float(truth[0])
         + float(estimate[1]) * float(truth[1]))
        / (en * tn)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _failure_stage(report):
    if report.get("task_complete"):
        return "success"
    if report.get("safely_abstained"):
        return "safe_abstention"
    grasp = report.get("grasp") or {}
    placement = report.get("placement") or {}
    if not grasp.get("pickup_verified"):
        return "pickup_failed"
    if not placement.get("carried_to_detected_bowl"):
        return "carry_failed"
    if not placement.get("inside_bowl"):
        return "placement_failed"
    return "incomplete"


def summarize(reports, expected_trials, min_success_rate=0.90):
    rows = []
    for path, report in reports:
        scene = report.get("requested_scene") or {}
        perception = report.get("perception") or {}
        battery = perception.get("selected_battery") or {}
        battery_pose = battery.get("pose") or {}
        zone = perception.get("selected_drop_zone") or {}
        wrist = perception.get("wrist_reobservation") or {}
        evaluation = report.get("evaluation_only") or {}
        grasp = report.get("grasp") or {}
        placement = report.get("placement") or {}
        source = battery.get("source_camera")
        if source is None and battery:
            source = "table"
        control_source = str(report.get("control_source") or "")
        row = {
            "trial": path.parent.name,
            "brand": scene.get("battery_brand"),
            "visual_roll_deg": scene.get("battery_visual_roll_deg"),
            "requested_x_m": scene.get("battery_x_m"),
            "requested_y_m": scene.get("battery_y_m"),
            "requested_yaw_deg": scene.get("battery_yaw_deg"),
            "task_complete": bool(report.get("task_complete")),
            "safely_abstained": bool(report.get("safely_abstained")),
            "failure_stage": _failure_stage(report),
            "failure_reason": report.get("reason"),
            "vision_only_control": bool(
                control_source.startswith(
                    "external_yolo_instance_masks_plus_calibrated_geometry"
                )
                and report.get("simulator_pose_used_for_control") is False
            ),
            "battery_source_camera": source,
            "battery_confidence": battery.get("confidence"),
            "wrist_triggered": bool(wrist.get("triggered")),
            "wrist_selected_scan": wrist.get("selected_scan"),
            "battery_center_error_mm": _xy_error_mm(
                battery_pose.get("center_xy_m"),
                evaluation.get("initial_position_world_m"),
            ),
            "battery_yaw_error_deg": _axis_error_deg(
                battery_pose.get("axis_xy"),
                evaluation.get("initial_axis_world"),
            ),
            "bowl_center_error_mm": _xy_error_mm(
                zone.get("center_xy_m"), evaluation.get("truth_bowl_xy_m")
            ),
            "pickup_verified": bool(grasp.get("pickup_verified")),
            "lift_mm": (
                None
                if grasp.get("lift_delta_m") is None
                else 1000.0 * float(grasp["lift_delta_m"])
            ),
            "carried_to_bowl": bool(placement.get("carried_to_detected_bowl")),
            "inside_bowl": bool(placement.get("inside_bowl")),
            "final_bowl_center_error_mm": (
                None
                if placement.get(
                    "final_center_distance_to_truth_bowl_m_evaluation_only"
                )
                is None
                else 1000.0
                * float(
                    placement[
                        "final_center_distance_to_truth_bowl_m_evaluation_only"
                    ]
                )
            ),
        }
        rows.append(row)

    successes = sum(row["task_complete"] for row in rows)
    success_rate = successes / expected_trials if expected_trials else 0.0
    brands = {}
    for brand in sorted({row["brand"] for row in rows if row["brand"]}):
        selected = [row for row in rows if row["brand"] == brand]
        brand_successes = sum(row["task_complete"] for row in selected)
        brands[brand] = {
            "trials": len(selected),
            "successes": brand_successes,
            "success_rate": brand_successes / len(selected),
            "mean_confidence": _stats(
                row["battery_confidence"] for row in selected
            )["mean"],
        }

    rolls = {}
    for roll in sorted(
        {row["visual_roll_deg"] for row in rows if row["visual_roll_deg"] is not None}
    ):
        selected = [row for row in rows if row["visual_roll_deg"] == roll]
        rolls[str(float(roll))] = {
            "trials": len(selected),
            "successes": sum(row["task_complete"] for row in selected),
        }

    reports_complete = len(rows) == expected_trials
    vision_only = bool(rows) and all(row["vision_only_control"] for row in rows)
    brand_coverage = set(brands) == {"amazon", "seven_eleven"}
    gates = {
        "all_reports_present": reports_complete,
        "vision_only_control_all_trials": vision_only,
        "both_brands_present": brand_coverage,
        "success_rate_at_least": {
            "threshold": min_success_rate,
            "actual": success_rate,
            "passed": success_rate >= min_success_rate,
        },
    }
    all_gates = bool(
        reports_complete
        and vision_only
        and brand_coverage
        and gates["success_rate_at_least"]["passed"]
    )
    summary = {
        "task": "aaa_yolo_full_pick_place_validation_sweep",
        "expected_trials": expected_trials,
        "reports_found": len(rows),
        "successes": successes,
        "success_rate": success_rate,
        "safe_abstentions": sum(row["safely_abstained"] for row in rows),
        "pickup_successes": sum(row["pickup_verified"] for row in rows),
        "carry_successes": sum(row["carried_to_bowl"] for row in rows),
        "placement_successes": sum(row["inside_bowl"] for row in rows),
        "table_camera_selections": sum(
            row["battery_source_camera"] == "table" for row in rows
        ),
        "wrist_camera_selections": sum(
            row["battery_source_camera"] == "wrist" for row in rows
        ),
        "brand_breakdown": brands,
        "visual_roll_breakdown": rolls,
        "battery_confidence": _stats(row["battery_confidence"] for row in rows),
        "battery_center_error_mm": _stats(
            row["battery_center_error_mm"] for row in rows
        ),
        "battery_yaw_error_deg": _stats(
            row["battery_yaw_error_deg"] for row in rows
        ),
        "bowl_center_error_mm": _stats(row["bowl_center_error_mm"] for row in rows),
        "final_bowl_center_error_mm": _stats(
            row["final_bowl_center_error_mm"] for row in rows
        ),
        "quality_gates": gates,
        "all_quality_gates_passed": all_gates,
        "failed_trials": [
            {
                "trial": row["trial"],
                "brand": row["brand"],
                "failure_stage": row["failure_stage"],
                "reason": row["failure_reason"],
            }
            for row in rows
            if not row["task_complete"]
        ],
        "trials": rows,
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    paths = sorted(root.glob("trial_*/battery_yolo_pick_place_report.json"))
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    summary, rows = summarize(reports, args.expected, args.min_success_rate)

    json_path = root / "pick_place_sweep_report.json"
    csv_path = root / "pick_place_sweep_trials.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["trial", "task_complete"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[pick-place-sweep] success={summary['successes']}/{args.expected} "
        f"quality_gates={summary['all_quality_gates_passed']} report={json_path}",
        flush=True,
    )
    if not summary["all_quality_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
