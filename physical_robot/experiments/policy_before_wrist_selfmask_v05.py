"""Stationary height-map safety audit for the physical FANUC scan route.

This v04 policy does not move the robot.  It holds the observed joints and
gripper while table depth audits every proposed low vertical target and every
high horizontal corridor in the safer future scan plan.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from depth_safety import TableDepthSafetyGuard


class Policy:
    CAMERAS = ["table"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "joints"
    PREFETCH = False
    DEPTH = ["table"]
    DT = 0.2

    OUTPUT_ROOT = Path(__file__).resolve().parent / "heightmap_safety_audits"
    MAX_SAMPLES = 20

    TABLE_Z_MM = -335.1
    SCAN_Z_MM = -235.0
    TRANSIT_Z_MM = -165.0
    VERTICAL_FOOTPRINT_RADIUS_MM = 60.0
    VERTICAL_MINIMUM_CLEARANCE_MM = 25.0
    TRANSIT_CORRIDOR_RADIUS_MM = 70.0
    TRANSIT_MINIMUM_CLEARANCE_MM = 50.0

    # This is the earlier bounded scan footprint.  A motion-enabled successor
    # will rise vertically, move between these points only at TRANSIT_Z_MM,
    # and descend only after the corresponding vertical check passes.
    SCAN_WAYPOINTS = [
        ("center_start", 450.0, 0.0),
        ("left_back", 395.0, -30.0),
        ("center_back", 450.0, -30.0),
        ("right_back", 505.0, -30.0),
        ("right_front", 505.0, 30.0),
        ("center_front", 450.0, 30.0),
        ("left_front", 395.0, 30.0),
        ("center_finish", 450.0, 0.0),
    ]

    def __init__(self, checkpoint: str | None = None):
        del checkpoint
        self._guard = TableDepthSafetyGuard()
        self._rollout_number = 0
        self._rollout_dir: Path | None = None
        self._sample_number = 0
        self._samples: list[dict] = []
        self._started = 0.0
        self._complete_printed = False

    def reset(self) -> None:
        self._rollout_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rollout_dir = self.OUTPUT_ROOT / (
            f"rollout_{self._rollout_number:02d}_{timestamp}"
        )
        self._rollout_dir.mkdir(parents=True, exist_ok=False)
        self._sample_number = 0
        self._samples = []
        self._started = time.monotonic()
        self._complete_printed = False
        print(
            "[heightmap] stationary route safety audit started; "
            f"results: {self._rollout_dir}",
            flush=True,
        )

    @staticmethod
    def _hold_current_joints(obs: dict) -> list[float]:
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        if joints.size < 7 or not np.all(np.isfinite(joints[:7])):
            raise RuntimeError(
                "joint or gripper state missing; refusing to command an action"
            )
        return joints[:7].tolist()

    @staticmethod
    def _observed_tcp_xy(obs: dict) -> tuple[float, float] | None:
        pose = np.asarray(obs.get("tcp_pose", []), dtype=float).reshape(-1)
        if pose.size < 2 or not np.all(np.isfinite(pose[:2])):
            return None
        return float(pose[0]), float(pose[1])

    def _audit(self, obs: dict) -> dict:
        assert self._rollout_dir is not None
        depth = np.asarray(obs.get("depth", {}).get("table"))
        snapshot = self._guard.snapshot(depth)

        vertical_checks = []
        for name, x_mm, y_mm in self.SCAN_WAYPOINTS:
            check = snapshot.evaluate_vertical_target(
                x_mm=x_mm,
                y_mm=y_mm,
                target_z_uf0_mm=self.SCAN_Z_MM,
                footprint_radius_mm=self.VERTICAL_FOOTPRINT_RADIUS_MM,
                minimum_clearance_mm=self.VERTICAL_MINIMUM_CLEARANCE_MM,
            )
            check["waypoint"] = name
            vertical_checks.append(check)

        corridor_checks = []
        previous = self._observed_tcp_xy(obs)
        initial_tcp_available = previous is not None
        if previous is None:
            previous = (
                self.SCAN_WAYPOINTS[0][1],
                self.SCAN_WAYPOINTS[0][2],
            )
        for name, x_mm, y_mm in self.SCAN_WAYPOINTS:
            current = (x_mm, y_mm)
            check = snapshot.evaluate_horizontal_corridor(
                start_xy_mm=previous,
                end_xy_mm=current,
                target_z_uf0_mm=self.TRANSIT_Z_MM,
                corridor_radius_mm=self.TRANSIT_CORRIDOR_RADIUS_MM,
                minimum_clearance_mm=self.TRANSIT_MINIMUM_CLEARANCE_MM,
            )
            check["destination_waypoint"] = name
            corridor_checks.append(check)
            previous = current

        all_checks = [*vertical_checks, *corridor_checks]
        vertical_clearances = [
            item["predicted_tcp_clearance_mm"]
            for item in vertical_checks
            if item["predicted_tcp_clearance_mm"] is not None
        ]
        corridor_clearances = [
            item["predicted_tcp_clearance_mm"]
            for item in corridor_checks
            if item["predicted_tcp_clearance_mm"] is not None
        ]
        record = {
            "sample": self._sample_number,
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "robot_motion_commanded": False,
            "initial_tcp_available": initial_tcp_available,
            "observed_tcp_pose": np.asarray(
                obs.get("tcp_pose", [])
            ).tolist(),
            "valid_depth_fraction": snapshot.valid_depth_fraction,
            "table_median_error_mm": snapshot.table_median_error_mm,
            "table_p95_absolute_error_mm": (
                snapshot.table_p95_absolute_error_mm
            ),
            "calibration_valid": snapshot.calibration_valid,
            "vertical_checks": vertical_checks,
            "corridor_checks": corridor_checks,
            "minimum_vertical_clearance_mm": (
                float(min(vertical_clearances))
                if vertical_clearances
                else None
            ),
            "minimum_corridor_clearance_mm": (
                float(min(corridor_clearances))
                if corridor_clearances
                else None
            ),
            "all_route_checks_passed": bool(
                snapshot.calibration_valid
                and all(item["passed"] for item in all_checks)
            ),
        }
        if self._sample_number == 0:
            np.savez_compressed(
                self._rollout_dir / "first_table_depth_mm.npz",
                depth_mm=depth,
            )
            (self._rollout_dir / "first_route_checks.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
        with (self._rollout_dir / "samples.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def _write_summary(self) -> None:
        assert self._rollout_dir is not None
        if not self._samples:
            return
        vertical = [
            item["minimum_vertical_clearance_mm"]
            for item in self._samples
            if item["minimum_vertical_clearance_mm"] is not None
        ]
        corridors = [
            item["minimum_corridor_clearance_mm"]
            for item in self._samples
            if item["minimum_corridor_clearance_mm"] is not None
        ]
        passed_frames = sum(
            item["all_route_checks_passed"] for item in self._samples
        )
        all_queries = [
            query
            for item in self._samples
            for query in [
                *item["vertical_checks"],
                *item["corridor_checks"],
            ]
        ]
        high_point_counts = [
            int(query.get("points_above_task_envelope", 0))
            for query in all_queries
        ]
        enough_samples = len(self._samples) >= 10
        tabletop_audit_passed = bool(
            enough_samples
            and passed_frames == len(self._samples)
            and all(item["calibration_valid"] for item in self._samples)
        )
        summary = {
            "task": "stationary_table_depth_route_safety_audit",
            "robot_motion_commanded": False,
            "samples": len(self._samples),
            "depth_requested": ["table"],
            "scan_z_uf0_mm": self.SCAN_Z_MM,
            "transit_z_uf0_mm": self.TRANSIT_Z_MM,
            "depth_uncertainty_margin_mm": (
                self._guard.config.depth_uncertainty_margin_mm
            ),
            "vertical_footprint_radius_mm": (
                self.VERTICAL_FOOTPRINT_RADIUS_MM
            ),
            "vertical_minimum_clearance_mm": (
                self.VERTICAL_MINIMUM_CLEARANCE_MM
            ),
            "transit_corridor_radius_mm": (
                self.TRANSIT_CORRIDOR_RADIUS_MM
            ),
            "transit_minimum_clearance_mm": (
                self.TRANSIT_MINIMUM_CLEARANCE_MM
            ),
            "calibration_valid_frames": int(
                sum(item["calibration_valid"] for item in self._samples)
            ),
            "route_passed_frames": int(passed_frames),
            "route_pass_fraction": float(
                passed_frames / len(self._samples)
            ),
            "maximum_task_obstacle_height_mm": (
                self._guard.config.maximum_task_obstacle_height_mm
            ),
            "queries_with_points_above_task_envelope": int(
                sum(count > 0 for count in high_point_counts)
            ),
            "maximum_points_above_task_envelope_per_query": int(
                max(high_point_counts, default=0)
            ),
            "controlled_scene_requirement": (
                "No external object taller than 120 mm; higher points from "
                "the fixed table view are treated as robot/self or out-of-scope."
            ),
            "minimum_vertical_clearance_across_frames_mm": (
                float(min(vertical)) if vertical else None
            ),
            "maximum_vertical_clearance_across_frames_mm": (
                float(max(vertical)) if vertical else None
            ),
            "minimum_corridor_clearance_across_frames_mm": (
                float(min(corridors)) if corridors else None
            ),
            "maximum_corridor_clearance_across_frames_mm": (
                float(max(corridors)) if corridors else None
            ),
            "tabletop_heightmap_audit_passed": tabletop_audit_passed,
            "heightmap_safety_audit_passed": tabletop_audit_passed,
            "motion_enable_recommended": False,
            "next_stage": (
                "wrist_depth_and_robot_self_mask_validation"
                if tabletop_audit_passed
                else "remain_stationary_and_review_failed_checks"
            ),
        }
        (self._rollout_dir / "heightmap_safety_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    def act(self, obs: dict):
        if self._rollout_dir is None:
            self.reset()
        if self._sample_number < self.MAX_SAMPLES:
            try:
                record = self._audit(obs)
                self._samples.append(record)
                self._sample_number += 1
                self._write_summary()
                print(
                    f"[heightmap] sample {self._sample_number}/"
                    f"{self.MAX_SAMPLES}: "
                    f"vertical_min="
                    f"{record['minimum_vertical_clearance_mm']:.1f} mm, "
                    f"corridor_min="
                    f"{record['minimum_corridor_clearance_mm']:.1f} mm, "
                    f"passed={record['all_route_checks_passed']}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    "[heightmap] safety audit failed closed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        elif not self._complete_printed:
            print(
                "[heightmap] audit complete; robot remained at observed joints",
                flush=True,
            )
            self._complete_printed = True
        return self._hold_current_joints(obs)
