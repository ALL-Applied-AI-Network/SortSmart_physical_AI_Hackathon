"""Stationary robust wrist-depth and table-camera robot-self validation (v06.1).

The wrist depth stream has a stable spatial error band on the real tabletop:
its global median is accurate, while its absolute p95 is about 18 mm.  This
diagnostic therefore validates the measured error envelope, the known-height
box over time, and the table-camera robot-self mask.  It does not command any
motion and it does not silently reinterpret the wrist stream as millimetre-
accurate collision geometry.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration.geometry import SURFACE_Z, depth_to_base, load
from robot_self_filter import classify_robot_self_points


class Policy:
    CAMERAS = ["table", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "joints"
    PREFETCH = False
    DEPTH = ["table", "wrist"]
    DT = 0.2

    OUTPUT_ROOT = Path(__file__).resolve().parent / "wrist_selfmask_diagnostics"
    MAX_SAMPLES = 20
    WORKSPACE_X_M = (0.15, 0.65)
    WORKSPACE_Y_M = (-0.45, 0.45)
    TASK_OBJECT_MAX_HEIGHT_MM = 120.0
    SELF_CAPSULE_RADIUS_M = 0.125
    MINIMUM_SELF_EXPLAINED_FRACTION = 0.95
    BOX_ROI_X_M = (0.24, 0.44)
    BOX_ROI_Y_M = (-0.14, 0.09)
    LOCAL_TABLE_X_M = (0.18, 0.50)
    LOCAL_TABLE_Y_M = (-0.22, 0.17)

    # Gates derived from the v05 real-camera capture.  The 25 mm guard margin
    # is deliberately larger than the accepted 20 mm p95 error envelope.
    WRIST_TABLE_MEDIAN_LIMIT_MM = 2.0
    WRIST_TABLE_P95_LIMIT_MM = 20.0
    WRIST_DEPTH_GUARD_MARGIN_MM = 25.0
    EXPECTED_BOX_HEIGHT_MM = (55.0, 56.0)
    BOX_HEIGHT_ACCEPTANCE_MM = (50.0, 61.0)
    FRAME_BOX_HEIGHT_MM = (48.0, 63.0)
    FRAME_TOP_PLANE_RMSE_LIMIT_MM = 6.0
    TEMPORAL_WINDOW = 5
    MINIMUM_FRAME_PASS_FRACTION = 0.90
    MAXIMUM_HEIGHT_STD_MM = 2.0
    MAXIMUM_TEMPORAL_MEDIAN_DEVIATION_MM = 2.5

    def __init__(self, checkpoint: str | None = None):
        del checkpoint
        cameras = load()
        self._table_camera = cameras["table_depth"]
        self._wrist_camera = cameras["wrist_depth"]
        self._rollout_number = 0
        self._rollout_dir: Path | None = None
        self._sample_number = 0
        self._samples: list[dict] = []
        self._started = 0.0
        self._complete_printed = False
        self._missing_tcp_printed = False

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
        self._missing_tcp_printed = False
        print(
            "[v06.1] stationary robust wrist-depth/self-mask validation started; "
            f"results: {self._rollout_dir}",
            flush=True,
        )

    @staticmethod
    def _hold_current_joints(obs: dict) -> list[float]:
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        if joints.size < 7 or not np.all(np.isfinite(joints[:7])):
            raise RuntimeError("joint state missing; refusing to command action")
        return joints[:7].tolist()

    @staticmethod
    def _has_finite_tcp_pose(obs: dict) -> bool:
        """The local mock has no FK and reports tcp_pose=None for joint holds."""
        pose = np.asarray(obs.get("tcp_pose", []), dtype=float).reshape(-1)
        return bool(pose.size >= 3 and np.all(np.isfinite(pose[:3])))

    @staticmethod
    def _robust_plane(points: np.ndarray) -> tuple[np.ndarray, float, int]:
        if len(points) < 100:
            raise RuntimeError(f"too few points for plane fit: {len(points)}")
        design = np.column_stack(
            [points[:, 0], points[:, 1], np.ones(len(points))]
        )
        z = points[:, 2]
        keep = np.ones(len(points), dtype=bool)
        for _ in range(5):
            coefficients = np.linalg.lstsq(
                design[keep], z[keep], rcond=None
            )[0]
            residual = (z - design @ coefficients) * 1000.0
            median = float(np.median(residual[keep]))
            mad = float(np.median(np.abs(residual[keep] - median)))
            keep = np.abs(residual - median) <= max(
                1.5, 3.0 * 1.4826 * mad
            )
        rmse = float(np.sqrt(np.mean(np.square(residual[keep]))))
        return coefficients, rmse, int(np.count_nonzero(keep))

    def _workspace(self, points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] >= self.WORKSPACE_X_M[0])
            & (points[:, 0] <= self.WORKSPACE_X_M[1])
            & (points[:, 1] >= self.WORKSPACE_Y_M[0])
            & (points[:, 1] <= self.WORKSPACE_Y_M[1])
        )

    def _table_self_mask(self, depth: np.ndarray, obs: dict) -> dict:
        points = depth_to_base(self._table_camera, depth, stride=2)
        points = points[self._workspace(points)]
        height = (points[:, 2] - SURFACE_Z) * 1000.0
        high_points = points[height > self.TASK_OBJECT_MAX_HEIGHT_MM]
        classified = classify_robot_self_points(
            high_points,
            obs.get("joints", []),
            obs.get("tcp_pose", []),
            capsule_radius_m=self.SELF_CAPSULE_RADIUS_M,
        )
        high_count = int(len(high_points))
        explained = int(np.count_nonzero(classified["self_mask"]))
        fraction = float(explained / high_count) if high_count else 1.0
        distances = classified["minimum_distance_m"]
        return {
            "table_workspace_points": int(len(points)),
            "table_points_above_task_envelope": high_count,
            "table_high_points_explained_as_robot_self": explained,
            "table_high_points_unexplained": high_count - explained,
            "table_robot_self_explained_fraction": fraction,
            "robot_self_capsule_radius_mm": self.SELF_CAPSULE_RADIUS_M * 1000.0,
            "robot_self_distance_p95_mm": (
                float(np.percentile(distances, 95.0) * 1000.0)
                if len(distances)
                else None
            ),
            "robot_self_mask_valid": bool(
                fraction >= self.MINIMUM_SELF_EXPLAINED_FRACTION
            ),
        }

    def _measure_wrist_box(self, points: np.ndarray) -> dict:
        height = (points[:, 2] - SURFACE_Z) * 1000.0
        in_box = (
            (points[:, 0] >= self.BOX_ROI_X_M[0])
            & (points[:, 0] <= self.BOX_ROI_X_M[1])
            & (points[:, 1] >= self.BOX_ROI_Y_M[0])
            & (points[:, 1] <= self.BOX_ROI_Y_M[1])
        )
        candidates = in_box & (height >= 35.0) & (height <= 75.0)
        count = int(np.count_nonzero(candidates))
        if count < 200:
            return {
                "wrist_box_detected": False,
                "wrist_box_candidate_points": count,
                "wrist_box_height_mm": None,
                "wrist_box_geometry_valid": False,
                "wrist_box_failure_reason": "too few box points",
            }
        cutoff = float(np.percentile(height[candidates], 50.0))
        top = candidates & (height >= cutoff)
        local = (
            (points[:, 0] >= self.LOCAL_TABLE_X_M[0])
            & (points[:, 0] <= self.LOCAL_TABLE_X_M[1])
            & (points[:, 1] >= self.LOCAL_TABLE_Y_M[0])
            & (points[:, 1] <= self.LOCAL_TABLE_Y_M[1])
        )
        table = local & ~in_box & (np.abs(height) <= 20.0)
        if np.count_nonzero(table) < 200:
            return {
                "wrist_box_detected": True,
                "wrist_box_candidate_points": count,
                "wrist_box_height_mm": None,
                "wrist_box_geometry_valid": False,
                "wrist_box_failure_reason": "too few local table points",
            }
        top_plane, top_rmse, top_inliers = self._robust_plane(points[top])
        table_plane, table_rmse, table_inliers = self._robust_plane(
            points[table]
        )
        center = np.median(points[top, :2], axis=0)
        center_h = np.array([center[0], center[1], 1.0])
        box_height = float(center_h @ (top_plane - table_plane) * 1000.0)
        # The real wrist stream has a repeatable spatial depth bias on the
        # tabletop.  Keep table RMSE as a diagnostic, but judge this known-
        # height target by its height and top-plane quality.  Temporal
        # consistency is enforced in _write_summary before this validation can
        # advance to any motion-enabled stage.
        valid = bool(
            self.FRAME_BOX_HEIGHT_MM[0]
            <= box_height
            <= self.FRAME_BOX_HEIGHT_MM[1]
            and top_rmse <= self.FRAME_TOP_PLANE_RMSE_LIMIT_MM
        )
        return {
            "wrist_box_detected": True,
            "wrist_box_candidate_points": count,
            "wrist_box_top_inliers": top_inliers,
            "wrist_local_table_inliers": table_inliers,
            "wrist_box_center_xy_base_m": center.tolist(),
            "wrist_box_height_mm": box_height,
            "wrist_box_top_plane_rmse_mm": top_rmse,
            "wrist_local_table_plane_rmse_mm": table_rmse,
            "wrist_box_geometry_valid": valid,
            "wrist_box_failure_reason": None if valid else "quality gate failed",
        }

    def _wrist_depth(self, depth: np.ndarray, obs: dict) -> dict:
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        points = depth_to_base(
            self._wrist_camera, depth, stride=2, joints=joints
        )
        workspace_points = points[self._workspace(points)]
        height = (workspace_points[:, 2] - SURFACE_Z) * 1000.0
        table = height[np.abs(height) <= 20.0]
        if len(table) < 100:
            table_median = None
            table_p95 = None
            calibration_valid = False
        else:
            table_median = float(np.median(table))
            table_p95 = float(np.percentile(np.abs(table), 95.0))
            calibration_valid = bool(
                abs(table_median) <= self.WRIST_TABLE_MEDIAN_LIMIT_MM
                and table_p95 <= self.WRIST_TABLE_P95_LIMIT_MM
            )
        result = {
            "wrist_workspace_points": int(len(workspace_points)),
            "wrist_table_points": int(len(table)),
            "wrist_table_median_error_mm": table_median,
            "wrist_table_p95_absolute_error_mm": table_p95,
            "wrist_table_p95_limit_mm": self.WRIST_TABLE_P95_LIMIT_MM,
            "wrist_depth_guard_margin_mm": self.WRIST_DEPTH_GUARD_MARGIN_MM,
            "wrist_calibration_valid": calibration_valid,
        }
        result.update(self._measure_wrist_box(workspace_points))
        return result

    def _measure(self, obs: dict) -> dict:
        assert self._rollout_dir is not None
        table_depth = np.asarray(obs.get("depth", {}).get("table"))
        wrist_depth = np.asarray(obs.get("depth", {}).get("wrist"))
        if table_depth.shape != (400, 640) or table_depth.dtype != np.uint16:
            raise RuntimeError("invalid table depth; expected uint16 (400, 640)")
        if wrist_depth.shape != (480, 848) or wrist_depth.dtype != np.uint16:
            raise RuntimeError("invalid wrist depth; expected uint16 (480, 848)")
        table_valid = float(np.count_nonzero(table_depth) / table_depth.size)
        wrist_valid = float(np.count_nonzero(wrist_depth) / wrist_depth.size)
        record = {
            "sample": self._sample_number,
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "robot_motion_commanded": False,
            "table_valid_depth_fraction": table_valid,
            "wrist_valid_depth_fraction": wrist_valid,
            "tcp_pose": np.asarray(obs.get("tcp_pose", [])).tolist(),
            "joints": np.asarray(obs.get("joints", [])).tolist(),
        }
        record.update(self._table_self_mask(table_depth, obs))
        record.update(self._wrist_depth(wrist_depth, obs))
        record["sample_passed"] = bool(
            record["robot_self_mask_valid"]
            and record["wrist_calibration_valid"]
            and record["wrist_box_geometry_valid"]
        )
        if self._sample_number == 0:
            np.savez_compressed(
                self._rollout_dir / "first_depth_frames_mm.npz",
                table_depth_mm=table_depth,
                wrist_depth_mm=wrist_depth,
            )
            (self._rollout_dir / "first_sample_report.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
        with (self._rollout_dir / "samples.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def _write_summary(self) -> None:
        assert self._rollout_dir is not None
        measured = [
            item for item in self._samples
            if item.get("wrist_box_height_mm") is not None
        ]
        heights = np.asarray(
            [item["wrist_box_height_mm"] for item in measured], dtype=float
        )
        passed = sum(item["sample_passed"] for item in self._samples)
        enough = len(self._samples) >= 10
        detection_fraction = float(len(measured) / len(self._samples))
        frame_pass_fraction = float(passed / len(self._samples))
        height_median = float(np.median(heights)) if len(heights) else None
        height_std = float(np.std(heights)) if len(heights) else None
        rolling_medians = np.asarray(
            [
                np.median(heights[index:index + self.TEMPORAL_WINDOW])
                for index in range(
                    max(0, len(heights) - self.TEMPORAL_WINDOW + 1)
                )
            ],
            dtype=float,
        )
        temporal_max_deviation = (
            float(np.max(np.abs(rolling_medians - height_median)))
            if len(rolling_medians) and height_median is not None
            else None
        )
        wrist_p95_values = [
            item["wrist_table_p95_absolute_error_mm"]
            for item in self._samples
            if item.get("wrist_table_p95_absolute_error_mm") is not None
        ]
        maximum_wrist_p95 = (
            float(max(wrist_p95_values)) if wrist_p95_values else None
        )
        validation_passed = bool(
            enough
            and frame_pass_fraction >= self.MINIMUM_FRAME_PASS_FRACTION
            and detection_fraction >= 0.90
            and height_median is not None
            and self.BOX_HEIGHT_ACCEPTANCE_MM[0]
            <= height_median
            <= self.BOX_HEIGHT_ACCEPTANCE_MM[1]
            and height_std is not None
            and height_std <= self.MAXIMUM_HEIGHT_STD_MM
            and temporal_max_deviation is not None
            and temporal_max_deviation
            <= self.MAXIMUM_TEMPORAL_MEDIAN_DEVIATION_MM
            and maximum_wrist_p95 is not None
            and maximum_wrist_p95 <= self.WRIST_TABLE_P95_LIMIT_MM
        )
        summary = {
            "task": "stationary_robust_wrist_depth_and_robot_self_mask_validation",
            "robot_motion_commanded": False,
            "samples": len(self._samples),
            "depth_requested": ["table", "wrist"],
            "sample_passed_frames": int(passed),
            "sample_pass_fraction": frame_pass_fraction,
            "minimum_table_robot_self_explained_fraction": float(
                min(
                    item["table_robot_self_explained_fraction"]
                    for item in self._samples
                )
            ),
            "maximum_unexplained_table_high_points": int(
                max(item["table_high_points_unexplained"] for item in self._samples)
            ),
            "mean_wrist_valid_depth_fraction": float(
                np.mean([item["wrist_valid_depth_fraction"] for item in self._samples])
            ),
            "wrist_calibration_valid_frames": int(
                sum(item["wrist_calibration_valid"] for item in self._samples)
            ),
            "wrist_box_detection_fraction": detection_fraction,
            "wrist_box_height_median_mm": height_median,
            "wrist_box_height_std_mm": height_std,
            "wrist_box_rolling_median_window": self.TEMPORAL_WINDOW,
            "wrist_box_temporal_max_median_deviation_mm": temporal_max_deviation,
            "maximum_wrist_table_p95_absolute_error_mm": maximum_wrist_p95,
            "wrist_table_p95_acceptance_limit_mm": self.WRIST_TABLE_P95_LIMIT_MM,
            "required_wrist_depth_guard_margin_mm": self.WRIST_DEPTH_GUARD_MARGIN_MM,
            "wrist_depth_and_self_mask_validation_passed": validation_passed,
            "motion_enable_recommended": False,
            "next_stage": (
                "limited_depth_guarded_scan_motion_with_25mm_wrist_margin"
                if validation_passed
                else "remain_stationary_and_review_v06_failures"
            ),
        }
        (self._rollout_dir / "wrist_robust_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    def act(self, obs: dict):
        if self._rollout_dir is None:
            self.reset()
        # A joint-hold policy cannot make the local mock synthesize TCP pose:
        # the mock only learns tcp_pose after receiving a Cartesian command.
        # Never send such a command merely to satisfy a mock.  Pause the depth
        # diagnostic, print once, and continue the API/action compatibility
        # check by returning the measured joint state.  On the real robot a
        # missing TCP likewise prevents validation from passing.
        if not self._has_finite_tcp_pose(obs):
            if not self._missing_tcp_printed:
                print(
                    "[v06.1] tcp_pose unavailable; local mock can only check "
                    "the action interface. Depth validation is paused and "
                    "the observed joints are held.",
                    flush=True,
                )
                self._missing_tcp_printed = True
            return self._hold_current_joints(obs)
        if self._sample_number < self.MAX_SAMPLES:
            try:
                record = self._measure(obs)
                self._samples.append(record)
                self._sample_number += 1
                self._write_summary()
                height = record.get("wrist_box_height_mm")
                height_text = "None" if height is None else f"{height:.2f} mm"
                print(
                    f"[v06.1] sample {self._sample_number}/{self.MAX_SAMPLES}: "
                    f"self={record['table_robot_self_explained_fraction']:.3f}, "
                    f"wrist_box={height_text}, passed={record['sample_passed']}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[v06.1] failed closed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        elif not self._complete_printed:
            print("[v06.1] complete; robot remained stationary", flush=True)
            self._complete_printed = True
        return self._hold_current_joints(obs)
