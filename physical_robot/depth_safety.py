"""Fail-closed table-depth safety checks for physical FANUC TCP targets.

The geometry module returns points in the robot base frame.  TCP targets use
UF0 millimetres, whose z coordinate is the base z coordinate shifted by
UF0_Z_MM.  This module never sends robot commands; it only evaluates proposed
vertical and horizontal motion against a calibrated table-depth snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.geometry import SURFACE_Z, UF0_Z_MM, depth_to_base, load


@dataclass(frozen=True)
class SafetyConfig:
    workspace_x_m: tuple[float, float] = (0.15, 0.65)
    workspace_y_m: tuple[float, float] = (-0.45, 0.45)
    obstacle_threshold_mm: float = 10.0
    # v04 audits the controlled battery-sorting tabletop, whose external
    # objects are required to be no taller than this.  Taller points are
    # recorded separately because the fixed table camera also sees the FANUC
    # arm itself.  They are not silently treated as ordinary task objects.
    maximum_task_obstacle_height_mm: float = 120.0
    depth_uncertainty_margin_mm: float = 10.0
    obstacle_percentile: float = 99.0
    minimum_local_points: int = 40
    # A stable depth-camera plane offset is corrected online from thousands of
    # dominant table points.  Larger offsets remain fail-closed.
    maximum_table_plane_offset_mm: float = 15.0
    maximum_table_p95_error_mm: float = 10.0


class HeightMapSnapshot:
    """One immutable calibrated depth snapshot and its safety queries."""

    def __init__(
        self,
        points_base_m: np.ndarray,
        valid_depth_fraction: float,
        config: SafetyConfig,
    ):
        self.points = np.asarray(points_base_m, dtype=float)
        self.config = config
        self.valid_depth_fraction = float(valid_depth_fraction)
        raw_height_mm = (self.points[:, 2] - SURFACE_Z) * 1000.0
        table_candidates = raw_height_mm[np.abs(raw_height_mm) <= 20.0]
        if len(table_candidates) < 100:
            raise RuntimeError(
                "too few calibrated table points in snapshot: "
                f"{len(table_candidates)}"
            )
        # Shallow task objects such as a lying AAA battery are only about
        # 10 mm high and therefore fall inside the broad +/-20 mm table band.
        # Do not let those legitimate objects invalidate the camera.  Estimate
        # the dominant table plane first, then evaluate calibration only on
        # its robust inliers.  A real global calibration shift still changes
        # the dominant median and remains fail-closed.
        initial_median = float(np.median(table_candidates))
        residual = np.abs(table_candidates - initial_median)
        robust_sigma = 1.4826 * float(np.median(residual))
        inlier_limit = float(np.clip(3.0 * robust_sigma, 4.0, 8.0))
        table = table_candidates[residual <= inlier_limit]
        if len(table) < 100:
            raise RuntimeError(
                f"too few robust table-plane inliers: {len(table)}"
            )
        self.table_plane_offset_mm = float(np.median(table))
        corrected_table = table - self.table_plane_offset_mm
        absolute = np.abs(corrected_table)
        self.table_candidate_points = int(len(table_candidates))
        self.table_inlier_points = int(len(table))
        self.table_inlier_fraction = float(len(table) / len(table_candidates))
        self.table_inlier_limit_mm = inlier_limit
        self.table_robust_sigma_mm = robust_sigma
        self.table_raw_p95_absolute_error_mm = float(
            np.percentile(np.abs(table), 95.0)
        )
        # Backwards-compatible field name: this is the observed raw plane
        # offset, before online correction.
        self.table_median_error_mm = self.table_plane_offset_mm
        self.table_p95_absolute_error_mm = float(
            np.percentile(absolute, 95.0)
        )
        self.height_mm = raw_height_mm - self.table_plane_offset_mm
        self.calibration_valid = bool(
            self.valid_depth_fraction >= 0.50
            and abs(self.table_plane_offset_mm)
            <= self.config.maximum_table_plane_offset_mm
            and self.table_p95_absolute_error_mm
            <= self.config.maximum_table_p95_error_mm
        )

    def _calibration_report(self) -> dict:
        return {
            "valid_depth_fraction": self.valid_depth_fraction,
            "table_candidate_points": self.table_candidate_points,
            "table_inlier_points": self.table_inlier_points,
            "table_inlier_fraction": self.table_inlier_fraction,
            "table_inlier_limit_mm": self.table_inlier_limit_mm,
            "table_robust_sigma_mm": self.table_robust_sigma_mm,
            "table_plane_offset_mm": self.table_plane_offset_mm,
            "table_median_error_mm": self.table_median_error_mm,
            "table_raw_p95_absolute_error_mm": (
                self.table_raw_p95_absolute_error_mm
            ),
            "table_p95_absolute_error_mm": self.table_p95_absolute_error_mm,
            "online_plane_offset_correction_applied": True,
            "calibration_valid": self.calibration_valid,
        }

    @property
    def table_z_uf0_mm(self) -> float:
        return float(SURFACE_Z * 1000.0 + UF0_Z_MM)

    def _evaluate_mask(
        self,
        mask: np.ndarray,
        target_z_uf0_mm: float,
        minimum_clearance_mm: float,
        query: dict,
        excluded_points_mask: np.ndarray | None = None,
    ) -> dict:
        original_local_count = int(np.count_nonzero(mask))
        excluded_local_count = 0
        if excluded_points_mask is not None:
            excluded = np.asarray(excluded_points_mask, dtype=bool).reshape(-1)
            if len(excluded) != len(self.points):
                raise ValueError("excluded_points_mask length does not match snapshot")
            excluded_local_count = int(np.count_nonzero(mask & excluded))
            mask = mask & ~excluded
        local_count = int(np.count_nonzero(mask))
        if local_count < self.config.minimum_local_points:
            return {
                **query,
                "table_depth_calibration": self._calibration_report(),
                "target_z_uf0_mm": float(target_z_uf0_mm),
                "local_depth_points_before_self_filter": original_local_count,
                "robot_self_points_excluded_locally": excluded_local_count,
                "local_depth_points": local_count,
                "obstacle_points": 0,
                "obstacle_top_height_mm": None,
                "inflated_obstacle_height_mm": None,
                "predicted_tcp_clearance_mm": None,
                "minimum_required_clearance_mm": float(
                    minimum_clearance_mm
                ),
                "passed": False,
                "reason": "insufficient depth coverage",
            }

        local_heights = self.height_mm[mask]
        elevated = local_heights[
            (local_heights >= self.config.obstacle_threshold_mm)
            & (
                local_heights
                <= self.config.maximum_task_obstacle_height_mm
            )
        ]
        above_task_envelope = local_heights[
            local_heights > self.config.maximum_task_obstacle_height_mm
        ]
        if len(elevated):
            obstacle_top_mm = float(
                np.percentile(
                    elevated, self.config.obstacle_percentile
                )
            )
        else:
            obstacle_top_mm = 0.0
        inflated_top_mm = (
            obstacle_top_mm + self.config.depth_uncertainty_margin_mm
        )
        target_height_mm = float(
            target_z_uf0_mm - self.table_z_uf0_mm
        )
        predicted_clearance_mm = target_height_mm - inflated_top_mm
        passed = bool(
            self.calibration_valid
            and predicted_clearance_mm >= minimum_clearance_mm
        )
        if not self.calibration_valid:
            reason = "table-depth calibration gate failed"
        elif not passed:
            reason = "predicted TCP clearance is below the required margin"
        else:
            reason = "clear"
        return {
            **query,
            "table_depth_calibration": self._calibration_report(),
            "target_z_uf0_mm": float(target_z_uf0_mm),
            "local_depth_points_before_self_filter": original_local_count,
            "robot_self_points_excluded_locally": excluded_local_count,
            "target_height_above_table_mm": target_height_mm,
            "local_depth_points": local_count,
            "obstacle_points": int(len(elevated)),
            "points_above_task_envelope": int(len(above_task_envelope)),
            "maximum_task_obstacle_height_mm": (
                self.config.maximum_task_obstacle_height_mm
            ),
            "obstacle_top_height_mm": obstacle_top_mm,
            "depth_uncertainty_margin_mm": (
                self.config.depth_uncertainty_margin_mm
            ),
            "inflated_obstacle_height_mm": inflated_top_mm,
            "predicted_tcp_clearance_mm": predicted_clearance_mm,
            "minimum_required_clearance_mm": float(
                minimum_clearance_mm
            ),
            "passed": passed,
            "reason": reason,
        }

    def evaluate_vertical_target(
        self,
        x_mm: float,
        y_mm: float,
        target_z_uf0_mm: float,
        footprint_radius_mm: float,
        minimum_clearance_mm: float,
        excluded_points_mask: np.ndarray | None = None,
    ) -> dict:
        x_m, y_m = x_mm / 1000.0, y_mm / 1000.0
        distance = np.hypot(
            self.points[:, 0] - x_m, self.points[:, 1] - y_m
        )
        mask = distance <= footprint_radius_mm / 1000.0
        return self._evaluate_mask(
            mask,
            target_z_uf0_mm,
            minimum_clearance_mm,
            {
                "query_type": "vertical_target",
                "target_xy_mm": [float(x_mm), float(y_mm)],
                "footprint_radius_mm": float(footprint_radius_mm),
            },
            excluded_points_mask,
        )

    def evaluate_horizontal_corridor(
        self,
        start_xy_mm: tuple[float, float],
        end_xy_mm: tuple[float, float],
        target_z_uf0_mm: float,
        corridor_radius_mm: float,
        minimum_clearance_mm: float,
        excluded_points_mask: np.ndarray | None = None,
    ) -> dict:
        start = np.asarray(start_xy_mm, dtype=float) / 1000.0
        end = np.asarray(end_xy_mm, dtype=float) / 1000.0
        segment = end - start
        denominator = float(segment @ segment)
        xy = self.points[:, :2]
        if denominator <= 1e-12:
            projection = np.repeat(start[None, :], len(xy), axis=0)
        else:
            fraction = np.clip(
                ((xy - start) @ segment) / denominator, 0.0, 1.0
            )
            projection = start + fraction[:, None] * segment
        distance = np.linalg.norm(xy - projection, axis=1)
        mask = distance <= corridor_radius_mm / 1000.0
        return self._evaluate_mask(
            mask,
            target_z_uf0_mm,
            minimum_clearance_mm,
            {
                "query_type": "horizontal_corridor",
                "start_xy_mm": [float(v) for v in start_xy_mm],
                "end_xy_mm": [float(v) for v in end_xy_mm],
                "corridor_radius_mm": float(corridor_radius_mm),
            },
            excluded_points_mask,
        )


class TableDepthSafetyGuard:
    """Convert one real table-depth frame into a queryable safety snapshot."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._camera = load()["table_depth"]

    def snapshot(self, depth_mm: np.ndarray) -> HeightMapSnapshot:
        depth = np.asarray(depth_mm)
        if depth.shape != (400, 640) or depth.dtype != np.uint16:
            raise RuntimeError(
                "expected table depth as uint16 (400, 640); "
                f"received dtype={depth.dtype}, shape={depth.shape}"
            )
        valid_fraction = float(np.count_nonzero(depth) / depth.size)
        if valid_fraction < 0.50:
            raise RuntimeError(
                f"table depth coverage too low: {valid_fraction:.3f}"
            )
        points = depth_to_base(self._camera, depth, stride=2)
        workspace = (
            (points[:, 0] >= self.config.workspace_x_m[0])
            & (points[:, 0] <= self.config.workspace_x_m[1])
            & (points[:, 1] >= self.config.workspace_y_m[0])
            & (points[:, 1] <= self.config.workspace_y_m[1])
        )
        points = points[workspace]
        if len(points) < 1000:
            raise RuntimeError(
                f"too few depth points in robot workspace: {len(points)}"
            )
        return HeightMapSnapshot(points, valid_fraction, self.config)
