"""Stationary known-height box diagnostic for the physical FANUC cell.

This policy never plans a new robot pose. It returns the observed joint state,
records the real table RGB/depth streams, converts depth points
to the calibrated robot-base frame, reports table-plane error, and measures a
55--56 mm reference box in a fixed central ROI. Run this diagnostic before
allowing depth or RGB detections to control grasp motion.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, depth_to_base, load, project


class Policy:
    CAMERAS = ["table"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "joints"
    PREFETCH = False
    DEPTH = ["table"]
    DT = 0.2

    OUTPUT_ROOT = Path(__file__).resolve().parent / "table_depth_diagnostics"
    MAX_SAMPLES = 20
    # Robot-base workspace used only for statistics, never for motion.
    WORKSPACE_X_M = (0.15, 0.65)
    WORKSPACE_Y_M = (-0.45, 0.45)

    # Put the flat 55--56 mm reference box in this central robot-base ROI.
    # These bounds include the box placement used in the validated real image,
    # while excluding the robot and the clutter at the table edge.
    BOX_ROI_X_M = (0.24, 0.44)
    BOX_ROI_Y_M = (-0.14, 0.09)
    LOCAL_TABLE_X_M = (0.18, 0.50)
    LOCAL_TABLE_Y_M = (-0.22, 0.17)
    BOX_CANDIDATE_HEIGHT_MM = (35.0, 75.0)
    EXPECTED_BOX_HEIGHT_MM = (55.0, 56.0)
    EXPECTED_HEIGHT_TOLERANCE_MM = 5.0
    MAX_HEIGHT_STD_MM = 2.0
    MAX_TOP_PLANE_RMSE_MM = 3.0

    def __init__(self, checkpoint: str | None = None):
        del checkpoint
        self._cameras = load()
        self._table_depth_camera = self._cameras["table_depth"]
        self._table_rgb_camera = self._cameras["table"]
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
            "[depth] stationary known-height box diagnostic started; "
            f"results: {self._rollout_dir}",
            flush=True,
        )

    def _hold_current_joints(self, obs: dict) -> list[float]:
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        if joints.size < 7 or not np.all(np.isfinite(joints[:7])):
            raise RuntimeError(
                "joint or gripper state missing; refusing to command an action"
            )
        # Reissue exactly the measured joints and measured gripper state. This
        # creates no scan, approach, descent, or gripper transition. It also
        # works with the mock's first observation, where tcp_pose is None until
        # a first TCP command has been received.
        return joints[:7].tolist()

    @staticmethod
    def _depth_preview(depth_mm: np.ndarray) -> np.ndarray:
        valid = depth_mm > 0
        preview = np.zeros(depth_mm.shape, dtype=np.uint8)
        if np.any(valid):
            lo, hi = np.percentile(depth_mm[valid], [2.0, 98.0])
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((depth_mm.astype(float) - lo) / (hi - lo), 0, 1)
            preview[valid] = np.rint(255.0 * (1.0 - scaled[valid])).astype(
                np.uint8
            )
        return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)

    def _registration_overlay(
        self,
        rgb: np.ndarray,
        points_base_m: np.ndarray,
    ) -> np.ndarray:
        """Project sparse depth points into RGB to inspect stream alignment."""

        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if len(points_base_m) == 0:
            return overlay
        points = points_base_m[:: max(1, len(points_base_m) // 12000)]
        uv = project(self._table_rgb_camera, points)
        native_w, native_h = self._table_rgb_camera["size"]
        height, width = rgb.shape[:2]
        finite = np.isfinite(uv).all(axis=1)
        u = np.zeros(len(uv), dtype=int)
        v = np.zeros(len(uv), dtype=int)
        u[finite] = np.rint(uv[finite, 0] * width / native_w).astype(int)
        v[finite] = np.rint(uv[finite, 1] * height / native_h).astype(int)
        inside = finite & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, points = u[inside], v[inside], points[inside]
        height_mm = (points[:, 2] - SURFACE_Z) * 1000.0
        table_like = np.abs(height_mm) <= 5.0
        elevated = height_mm > 5.0
        below = height_mm < -5.0
        overlay[v[table_like], u[table_like]] = (0, 220, 0)
        overlay[v[elevated], u[elevated]] = (0, 0, 255)
        overlay[v[below], u[below]] = (255, 100, 0)
        cv2.putText(
            overlay,
            "green: table +/-5mm | red: above | blue: below",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    @staticmethod
    def _robust_plane(points: np.ndarray) -> tuple[np.ndarray, float, int]:
        """Fit z = ax + by + c with iterative MAD outlier rejection."""

        if len(points) < 100:
            raise RuntimeError(f"too few points for plane fit: {len(points)}")
        design = np.column_stack(
            [points[:, 0], points[:, 1], np.ones(len(points))]
        )
        z = points[:, 2]
        keep = np.ones(len(points), dtype=bool)
        coefficients = np.zeros(3, dtype=float)
        residual_mm = np.zeros(len(points), dtype=float)
        for _ in range(5):
            if np.count_nonzero(keep) < 100:
                raise RuntimeError("plane fit rejected too many points")
            coefficients = np.linalg.lstsq(
                design[keep], z[keep], rcond=None
            )[0]
            residual_mm = (z - design @ coefficients) * 1000.0
            median = float(np.median(residual_mm[keep]))
            mad = float(np.median(np.abs(residual_mm[keep] - median)))
            threshold = max(1.5, 3.0 * 1.4826 * mad)
            keep = np.abs(residual_mm - median) <= threshold
        rmse_mm = float(np.sqrt(np.mean(np.square(residual_mm[keep]))))
        return coefficients, rmse_mm, int(np.count_nonzero(keep))

    def _measure_box_height(self, points: np.ndarray) -> dict:
        """Measure the box top relative to a nearby fitted table plane."""

        height_mm = (points[:, 2] - SURFACE_Z) * 1000.0
        in_box_xy = (
            (points[:, 0] >= self.BOX_ROI_X_M[0])
            & (points[:, 0] <= self.BOX_ROI_X_M[1])
            & (points[:, 1] >= self.BOX_ROI_Y_M[0])
            & (points[:, 1] <= self.BOX_ROI_Y_M[1])
        )
        in_height_band = (
            (height_mm >= self.BOX_CANDIDATE_HEIGHT_MM[0])
            & (height_mm <= self.BOX_CANDIDATE_HEIGHT_MM[1])
        )
        candidates = in_box_xy & in_height_band
        candidate_count = int(np.count_nonzero(candidates))
        if candidate_count < 200:
            return {
                "box_detected": False,
                "box_candidate_points": candidate_count,
                "box_height_mm": None,
                "box_top_plane_rmse_mm": None,
                "local_table_plane_rmse_mm": None,
                "box_geometry_valid": False,
                "box_failure_reason": "too few elevated points in box ROI",
            }

        # Side/edge pixels are lower than the horizontal lid. Retain the upper
        # half, then use robust plane fitting to reject the remaining edges.
        cutoff_mm = float(np.percentile(height_mm[candidates], 50.0))
        top_mask = candidates & (height_mm >= cutoff_mm)

        in_local_xy = (
            (points[:, 0] >= self.LOCAL_TABLE_X_M[0])
            & (points[:, 0] <= self.LOCAL_TABLE_X_M[1])
            & (points[:, 1] >= self.LOCAL_TABLE_Y_M[0])
            & (points[:, 1] <= self.LOCAL_TABLE_Y_M[1])
        )
        # Use a surrounding ring so the box cannot bias the local table fit.
        local_table_mask = (
            in_local_xy
            & ~in_box_xy
            & (np.abs(height_mm) <= 15.0)
        )
        if np.count_nonzero(local_table_mask) < 500:
            return {
                "box_detected": True,
                "box_candidate_points": candidate_count,
                "box_top_points": int(np.count_nonzero(top_mask)),
                "box_height_mm": None,
                "box_top_plane_rmse_mm": None,
                "local_table_plane_rmse_mm": None,
                "box_geometry_valid": False,
                "box_failure_reason": "too few surrounding table points",
            }

        top_plane, top_rmse_mm, top_inliers = self._robust_plane(
            points[top_mask]
        )
        table_plane, table_rmse_mm, table_inliers = self._robust_plane(
            points[local_table_mask]
        )
        center_xy = np.median(points[top_mask, :2], axis=0)
        homogeneous_center = np.array([center_xy[0], center_xy[1], 1.0])
        measured_height_mm = float(
            homogeneous_center @ (top_plane - table_plane) * 1000.0
        )
        expected_low, expected_high = self.EXPECTED_BOX_HEIGHT_MM
        within_expected_tolerance = bool(
            expected_low - self.EXPECTED_HEIGHT_TOLERANCE_MM
            <= measured_height_mm
            <= expected_high + self.EXPECTED_HEIGHT_TOLERANCE_MM
        )
        geometry_valid = bool(
            top_rmse_mm <= self.MAX_TOP_PLANE_RMSE_MM
            and table_rmse_mm <= 4.0
            and within_expected_tolerance
        )
        return {
            "box_detected": True,
            "box_candidate_points": candidate_count,
            "box_top_points": int(np.count_nonzero(top_mask)),
            "box_top_plane_inliers": top_inliers,
            "local_table_plane_points": int(
                np.count_nonzero(local_table_mask)
            ),
            "local_table_plane_inliers": table_inliers,
            "box_center_xy_base_m": center_xy.tolist(),
            "box_height_mm": measured_height_mm,
            "box_top_plane_rmse_mm": top_rmse_mm,
            "local_table_plane_rmse_mm": table_rmse_mm,
            "expected_box_height_mm": list(self.EXPECTED_BOX_HEIGHT_MM),
            "expected_height_tolerance_mm": (
                self.EXPECTED_HEIGHT_TOLERANCE_MM
            ),
            "box_height_within_expected_tolerance": (
                within_expected_tolerance
            ),
            "box_geometry_valid": geometry_valid,
            "box_failure_reason": None if geometry_valid else (
                "height or plane-fit quality gate failed"
            ),
        }

    def _measure(self, obs: dict) -> dict:
        assert self._rollout_dir is not None
        depth = np.asarray(obs.get("depth", {}).get("table"))
        if depth.shape != (400, 640) or depth.dtype != np.uint16:
            raise RuntimeError(
                "expected obs['depth']['table'] as uint16 (400, 640), "
                f"received dtype={depth.dtype}, shape={depth.shape}"
            )
        rgb = np.asarray(obs.get("images", {}).get("table"))
        if rgb.shape != (360, 640, 3):
            raise RuntimeError(
                "expected obs['images']['table'] with shape (360, 640, 3), "
                f"received {rgb.shape}"
            )
        rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)

        valid_fraction = float(np.count_nonzero(depth) / depth.size)
        points = depth_to_base(self._table_depth_camera, depth, stride=2)
        if len(points) == 0:
            raise RuntimeError("table depth contains no valid pixels")

        in_workspace = (
            (points[:, 0] >= self.WORKSPACE_X_M[0])
            & (points[:, 0] <= self.WORKSPACE_X_M[1])
            & (points[:, 1] >= self.WORKSPACE_Y_M[0])
            & (points[:, 1] <= self.WORKSPACE_Y_M[1])
        )
        workspace_points = points[in_workspace]
        residual_mm = (workspace_points[:, 2] - SURFACE_Z) * 1000.0
        # Most workspace pixels are the table. Reject objects and distant
        # background with a broad band, then report robust plane statistics.
        table_band = residual_mm[np.abs(residual_mm) <= 20.0]
        if len(table_band) < 100:
            raise RuntimeError(
                f"too few table-plane points after calibration: {len(table_band)}"
            )
        median_error = float(np.median(table_band))
        absolute_error = np.abs(table_band)
        p05, p95 = np.percentile(table_band, [5.0, 95.0])
        record = {
            "sample": self._sample_number,
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "depth_shape": list(depth.shape),
            "depth_dtype": str(depth.dtype),
            "valid_depth_fraction": valid_fraction,
            "valid_depth_pixels": int(np.count_nonzero(depth)),
            "workspace_points_stride_2": int(len(workspace_points)),
            "table_band_points": int(len(table_band)),
            "expected_surface_z_base_mm": float(SURFACE_Z * 1000.0),
            "median_surface_error_mm": median_error,
            "median_absolute_error_mm": float(np.median(absolute_error)),
            "p95_absolute_error_mm": float(np.percentile(absolute_error, 95.0)),
            "surface_error_p05_mm": float(p05),
            "surface_error_p95_mm": float(p95),
            "within_3mm_fraction": float(np.mean(absolute_error <= 3.0)),
            "within_5mm_fraction": float(np.mean(absolute_error <= 5.0)),
            "calibration_plausible": bool(
                valid_fraction >= 0.50
                and abs(median_error) <= 5.0
                and float(np.percentile(absolute_error, 95.0)) <= 10.0
            ),
            "tcp_pose": np.asarray(obs.get("tcp_pose", [])).tolist(),
            "joints": np.asarray(obs.get("joints", [])).tolist(),
        }
        record.update(self._measure_box_height(points))

        prefix = f"sample_{self._sample_number:03d}"
        if self._sample_number == 0:
            np.savez_compressed(
                self._rollout_dir / "first_table_depth_mm.npz",
                depth_mm=depth,
            )
        cv2.imwrite(
            str(self._rollout_dir / f"{prefix}_table_rgb.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(self._rollout_dir / f"{prefix}_depth_preview.png"),
            self._depth_preview(depth),
        )
        cv2.imwrite(
            str(self._rollout_dir / f"{prefix}_depth_on_rgb.png"),
            self._registration_overlay(rgb, points),
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
        medians = np.asarray(
            [item["median_surface_error_mm"] for item in self._samples]
        )
        measured = [
            item
            for item in self._samples
            if item.get("box_height_mm") is not None
        ]
        heights = np.asarray(
            [item["box_height_mm"] for item in measured], dtype=float
        )
        table_ready = bool(
            len(self._samples) >= 10
            and np.max(np.abs(medians)) <= 5.0
            and all(item["calibration_plausible"] for item in self._samples)
        )
        detection_fraction = float(len(measured) / len(self._samples))
        if len(heights):
            height_median = float(np.median(heights))
            height_mean = float(np.mean(heights))
            height_std = float(np.std(heights))
            height_min = float(np.min(heights))
            height_max = float(np.max(heights))
            height_peak_to_peak = float(np.ptp(heights))
            max_top_rmse = float(
                max(item["box_top_plane_rmse_mm"] for item in measured)
            )
        else:
            height_median = None
            height_mean = None
            height_std = None
            height_min = None
            height_max = None
            height_peak_to_peak = None
            max_top_rmse = None
        expected_low, expected_high = self.EXPECTED_BOX_HEIGHT_MM
        median_height_passes = bool(
            height_median is not None
            and expected_low - self.EXPECTED_HEIGHT_TOLERANCE_MM
            <= height_median
            <= expected_high + self.EXPECTED_HEIGHT_TOLERANCE_MM
        )
        object_height_passed = bool(
            table_ready
            and detection_fraction >= 0.90
            and median_height_passes
            and height_std is not None
            and height_std <= self.MAX_HEIGHT_STD_MM
            and max_top_rmse is not None
            and max_top_rmse <= self.MAX_TOP_PLANE_RMSE_MM
            and all(item["box_geometry_valid"] for item in measured)
        )
        summary = {
            "task": "stationary_known_height_box_diagnostic",
            "robot_motion_commanded": False,
            "depth_requested": ["table"],
            "depth_units": "uint16 millimetres along camera optical axis",
            "zero_means_invalid": True,
            "samples": len(self._samples),
            "expected_surface_z_base_mm": float(SURFACE_Z * 1000.0),
            "median_surface_error_across_samples_mm": float(np.median(medians)),
            "maximum_absolute_sample_median_error_mm": float(
                np.max(np.abs(medians))
            ),
            "mean_valid_depth_fraction": float(
                np.mean([item["valid_depth_fraction"] for item in self._samples])
            ),
            "all_samples_calibration_plausible": bool(
                all(item["calibration_plausible"] for item in self._samples)
            ),
            "acceptance_gate": {
                "minimum_samples": 10,
                "maximum_absolute_median_surface_error_mm": 5.0,
                "all_samples_calibration_plausible": True,
            },
            "ready_for_object_height_validation": table_ready,
            "expected_box_height_mm": list(self.EXPECTED_BOX_HEIGHT_MM),
            "box_roi_x_base_m": list(self.BOX_ROI_X_M),
            "box_roi_y_base_m": list(self.BOX_ROI_Y_M),
            "box_height_frames": len(measured),
            "box_detection_fraction": detection_fraction,
            "box_height_median_mm": height_median,
            "box_height_mean_mm": height_mean,
            "box_height_std_mm": height_std,
            "box_height_min_mm": height_min,
            "box_height_max_mm": height_max,
            "box_height_peak_to_peak_mm": height_peak_to_peak,
            "maximum_top_plane_rmse_mm": max_top_rmse,
            "object_height_acceptance_gate": {
                "minimum_samples": 10,
                "minimum_detection_fraction": 0.90,
                "expected_height_tolerance_mm": (
                    self.EXPECTED_HEIGHT_TOLERANCE_MM
                ),
                "maximum_height_std_mm": self.MAX_HEIGHT_STD_MM,
                "maximum_top_plane_rmse_mm": (
                    self.MAX_TOP_PLANE_RMSE_MM
                ),
            },
            "object_height_validation_passed": object_height_passed,
        }
        encoded = json.dumps(summary, indent=2) + "\n"
        (self._rollout_dir / "box_height_summary.json").write_text(
            encoded, encoding="utf-8"
        )
        # Keep the original filename for scripts written for v02.
        (self._rollout_dir / "table_depth_summary.json").write_text(
            encoded, encoding="utf-8"
        )

    def act(self, obs: dict):
        if self._rollout_dir is None:
            self.reset()
        if self._sample_number < self.MAX_SAMPLES:
            try:
                record = self._measure(obs)
                self._samples.append(record)
                self._sample_number += 1
                self._write_summary()
                box_height = record.get("box_height_mm")
                box_text = (
                    "None"
                    if box_height is None
                    else f"{float(box_height):.2f} mm"
                )
                print(
                    f"[depth] sample {self._sample_number}/{self.MAX_SAMPLES}: "
                    f"valid={record['valid_depth_fraction']:.3f}, "
                    f"table_error={record['median_surface_error_mm']:+.2f} mm, "
                    f"box_height={box_text}, "
                    f"box_valid={record.get('box_geometry_valid')}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[depth] diagnostic sample failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        elif not self._complete_printed:
            print(
                "[depth] collection complete; robot remained at observed joints",
                flush=True,
            )
            self._complete_printed = True
        return self._hold_current_joints(obs)
