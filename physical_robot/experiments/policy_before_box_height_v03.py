"""Stationary table-depth diagnostic for the physical FANUC cell.

This policy never plans a new robot pose. It returns the observed joint state,
records the real table RGB/depth streams, converts depth points
to the calibrated robot-base frame, and reports table-plane error. Run this
diagnostic before allowing depth or RGB detections to control grasp motion.
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
            "[depth] stationary table-depth diagnostic started; "
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
        summary = {
            "task": "stationary_table_depth_diagnostic",
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
            "ready_for_object_height_validation": bool(
                len(self._samples) >= 10
                and np.max(np.abs(medians)) <= 5.0
                and all(item["calibration_plausible"] for item in self._samples)
            ),
        }
        (self._rollout_dir / "table_depth_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
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
                print(
                    f"[depth] sample {self._sample_number}/{self.MAX_SAMPLES}: "
                    f"valid={record['valid_depth_fraction']:.3f}, "
                    f"table_error={record['median_surface_error_mm']:+.2f} mm, "
                    f"plausible={record['calibration_plausible']}",
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
