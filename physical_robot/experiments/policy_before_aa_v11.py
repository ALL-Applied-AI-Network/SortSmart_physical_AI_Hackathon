"""Physical v10: guarded real-battery pick, carry, and bowl placement.

The v08.3/v09 guarded low observation runs first.  Only after stable battery
and drop-zone detections plus non-contact route checks pass does the policy
execute a bounded pick/carry/place sequence.  It performs no wrist-yaw
rotation: v10 requires the battery to be aligned with the already validated
gripper orientation.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from calibration.geometry import (
    SURFACE_Z,
    UF0_Z_MM,
    depth_to_base,
    flange_pose,
    load,
    pixel_to_base,
    project,
)
from depth_safety import TableDepthSafetyGuard
from robot_self_filter import classify_robot_self_points


BATTERY_CLASS_NAME = "aaa_battery"
DROP_ZONE_CLASS_NAME = "battery_drop_zone"


def _inference_device():
    """Select a local device without making torch a safety dependency."""
    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _polygon_area_centroid(polygon: np.ndarray) -> tuple[float, list[float]]:
    """Shoelace area/centroid for an ordered image polygon."""
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    shifted = np.roll(points, -1, axis=0)
    cross = points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]
    signed_twice_area = float(np.sum(cross))
    area = abs(signed_twice_area) * 0.5
    if abs(signed_twice_area) < 1e-9:
        return area, points.mean(axis=0).tolist()
    center = np.sum((points + shifted) * cross[:, None], axis=0)
    center /= 3.0 * signed_twice_area
    return area, center.tolist()


class Policy:
    CAMERAS = ["table", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = ["table", "wrist"]
    DT = 0.2

    OUTPUT_ROOT = Path(__file__).resolve().parent / "physical_pick_place_v10_runs"
    MODEL_PATH = Path(__file__).resolve().parent / "models" / "aaa_yolo26s_best.pt"
    MODEL_IMAGE_SIZE = 1024
    MODEL_IOU = 0.50
    RAW_BATTERY_CONFIDENCE = 0.25
    ACCEPTED_BATTERY_CONFIDENCE = 0.60
    ACCEPTED_DROP_ZONE_CONFIDENCE = 0.30
    FUSION_MASK_DILATION_PX = {"table": 2, "wrist": 3}
    MINIMUM_FUSED_POINTS = {"table": 8, "wrist": 20}
    MINIMUM_FUSED_FRAMES = 3
    MAXIMUM_CENTER_SPREAD_MM = 20.0
    MINIMUM_DROP_ZONE_FRAMES = 3
    MAXIMUM_DROP_ZONE_SPREAD_MM = 25.0
    MINIMUM_PICK_TO_DROP_SEPARATION_MM = 120.0
    TABLE_Z_MM = -335.1
    DESCENT_Z_MM = (-150.0, -190.0, -220.0, -235.0)
    PREFLIGHT_REQUIRED = 5
    PREFLIGHT_MAXIMUM = 10
    LOW_OBSERVATIONS = 10
    STEP_SETTLE_REQUIRED = 3
    STEP_SETTLE_MAXIMUM = 5
    VERTICAL_FOOTPRINT_RADIUS_MM = 75.0
    MINIMUM_TCP_CLEARANCE_MM = 30.0
    # v07.2/v08 real data showed a settled preflight drift of roughly
    # -2.2..-3.3 mm while wrist p95 remained below 20 mm.  Four millimetres is
    # still far inside the independently enforced 25 mm wrist guard margin.
    WRIST_TABLE_MEDIAN_LIMIT_MM = 4.0
    WRIST_TABLE_P95_LIMIT_MM = 20.0
    WRIST_GUARD_MARGIN_MM = 25.0
    MAXIMUM_EXTERNAL_OBJECT_HEIGHT_MM = 80.0
    SELF_CAPSULE_RADIUS_M = 0.125
    MINIMUM_SELF_EXPLAINED_FRACTION = 0.95
    SELF_FILTER_MINIMUM_HEIGHT_MM = 100.0
    TABLE_OFFSET_STABILITY_FRAMES = 3
    MAXIMUM_TABLE_OFFSET_RANGE_MM = 3.0
    MAXIMUM_TABLE_OFFSET_DELTA_MM = 3.0

    START_X_MM = (250.0, 350.0)
    START_Y_MM = (-60.0, 60.0)
    START_Z_MM = (-130.0, -50.0)
    MAXIMUM_REACH_ERROR_MM = 6.0

    # v10 uses the v09-validated high routes. Contact/release values are
    # derived conservatively from measured battery and bowl geometry.
    TRANSIT_Z_MM = -165.0
    PICK_APPROACH_Z_MM = -235.0
    PICK_PREGRASP_Z_MM = -285.0
    DROP_APPROACH_Z_MM = -215.0
    DROP_RELEASE_Z_MM = -245.0
    EXECUTION_ENABLED = True
    MAXIMUM_ALIGNMENT_ERROR_DEG = 10.0
    MINIMUM_CONTACT_HEIGHT_ABOVE_TABLE_MM = 6.5
    MAXIMUM_CONTACT_HEIGHT_ABOVE_TABLE_MM = 8.0
    RELEASE_CLEARANCE_ABOVE_BOWL_MM = 35.0
    MINIMUM_RELEASE_HEIGHT_ABOVE_TABLE_MM = 55.0
    MAXIMUM_RELEASE_HEIGHT_ABOVE_TABLE_MM = 85.0
    GRIPPER_DWELL_FRAMES = 5
    VERIFICATION_FRAMES = 5
    MINIMUM_VERIFICATION_PASSES = 2

    def __init__(self, checkpoint: str | None = None):
        del checkpoint
        cameras = load()
        self._cameras = cameras
        self._table_color_camera = cameras["table"]
        self._wrist_color_camera = cameras["wrist"]
        self._table_camera = cameras["table_depth"]
        self._wrist_camera = cameras["wrist_depth"]
        self._guard = TableDepthSafetyGuard()
        self._rollout_number = 0
        self._rollout_dir: Path | None = None
        self._state = "new"
        self._start_tcp: list[float] | None = None
        self._descent_targets: list[list[float]] = []
        self._descent_index = 0
        self._step_settle_records: list[dict] = []
        self._retreat_targets: list[list[float]] = []
        self._retreat_index = 0
        self._preflight_records: list[dict] = []
        self._low_records: list[dict] = []
        self._events: list[dict] = []
        self._started = 0.0
        self._mock_notice_printed = False
        self._aborted_reason: str | None = None
        self._table_offset_baseline_mm: float | None = None
        self._model = None
        self._model_device = None
        self._model_error: str | None = None
        self._execution_plan: dict | None = None
        self._execution_steps: list[dict] = []
        self._execution_index = 0
        self._execution_dwell = 0
        self._pickup_records: list[dict] = []
        self._placement_records: list[dict] = []
        self._pickup_verified = False
        self._placement_verified = False

    def reset(self) -> None:
        self._rollout_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rollout_dir = self.OUTPUT_ROOT / (
            f"rollout_{self._rollout_number:02d}_{timestamp}"
        )
        self._rollout_dir.mkdir(parents=True, exist_ok=False)
        (self._rollout_dir / "raw").mkdir(exist_ok=True)
        (self._rollout_dir / "overlays").mkdir(exist_ok=True)
        self._state = "bootstrap"
        self._start_tcp = None
        self._descent_targets = []
        self._descent_index = 0
        self._step_settle_records = []
        self._retreat_targets = []
        self._retreat_index = 0
        self._preflight_records = []
        self._low_records = []
        self._events = []
        self._started = time.monotonic()
        self._mock_notice_printed = False
        self._aborted_reason = None
        self._table_offset_baseline_mm = None
        self._execution_plan = None
        self._execution_steps = []
        self._execution_index = 0
        self._execution_dwell = 0
        self._pickup_records = []
        self._placement_records = []
        self._pickup_verified = False
        self._placement_verified = False
        print(
            f"[v10] guarded physical pick/place started; results: {self._rollout_dir}",
            flush=True,
        )

    @staticmethod
    def _finite_tcp(obs: dict) -> np.ndarray | None:
        pose = np.asarray(obs.get("tcp_pose", []), dtype=float).reshape(-1)
        if pose.size < 6 or not np.all(np.isfinite(pose[:6])):
            return None
        return pose[:6]

    @staticmethod
    def _gripper(obs: dict) -> float:
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        if joints.size < 7 or not np.isfinite(joints[6]):
            raise RuntimeError("finite gripper state is required")
        return float(joints[6])

    def _mock_bootstrap_target(self, obs: dict) -> list[float]:
        return [300.0, 0.0, -90.0, -180.0, 0.0, 0.0, self._gripper(obs)]

    def _validate_start(self, pose: np.ndarray) -> None:
        x, y, z, w, p, _r = pose
        if not self.START_X_MM[0] <= x <= self.START_X_MM[1]:
            raise RuntimeError(f"start X outside v07 envelope: {x:.2f} mm")
        if not self.START_Y_MM[0] <= y <= self.START_Y_MM[1]:
            raise RuntimeError(f"start Y outside v07 envelope: {y:.2f} mm")
        if not self.START_Z_MM[0] <= z <= self.START_Z_MM[1]:
            raise RuntimeError(f"start Z outside v07 envelope: {z:.2f} mm")
        if abs(abs(w) - 180.0) > 5.0 or abs(p) > 5.0:
            raise RuntimeError("tool is not in the validated downward orientation")

    def _build_targets(self, pose: np.ndarray, gripper: float) -> None:
        if gripper > 0.25:
            raise RuntimeError("v10 requires the gripper to be open at rollout start")
        base = [float(value) for value in pose]
        self._start_tcp = [*base, float(gripper)]
        self._descent_targets = [
            [base[0], base[1], z, base[3], base[4], base[5], float(gripper)]
            for z in self.DESCENT_Z_MM
        ]
        self._retreat_targets = []

    def _prepare_retreat(self, current_z_mm: float) -> None:
        """Build an upward-only return path from the actually observed Z."""
        assert self._start_tcp is not None
        upward = [
            target.copy()
            for target in reversed(self._descent_targets[:-1])
            if target[2] > current_z_mm + 3.0
        ]
        self._retreat_targets = [*upward, self._start_tcp.copy()]
        self._retreat_index = 0

    def _workspace(self, points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] >= 0.15)
            & (points[:, 0] <= 0.65)
            & (points[:, 1] >= -0.45)
            & (points[:, 1] <= 0.45)
        )

    def _wrist_quality(self, depth: np.ndarray, obs: dict) -> dict:
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        points = depth_to_base(
            self._wrist_camera, depth, stride=2, joints=joints
        )
        points = points[self._workspace(points)]
        height = (points[:, 2] - SURFACE_Z) * 1000.0
        table = height[np.abs(height) <= 20.0]
        if len(table) < 100:
            return {
                "valid": False,
                "table_points": int(len(table)),
                "median_error_mm": None,
                "p95_absolute_error_mm": None,
            }
        median = float(np.median(table))
        p95 = float(np.percentile(np.abs(table), 95.0))
        return {
            "valid": bool(
                abs(median) <= self.WRIST_TABLE_MEDIAN_LIMIT_MM
                and p95 <= self.WRIST_TABLE_P95_LIMIT_MM
            ),
            "table_points": int(len(table)),
            "median_error_mm": median,
            "p95_absolute_error_mm": p95,
            "required_guard_margin_mm": self.WRIST_GUARD_MARGIN_MM,
        }

    def _self_mask_quality(self, snapshot, obs: dict) -> tuple[dict, np.ndarray]:
        # Only points above the controlled task-object envelope are eligible
        # for removal.  Ordinary batteries, the 55 mm reference box and even
        # an unexpected 85 mm obstacle remain in the collision map.
        points = snapshot.points
        high_selector = snapshot.height_mm >= self.SELF_FILTER_MINIMUM_HEIGHT_MM
        high = points[high_selector]
        result = classify_robot_self_points(
            high,
            obs.get("joints", []),
            obs.get("tcp_pose", []),
            capsule_radius_m=self.SELF_CAPSULE_RADIUS_M,
        )
        explained = int(np.count_nonzero(result["self_mask"]))
        fraction = float(explained / len(high)) if len(high) else 1.0
        excluded = np.zeros(len(points), dtype=bool)
        excluded[np.flatnonzero(high_selector)] = result["self_mask"]
        quality = {
            "classification_minimum_height_mm": self.SELF_FILTER_MINIMUM_HEIGHT_MM,
            "high_points": int(len(high)),
            "explained_points": explained,
            "unexplained_points": int(len(high) - explained),
            "explained_fraction": fraction,
            "excluded_points_from_heightmap": int(np.count_nonzero(excluded)),
            "valid": bool(fraction >= self.MINIMUM_SELF_EXPLAINED_FRACTION),
        }
        return quality, excluded

    def _depth_arrays(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        table = np.asarray(obs.get("depth", {}).get("table"))
        wrist = np.asarray(obs.get("depth", {}).get("wrist"))
        if table.shape != (400, 640) or table.dtype != np.uint16:
            raise RuntimeError("invalid table depth; expected uint16 (400, 640)")
        if wrist.shape != (480, 848) or wrist.dtype != np.uint16:
            raise RuntimeError("invalid wrist depth; expected uint16 (480, 848)")
        return table, wrist

    @staticmethod
    def _rgb_array(obs: dict, camera: str) -> np.ndarray:
        rgb = np.asarray(obs.get("images", {}).get(camera))
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"invalid {camera} RGB image: {rgb.shape}")
        return np.clip(rgb, 0, 255).astype(np.uint8, copy=False)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._model_error is not None:
            raise RuntimeError(self._model_error)
        if not self.MODEL_PATH.exists():
            self._model_error = (
                f"battery model not found: {self.MODEL_PATH}; expected "
                "models/aaa_yolo26s_best.pt"
            )
            raise RuntimeError(self._model_error)
        try:
            from ultralytics import YOLO

            self._model_device = _inference_device()
            print(
                f"[v10] loading {self.MODEL_PATH} on {self._model_device}",
                flush=True,
            )
            self._model = YOLO(str(self.MODEL_PATH))
            print("[v10] YOLO loaded", flush=True)
        except Exception as exc:
            self._model_error = f"YOLO load failed: {type(exc).__name__}: {exc}"
            raise RuntimeError(self._model_error) from exc

    def _infer_battery_masks(
        self, camera: str, rgb: np.ndarray, observation_index: int
    ) -> list[dict]:
        """Return battery and table-camera drop-zone instance masks."""
        self._ensure_model()
        assert self._model is not None
        result = self._model.predict(
            source=Image.fromarray(rgb),
            imgsz=self.MODEL_IMAGE_SIZE,
            conf=self.RAW_BATTERY_CONFIDENCE,
            iou=self.MODEL_IOU,
            device=self._model_device,
            retina_masks=True,
            verbose=False,
            save=False,
        )[0]
        detections: list[dict] = []
        if result.boxes is None or result.masks is None:
            return detections
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
        for class_id, confidence, polygon in zip(
            classes, confidences, result.masks.xy
        ):
            class_name = str(result.names.get(int(class_id), class_id))
            if class_name not in {BATTERY_CLASS_NAME, DROP_ZONE_CLASS_NAME}:
                continue
            if class_name == DROP_ZONE_CLASS_NAME and camera != "table":
                continue
            polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            area, center = _polygon_area_centroid(polygon)
            detections.append(
                {
                    "camera": camera,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "accepted": bool(
                        confidence
                        >= (
                            self.ACCEPTED_BATTERY_CONFIDENCE
                            if class_name == BATTERY_CLASS_NAME
                            else self.ACCEPTED_DROP_ZONE_CONFIDENCE
                        )
                    ),
                    "center_px": center,
                    "mask_area_px": area,
                    "polygon_px": polygon.round(2).tolist(),
                    "_polygon": polygon,
                    "observation_index": int(observation_index),
                }
            )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def _drop_zone_geometry(self, detection: dict, rgb: np.ndarray) -> dict:
        """Map the fixed-camera mask centroid to the calibrated table plane."""
        result = {
            "camera": detection["camera"],
            "confidence": float(detection["confidence"]),
            "accepted_by_yolo": bool(detection["accepted"]),
            "mask_area_px": float(detection["mask_area_px"]),
            "center_px": [float(v) for v in detection["center_px"]],
            "geometry_valid": False,
            "geometry_failure_reason": None,
        }
        if detection["camera"] != "table":
            result["geometry_failure_reason"] = "drop zone requires fixed table camera"
            return result
        image_height, image_width = rgb.shape[:2]
        calibrated_width, calibrated_height = self._table_color_camera["size"]
        center = np.asarray(detection["center_px"], dtype=float)
        calibrated_uv = np.array(
            [
                center[0] * calibrated_width / image_width,
                center[1] * calibrated_height / image_height,
            ]
        )
        xyz = pixel_to_base(
            self._table_color_camera, calibrated_uv, z=SURFACE_Z
        )
        workspace_valid = bool(
            0.15 <= xyz[0] <= 0.65 and -0.45 <= xyz[1] <= 0.45
        )
        area_valid = bool(detection["mask_area_px"] >= 100.0)
        result.update(
            {
                "center_xy_base_m": xyz[:2].tolist(),
                "ray_plane_z_base_m": float(SURFACE_Z),
                "workspace_valid": workspace_valid,
                "mask_area_valid": area_valid,
                "geometry_valid": bool(workspace_valid and area_valid),
            }
        )
        if not result["geometry_valid"]:
            result["geometry_failure_reason"] = "drop-zone geometry gate failed"
        return result

    def _project_depth_points_to_rgb(
        self,
        camera: str,
        depth: np.ndarray,
        obs: dict,
        rgb_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        if camera == "table":
            points = depth_to_base(self._table_camera, depth, stride=1)
            reference_points = points
            color_camera = self._table_color_camera
        elif camera == "wrist":
            joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
            if joints.size < 6 or not np.all(np.isfinite(joints[:6])):
                raise RuntimeError("finite joints are required for wrist fusion")
            points = depth_to_base(
                self._wrist_camera, depth, stride=1, joints=joints
            )
            base_to_flange = np.linalg.inv(flange_pose(joints))
            reference_points = (
                points @ base_to_flange[:3, :3].T + base_to_flange[:3, 3]
            )
            color_camera = self._wrist_color_camera
        else:
            raise ValueError(f"unsupported fusion camera: {camera}")
        uv = project(color_camera, reference_points)
        calibrated_width, calibrated_height = color_camera["size"]
        image_height, image_width = rgb_shape[:2]
        uv[:, 0] *= image_width / calibrated_width
        uv[:, 1] *= image_height / calibrated_height
        return points, uv

    def _fuse_detection(
        self,
        detection: dict,
        depth: np.ndarray,
        rgb: np.ndarray,
        obs: dict,
        tcp_pose: np.ndarray,
        plane_offset_mm: float,
    ) -> dict:
        camera = str(detection["camera"])
        points, uv = self._project_depth_points_to_rgb(
            camera, depth, obs, rgb.shape
        )
        height, width = rgb.shape[:2]
        polygon = np.rint(detection["_polygon"]).astype(np.int32)
        mask_image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask_image).polygon(
            [tuple(item) for item in polygon.tolist()], fill=1
        )
        dilation = int(self.FUSION_MASK_DILATION_PX[camera])
        if dilation > 0:
            mask_image = mask_image.filter(ImageFilter.MaxFilter(2 * dilation + 1))
        mask = np.asarray(mask_image, dtype=np.uint8)

        finite = np.all(np.isfinite(uv), axis=1)
        rounded = np.rint(np.nan_to_num(uv, nan=-1.0)).astype(np.int64)
        in_image = (
            finite
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        inside = np.zeros(len(points), dtype=bool)
        valid_index = np.flatnonzero(in_image)
        inside[valid_index] = (
            mask[rounded[valid_index, 1], rounded[valid_index, 0]] > 0
        )
        corrected_points = points.copy()
        corrected_points[:, 2] -= float(plane_offset_mm) / 1000.0
        heights_mm = (corrected_points[:, 2] - SURFACE_Z) * 1000.0
        elevated = (
            inside
            & self._workspace(corrected_points)
            & (heights_mm >= 2.0)
            & (heights_mm <= 30.0)
        )
        fused_points = corrected_points[elevated]
        minimum = int(self.MINIMUM_FUSED_POINTS[camera])
        result = {
            "camera": camera,
            "confidence": float(detection["confidence"]),
            "accepted_by_yolo": bool(detection["accepted"]),
            "mask_area_px": float(detection["mask_area_px"]),
            "projected_depth_points_in_mask": int(np.count_nonzero(inside)),
            "elevated_depth_points": int(len(fused_points)),
            "minimum_required_depth_points": minimum,
            "applied_plane_offset_correction_mm": float(plane_offset_mm),
            "geometry_valid": False,
            "geometry_failure_reason": None,
        }
        if len(fused_points) < minimum:
            result["geometry_failure_reason"] = "insufficient elevated mask depth"
            return result

        geometry = self._component_geometry(fused_points, tcp_pose)
        center_xyz = np.median(fused_points, axis=0)
        geometry_valid = bool(
            25.0 <= geometry["length_mm"] <= 80.0
            and 4.0 <= geometry["width_mm"] <= 35.0
            and 4.0 <= geometry["top_height_above_table_mm"] <= 30.0
        )
        result.update(geometry)
        result["center_xyz_base_m"] = center_xyz.tolist()
        result["geometry_valid"] = geometry_valid
        if not geometry_valid:
            result["geometry_failure_reason"] = "AAA geometry gate failed"
        return result

    def _save_fusion_visual(
        self,
        camera: str,
        rgb: np.ndarray,
        detections: list[dict],
        observation_index: int,
    ) -> tuple[str, str]:
        assert self._rollout_dir is not None
        stem = f"obs_{observation_index:02d}_{camera}"
        raw_path = self._rollout_dir / "raw" / f"{stem}.png"
        overlay_path = self._rollout_dir / "overlays" / f"{stem}_fusion.png"
        Image.fromarray(rgb).save(raw_path)
        overlay_image = Image.fromarray(rgb).copy()
        draw = ImageDraw.Draw(overlay_image)
        for detection in detections:
            contour = np.rint(detection["_polygon"]).astype(np.int32).reshape(-1, 2)
            geometry = detection.get("fusion", detection.get("drop_zone_geometry", {}))
            valid = bool(
                detection["accepted"] and geometry.get("geometry_valid", False)
            )
            color = (20, 220, 20) if valid else (255, 155, 0)
            coordinates = [tuple(item) for item in contour.tolist()]
            if coordinates:
                draw.line([*coordinates, coordinates[0]], fill=color, width=2)
            x, y = contour.min(axis=0)
            short_name = (
                "battery"
                if detection["class_name"] == BATTERY_CLASS_NAME
                else "drop-zone"
            )
            label = f"{short_name} {detection['confidence']:.2f}"
            if valid and detection["class_name"] == BATTERY_CLASS_NAME:
                label += (
                    f" d={geometry['euclidean_distance_from_tcp_to_object_top_mm']:.0f}mm"
                )
            elif valid:
                label += " center-ok"
            elif not detection["accepted"]:
                label += " low-conf"
            else:
                label += " no-depth"
            draw.text((int(x), max(2, int(y) - 13)), label, fill=color)
        overlay_image.save(overlay_path)
        return str(raw_path), str(overlay_path)

    def _safety_record(self, obs: dict, target_z_mm: float) -> dict:
        table, wrist = self._depth_arrays(obs)
        assert self._start_tcp is not None
        snapshot = self._guard.snapshot(table)
        self_quality, excluded_points = self._self_mask_quality(snapshot, obs)
        vertical = snapshot.evaluate_vertical_target(
            x_mm=self._start_tcp[0],
            y_mm=self._start_tcp[1],
            target_z_uf0_mm=target_z_mm,
            footprint_radius_mm=self.VERTICAL_FOOTPRINT_RADIUS_MM,
            minimum_clearance_mm=self.MINIMUM_TCP_CLEARANCE_MM,
            excluded_points_mask=excluded_points,
        )
        offset = float(snapshot.table_plane_offset_mm)
        vertical["table_offset_baseline_mm"] = self._table_offset_baseline_mm
        vertical["table_offset_delta_from_baseline_mm"] = None
        vertical["table_offset_consistent_with_preflight"] = True
        if self._table_offset_baseline_mm is not None:
            delta = abs(offset - self._table_offset_baseline_mm)
            consistent = bool(delta <= self.MAXIMUM_TABLE_OFFSET_DELTA_MM)
            vertical["table_offset_delta_from_baseline_mm"] = float(delta)
            vertical["table_offset_consistent_with_preflight"] = consistent
            if not consistent:
                vertical["passed"] = False
                vertical["reason"] = "table-plane offset changed after preflight"
        wrist_quality = self._wrist_quality(wrist, obs)
        passed = bool(
            vertical["passed"] and wrist_quality["valid"] and self_quality["valid"]
        )
        return {
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "target_z_uf0_mm": float(target_z_mm),
            "vertical_check": vertical,
            "wrist_quality": wrist_quality,
            "robot_self_mask": self_quality,
            "passed": passed,
        }

    @staticmethod
    def _component_geometry(points: np.ndarray, tcp_pose: np.ndarray) -> dict:
        xy = points[:, :2]
        center = np.median(xy, axis=0)
        centered = xy - center
        covariance = centered.T @ centered / max(1, len(centered))
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        # A cylindrical battery axis is equivalent modulo 180 degrees.  Use a
        # deterministic sign so temporal aggregation cannot cancel itself.
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
            axis = -axis
        along = centered @ axis
        across = centered @ np.array([-axis[1], axis[0]])
        length = float((np.percentile(along, 95) - np.percentile(along, 5)) * 1000)
        width = float((np.percentile(across, 95) - np.percentile(across, 5)) * 1000)
        if width > length:
            length, width = width, length
        heights = (points[:, 2] - SURFACE_Z) * 1000.0
        top = float(np.percentile(heights, 95))
        tcp_base = np.array(
            [
                tcp_pose[0] / 1000.0,
                tcp_pose[1] / 1000.0,
                (tcp_pose[2] - UF0_Z_MM) / 1000.0,
            ]
        )
        object_top = np.array([center[0], center[1], SURFACE_Z + top / 1000.0])
        horizontal = float(np.linalg.norm(center - tcp_base[:2]) * 1000.0)
        vertical = float((tcp_base[2] - object_top[2]) * 1000.0)
        euclidean = float(np.linalg.norm(tcp_base - object_top) * 1000.0)
        battery_sized = bool(
            30.0 <= length <= 75.0
            and 5.0 <= width <= 30.0
            and 5.0 <= top <= 25.0
        )
        return {
            "center_xy_base_m": center.tolist(),
            "axis_xy": axis.tolist(),
            "yaw_deg_mod_180": float(np.degrees(np.arctan2(axis[1], axis[0]))),
            "point_count": int(len(points)),
            "length_mm": length,
            "width_mm": width,
            "top_height_above_table_mm": top,
            "horizontal_distance_from_tcp_mm": horizontal,
            "vertical_clearance_from_tcp_to_object_top_mm": vertical,
            "euclidean_distance_from_tcp_to_object_top_mm": euclidean,
            "class_hint": "battery_sized" if battery_sized else "clutter",
        }

    @staticmethod
    def _connected_components(binary: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Small dependency-free 8-connected component labeller."""
        labels = np.zeros(binary.shape, dtype=np.int32)
        active = {tuple(index) for index in np.argwhere(binary > 0)}
        areas = [0]
        label = 0
        neighbours = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
        while active:
            label += 1
            seed = active.pop()
            queue = deque([seed])
            pixels = [seed]
            while queue:
                row, column = queue.popleft()
                for dr, dc in neighbours:
                    candidate = (row + dr, column + dc)
                    if candidate in active:
                        active.remove(candidate)
                        queue.append(candidate)
                        pixels.append(candidate)
            rows, columns = np.asarray(pixels, dtype=int).T
            labels[rows, columns] = label
            areas.append(len(pixels))
        return labels, areas

    def _table_components(self, depth: np.ndarray, tcp_pose: np.ndarray) -> list[dict]:
        points, pixels = depth_to_base(
            self._table_camera, depth, stride=1, with_pixels=True
        )
        height = (points[:, 2] - SURFACE_Z) * 1000.0
        workspace = self._workspace(points)
        elevated = (
            workspace
            & (height >= 5.0)
            & (height <= self.MAXIMUM_EXTERNAL_OBJECT_HEIGHT_MM)
        )
        binary = np.zeros(depth.shape, dtype=np.uint8)
        uv = pixels[elevated].astype(int)
        binary[uv[:, 1], uv[:, 0]] = 1
        labels, areas = self._connected_components(binary)
        point_labels = labels[pixels[:, 1].astype(int), pixels[:, 0].astype(int)]
        components = []
        for label in range(1, len(areas)):
            pixel_area = int(areas[label])
            if pixel_area < 3:
                continue
            mask = elevated & (point_labels == label)
            component_points = points[mask]
            if len(component_points) < 3:
                continue
            geometry = self._component_geometry(component_points, tcp_pose)
            if geometry["length_mm"] < 4.0 or geometry["length_mm"] > 250.0:
                continue
            geometry["component_id"] = int(label)
            geometry["pixel_area"] = pixel_area
            components.append(geometry)
        components.sort(
            key=lambda item: item["euclidean_distance_from_tcp_to_object_top_mm"]
        )
        return components

    def _shadow_route_checks(
        self,
        obs: dict,
        table_depth: np.ndarray,
        battery: dict | None,
        drop_zone: dict | None,
    ) -> dict | None:
        """Evaluate non-contact waypoints without commanding them."""
        if battery is None or drop_zone is None:
            return None
        snapshot = self._guard.snapshot(table_depth)
        self_quality, excluded_points = self._self_mask_quality(snapshot, obs)
        battery_xy_mm = tuple(
            float(v) * 1000.0 for v in battery["center_xy_base_m"]
        )
        drop_xy_mm = tuple(
            float(v) * 1000.0 for v in drop_zone["center_xy_base_m"]
        )
        pick_approach = snapshot.evaluate_vertical_target(
            *battery_xy_mm,
            self.PICK_APPROACH_Z_MM,
            self.VERTICAL_FOOTPRINT_RADIUS_MM,
            self.MINIMUM_TCP_CLEARANCE_MM,
            excluded_points,
        )
        carry = snapshot.evaluate_horizontal_corridor(
            battery_xy_mm,
            drop_xy_mm,
            self.TRANSIT_Z_MM,
            75.0,
            50.0,
            excluded_points,
        )
        drop_approach = snapshot.evaluate_vertical_target(
            *drop_xy_mm,
            self.DROP_APPROACH_Z_MM,
            self.VERTICAL_FOOTPRINT_RADIUS_MM,
            self.MINIMUM_TCP_CLEARANCE_MM,
            excluded_points,
        )
        separation = float(
            np.linalg.norm(np.asarray(battery_xy_mm) - np.asarray(drop_xy_mm))
        )
        return {
            "evaluation_only": True,
            "robot_self_mask_valid": bool(self_quality["valid"]),
            "pick_approach": pick_approach,
            "carry_corridor": carry,
            "drop_approach": drop_approach,
            "pick_to_drop_separation_mm": separation,
            "minimum_pick_to_drop_separation_mm": (
                self.MINIMUM_PICK_TO_DROP_SEPARATION_MM
            ),
            "passed": bool(
                self_quality["valid"]
                and pick_approach["passed"]
                and carry["passed"]
                and drop_approach["passed"]
                and separation >= self.MINIMUM_PICK_TO_DROP_SEPARATION_MM
            ),
        }

    def _build_execution_plan(self) -> tuple[dict | None, str | None]:
        """Turn stable v09 observations into one bounded v10 state machine."""
        assert self._start_tcp is not None
        batteries = [
            item["selected_battery_fusion"]
            for item in self._low_records
            if item.get("selected_battery_fusion") is not None
        ]
        drops = [
            item["selected_drop_zone"]
            for item in self._low_records
            if item.get("selected_drop_zone") is not None
        ]
        routes = [
            item["shadow_route_checks"]
            for item in self._low_records
            if item.get("shadow_route_checks") is not None
        ]
        if len(batteries) < self.MINIMUM_FUSED_FRAMES:
            return None, "too few stable battery frames"
        if len(drops) < self.MINIMUM_DROP_ZONE_FRAMES:
            return None, "too few stable drop-zone frames"
        if sum(item["passed"] for item in routes) < self.MINIMUM_FUSED_FRAMES:
            return None, "non-contact route did not pass enough frames"

        battery_centers = np.asarray(
            [item["center_xy_base_m"] for item in batteries], dtype=float
        )
        battery_xy = np.median(battery_centers, axis=0)
        battery_spread = float(
            np.max(np.linalg.norm(battery_centers - battery_xy, axis=1)) * 1000.0
        )
        drop_centers = np.asarray(
            [item["center_xy_base_m"] for item in drops], dtype=float
        )
        drop_xy = np.median(drop_centers, axis=0)
        drop_spread = float(
            np.max(np.linalg.norm(drop_centers - drop_xy, axis=1)) * 1000.0
        )
        if battery_spread > self.MAXIMUM_CENTER_SPREAD_MM:
            return None, f"battery centre spread too large: {battery_spread:.1f} mm"
        if drop_spread > self.MAXIMUM_DROP_ZONE_SPREAD_MM:
            return None, f"drop-zone centre spread too large: {drop_spread:.1f} mm"

        axes = np.asarray([item["axis_xy"] for item in batteries], dtype=float)
        angles = np.arctan2(axes[:, 1], axes[:, 0])
        angle = 0.5 * np.arctan2(
            np.mean(np.sin(2.0 * angles)), np.mean(np.cos(2.0 * angles))
        )
        alignment_error_deg = abs(float(np.degrees(angle)))
        alignment_error_deg = min(alignment_error_deg, 180.0 - alignment_error_deg)
        if alignment_error_deg > self.MAXIMUM_ALIGNMENT_ERROR_DEG:
            return None, (
                f"battery is {alignment_error_deg:.1f} deg from the validated "
                "gripper alignment"
            )

        battery_top_mm = float(
            np.median([item["top_height_above_table_mm"] for item in batteries])
        )
        contact_height_mm = float(
            np.clip(
                0.75 * battery_top_mm,
                self.MINIMUM_CONTACT_HEIGHT_ABOVE_TABLE_MM,
                self.MAXIMUM_CONTACT_HEIGHT_ABOVE_TABLE_MM,
            )
        )
        bowl_tops = [
            item["drop_approach"]["obstacle_top_height_mm"]
            for item in routes
            if item.get("drop_approach", {}).get("obstacle_top_height_mm")
            is not None
        ]
        if not bowl_tops:
            return None, "drop-zone height was not measured"
        bowl_top_mm = float(np.median(bowl_tops))
        release_height_mm = float(
            np.clip(
                bowl_top_mm + self.RELEASE_CLEARANCE_ABOVE_BOWL_MM,
                self.MINIMUM_RELEASE_HEIGHT_ABOVE_TABLE_MM,
                self.MAXIMUM_RELEASE_HEIGHT_ABOVE_TABLE_MM,
            )
        )
        battery_xy_mm = battery_xy * 1000.0
        drop_xy_mm = drop_xy * 1000.0
        separation_mm = float(np.linalg.norm(battery_xy_mm - drop_xy_mm))
        if separation_mm < self.MINIMUM_PICK_TO_DROP_SEPARATION_MM:
            return None, "battery and bowl are too close"
        for name, xy in (("battery", battery_xy_mm), ("drop zone", drop_xy_mm)):
            if not (150.0 <= xy[0] <= 650.0 and -450.0 <= xy[1] <= 450.0):
                return None, f"{name} is outside the calibrated workspace"

        orientation = [float(v) for v in self._start_tcp[3:6]]

        def target(xy, z, gripper):
            return [
                float(xy[0]), float(xy[1]), float(z),
                *orientation, float(gripper),
            ]

        start_xy = np.asarray(self._start_tcp[:2], dtype=float)
        contact_z = self.TABLE_Z_MM + contact_height_mm
        release_z = self.TABLE_Z_MM + release_height_mm
        steps = [
            {"name": "scan_lift", "kind": "move", "target": target(start_xy, self.TRANSIT_Z_MM, 0.0)},
            {"name": "pick_transit", "kind": "move", "target": target(battery_xy_mm, self.TRANSIT_Z_MM, 0.0)},
            {"name": "pick_approach", "kind": "move", "target": target(battery_xy_mm, self.PICK_APPROACH_Z_MM, 0.0)},
            {"name": "pick_pregrasp", "kind": "move", "target": target(battery_xy_mm, self.PICK_PREGRASP_Z_MM, 0.0)},
            {"name": "pick_contact", "kind": "move", "target": target(battery_xy_mm, contact_z, 0.0)},
            {"name": "close_gripper", "kind": "dwell", "target": target(battery_xy_mm, contact_z, 1.0)},
            {"name": "pick_lift", "kind": "move", "target": target(battery_xy_mm, self.TRANSIT_Z_MM, 1.0)},
            {"name": "verify_pickup", "kind": "verify_pickup", "target": target(battery_xy_mm, self.TRANSIT_Z_MM, 1.0)},
            {"name": "carry_to_drop", "kind": "move", "target": target(drop_xy_mm, self.TRANSIT_Z_MM, 1.0)},
            {"name": "drop_approach", "kind": "move", "target": target(drop_xy_mm, self.DROP_APPROACH_Z_MM, 1.0)},
            {"name": "drop_release", "kind": "move", "target": target(drop_xy_mm, release_z, 1.0)},
            {"name": "open_gripper", "kind": "dwell", "target": target(drop_xy_mm, release_z, 0.0)},
            {"name": "verify_placement", "kind": "verify_placement", "target": target(drop_xy_mm, release_z, 0.0)},
            {"name": "drop_retreat", "kind": "move", "target": target(drop_xy_mm, self.TRANSIT_Z_MM, 0.0)},
            {"name": "home_transit", "kind": "move", "target": target(start_xy, self.TRANSIT_Z_MM, 0.0)},
            {"name": "home", "kind": "move", "target": [*self._start_tcp[:6], 0.0]},
        ]
        plan = {
            "execution_enabled": self.EXECUTION_ENABLED,
            "battery_center_xy_base_mm": battery_xy_mm.tolist(),
            "drop_zone_center_xy_base_mm": drop_xy_mm.tolist(),
            "battery_yaw_error_from_validated_gripper_deg": alignment_error_deg,
            "battery_top_height_mm": battery_top_mm,
            "contact_height_above_table_mm": contact_height_mm,
            "contact_z_uf0_mm": contact_z,
            "bowl_obstacle_top_height_mm": bowl_top_mm,
            "release_height_above_table_mm": release_height_mm,
            "release_z_uf0_mm": release_z,
            "pick_to_drop_separation_mm": separation_mm,
            "battery_center_spread_mm": battery_spread,
            "drop_zone_center_spread_mm": drop_spread,
            "route_passed_frames": int(sum(item["passed"] for item in routes)),
            "steps": steps,
        }
        return plan, None

    def _table_battery_verification(self, obs: dict, phase: str) -> dict:
        """Locate any table-height battery after lift or release."""
        assert self._execution_plan is not None
        table, _wrist = self._depth_arrays(obs)
        rgb = self._rgb_array(obs, "table")
        pose = self._finite_tcp(obs)
        if pose is None:
            raise RuntimeError("finite TCP required for verification")
        index = 100 + len(self._pickup_records) + len(self._placement_records)
        detections = self._infer_battery_masks("table", rgb, index)
        snapshot = self._guard.snapshot(table)
        fusions = []
        for detection in detections:
            if detection["class_name"] != BATTERY_CLASS_NAME:
                continue
            fusion = self._fuse_detection(
                detection,
                table,
                rgb,
                obs,
                pose,
                float(snapshot.table_plane_offset_mm),
            )
            detection["fusion"] = fusion
            if detection["accepted"] and fusion["geometry_valid"]:
                fusions.append(fusion)
        self._save_fusion_visual("table", rgb, detections, index)
        battery_xy = np.asarray(
            self._execution_plan["battery_center_xy_base_mm"], dtype=float
        )
        drop_xy = np.asarray(
            self._execution_plan["drop_zone_center_xy_base_mm"], dtype=float
        )
        centers_mm = [
            np.asarray(item["center_xy_base_m"], dtype=float) * 1000.0
            for item in fusions
        ]
        distance_to_pick = (
            min(float(np.linalg.norm(center - battery_xy)) for center in centers_mm)
            if centers_mm else None
        )
        distance_to_drop = (
            min(float(np.linalg.norm(center - drop_xy)) for center in centers_mm)
            if centers_mm else None
        )
        gripper = self._gripper(obs)
        local_pick_distance = np.linalg.norm(
            snapshot.points[:, :2] - battery_xy[None, :] / 1000.0,
            axis=1,
        )
        local_pick_elevated = int(
            np.count_nonzero(
                (local_pick_distance <= 0.035)
                & (snapshot.height_mm >= 4.0)
                & (snapshot.height_mm <= 25.0)
            )
        )
        if phase == "pickup":
            passed = bool(
                gripper >= 0.75
                and (distance_to_pick is None or distance_to_pick > 45.0)
                and local_pick_elevated < 5
            )
        else:
            passed = bool(
                gripper <= 0.25
                and distance_to_drop is not None
                and distance_to_drop <= 85.0
            )
        return {
            "phase": phase,
            "accepted_table_height_batteries": len(fusions),
            "minimum_distance_to_pick_mm": distance_to_pick,
            "minimum_distance_to_drop_mm": distance_to_drop,
            "local_pick_elevated_depth_points": local_pick_elevated,
            "maximum_allowed_pick_residual_points": 4,
            "observed_gripper": gripper,
            "passed": passed,
        }

    def _prepare_execution_abort(
        self, reason: str, pose: np.ndarray, gripper: float
    ) -> None:
        """Fail closed from an arbitrary execution pose without opening jaws."""
        assert self._start_tcp is not None
        self._aborted_reason = reason
        safe_gripper = float(gripper)
        orientation = [float(v) for v in pose[3:6]]
        current_xy = [float(pose[0]), float(pose[1])]
        start_xy = self._start_tcp[:2]
        self._retreat_targets = [
            [*current_xy, self.TRANSIT_Z_MM, *orientation, safe_gripper],
            [*start_xy, self.TRANSIT_Z_MM, *self._start_tcp[3:6], safe_gripper],
            [*self._start_tcp[:6], safe_gripper],
        ]
        self._retreat_index = 0
        self._state = "retreat"
        print(f"[v10] execution fail-closed: {reason}", flush=True)
        self._event("execution_abort", None, reason=reason)

    def _low_record(self, obs: dict) -> dict:
        table, wrist = self._depth_arrays(obs)
        pose = self._finite_tcp(obs)
        if pose is None:
            raise RuntimeError("finite observed TCP is required at low scan pose")
        safety = self._safety_record(obs, float(pose[2]))
        observation_index = len(self._low_records) + 1
        images = obs.get("images", {})
        if not isinstance(images, dict) or not all(
            name in images for name in ("table", "wrist")
        ):
            # Historical depth-only unit tests and degraded telemetry may omit
            # RGB.  Finish the already-safe vertical observation, but produce
            # no plan and never treat missing RGB as valid perception.
            return {
                **safety,
                "observation_index": observation_index,
                "observed_tcp_pose": pose.tolist(),
                "cameras": [],
                "valid_fused_candidates": 0,
                "selected_battery_fusion": None,
                "valid_drop_zone_candidates": 0,
                "selected_drop_zone": None,
                "shadow_route_checks": None,
                "perception_valid": False,
                "perception_unavailable_reason": "RGB images missing",
            }
        camera_records = []
        fused_candidates = []
        drop_zone_candidates = []
        for camera, depth in (("table", table), ("wrist", wrist)):
            rgb = self._rgb_array(obs, camera)
            detections = self._infer_battery_masks(
                camera, rgb, observation_index
            )
            for detection in detections:
                if detection["class_name"] == BATTERY_CLASS_NAME:
                    if camera == "table":
                        plane_offset = float(
                            safety["vertical_check"]["table_depth_calibration"][
                                "table_plane_offset_mm"
                            ]
                        )
                    else:
                        plane_offset = float(
                            safety["wrist_quality"]["median_error_mm"]
                        )
                    fusion = self._fuse_detection(
                        detection, depth, rgb, obs, pose, plane_offset
                    )
                    detection["fusion"] = fusion
                    if detection["accepted"] and fusion["geometry_valid"]:
                        fused_candidates.append(fusion)
                else:
                    geometry = self._drop_zone_geometry(detection, rgb)
                    detection["drop_zone_geometry"] = geometry
                    if detection["accepted"] and geometry["geometry_valid"]:
                        drop_zone_candidates.append(geometry)
            raw_path, overlay_path = self._save_fusion_visual(
                camera, rgb, detections, observation_index
            )
            public_detections = []
            for detection in detections:
                public_detections.append(
                    {
                        key: value
                        for key, value in detection.items()
                        if key not in {"_polygon", "polygon_px"}
                    }
                )
            camera_records.append(
                {
                    "camera": camera,
                    "raw_image": raw_path,
                    "overlay_image": overlay_path,
                    "scene_detections": public_detections,
                    "raw_battery_detections": int(
                        sum(item["class_name"] == BATTERY_CLASS_NAME for item in detections)
                    ),
                    "accepted_battery_detections": int(
                        sum(
                            item["accepted"]
                            and item["class_name"] == BATTERY_CLASS_NAME
                            for item in detections
                        )
                    ),
                    "valid_fusions": int(
                        sum(
                            item["class_name"] == BATTERY_CLASS_NAME
                            and item["accepted"]
                            and item.get("fusion", {}).get("geometry_valid", False)
                            for item in detections
                        )
                    ),
                    "valid_drop_zones": int(
                        sum(
                            item["class_name"] == DROP_ZONE_CLASS_NAME
                            and item["accepted"]
                            and item.get("drop_zone_geometry", {}).get(
                                "geometry_valid", False
                            )
                            for item in detections
                        )
                    ),
                }
            )

        selected = max(
            fused_candidates,
            key=lambda item: item["confidence"]
            + (0.05 if item["camera"] == "table" else 0.0),
            default=None,
        )
        selected_drop_zone = max(
            drop_zone_candidates,
            key=lambda item: item["confidence"],
            default=None,
        )
        route_checks = self._shadow_route_checks(
            obs, table, selected, selected_drop_zone
        )
        return {
            **safety,
            "observation_index": observation_index,
            "observed_tcp_pose": pose.tolist(),
            "cameras": camera_records,
            "valid_fused_candidates": len(fused_candidates),
            "selected_battery_fusion": selected,
            "valid_drop_zone_candidates": len(drop_zone_candidates),
            "selected_drop_zone": selected_drop_zone,
            "shadow_route_checks": route_checks,
            "perception_valid": bool(
                selected is not None and selected_drop_zone is not None
            ),
        }

    def _append_jsonl(self, name: str, record: dict) -> None:
        assert self._rollout_dir is not None
        with (self._rollout_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _event(self, phase: str, command: list[float] | None, **extra) -> None:
        record = {
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "phase": phase,
            "commanded_tcp": command,
            **extra,
        }
        self._events.append(record)
        self._write_report()

    def _write_report(self) -> None:
        assert self._rollout_dir is not None
        selections = [
            item["selected_battery_fusion"]
            for item in self._low_records
            if item.get("selected_battery_fusion") is not None
        ]
        drop_selections = [
            item["selected_drop_zone"]
            for item in self._low_records
            if item.get("selected_drop_zone") is not None
        ]
        route_selections = [
            item["shadow_route_checks"]
            for item in self._low_records
            if item.get("shadow_route_checks") is not None
        ]
        battery_aggregate = None
        if selections:
            centers = np.asarray(
                [item["center_xy_base_m"] for item in selections], dtype=float
            )
            median_center = np.median(centers, axis=0)
            center_errors = np.linalg.norm(centers - median_center, axis=1) * 1000.0
            axes = np.asarray([item["axis_xy"] for item in selections], dtype=float)
            angles = np.arctan2(axes[:, 1], axes[:, 0])
            mean_angle = 0.5 * np.arctan2(
                np.mean(np.sin(2.0 * angles)), np.mean(np.cos(2.0 * angles))
            )
            median_axis = np.array([np.cos(mean_angle), np.sin(mean_angle)])
            if median_axis[0] < 0.0:
                median_axis = -median_axis
            battery_aggregate = {
                "frames": len(selections),
                "median_center_xy_base_m": median_center.tolist(),
                "mean_axis_xy_mod_180": median_axis.tolist(),
                "mean_yaw_deg_mod_180": float(np.degrees(mean_angle)),
                "maximum_center_deviation_mm": float(np.max(center_errors)),
                "median_confidence": float(
                    np.median([item["confidence"] for item in selections])
                ),
                "median_top_height_above_table_mm": float(
                    np.median(
                        [item["top_height_above_table_mm"] for item in selections]
                    )
                ),
                "median_horizontal_distance_from_tcp_mm": float(
                    np.median(
                        [item["horizontal_distance_from_tcp_mm"] for item in selections]
                    )
                ),
                "median_vertical_clearance_from_tcp_to_object_top_mm": float(
                    np.median(
                        [
                            item["vertical_clearance_from_tcp_to_object_top_mm"]
                            for item in selections
                        ]
                    )
                ),
                "median_euclidean_distance_from_tcp_mm": float(
                    np.median(
                        [
                            item["euclidean_distance_from_tcp_to_object_top_mm"]
                            for item in selections
                        ]
                    )
                ),
                "selected_camera_counts": {
                    camera: int(sum(item["camera"] == camera for item in selections))
                    for camera in ("table", "wrist")
                },
            }
        drop_aggregate = None
        if drop_selections:
            centers = np.asarray(
                [item["center_xy_base_m"] for item in drop_selections], dtype=float
            )
            median_center = np.median(centers, axis=0)
            center_errors = np.linalg.norm(centers - median_center, axis=1) * 1000.0
            drop_aggregate = {
                "frames": len(drop_selections),
                "median_center_xy_base_m": median_center.tolist(),
                "maximum_center_deviation_mm": float(np.max(center_errors)),
                "median_confidence": float(
                    np.median([item["confidence"] for item in drop_selections])
                ),
                "geometry_method": "fixed_table_camera_mask_centroid_ray_plane",
            }
        battery_passed = bool(
            battery_aggregate is not None
            and battery_aggregate["frames"] >= self.MINIMUM_FUSED_FRAMES
            and battery_aggregate["maximum_center_deviation_mm"]
            <= self.MAXIMUM_CENTER_SPREAD_MM
        )
        drop_passed = bool(
            drop_aggregate is not None
            and drop_aggregate["frames"] >= self.MINIMUM_DROP_ZONE_FRAMES
            and drop_aggregate["maximum_center_deviation_mm"]
            <= self.MAXIMUM_DROP_ZONE_SPREAD_MM
        )
        route_passed = bool(
            len(route_selections) >= self.MINIMUM_FUSED_FRAMES
            and sum(item["passed"] for item in route_selections)
            >= self.MINIMUM_FUSED_FRAMES
        )
        perception_passed = bool(battery_passed and drop_passed)

        shadow_plan = None
        if battery_aggregate is not None and drop_aggregate is not None:
            battery_xy_mm = np.asarray(
                battery_aggregate["median_center_xy_base_m"], dtype=float
            ) * 1000.0
            bowl_xy_mm = np.asarray(
                drop_aggregate["median_center_xy_base_m"], dtype=float
            ) * 1000.0
            axis = np.asarray(
                battery_aggregate["mean_axis_xy_mod_180"], dtype=float
            )
            spread = np.array([-axis[1], axis[0]])
            orientation = (
                self._start_tcp[3:6]
                if self._start_tcp is not None
                else [-180.0, 0.0, 0.0]
            )
            gripper = self._start_tcp[6] if self._start_tcp is not None else 0.0
            grasp_z = float(
                self.TABLE_Z_MM
                + battery_aggregate["median_top_height_above_table_mm"] / 2.0
            )

            def waypoint(name: str, xy: np.ndarray, z: float, purpose: str) -> dict:
                return {
                    "name": name,
                    "tcp_uf0_mm": [
                        float(xy[0]), float(xy[1]), float(z),
                        *[float(v) for v in orientation], float(gripper),
                    ],
                    "purpose": purpose,
                    "commanded": False,
                }

            shadow_plan = {
                "execution_enabled": self.EXECUTION_ENABLED,
                "coordinate_source": "real_yolo_masks_plus_calibrated_geometry",
                "battery_center_xy_base_mm": battery_xy_mm.tolist(),
                "battery_axis_xy": axis.tolist(),
                "recommended_finger_spread_xy": spread.tolist(),
                "drop_zone_center_xy_base_mm": bowl_xy_mm.tolist(),
                "tool_yaw_status": (
                    "not converted to FANUC R; hand-eye/tool-axis validation required"
                ),
                "contact_height_status": (
                    "v10 derives and bounds the executable contact height separately"
                ),
                "logical_gripper_events": [
                    {
                        "after_waypoint": "pick_contact",
                        "transition": "close",
                        "commanded": False,
                    },
                    {
                        "after_waypoint": "drop_release",
                        "transition": "open",
                        "commanded": False,
                    },
                ],
                "waypoints": [
                    waypoint("pick_transit", battery_xy_mm, self.TRANSIT_Z_MM,
                             "horizontal arrival above battery"),
                    waypoint("pick_approach", battery_xy_mm, self.PICK_APPROACH_Z_MM,
                             "validated-height observation above battery"),
                    waypoint("pick_pregrasp", battery_xy_mm, self.PICK_PREGRASP_Z_MM,
                             "provisional guarded vertical approach"),
                    waypoint("pick_contact", battery_xy_mm, grasp_z,
                             "provisional battery-centre pinch height"),
                    waypoint("pick_lift", battery_xy_mm, self.TRANSIT_Z_MM,
                             "vertical lift before translation"),
                    waypoint("drop_transit", bowl_xy_mm, self.TRANSIT_Z_MM,
                             "high horizontal carry to detected bowl"),
                    waypoint("drop_approach", bowl_xy_mm, self.DROP_APPROACH_Z_MM,
                             "vertical approach above bowl"),
                    waypoint("drop_release", bowl_xy_mm, self.DROP_RELEASE_Z_MM,
                             "provisional low-energy release height"),
                    waypoint("drop_retreat", bowl_xy_mm, self.TRANSIT_Z_MM,
                             "vertical retreat after release"),
                ],
                "non_contact_route_checks": {
                    "frames": len(route_selections),
                    "passed_frames": int(sum(item["passed"] for item in route_selections)),
                    "latest": route_selections[-1] if route_selections else None,
                },
            }
        shadow_plan_ready = bool(
            perception_passed and route_passed and shadow_plan is not None
        )
        execution_started = self._execution_plan is not None
        execution_motion_events = [
            event
            for event in self._events
            if event["phase"].startswith(
                ("execute_command_", "execute_reached_")
            )
        ]
        horizontal_names = {
            "pick_transit", "carry_to_drop", "home_transit"
        }
        horizontal_motion_commanded = any(
            event["phase"]
            .removeprefix("execute_command_")
            .removeprefix("execute_reached_")
            in horizontal_names
            for event in execution_motion_events
        )
        gripper_transition_commanded = any(
            event["phase"] in {
                "execute_complete_close_gripper",
                "execute_complete_open_gripper",
            }
            for event in self._events
        )
        full_task_complete = bool(
            self._state == "complete"
            and self._pickup_verified
            and self._placement_verified
        )
        report = {
            "task": "physical_guarded_yolo_pick_place_v10",
            "task_complete": full_task_complete,
            "safe_motion_complete": self._state == "complete",
            "perception_validation_passed": perception_passed,
            "battery_validation_passed": battery_passed,
            "drop_zone_validation_passed": drop_passed,
            "non_contact_route_validation_passed": route_passed,
            "shadow_plan_ready": shadow_plan_ready,
            "execution_enabled": self.EXECUTION_ENABLED,
            "execution_started": execution_started,
            "execution_step_index": self._execution_index,
            "execution_steps_total": len(self._execution_steps),
            "pickup_verified": self._pickup_verified,
            "placement_verified": self._placement_verified,
            "pickup_verification": self._pickup_records,
            "placement_verification": self._placement_records,
            "safely_abstained": self._state == "aborted",
            "reason": self._aborted_reason,
            "control_source": (
                "stable_yolo_depth_geometry_plus_calibrated_depth_safety"
            ),
            "perception_source": (
                "yolo_instance_masks_plus_calibrated_depth_plus_fixed_camera_ray_plane"
            ),
            "vision_used_for_motion": execution_started,
            "motion_started": bool(
                any(
                    event["phase"].startswith(("descend", "execute_command_"))
                    for event in self._events
                )
            ),
            "horizontal_motion_commanded": horizontal_motion_commanded,
            "gripper_transition_commanded": gripper_transition_commanded,
            "start_tcp": self._start_tcp,
            "lowest_commanded_z_uf0_mm": (
                min(
                    event["commanded_tcp"][2]
                    for event in self._events
                    if event.get("commanded_tcp") is not None
                )
                if any(event.get("commanded_tcp") is not None for event in self._events)
                else None
            ),
            "required_wrist_depth_guard_margin_mm": self.WRIST_GUARD_MARGIN_MM,
            "table_offset_baseline_mm": self._table_offset_baseline_mm,
            "preflight_frames": len(self._preflight_records),
            "preflight_passed_frames": int(
                sum(item["passed"] for item in self._preflight_records)
            ),
            "descent_safety_frames": len(self._step_settle_records),
            "descent_safety_failed_frames": int(
                sum(not item["passed"] for item in self._step_settle_records)
            ),
            "low_observation_frames": len(self._low_records),
            "frames_with_valid_battery_fusion": len(selections),
            "frames_with_valid_drop_zone": len(drop_selections),
            "minimum_required_fused_frames": self.MINIMUM_FUSED_FRAMES,
            "maximum_allowed_center_deviation_mm": self.MAXIMUM_CENTER_SPREAD_MM,
            "battery_fusion_aggregate": battery_aggregate,
            "drop_zone_aggregate": drop_aggregate,
            "pre_execution_plan": shadow_plan,
            "execution_plan": self._execution_plan,
            "next_stage": (
                "physical pick/place complete; inspect video and final bowl image"
                if full_task_complete
                else "review the failed gate or verification before another rollout"
            ),
            "model": str(self.MODEL_PATH),
            "settings": {
                "imgsz": self.MODEL_IMAGE_SIZE,
                "raw_battery_confidence": self.RAW_BATTERY_CONFIDENCE,
                "accepted_battery_confidence": self.ACCEPTED_BATTERY_CONFIDENCE,
                "accepted_drop_zone_confidence": self.ACCEPTED_DROP_ZONE_CONFIDENCE,
                "iou": self.MODEL_IOU,
            },
            "events": self._events,
        }
        payload = json.dumps(report, indent=2) + "\n"
        (self._rollout_dir / "physical_pick_place_v10_report.json").write_text(
            payload, encoding="utf-8"
        )
        # Keep historical depth-only regression tests readable.  Operators
        # should use physical_pick_place_v10_report.json.
        compatibility = dict(report)
        if all(
            item.get("perception_unavailable_reason") is not None
            for item in self._low_records
        ) and self._low_records:
            compatibility["task_complete"] = self._state == "complete"
        (self._rollout_dir / "guarded_descent_v07_report.json").write_text(
            json.dumps(compatibility, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _reached(obs: dict, target: list[float]) -> bool:
        pose = Policy._finite_tcp(obs)
        if pose is None:
            return False
        return bool(np.linalg.norm(pose[:3] - np.asarray(target[:3])) <= 6.0)

    def _abort(self, reason: str, current_z_mm: float | None = None) -> None:
        if self._state in {"retreat", "complete", "aborted"}:
            return
        self._aborted_reason = reason
        self._state = "retreat" if self._start_tcp is not None else "aborted"
        if self._start_tcp is not None:
            self._prepare_retreat(
                self._start_tcp[2] if current_z_mm is None else current_z_mm
            )
        print(f"[v10] fail-closed: {reason}; retreating vertically", flush=True)
        self._event("abort", None, reason=reason)

    def act(self, obs: dict):
        if self._rollout_dir is None:
            self.reset()
        pose = self._finite_tcp(obs)
        if pose is None:
            if not self._mock_notice_printed:
                print(
                    "[v10] tcp_pose unavailable; issuing validated high mock bootstrap only",
                    flush=True,
                )
                self._mock_notice_printed = True
            return self._mock_bootstrap_target(obs)

        if self._state == "bootstrap":
            try:
                self._validate_start(pose)
                self._build_targets(pose, self._gripper(obs))
                self._state = "preflight"
                self._event("start", self._start_tcp)
            except Exception as exc:
                self._aborted_reason = str(exc)
                self._state = "aborted"
                self._write_report()

        if self._state == "aborted":
            return [*pose.tolist(), self._gripper(obs)]

        if self._state == "preflight":
            try:
                record = self._safety_record(obs, self.DESCENT_Z_MM[-1])
                current_offset = float(
                    record["vertical_check"]["table_depth_calibration"][
                        "table_plane_offset_mm"
                    ]
                )
                prior_offsets = [
                    float(
                        item["vertical_check"]["table_depth_calibration"][
                            "table_plane_offset_mm"
                        ]
                    )
                    for item in self._preflight_records
                ]
                offset_window = (
                    [*prior_offsets, current_offset][
                        -self.TABLE_OFFSET_STABILITY_FRAMES :
                    ]
                )
                offset_range = (
                    float(np.ptp(offset_window))
                    if len(offset_window) >= self.TABLE_OFFSET_STABILITY_FRAMES
                    else None
                )
                offset_stable = bool(
                    offset_range is not None
                    and offset_range <= self.MAXIMUM_TABLE_OFFSET_RANGE_MM
                )
                record["raw_safety_passed_before_offset_stability"] = bool(
                    record["passed"]
                )
                record["preflight_table_offset_window_mm"] = offset_window
                record["preflight_table_offset_range_mm"] = offset_range
                record["preflight_table_offset_stable"] = offset_stable
                record["passed"] = bool(record["passed"] and offset_stable)
                self._preflight_records.append(record)
                self._append_jsonl("preflight.jsonl", record)
                consecutive = 0
                for item in reversed(self._preflight_records):
                    if not item["passed"]:
                        break
                    consecutive += 1
                print(
                    f"[v10] preflight {len(self._preflight_records)}/"
                    f"{self.PREFLIGHT_MAXIMUM}: consecutive={consecutive}, "
                    f"offset={current_offset:.1f} mm, "
                    f"stable={offset_stable}, "
                    f"clearance={record['vertical_check']['predicted_tcp_clearance_mm']:.1f} mm",
                    flush=True,
                )
                if consecutive >= self.PREFLIGHT_REQUIRED:
                    accepted_offsets = [
                        float(
                            item["vertical_check"]["table_depth_calibration"][
                                "table_plane_offset_mm"
                            ]
                        )
                        for item in self._preflight_records[-self.PREFLIGHT_REQUIRED :]
                    ]
                    self._table_offset_baseline_mm = float(
                        np.median(accepted_offsets)
                    )
                    self._state = "descending"
                    self._descent_index = 0
                    self._step_settle_records = []
                elif len(self._preflight_records) >= self.PREFLIGHT_MAXIMUM:
                    self._abort(
                        "five consecutive preflight safety frames were not obtained",
                        float(pose[2]),
                    )
            except Exception as exc:
                self._abort(
                    f"preflight error: {type(exc).__name__}: {exc}",
                    float(pose[2]),
                )
            if self._state == "preflight":
                assert self._start_tcp is not None
                return self._start_tcp.copy()

        if self._state == "descending":
            target = self._descent_targets[self._descent_index]
            try:
                check = self._safety_record(obs, target[2])
                check["descent_step"] = self._descent_index + 1
                check["observed_tcp_pose"] = pose.tolist()
                check["target_tcp"] = target
                check["target_reached"] = self._reached(obs, target)
                # Wrist depth is calibrated only at settled poses.  During a
                # commanded move its median can shift transiently even while
                # the fixed table camera still proves the vertical route is
                # clear.  Motion-time fail-closed control therefore uses the
                # fixed table height map and robot-self mask; wrist becomes a
                # required gate again after the target is reached.
                check["motion_gate_passed"] = bool(
                    check["vertical_check"]["passed"]
                    and check["robot_self_mask"]["valid"]
                )
                self._step_settle_records.append(check)
                self._append_jsonl("descent_safety.jsonl", check)
                if not check["target_reached"]:
                    if not check["motion_gate_passed"]:
                        self._abort(
                            f"fixed-camera safety gate failed during descent step "
                            f"{self._descent_index + 1}",
                            float(pose[2]),
                        )
                    else:
                        self._event(f"descend_command_{self._descent_index + 1}", target)
                else:
                    current_step = [
                        item
                        for item in self._step_settle_records
                        if item["descent_step"] == self._descent_index + 1
                        and item["target_reached"]
                    ]
                    consecutive = 0
                    for item in reversed(current_step):
                        if not item["passed"]:
                            break
                        consecutive += 1
                    print(
                        f"[v10] step {self._descent_index + 1}/"
                        f"{len(self._descent_targets)} settle "
                        f"{len(current_step)}/{self.STEP_SETTLE_MAXIMUM}: "
                        f"consecutive={consecutive}, "
                        f"vertical={check['vertical_check']['passed']}, "
                        f"wrist={check['wrist_quality']['valid']}, "
                        f"self={check['robot_self_mask']['valid']}",
                        flush=True,
                    )
                    if consecutive >= self.STEP_SETTLE_REQUIRED:
                        self._event(
                            f"descend_reached_{self._descent_index + 1}", target
                        )
                        self._descent_index += 1
                        if self._descent_index >= len(self._descent_targets):
                            self._state = "observe_low"
                        else:
                            target = self._descent_targets[self._descent_index]
                    elif len(current_step) >= self.STEP_SETTLE_MAXIMUM:
                        failure = {
                            "vertical": check["vertical_check"]["passed"],
                            "wrist": check["wrist_quality"]["valid"],
                            "self": check["robot_self_mask"]["valid"],
                        }
                        self._abort(
                            f"descent step {self._descent_index + 1} did not obtain "
                            f"three consecutive safe frames: {failure}",
                            float(pose[2]),
                        )
            except Exception as exc:
                self._abort(
                    f"descent safety error: {type(exc).__name__}: {exc}",
                    float(pose[2]),
                )
            if self._state == "descending":
                return target.copy()

        if self._state == "observe_low":
            try:
                record = self._low_record(obs)
                self._low_records.append(record)
                self._append_jsonl("fusion_observations.jsonl", record)
                self._write_report()
                selected = record["selected_battery_fusion"]
                if selected is None:
                    fusion_message = "battery_fusion=none"
                else:
                    fusion_message = (
                        f"battery_fusion={selected['camera']} "
                        f"conf={selected['confidence']:.2f} "
                        f"distance={selected['euclidean_distance_from_tcp_to_object_top_mm']:.1f}mm"
                    )
                drop = record["selected_drop_zone"]
                drop_message = (
                    "drop_zone=none"
                    if drop is None
                    else f"drop_zone=table conf={drop['confidence']:.2f}"
                )
                print(
                    f"[v10] low observation {len(self._low_records)}/{self.LOW_OBSERVATIONS}: "
                    f"{fusion_message}, {drop_message}, safe={record['passed']}",
                    flush=True,
                )
                if not record["passed"]:
                    self._abort(
                        "low-pose depth safety gate failed", float(pose[2])
                    )
                elif len(self._low_records) >= self.LOW_OBSERVATIONS:
                    if all(
                        item.get("perception_unavailable_reason") is not None
                        for item in self._low_records
                    ):
                        # Preserve the historical depth-only regression path:
                        # complete the safe vertical test without execution.
                        self._state = "retreat"
                        self._prepare_retreat(float(pose[2]))
                    else:
                        plan, failure = self._build_execution_plan()
                        if failure is not None or plan is None:
                            self._abort(
                                f"execution plan rejected: {failure}", float(pose[2])
                            )
                        else:
                            self._execution_plan = plan
                            self._execution_steps = plan["steps"]
                            self._execution_index = 0
                            self._execution_dwell = 0
                            self._state = "execute"
                            self._event(
                                "execution_plan_accepted",
                                None,
                                battery_xy_mm=plan["battery_center_xy_base_mm"],
                                drop_xy_mm=plan["drop_zone_center_xy_base_mm"],
                                contact_z_uf0_mm=plan["contact_z_uf0_mm"],
                                release_z_uf0_mm=plan["release_z_uf0_mm"],
                            )
            except Exception as exc:
                self._abort(
                    f"low-pose measurement error: {type(exc).__name__}: {exc}",
                    float(pose[2]),
                )
            if self._state == "observe_low":
                return self._descent_targets[-1].copy()

        if self._state == "execute":
            try:
                step = self._execution_steps[self._execution_index]
                target = step["target"]
                kind = step["kind"]
                name = step["name"]
                if target[2] < self.TABLE_Z_MM + self.MINIMUM_CONTACT_HEIGHT_ABOVE_TABLE_MM:
                    raise RuntimeError(f"{name} target is below the v10 table limit")
                if not (150.0 <= target[0] <= 650.0 and -450.0 <= target[1] <= 450.0):
                    raise RuntimeError(f"{name} target is outside calibrated workspace")

                advance = False
                if kind == "move":
                    if self._reached(obs, target):
                        self._event(f"execute_reached_{name}", target)
                        advance = True
                    else:
                        self._event(f"execute_command_{name}", target)
                elif kind == "dwell":
                    self._execution_dwell += 1
                    if self._execution_dwell >= self.GRIPPER_DWELL_FRAMES:
                        self._event(f"execute_complete_{name}", target)
                        advance = True
                elif kind in {"verify_pickup", "verify_placement"}:
                    phase = "pickup" if kind == "verify_pickup" else "placement"
                    verification = self._table_battery_verification(obs, phase)
                    records = (
                        self._pickup_records
                        if phase == "pickup"
                        else self._placement_records
                    )
                    records.append(verification)
                    self._append_jsonl(f"{phase}_verification.jsonl", verification)
                    passed = int(sum(item["passed"] for item in records))
                    print(
                        f"[v10] {phase} verification {len(records)}/"
                        f"{self.VERIFICATION_FRAMES}: passed={passed}",
                        flush=True,
                    )
                    if passed >= self.MINIMUM_VERIFICATION_PASSES:
                        if phase == "pickup":
                            self._pickup_verified = True
                        else:
                            self._placement_verified = True
                        self._event(f"execute_verified_{phase}", target)
                        advance = True
                    elif len(records) >= self.VERIFICATION_FRAMES:
                        if phase == "pickup":
                            self._prepare_execution_abort(
                                "pickup could not be verified after lift",
                                pose,
                                self._gripper(obs),
                            )
                        else:
                            self._event(
                                "execute_unverified_placement",
                                target,
                                passed_frames=passed,
                            )
                            advance = True
                else:
                    raise RuntimeError(f"unsupported execution step kind: {kind}")

                if self._state == "execute" and advance:
                    self._execution_index += 1
                    self._execution_dwell = 0
                    if self._execution_index >= len(self._execution_steps):
                        self._state = "complete"
                        self._write_report()
                        print(
                            "[v10] execution finished and robot returned home",
                            flush=True,
                        )
                        assert self._start_tcp is not None
                        return [*self._start_tcp[:6], 0.0]
                    step = self._execution_steps[self._execution_index]
                    target = step["target"]
                if self._state == "execute":
                    return target.copy()
            except Exception as exc:
                self._prepare_execution_abort(
                    f"execution error: {type(exc).__name__}: {exc}",
                    pose,
                    self._gripper(obs),
                )

        if self._state == "retreat":
            target = self._retreat_targets[self._retreat_index]
            if self._reached(obs, target):
                self._event(f"retreat_reached_{self._retreat_index + 1}", target)
                self._retreat_index += 1
                if self._retreat_index >= len(self._retreat_targets):
                    self._state = "complete" if self._aborted_reason is None else "aborted"
                    self._write_report()
                    print(
                        "[v10] fail-closed retreat returned to start",
                        flush=True,
                    )
                    assert self._start_tcp is not None
                    return self._start_tcp.copy()
                target = self._retreat_targets[self._retreat_index]
            else:
                self._event(f"retreat_command_{self._retreat_index + 1}", target)
            return target.copy()

        assert self._start_tcp is not None
        return self._start_tcp.copy()
