"""Physical v08.2: guarded descent plus YOLO-mask/depth battery fusion.

This policy performs no horizontal motion and no grasp.  It validates five
consecutive depth frames, descends along the observed TCP vertical line in four
bounded steps, observes the low scene for ten frames, and retreats along the
same line.  Every commanded descent step is checked by the calibrated table
depth height map.  At the settled low pose, YOLO segmentation identifies AAA
battery pixels and calibrated table/wrist depth measures only points inside
those masks.  Vision results are diagnostic only: they never command horizontal
motion or a gripper transition in this version.
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
    project,
)
from depth_safety import TableDepthSafetyGuard
from robot_self_filter import classify_robot_self_points


BATTERY_CLASS_NAME = "aaa_battery"


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

    OUTPUT_ROOT = Path(__file__).resolve().parent / "guarded_yolo_depth_v08_2_runs"
    MODEL_PATH = Path(__file__).resolve().parent / "models" / "aaa_yolo26s_best.pt"
    MODEL_IMAGE_SIZE = 1024
    MODEL_IOU = 0.50
    RAW_BATTERY_CONFIDENCE = 0.25
    ACCEPTED_BATTERY_CONFIDENCE = 0.60
    FUSION_MASK_DILATION_PX = {"table": 2, "wrist": 3}
    MINIMUM_FUSED_POINTS = {"table": 8, "wrist": 20}
    MINIMUM_FUSED_FRAMES = 3
    MAXIMUM_CENTER_SPREAD_MM = 20.0
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
    TABLE_OFFSET_STABILITY_FRAMES = 3
    MAXIMUM_TABLE_OFFSET_RANGE_MM = 3.0
    MAXIMUM_TABLE_OFFSET_DELTA_MM = 3.0

    START_X_MM = (250.0, 350.0)
    START_Y_MM = (-60.0, 60.0)
    START_Z_MM = (-130.0, -50.0)
    MAXIMUM_REACH_ERROR_MM = 6.0

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
        print(
            f"[v08.2] guarded YOLO-depth fusion started; results: {self._rollout_dir}",
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

    def _self_mask_quality(self, depth: np.ndarray, obs: dict) -> dict:
        points = depth_to_base(self._table_camera, depth, stride=2)
        points = points[self._workspace(points)]
        height = (points[:, 2] - SURFACE_Z) * 1000.0
        high = points[height > 120.0]
        result = classify_robot_self_points(
            high,
            obs.get("joints", []),
            obs.get("tcp_pose", []),
            capsule_radius_m=self.SELF_CAPSULE_RADIUS_M,
        )
        explained = int(np.count_nonzero(result["self_mask"]))
        fraction = float(explained / len(high)) if len(high) else 1.0
        return {
            "high_points": int(len(high)),
            "explained_points": explained,
            "unexplained_points": int(len(high) - explained),
            "explained_fraction": fraction,
            "valid": bool(fraction >= self.MINIMUM_SELF_EXPLAINED_FRACTION),
        }

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
                f"[v08.2] loading {self.MODEL_PATH} on {self._model_device}",
                flush=True,
            )
            self._model = YOLO(str(self.MODEL_PATH))
            print("[v08.2] YOLO loaded", flush=True)
        except Exception as exc:
            self._model_error = f"YOLO load failed: {type(exc).__name__}: {exc}"
            raise RuntimeError(self._model_error) from exc

    def _infer_battery_masks(
        self, camera: str, rgb: np.ndarray, observation_index: int
    ) -> list[dict]:
        """Return raw AAA detections; confidence acceptance is explicit."""
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
            if class_name != BATTERY_CLASS_NAME:
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
                        confidence >= self.ACCEPTED_BATTERY_CONFIDENCE
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
            fusion = detection.get("fusion", {})
            valid = bool(
                detection["accepted"] and fusion.get("geometry_valid", False)
            )
            color = (20, 220, 20) if valid else (255, 155, 0)
            coordinates = [tuple(item) for item in contour.tolist()]
            if coordinates:
                draw.line([*coordinates, coordinates[0]], fill=color, width=2)
            x, y = contour.min(axis=0)
            label = f"battery {detection['confidence']:.2f}"
            if valid:
                label += f" d={fusion['euclidean_distance_from_tcp_to_object_top_mm']:.0f}mm"
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
        vertical = snapshot.evaluate_vertical_target(
            x_mm=self._start_tcp[0],
            y_mm=self._start_tcp[1],
            target_z_uf0_mm=target_z_mm,
            footprint_radius_mm=self.VERTICAL_FOOTPRINT_RADIUS_MM,
            minimum_clearance_mm=self.MINIMUM_TCP_CLEARANCE_MM,
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
        self_quality = self._self_mask_quality(table, obs)
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

    def _low_record(self, obs: dict) -> dict:
        table, wrist = self._depth_arrays(obs)
        pose = self._finite_tcp(obs)
        if pose is None:
            raise RuntimeError("finite observed TCP is required at low scan pose")
        safety = self._safety_record(obs, float(pose[2]))
        observation_index = len(self._low_records) + 1
        camera_records = []
        fused_candidates = []
        for camera, depth in (("table", table), ("wrist", wrist)):
            rgb = self._rgb_array(obs, camera)
            detections = self._infer_battery_masks(
                camera, rgb, observation_index
            )
            for detection in detections:
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
                    "battery_detections": public_detections,
                    "raw_battery_detections": len(detections),
                    "accepted_battery_detections": int(
                        sum(item["accepted"] for item in detections)
                    ),
                    "valid_fusions": int(
                        sum(
                            item["accepted"]
                            and item["fusion"]["geometry_valid"]
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
        return {
            **safety,
            "observation_index": observation_index,
            "observed_tcp_pose": pose.tolist(),
            "cameras": camera_records,
            "valid_fused_candidates": len(fused_candidates),
            "selected_battery_fusion": selected,
            "perception_valid": selected is not None,
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
        aggregate = None
        if selections:
            centers = np.asarray(
                [item["center_xy_base_m"] for item in selections], dtype=float
            )
            median_center = np.median(centers, axis=0)
            center_errors = np.linalg.norm(centers - median_center, axis=1) * 1000.0
            aggregate = {
                "frames": len(selections),
                "median_center_xy_base_m": median_center.tolist(),
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
        perception_passed = bool(
            aggregate is not None
            and aggregate["frames"] >= self.MINIMUM_FUSED_FRAMES
            and aggregate["maximum_center_deviation_mm"]
            <= self.MAXIMUM_CENTER_SPREAD_MM
        )
        report = {
            "task": "guarded_descent_yolo_mask_depth_fusion_v08_2",
            "task_complete": self._state == "complete",
            "safe_motion_complete": self._state == "complete",
            "perception_validation_passed": perception_passed,
            "safely_abstained": self._state == "aborted",
            "reason": self._aborted_reason,
            "control_source": "calibrated_depth_safety_only",
            "perception_source": "yolo_instance_masks_plus_calibrated_depth",
            "vision_used_for_motion": False,
            "motion_started": bool(
                any(event["phase"].startswith("descend") for event in self._events)
            ),
            "horizontal_motion_commanded": False,
            "gripper_transition_commanded": False,
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
            "minimum_required_fused_frames": self.MINIMUM_FUSED_FRAMES,
            "maximum_allowed_center_deviation_mm": self.MAXIMUM_CENTER_SPREAD_MM,
            "battery_fusion_aggregate": aggregate,
            "model": str(self.MODEL_PATH),
            "settings": {
                "imgsz": self.MODEL_IMAGE_SIZE,
                "raw_battery_confidence": self.RAW_BATTERY_CONFIDENCE,
                "accepted_battery_confidence": self.ACCEPTED_BATTERY_CONFIDENCE,
                "iou": self.MODEL_IOU,
            },
            "events": self._events,
        }
        (self._rollout_dir / "guarded_yolo_depth_v08_2_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
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
        print(f"[v08.2] fail-closed: {reason}; retreating vertically", flush=True)
        self._event("abort", None, reason=reason)

    def act(self, obs: dict):
        if self._rollout_dir is None:
            self.reset()
        pose = self._finite_tcp(obs)
        if pose is None:
            if not self._mock_notice_printed:
                print(
                    "[v08.2] tcp_pose unavailable; issuing validated high mock bootstrap only",
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
                    f"[v08.2] preflight {len(self._preflight_records)}/"
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
                        f"[v08.2] step {self._descent_index + 1}/"
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
                print(
                    f"[v08.2] low observation {len(self._low_records)}/{self.LOW_OBSERVATIONS}: "
                    f"{fusion_message}, safe={record['passed']}",
                    flush=True,
                )
                if not record["passed"]:
                    self._abort(
                        "low-pose depth safety gate failed", float(pose[2])
                    )
                elif len(self._low_records) >= self.LOW_OBSERVATIONS:
                    self._state = "retreat"
                    self._prepare_retreat(float(pose[2]))
            except Exception as exc:
                self._abort(
                    f"low-pose measurement error: {type(exc).__name__}: {exc}",
                    float(pose[2]),
                )
            if self._state == "observe_low":
                return self._descent_targets[-1].copy()

        if self._state == "retreat":
            target = self._retreat_targets[self._retreat_index]
            if self._reached(obs, target):
                self._event(f"retreat_reached_{self._retreat_index + 1}", target)
                self._retreat_index += 1
                if self._retreat_index >= len(self._retreat_targets):
                    self._state = "complete" if self._aborted_reason is None else "aborted"
                    self._write_report()
                    print(
                        "[v08.2] returned to the observed start pose; no grasp attempted",
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
