"""Safe wrist-camera scan plus AAA battery detection for the FANUC robot.

The arm follows a fixed, bounded scan path above the table.  At every reached
pose, the policy runs YOLO segmentation on table, side, and wrist RGB images.
Detected image coordinates never control motion in this version: the gripper
stays open and no grasp is attempted.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


BATTERY_CLASS = 0
DROP_ZONE_CLASS = 1


def _device() -> str | int:
    """Choose a local inference device without assuming CUDA is present."""
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _polygon_geometry(polygon: np.ndarray) -> dict:
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx, cy = map(float, polygon.mean(axis=0))

    (_, _), (width, height), angle = cv2.minAreaRect(contour)
    if width < height:
        angle += 90.0
    long_axis = max(float(width), float(height))
    short_axis = min(float(width), float(height))
    yaw = ((float(angle) + 90.0) % 180.0) - 90.0
    return {
        "center_px": [cx, cy],
        "yaw_image_deg_clockwise": yaw,
        "long_axis_px": long_axis,
        "short_axis_px": short_axis,
        "aspect_ratio": long_axis / max(short_axis, 1e-6),
        "mask_area_px2": area,
    }


def _center_inside(center: list[float], polygon: np.ndarray) -> bool:
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    return (
        cv2.pointPolygonTest(
            contour, (float(center[0]), float(center[1])), False
        )
        >= 0
    )


def _draw_detection(
    image_rgb: np.ndarray,
    detection: dict,
    polygon: np.ndarray,
) -> None:
    class_id = int(detection["class_id"])
    if class_id == DROP_ZONE_CLASS:
        color = (0, 220, 220)  # RGB cyan
    elif detection.get("pick_candidate", False):
        color = (20, 220, 20)  # RGB green
    else:
        color = (255, 155, 0)  # RGB orange

    contour = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
    transparent = image_rgb.copy()
    cv2.fillPoly(transparent, [contour], color)
    cv2.addWeighted(transparent, 0.22, image_rgb, 0.78, 0.0, dst=image_rgb)
    cv2.polylines(image_rgb, [contour], True, color, 2, cv2.LINE_AA)

    cx, cy = detection["center_px"]
    center = (int(round(cx)), int(round(cy)))
    cv2.circle(image_rgb, center, 4, color, -1, cv2.LINE_AA)

    if class_id == BATTERY_CLASS:
        yaw = math.radians(float(detection["yaw_image_deg_clockwise"]))
        radius = 0.5 * float(detection["long_axis_px"])
        dx, dy = radius * math.cos(yaw), radius * math.sin(yaw)
        point_1 = (int(round(cx - dx)), int(round(cy - dy)))
        point_2 = (int(round(cx + dx)), int(round(cy + dy)))
        cv2.line(image_rgb, point_1, point_2, color, 2, cv2.LINE_AA)

    label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
    if class_id == BATTERY_CLASS:
        label += f' yaw={detection["yaw_image_deg_clockwise"]:.1f}'
        if not detection.get("pick_candidate", False):
            label += f' [{detection.get("pick_reason", "rejected")}]'
    x, y = contour.reshape(-1, 2).min(axis=0)
    cv2.putText(
        image_rgb,
        label,
        (int(x), max(22, int(y) - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )


class Policy:
    # Keep all three cameras for comparison.  Only the wrist camera moves.
    CAMERAS = ["table", "side", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = []

    MODEL_PATH = Path(__file__).resolve().parent / "models" / "aaa_yolo26s_best.pt"
    OUTPUT_ROOT = Path(__file__).resolve().parent / "battery_detection_runs"

    MODEL_IMAGE_SIZE = 1024
    MODEL_IOU = 0.50
    BATTERY_CONFIDENCE = 0.60
    DROP_ZONE_CONFIDENCE = 0.30

    # TABLE_Z_MM is the table surface, not a commanded TCP height.
    TABLE_Z_MM = -335.0
    SCAN_CLEARANCE_MM = 100.0
    TRANSIT_CLEARANCE_MM = 170.0
    SCAN_Z_MM = TABLE_Z_MM + SCAN_CLEARANCE_MM        # -235 mm
    TRANSIT_Z_MM = TABLE_Z_MM + TRANSIT_CLEARANCE_MM  # -165 mm

    # Fixed table center from the client package's documented TCP example.
    # The scan is deliberately small: 110 mm left-to-right and 60 mm front-to-back.
    SCAN_CENTER_X_MM = 450.0
    SCAN_CENTER_Y_MM = 0.0
    SCAN_X_OFFSET_MM = 55.0
    SCAN_Y_OFFSET_MM = 30.0

    # Hard software envelope.  Every generated target is validated before use.
    SAFE_X_MIN_MM = 375.0
    SAFE_X_MAX_MM = 525.0
    SAFE_Y_MIN_MM = -90.0
    SAFE_Y_MAX_MM = 90.0
    SAFE_Z_MIN_MM = TABLE_Z_MM + 80.0
    SAFE_Z_MAX_MM = -100.0

    TOOL_W_DEG = -180.0
    TOOL_P_DEG = 0.0
    TOOL_R_DEG = 0.0
    OPEN_GRIPPER = 0.0
    ENABLE_SCAN_MOTION = True

    def __init__(self, checkpoint: str | None = None):
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Battery model not found: {self.MODEL_PATH}. "
                "Copy best.pt to models/aaa_yolo26s_best.pt."
            )
        self._device = _device()
        print(
            f"[battery] loading {self.MODEL_PATH} on {self._device}",
            flush=True,
        )
        self._model = YOLO(str(self.MODEL_PATH))
        print("[battery] model loaded; waiting for rollout", flush=True)
        self._rollout_number = 0
        self._rollout_dir: Path | None = None
        self._started = 0.0
        self._frame_set = 0
        self._complete_printed = False
        self._trial_written = False
        self._initial_frame_processed = False
        self._pending_target: dict | None = None
        self._next_target_index = 0
        self._aborted = False
        self._scan_targets = self._build_scan_targets()

    def _build_scan_targets(self) -> list[dict]:
        """Create and validate a bounded serpentine scan path."""
        cx = self.SCAN_CENTER_X_MM
        cy = self.SCAN_CENTER_Y_MM
        dx = self.SCAN_X_OFFSET_MM
        dy = self.SCAN_Y_OFFSET_MM
        raw_targets = [
            ("transit_center_start", cx, cy, self.TRANSIT_Z_MM),
            ("scan_center_start", cx, cy, self.SCAN_Z_MM),
            ("scan_left_back", cx - dx, cy - dy, self.SCAN_Z_MM),
            ("scan_center_back", cx, cy - dy, self.SCAN_Z_MM),
            ("scan_right_back", cx + dx, cy - dy, self.SCAN_Z_MM),
            ("scan_right_front", cx + dx, cy + dy, self.SCAN_Z_MM),
            ("scan_center_front", cx, cy + dy, self.SCAN_Z_MM),
            ("scan_left_front", cx - dx, cy + dy, self.SCAN_Z_MM),
            ("scan_center_finish", cx, cy, self.SCAN_Z_MM),
            ("transit_center_finish", cx, cy, self.TRANSIT_Z_MM),
        ]
        targets: list[dict] = []
        for name, x, y, z in raw_targets:
            if not self.SAFE_X_MIN_MM <= x <= self.SAFE_X_MAX_MM:
                raise ValueError(f"Unsafe scan X target {name}: {x}")
            if not self.SAFE_Y_MIN_MM <= y <= self.SAFE_Y_MAX_MM:
                raise ValueError(f"Unsafe scan Y target {name}: {y}")
            if not self.SAFE_Z_MIN_MM <= z <= self.SAFE_Z_MAX_MM:
                raise ValueError(f"Unsafe scan Z target {name}: {z}")
            targets.append(
                {
                    "name": name,
                    "tcp": [
                        float(x),
                        float(y),
                        float(z),
                        self.TOOL_W_DEG,
                        self.TOOL_P_DEG,
                        self.TOOL_R_DEG,
                        self.OPEN_GRIPPER,
                    ],
                }
            )
        return targets

    def reset(self) -> None:
        """Called by the client at the beginning of each rollout."""
        self._rollout_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rollout_dir = self.OUTPUT_ROOT / (
            f"rollout_{self._rollout_number:02d}_{timestamp}"
        )
        (self._rollout_dir / "raw").mkdir(parents=True, exist_ok=False)
        (self._rollout_dir / "overlays").mkdir(parents=True, exist_ok=True)
        self._started = time.monotonic()
        self._frame_set = 0
        self._complete_printed = False
        self._trial_written = False
        self._initial_frame_processed = False
        self._pending_target = None
        self._next_target_index = 0
        self._aborted = False
        self._scan_targets = self._build_scan_targets()
        (self._rollout_dir / "scan_plan.json").write_text(
            json.dumps(
                {
                    "table_z_mm": self.TABLE_Z_MM,
                    "scan_clearance_mm": self.SCAN_CLEARANCE_MM,
                    "transit_clearance_mm": self.TRANSIT_CLEARANCE_MM,
                    "motion_enabled": self.ENABLE_SCAN_MOTION,
                    "targets": self._scan_targets,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[battery] scan-and-detect rollout {self._rollout_number} started; "
            f"results: {self._rollout_dir}",
            flush=True,
        )
        print(
            f"[motion] table_z={self.TABLE_Z_MM:.1f}, "
            f"scan_z={self.SCAN_Z_MM:.1f}, "
            f"transit_z={self.TRANSIT_Z_MM:.1f}, "
            f"targets={len(self._scan_targets)}",
            flush=True,
        )

    def _infer_camera(
        self,
        camera: str,
        rgb: np.ndarray,
        elapsed: float,
    ) -> dict:
        """Run one camera through YOLO and save raw/overlay images."""
        assert self._rollout_dir is not None
        prefix = f"set_{self._frame_set:03d}_{elapsed:05.1f}s_{camera}"
        raw_path = self._rollout_dir / "raw" / f"{prefix}.png"
        overlay_path = self._rollout_dir / "overlays" / f"{prefix}_pred.png"
        Image.fromarray(rgb).save(raw_path)

        # PIL explicitly preserves the RGB channel convention used by obs.
        result = self._model.predict(
            source=Image.fromarray(rgb),
            imgsz=self.MODEL_IMAGE_SIZE,
            conf=min(self.BATTERY_CONFIDENCE, self.DROP_ZONE_CONFIDENCE),
            iou=self.MODEL_IOU,
            device=self._device,
            retina_masks=True,
            verbose=False,
            save=False,
        )[0]

        raw_detections: list[tuple[dict, np.ndarray]] = []
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            for class_id, confidence, polygon in zip(
                classes, confidences, result.masks.xy
            ):
                polygon = np.asarray(polygon, dtype=np.float32)
                if len(polygon) < 3:
                    continue
                name = str(result.names.get(int(class_id), class_id))
                detection = {
                    "class_id": int(class_id),
                    "class_name": name,
                    "confidence": float(confidence),
                    **_polygon_geometry(polygon),
                    "polygon_px": polygon.round(2).tolist(),
                }
                raw_detections.append((detection, polygon))

        accepted_drop_zones = [
            polygon
            for detection, polygon in raw_detections
            if detection["class_id"] == DROP_ZONE_CLASS
            and detection["confidence"] >= self.DROP_ZONE_CONFIDENCE
        ]

        overlay = rgb.copy()
        detections: list[dict] = []
        for detection, polygon in raw_detections:
            class_id = int(detection["class_id"])
            if class_id == BATTERY_CLASS:
                inside = any(
                    _center_inside(detection["center_px"], zone)
                    for zone in accepted_drop_zones
                )
                detection["inside_drop_zone"] = inside
                if detection["confidence"] < self.BATTERY_CONFIDENCE:
                    reason = "low_confidence"
                elif inside:
                    reason = "inside_drop_zone"
                else:
                    reason = "candidate"
                detection["pick_reason"] = reason
                detection["pick_candidate"] = reason == "candidate"
            elif class_id == DROP_ZONE_CLASS:
                detection["accepted"] = (
                    detection["confidence"] >= self.DROP_ZONE_CONFIDENCE
                )
            detections.append(detection)
            _draw_detection(overlay, detection, polygon)

        Image.fromarray(overlay).save(overlay_path)
        candidates = [
            detection
            for detection in detections
            if detection.get("pick_candidate", False)
        ]
        selected = max(candidates, key=lambda item: item["confidence"], default=None)
        return {
            "camera": camera,
            "raw_image": str(raw_path),
            "overlay_image": str(overlay_path),
            "detections": detections,
            # This is diagnostic only.  It is never converted into an action.
            "selected_candidate": selected,
        }

    def _process_observation(
        self,
        obs: dict,
        pose_label: str,
        commanded_tcp: list[float] | None,
    ) -> None:
        """Run all requested cameras once at a reached scan pose."""
        assert self._rollout_dir is not None
        elapsed = time.monotonic() - self._started
        camera_records: list[dict] = []
        for camera in self.CAMERAS:
            image = obs.get("images", {}).get(camera)
            if image is None:
                print(f"[battery] warning: no {camera} image", flush=True)
                continue
            rgb = np.asarray(image)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                print(
                    f"[battery] warning: unexpected {camera} shape {rgb.shape}",
                    flush=True,
                )
                continue
            rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
            record = self._infer_camera(camera, rgb, elapsed)
            camera_records.append(record)

            candidates = [
                item
                for item in record["detections"]
                if item.get("pick_candidate", False)
            ]
            drop_zones = [
                item
                for item in record["detections"]
                if item["class_id"] == DROP_ZONE_CLASS
                and item.get("accepted", False)
            ]
            message = (
                f"[battery] pose={pose_label} {camera}: "
                f"candidates={len(candidates)}, drop_zones={len(drop_zones)}"
            )
            if record["selected_candidate"] is not None:
                target = record["selected_candidate"]
                message += (
                    f", selected center={target['center_px']}, "
                    f"yaw={target['yaw_image_deg_clockwise']:.1f}, "
                    f"conf={target['confidence']:.2f}"
                )
            print(message, flush=True)

        frame_record = {
            "frame_set": self._frame_set,
            "pose_label": pose_label,
            "elapsed_s": round(elapsed, 3),
            "commanded_tcp": commanded_tcp,
            "observed_tcp_pose": np.asarray(obs.get("tcp_pose", [])).tolist(),
            "joints": np.asarray(obs.get("joints", [])).tolist(),
            "cameras": camera_records,
        }
        with (self._rollout_dir / "detections.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(frame_record) + "\n")
        self._frame_set += 1

    def _hold_current_tcp(self, obs: dict) -> list[float]:
        tcp = np.asarray(obs.get("tcp_pose", []), dtype=float).reshape(-1)
        if tcp.size < 6:
            # This should not occur on the physical client.  The fallback is a
            # validated high center pose rather than an unbounded target.
            return self._scan_targets[0]["tcp"].copy()
        return tcp[:6].tolist() + [self.OPEN_GRIPPER]

    def act(self, obs: dict):
        """Process each reached pose, then command the next fixed scan pose."""
        if self._rollout_dir is None:
            self.reset()

        assert self._rollout_dir is not None
        if not self._trial_written:
            (self._rollout_dir / "trial.json").write_text(
                json.dumps(obs.get("trial", {}), indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            self._trial_written = True

        try:
            if not self._initial_frame_processed:
                self._process_observation(obs, "initial", None)
                self._initial_frame_processed = True
            elif self._pending_target is not None:
                reached = self._pending_target
                self._process_observation(obs, reached["name"], reached["tcp"])
                self._pending_target = None
        except Exception as exc:
            self._aborted = True
            print(
                f"[safety] inference failed; scan aborted and current TCP held: {exc!r}",
                flush=True,
            )

        if self._aborted or not self.ENABLE_SCAN_MOTION:
            return self._hold_current_tcp(obs)

        if self._next_target_index < len(self._scan_targets):
            target = self._scan_targets[self._next_target_index]
            self._next_target_index += 1
            self._pending_target = target
            print(
                f"[motion] target {self._next_target_index}/"
                f"{len(self._scan_targets)} {target['name']}: {target['tcp']}",
                flush=True,
            )
            return target["tcp"].copy()

        if not self._complete_printed:
            print(
                f"[battery] scan complete: {self._frame_set} observation sets; "
                "the gripper remained open and no grasp was attempted.",
                flush=True,
            )
            self._complete_printed = True

        # Hold the validated elevated center pose after the scan finishes.
        return self._scan_targets[-1]["tcp"].copy()
