"""Physical v07.2: guarded vertical descent and depth-only object ranging.

This policy performs no horizontal motion and no grasp.  It validates five
consecutive depth frames, descends along the observed TCP vertical line in four
bounded steps, measures tabletop components for ten frames, and retreats along
the same line.  Every commanded descent step is checked by the calibrated
table-depth height map.  Wrist depth is a secondary consistency gate and uses
the 25 mm uncertainty margin established by v06.1.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration.geometry import SURFACE_Z, UF0_Z_MM, depth_to_base, load
from depth_safety import TableDepthSafetyGuard
from robot_self_filter import classify_robot_self_points


class Policy:
    CAMERAS = ["table", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = ["table", "wrist"]
    DT = 0.2

    OUTPUT_ROOT = Path(__file__).resolve().parent / "guarded_descent_v07_runs"
    TABLE_Z_MM = -335.1
    DESCENT_Z_MM = (-150.0, -190.0, -220.0, -235.0)
    PREFLIGHT_REQUIRED = 5
    PREFLIGHT_MAXIMUM = 10
    LOW_OBSERVATIONS = 10
    STEP_SETTLE_REQUIRED = 3
    STEP_SETTLE_MAXIMUM = 5
    VERTICAL_FOOTPRINT_RADIUS_MM = 75.0
    MINIMUM_TCP_CLEARANCE_MM = 30.0
    WRIST_TABLE_MEDIAN_LIMIT_MM = 2.0
    WRIST_TABLE_P95_LIMIT_MM = 20.0
    WRIST_GUARD_MARGIN_MM = 25.0
    MAXIMUM_EXTERNAL_OBJECT_HEIGHT_MM = 80.0
    SELF_CAPSULE_RADIUS_M = 0.125
    MINIMUM_SELF_EXPLAINED_FRACTION = 0.95

    START_X_MM = (250.0, 350.0)
    START_Y_MM = (-60.0, 60.0)
    START_Z_MM = (-130.0, -50.0)
    MAXIMUM_REACH_ERROR_MM = 6.0

    def __init__(self, checkpoint: str | None = None):
        del checkpoint
        cameras = load()
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

    def reset(self) -> None:
        self._rollout_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rollout_dir = self.OUTPUT_ROOT / (
            f"rollout_{self._rollout_number:02d}_{timestamp}"
        )
        self._rollout_dir.mkdir(parents=True, exist_ok=False)
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
        print(
            f"[v07.2] guarded vertical ranging started; results: {self._rollout_dir}",
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
        components = self._table_components(table, pose)
        valid_wrist = wrist[wrist > 0]
        wrist_range = float(np.percentile(valid_wrist, 2.0)) if len(valid_wrist) else None
        return {
            **safety,
            "observed_tcp_pose": pose.tolist(),
            "table_components": components,
            "battery_sized_candidates": int(
                sum(item["class_hint"] == "battery_sized" for item in components)
            ),
            "nearest_valid_wrist_depth_p02_mm": wrist_range,
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
        all_components = [
            component
            for record in self._low_records
            for component in record.get("table_components", [])
        ]
        report = {
            "task": "guarded_vertical_descent_and_depth_object_ranging_v07",
            "task_complete": self._state == "complete",
            "safely_abstained": self._state == "aborted",
            "reason": self._aborted_reason,
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
            "preflight_frames": len(self._preflight_records),
            "preflight_passed_frames": int(
                sum(item["passed"] for item in self._preflight_records)
            ),
            "descent_safety_frames": len(self._step_settle_records),
            "descent_safety_failed_frames": int(
                sum(not item["passed"] for item in self._step_settle_records)
            ),
            "low_observation_frames": len(self._low_records),
            "frames_with_battery_sized_candidate": int(
                sum(item["battery_sized_candidates"] > 0 for item in self._low_records)
            ),
            "maximum_components_in_one_frame": int(
                max((len(item["table_components"]) for item in self._low_records), default=0)
            ),
            "nearest_measured_object_distance_mm": (
                min(
                    item["euclidean_distance_from_tcp_to_object_top_mm"]
                    for item in all_components
                )
                if all_components
                else None
            ),
            "events": self._events,
        }
        (self._rollout_dir / "guarded_descent_v07_report.json").write_text(
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
        print(f"[v07.2] fail-closed: {reason}; retreating vertically", flush=True)
        self._event("abort", None, reason=reason)

    def act(self, obs: dict):
        if self._rollout_dir is None:
            self.reset()
        pose = self._finite_tcp(obs)
        if pose is None:
            if not self._mock_notice_printed:
                print(
                    "[v07.2] tcp_pose unavailable; issuing validated high mock bootstrap only",
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
                self._preflight_records.append(record)
                self._append_jsonl("preflight.jsonl", record)
                consecutive = 0
                for item in reversed(self._preflight_records):
                    if not item["passed"]:
                        break
                    consecutive += 1
                print(
                    f"[v07.2] preflight {len(self._preflight_records)}/"
                    f"{self.PREFLIGHT_MAXIMUM}: consecutive={consecutive}, "
                    f"clearance={record['vertical_check']['predicted_tcp_clearance_mm']:.1f} mm",
                    flush=True,
                )
                if consecutive >= self.PREFLIGHT_REQUIRED:
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
                        f"[v07.2] step {self._descent_index + 1}/"
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
                self._append_jsonl("low_observations.jsonl", record)
                self._write_report()
                print(
                    f"[v07.2] low observation {len(self._low_records)}/{self.LOW_OBSERVATIONS}: "
                    f"components={len(record['table_components'])}, "
                    f"battery_sized={record['battery_sized_candidates']}, "
                    f"safe={record['passed']}",
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
                        "[v07.2] returned to the observed start pose; no grasp attempted",
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
