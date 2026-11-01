"""Conservative three-camera capture policy with a close wrist-camera scan.

The scan is centered on the robot's TCP position at the beginning of the
rollout.  It mainly moves vertically and uses only +/-30 mm XY offsets.  The
robot rises to a local transit height between every close observation point.

DRY_RUN is intentionally True by default.  Run once to inspect the dynamic
targets, then test with an empty table, low speed, and an operator at E-stop.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


class Policy:
    CAMERAS = ["table", "side", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False

    # Table depth supports global distance/height estimation; wrist depth gives
    # a closer local measurement.  The platform's normal RGB video download
    # does not contain these depth maps, so save both explicitly on this client.
    DEPTH = ["table", "wrist"]

    CAPTURE_ROOT = Path("captured_wrist_close_scan")
    DRY_RUN = False

    TABLE_Z_MM = -335.0
    LOCAL_TRANSIT_Z_MM = -165.0   # about 170 mm above the table
    CAPTURE_Z_MM = -245.0         # about 90 mm above the table

    # Small 3 x 3 scan, relative to the rollout's initial TCP XY.  The points
    # form a short spiral so each horizontal step is only 30 mm.  Total coverage
    # is 60 x 60 mm, still far smaller than the first absolute-grid policy.
    LOCAL_OFFSETS_XY_MM = [
        (0.0, 0.0),
        (30.0, 0.0),
        (30.0, 30.0),
        (0.0, 30.0),
        (-30.0, 30.0),
        (-30.0, 0.0),
        (-30.0, -30.0),
        (0.0, -30.0),
        (30.0, -30.0),
    ]

    # Official reach is about 550 mm from the base axis.  Keep a margin and
    # refuse to move if any generated target exceeds this radius.
    MAX_TARGET_RADIUS_MM = 520.0

    SETTLE_SECONDS = 0.8
    CAPTURES_PER_WAYPOINT = 3
    CAPTURE_GAP_SECONDS = 0.4

    def __init__(self, checkpoint: str | None = None):
        self._rollout_number = 0
        self._capture_dir: Path | None = None
        self._capture_started = 0.0
        self._trial_written = False
        self._dry_run_printed = False
        self._saved_sets = 0
        self._phase = "idle"
        self._waypoint_index = 0
        self._captures_at_waypoint = 0
        self._center_xy: tuple[float, float] | None = None
        self._tool_wpr: tuple[float, float, float] | None = None
        self._gripper = 0.0
        self._safety_blocked = False

    def reset(self) -> None:
        self._rollout_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._capture_dir = self.CAPTURE_ROOT / (
            f"rollout_{self._rollout_number:02d}_{timestamp}"
        )
        self._capture_dir.mkdir(parents=True, exist_ok=False)
        self._capture_started = time.monotonic()
        self._trial_written = False
        self._dry_run_printed = False
        self._saved_sets = 0
        self._phase = "initialize"
        self._waypoint_index = 0
        self._captures_at_waypoint = 0
        self._center_xy = None
        self._tool_wpr = None
        self._gripper = 0.0
        self._safety_blocked = False
        print(
            f"[close-scan] rollout {self._rollout_number} initialized; "
            f"DRY_RUN={self.DRY_RUN}; output={self._capture_dir}",
            flush=True,
        )

    @staticmethod
    def _current_tcp(obs: dict) -> np.ndarray:
        pose = np.asarray(obs.get("tcp_pose", []), dtype=float).reshape(-1)
        if pose.size != 6 or not np.isfinite(pose).all():
            raise RuntimeError(
                "Expected obs['tcp_pose'] = [x, y, z, w, p, r]; "
                f"received {pose.tolist()}"
            )
        return pose

    @staticmethod
    def _hold_current_tcp(obs: dict) -> list[float]:
        pose = Policy._current_tcp(obs)
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        gripper = float(joints[-1]) if joints.size >= 7 else 0.0
        return [*pose.tolist(), float(np.clip(gripper, 0.0, 1.0))]

    def _target(self, x_mm: float, y_mm: float, z_mm: float) -> list[float]:
        if self._tool_wpr is None:
            raise RuntimeError("Scan geometry has not been initialized")
        w_deg, p_deg, r_deg = self._tool_wpr
        return [
            float(x_mm),
            float(y_mm),
            float(z_mm),
            float(w_deg),
            float(p_deg),
            float(r_deg),
            float(self._gripper),
        ]

    def _transit_target(self) -> list[float]:
        if self._center_xy is None:
            raise RuntimeError("Scan geometry has not been initialized")
        return self._target(
            self._center_xy[0], self._center_xy[1], self.LOCAL_TRANSIT_Z_MM
        )

    def _scan_target(self, index: int) -> list[float]:
        if self._center_xy is None:
            raise RuntimeError("Scan geometry has not been initialized")
        dx_mm, dy_mm = self.LOCAL_OFFSETS_XY_MM[index]
        return self._target(
            self._center_xy[0] + dx_mm,
            self._center_xy[1] + dy_mm,
            self.CAPTURE_Z_MM,
        )

    def _scan_plan(self) -> list[dict]:
        return [
            {
                "waypoint_index": index,
                "name": f"close_{index + 1:02d}",
                "offset_xy_mm": list(offset),
                "tcp_target": self._scan_target(index),
            }
            for index, offset in enumerate(self.LOCAL_OFFSETS_XY_MM)
        ]

    def _initialize_geometry(self, obs: dict) -> None:
        pose = self._current_tcp(obs)
        self._center_xy = (float(pose[0]), float(pose[1]))
        # Preserve the known working wrist orientation instead of imposing a
        # new rotation while close to the table.
        self._tool_wpr = (float(pose[3]), float(pose[4]), float(pose[5]))
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        if joints.size >= 7:
            self._gripper = float(np.clip(joints[-1], 0.0, 1.0))

        targets = [self._transit_target(), *[p["tcp_target"] for p in self._scan_plan()]]
        invalid = [
            target
            for target in targets
            if math.hypot(float(target[0]), float(target[1]))
            > self.MAX_TARGET_RADIUS_MM
        ]
        if invalid:
            self._safety_blocked = True
            print(
                "[close-scan] SAFETY BLOCK: at least one local target exceeds "
                f"the {self.MAX_TARGET_RADIUS_MM:.0f} mm reach margin: {invalid}",
                flush=True,
            )

        assert self._capture_dir is not None
        plan = {
            "dry_run": self.DRY_RUN,
            "initial_tcp_pose": pose.tolist(),
            "center_xy_mm": list(self._center_xy),
            "table_z_mm": self.TABLE_Z_MM,
            "local_transit_target": self._transit_target(),
            "capture_z_mm": self.CAPTURE_Z_MM,
            "capture_clearance_above_table_mm": self.CAPTURE_Z_MM - self.TABLE_Z_MM,
            "max_target_radius_mm": self.MAX_TARGET_RADIUS_MM,
            "captures_per_waypoint": self.CAPTURES_PER_WAYPOINT,
            "cameras": self.CAMERAS,
            "depth": self.DEPTH,
            "waypoints": self._scan_plan(),
            "safety_blocked": self._safety_blocked,
        }
        (self._capture_dir / "scan_plan.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )

    def _write_trial_once(self, obs: dict) -> None:
        if self._trial_written:
            return
        assert self._capture_dir is not None
        trial = obs.get("trial", {})
        (self._capture_dir / "trial.json").write_text(
            json.dumps(trial, indent=2, default=str) + "\n", encoding="utf-8"
        )
        self._trial_written = True

    def _save_frame_set(self, obs: dict) -> None:
        assert self._capture_dir is not None
        elapsed = time.monotonic() - self._capture_started
        waypoint_name = f"close_{self._waypoint_index + 1:02d}"
        saved_files: dict[str, str] = {}
        depth_files: dict[str, str] = {}

        for camera in self.CAMERAS:
            image = obs.get("images", {}).get(camera)
            if image is None:
                print(f"[close-scan] warning: no {camera} image", flush=True)
                continue
            rgb = np.asarray(image)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                print(
                    f"[close-scan] warning: unexpected {camera} shape {rgb.shape}",
                    flush=True,
                )
                continue
            rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
            filename = (
                f"{waypoint_name}_{camera}_"
                f"capture_{self._captures_at_waypoint + 1:02d}.png"
            )
            Image.fromarray(rgb).save(self._capture_dir / filename)
            saved_files[camera] = filename

        for camera in self.DEPTH:
            depth = obs.get("depth", {}).get(camera)
            if depth is None:
                continue
            depth_array = np.asarray(depth)
            if depth_array.ndim != 2:
                print(
                    f"[close-scan] warning: unexpected {camera} depth shape "
                    f"{depth_array.shape}",
                    flush=True,
                )
                continue
            depth_array = np.clip(depth_array, 0, 65535).astype(np.uint16)
            filename = (
                f"{waypoint_name}_{camera}_depth_"
                f"capture_{self._captures_at_waypoint + 1:02d}.png"
            )
            Image.fromarray(depth_array).save(self._capture_dir / filename)
            depth_files[camera] = filename

        metadata = {
            "frame_set": self._saved_sets,
            "elapsed_s": round(elapsed, 3),
            "waypoint_index": self._waypoint_index,
            "waypoint_name": waypoint_name,
            "capture_at_waypoint": self._captures_at_waypoint,
            "target_tcp": self._scan_target(self._waypoint_index),
            "files": saved_files,
            "depth_files": depth_files,
            "joints": np.asarray(obs.get("joints", [])).tolist(),
            "tcp_pose": np.asarray(obs.get("tcp_pose", [])).tolist(),
        }
        with (self._capture_dir / "frames.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(metadata) + "\n")

        self._saved_sets += 1
        print(
            f"[close-scan] {waypoint_name}: saved set "
            f"{self._captures_at_waypoint + 1}/{self.CAPTURES_PER_WAYPOINT}",
            flush=True,
        )

    def act(self, obs: dict):
        if self._capture_dir is None:
            self.reset()
        self._write_trial_once(obs)

        if self._center_xy is None:
            self._initialize_geometry(obs)

        if self.DRY_RUN or self._safety_blocked:
            if not self._dry_run_printed:
                print("[close-scan] robot motion disabled.", flush=True)
                print(
                    f"[close-scan] current-center transit: {self._transit_target()}",
                    flush=True,
                )
                for item in self._scan_plan():
                    print(
                        f"[close-scan] {item['name']}: {item['tcp_target']}",
                        flush=True,
                    )
                if self.DRY_RUN and not self._safety_blocked:
                    print(
                        "[close-scan] Validate these local targets, then set "
                        "Policy.DRY_RUN = False.",
                        flush=True,
                    )
                self._dry_run_printed = True
            return self._hold_current_tcp(obs)

        if self._phase in {"initialize", "command_transit"}:
            self._phase = "command_scan"
            print("[close-scan] moving vertically to local transit", flush=True)
            return self._transit_target()

        if self._phase == "command_scan":
            self._phase = "settle"
            print(
                f"[close-scan] descending to close_{self._waypoint_index + 1:02d}",
                flush=True,
            )
            return self._scan_target(self._waypoint_index)

        if self._phase == "settle":
            time.sleep(self.SETTLE_SECONDS)
            self._phase = "capture"
            return self._scan_target(self._waypoint_index)

        if self._phase == "capture":
            self._save_frame_set(obs)
            self._captures_at_waypoint += 1
            if self._captures_at_waypoint < self.CAPTURES_PER_WAYPOINT:
                time.sleep(self.CAPTURE_GAP_SECONDS)
                return self._scan_target(self._waypoint_index)

            self._captures_at_waypoint = 0
            self._waypoint_index += 1
            if self._waypoint_index < len(self.LOCAL_OFFSETS_XY_MM):
                # Always rise before the next small lateral move.
                self._phase = "command_scan"
                print("[close-scan] rising to local transit", flush=True)
                return self._transit_target()

            self._phase = "returning"
            print("[close-scan] capture complete; rising to transit", flush=True)
            return self._transit_target()

        if self._phase == "returning":
            self._phase = "done"
            print(
                f"[close-scan] rollout complete: {self._saved_sets} synchronized "
                f"sets saved in {self._capture_dir}. Operator may stop rollout.",
                flush=True,
            )
            return self._transit_target()

        if self._phase == "done":
            return self._transit_target()

        raise RuntimeError(f"Unknown phase: {self._phase}")
