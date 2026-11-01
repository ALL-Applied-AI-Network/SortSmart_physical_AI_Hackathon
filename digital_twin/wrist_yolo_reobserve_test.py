"""Truth-independent wrist-camera re-observation test for a missed AAA cell.

The arm visits a fixed, high-clearance scan grid.  External YOLO masks plus
calibrated wrist-camera geometry are the only detection/control inputs.  The
simulator object pose is read only after scanning, for evaluation.  This test
never descends for a grasp and never closes the jaws.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True)
parser.add_argument("--ipc-dir", required=True)
parser.add_argument("--timeout-seconds", type=float, default=90.0)
parser.add_argument("--x-m", type=float, default=0.34)
parser.add_argument("--y-m", type=float, default=-0.12)
parser.add_argument("--yaw-deg", type=float, default=7.0)
parser.add_argument(
    "--battery-brand",
    choices=("amazon", "seven_eleven", "random"),
    default="amazon",
)
parser.add_argument("--battery-visual-roll-deg", type=float, default=90.0)
args = parser.parse_args()
args.out = os.path.abspath(args.out)
args.ipc_dir = os.path.abspath(args.ipc_dir)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

try:
    import cv2
    import numpy as np
    import omni.usd
    from pxr import UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import battery_drop_bowl as bowl
    import cell
    import fk
    import ik
    import scene_config as C
    from backproject import CameraModel
    from battery_2d_perception import estimate_pose_from_mask
    from wrist_scan_planning import scan_tip_targets_m
    from yolo_ipc_client import polygon_mask, request_instances

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Use an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    if not (0.18 <= args.x_m <= 0.55 and -0.30 <= args.y_m <= 0.30):
        raise ValueError("Requested battery XY is outside the validated workspace")

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    battery_path = cell.add_batteries(
        stage,
        [(args.x_m, args.y_m, args.yaw_deg)],
        brand=args.battery_brand,
        visual_roll_deg=args.battery_visual_roll_deg,
    )[0]
    bowl.add_battery_drop_bowl(stage)
    battery_prim = stage.GetPrimAtPath(battery_path)
    resolved_brand = battery_prim.GetAttribute("batteryBrand").Get()
    visual_roll_deg = battery_prim.GetAttribute("batteryVisualRollDeg").Get()
    UsdPhysics.RigidBodyAPI(battery_prim).CreateKinematicEnabledAttr().Set(True)

    wrist_camera = Camera(
        prim_path=cell.WRIST_CAM_PATH,
        resolution=C.WRIST_CAM["resolution"],
    )
    cell.enable_ccd(world)
    world.reset()
    wrist_camera.initialize()

    arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    names = list(arm.dof_names)
    arm_indices = [names.index(f"J{i}") for i in range(1, 7)]
    joint_index = {name: index for index, name in enumerate(names)}
    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    tool_offset = np.asarray(C.TOOL_AXIS, dtype=float) * cell.tip_offset()
    downward = np.array([0.0, 0.0, -1.0])
    scan_spread = np.array([0.0, 1.0, 0.0])
    configuration = np.deg2rad(
        [C.CAPTURE_POSE_DEG[f"J{i}"] for i in range(1, 7)]
    )

    def command(q6):
        complete = arm.get_joint_positions().copy()
        complete[arm_indices] = q6
        for name, value in cell.jaw_targets(C.JAW_OPEN).items():
            complete[joint_index[name]] = value
        arm.apply_action(ArticulationAction(joint_positions=complete))

    complete = arm.get_joint_positions().copy()
    complete[arm_indices] = configuration
    for name, value in cell.jaw_targets(C.JAW_OPEN).items():
        complete[joint_index[name]] = value
    arm.set_joint_positions(complete)
    arm.apply_action(ArticulationAction(joint_positions=complete))

    # Match the validated perception timing before observation motion.
    for _ in range(90):
        world.step(render=True)
    UsdPhysics.RigidBodyAPI(battery_prim).CreateKinematicEnabledAttr().Set(False)
    for _ in range(60):
        world.step(render=True)

    def ease(value):
        return value * value * (3.0 - 2.0 * value)

    def horizontal_spread(q6):
        rotation, _ = fk.fk(q6)
        value = rotation @ np.asarray(C.FINGER_SPREAD_AXIS, dtype=float)
        value[2] = 0.0
        value /= np.linalg.norm(value)
        return value

    def turn(start, finish, progress):
        angle = math.atan2(
            start[0] * finish[1] - start[1] * finish[0],
            float(np.dot(start[:2], finish[:2])),
        )
        c = math.cos(angle * progress)
        s = math.sin(angle * progress)
        return np.array(
            [start[0] * c - start[1] * s, start[0] * s + start[1] * c, 0.0]
        )

    move_configuration = configuration.copy()
    _, move_tip = fk.tool_pose(move_configuration, tool_offset)
    move_spread = horizontal_spread(move_configuration)

    def move_to(target, seconds=1.2):
        global move_configuration, move_tip, move_spread
        start = np.asarray(move_tip, dtype=float)
        finish = np.asarray(target, dtype=float)
        start_spread = np.asarray(move_spread, dtype=float)
        frames = max(1, round(seconds * 60))
        local_q = move_configuration
        for frame_index in range(frames):
            progress = ease((frame_index + 1) / frames)
            point = start + (finish - start) * progress
            direction = turn(start_spread, scan_spread, progress)
            local_q, solved = ik.solve(
                local_q,
                point,
                downward,
                limits,
                tool_offset=tool_offset,
                spread_dir=direction,
                iters=500,
                damping=0.02,
            )
            rotation, actual = fk.tool_pose(local_q, tool_offset)
            actual_spread = rotation @ np.asarray(C.FINGER_SPREAD_AXIS, dtype=float)
            aligned = abs(float(np.dot(actual_spread, direction))) > 0.999
            if not solved or np.linalg.norm(actual - point) > 0.001 or not aligned:
                raise RuntimeError(
                    f"IK failed during wrist observation at frame {frame_index}"
                )
            command(local_q)
            world.step(render=True)
        move_configuration = local_q
        move_tip = finish
        move_spread = scan_spread.copy()
        for _ in range(8):
            command(move_configuration)
            world.step(render=True)

    scan_records = []
    selected_battery = None
    selected_scan = None
    width, height = C.WRIST_CAM["resolution"]

    for scan_index, target in enumerate(scan_tip_targets_m(), start=1):
        move_to(target)
        rgba = wrist_camera.get_rgba()
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
            raise RuntimeError("Wrist camera returned no usable RGB image")
        bgr = cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        image_path = out / f"wrist_rgb_scan_{scan_index:02d}.png"
        cv2.imwrite(str(image_path), bgr)

        camera_prim = stage.GetPrimAtPath(cell.WRIST_CAM_PATH)
        usd_camera = UsdGeom.Camera(camera_prim)
        transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
        camera_json = {
            "camera": "wrist",
            "resolution": [width, height],
            "focal_length_mm": float(usd_camera.GetFocalLengthAttr().Get()),
            "horizontal_aperture_mm": float(
                usd_camera.GetHorizontalApertureAttr().Get()
            ),
            "vertical_aperture_mm": float(
                usd_camera.GetVerticalApertureAttr().Get()
            ),
            "camera_to_world": [
                [float(transform[row][column]) for column in range(4)]
                for row in range(4)
            ],
            "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
            "robot_base_xy": [0.0, 0.0],
        }
        (out / f"wrist_camera_scan_{scan_index:02d}.json").write_text(
            json.dumps(camera_json, indent=2) + "\n", encoding="utf-8"
        )
        calibrated_camera = CameraModel(camera_json)
        response = request_instances(
            bgr,
            args.ipc_dir,
            timeout_seconds=args.timeout_seconds,
            roi_xyxy=None,
        )
        (out / f"wrist_ipc_response_scan_{scan_index:02d}.json").write_text(
            json.dumps(response, indent=2) + "\n", encoding="utf-8"
        )

        overlay = bgr.copy()
        candidates = []
        raw_battery = [
            item
            for item in response.get("detections", [])
            if item.get("class_id") == 0
        ]
        for detection in raw_battery:
            contour = np.rint(np.asarray(detection["polygon_px"])).astype(np.int32)
            colour = (0, 220, 0) if detection.get("accepted") else (0, 165, 255)
            cv2.polylines(overlay, [contour], True, colour, 3, cv2.LINE_AA)
            if not detection.get("accepted"):
                continue
            mask = polygon_mask((height, width), detection["polygon_px"])
            try:
                pose = estimate_pose_from_mask(
                    mask,
                    calibrated_camera,
                    plane_z=calibrated_camera.table_z + C.BATTERY_RADIUS,
                    expected_length_m=C.BATTERY_BODY_LENGTH,
                    expected_width_m=C.BATTERY_DIAMETER,
                    workspace_xy=((0.18, 0.55), (-0.30, 0.30)),
                )
                candidates.append(
                    {
                        "confidence": float(detection["confidence"]),
                        "pose": pose.to_dict(),
                        "geometry_valid": True,
                    }
                )
            except Exception as exc:
                candidates.append(
                    {
                        "confidence": float(detection["confidence"]),
                        "geometry_valid": False,
                        "geometry_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        cv2.imwrite(str(out / f"wrist_overlay_scan_{scan_index:02d}.png"), overlay)
        valid = [item for item in candidates if item.get("geometry_valid")]
        best = max(valid, key=lambda item: item["confidence"], default=None)
        best_raw_confidence = max(
            (float(item["confidence"]) for item in raw_battery), default=None
        )
        scan_record = {
            "scan": scan_index,
            "commanded_tip_robot_m": list(target),
            "battery_raw_confidence": best_raw_confidence,
            "battery_accepted_by_yolo": any(
                bool(item.get("accepted")) for item in raw_battery
            ),
            "geometry_valid_candidates": len(valid),
            "selected_battery": best,
            "inference_seconds": response.get("inference_seconds"),
        }
        scan_records.append(scan_record)
        if best is not None:
            selected_battery = best
            selected_scan = scan_index
            break

    # Truth is read only after every observation/control decision is complete.
    battery = SingleXFormPrim(battery_path)
    truth_position, truth_quaternion = battery.get_world_pose()
    w, x, y, z = truth_quaternion
    truth_rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    truth_axis = truth_rotation[:, 0]
    truth_axis[2] = 0.0
    truth_axis /= np.linalg.norm(truth_axis)
    evaluation = {
        "battery_truth_position_world_m": [float(value) for value in truth_position],
        "battery_truth_axis_xy": [float(value) for value in truth_axis[:2]],
        "battery_center_error_mm": None,
        "battery_yaw_error_deg": None,
    }
    if selected_battery is not None:
        pose = selected_battery["pose"]
        detected_xy = np.asarray(pose["center_xy_m"], dtype=float)
        evaluation["battery_center_error_mm"] = float(
            np.linalg.norm(detected_xy - np.asarray(truth_position[:2])) * 1000.0
        )
        detected_axis = np.asarray(pose["axis_xy"], dtype=float)
        dot = float(np.clip(abs(np.dot(detected_axis, truth_axis[:2])), -1.0, 1.0))
        evaluation["battery_yaw_error_deg"] = float(math.degrees(math.acos(dot)))

    report = {
        "task": "aaa_yolo_wrist_reobserve_test",
        "task_complete": selected_battery is not None,
        "control_source": "fixed_safe_scan_plus_external_yolo_masks_plus_calibrated_geometry",
        "simulator_pose_used_for_control": False,
        "observation_motion_started": True,
        "grasp_motion_started": False,
        "jaws_closed": False,
        "requested_scene": {
            "battery_x_m": float(args.x_m),
            "battery_y_m": float(args.y_m),
            "battery_yaw_deg": float(args.yaw_deg),
            "battery_brand": resolved_brand,
            "battery_visual_roll_deg": float(visual_roll_deg),
        },
        "scan_attempts": scan_records,
        "selected_scan": selected_scan,
        "selected_battery": selected_battery,
        "evaluation_only": evaluation,
    }
    report_path = out / "wrist_yolo_reobserve_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[wrist-reobserve] report={report_path}", flush=True)
    raise SystemExit(0 if selected_battery is not None else 2)
finally:
    app.close()
