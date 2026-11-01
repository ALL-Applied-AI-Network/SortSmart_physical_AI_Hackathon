"""Isaac-to-external-YOLO perception integration test for one AAA and bowl.

No grasp is attempted.  The controller receives only table-camera pixels and
the IPC response.  Simulator object poses and the configured bowl centre are
read afterwards strictly to score perception error.
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
parser.add_argument("--x-m", type=float, default=None)
parser.add_argument("--y-m", type=float, default=None)
parser.add_argument("--yaw-deg", type=float, default=None)
parser.add_argument(
    "--battery-brand",
    choices=("amazon", "seven_eleven", "random"),
    default="random",
)
parser.add_argument(
    "--battery-visual-roll-deg",
    type=float,
    default=None,
    help="wrapper roll about the battery axis; default is deterministic randomisation",
)
parser.add_argument(
    "--disable-workspace-roi",
    action="store_true",
    help="send the full camera frame to YOLO (A/B diagnostic only)",
)
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
    import scene_config as C
    from backproject import CameraModel
    from battery_2d_perception import estimate_pose_from_mask
    from drop_zone_geometry import estimate_drop_zone_from_polygon
    from workspace_roi import projected_workspace_roi
    from yolo_ipc_client import polygon_mask, request_instances

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Use an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    x = C.BATTERY_TEST_XY[0] if args.x_m is None else args.x_m
    y = C.BATTERY_TEST_XY[1] if args.y_m is None else args.y_m
    yaw_deg = C.BATTERY_TEST_YAW_DEG if args.yaw_deg is None else args.yaw_deg
    if not (0.18 <= x <= 0.55 and -0.30 <= y <= 0.30):
        raise ValueError("Requested battery XY is outside the validated workspace")

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    battery_path = cell.add_batteries(
        stage,
        [(x, y, yaw_deg)],
        brand=args.battery_brand,
        visual_roll_deg=args.battery_visual_roll_deg,
    )[0]
    bowl.add_battery_drop_bowl(stage)
    battery_prim = stage.GetPrimAtPath(battery_path)
    resolved_brand = battery_prim.GetAttribute("batteryBrand").Get()
    visual_roll_deg = battery_prim.GetAttribute("batteryVisualRollDeg").Get()
    UsdPhysics.RigidBodyAPI(battery_prim).CreateKinematicEnabledAttr().Set(True)

    camera = Camera(
        prim_path=cell.TABLE_CAM_PATH,
        resolution=C.TABLE_CAM["resolution"],
    )
    cell.enable_ccd(world)
    world.reset()
    camera.initialize()

    arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    names = list(arm.dof_names)
    joints = arm.get_joint_positions().copy()
    for joint, degrees in C.CAPTURE_POSE_DEG.items():
        joints[names.index(joint)] = np.deg2rad(degrees)
    for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
        joints[names.index(joint)] = value
    arm.set_joint_positions(joints)
    arm.apply_action(ArticulationAction(joint_positions=joints))

    for _ in range(90):
        world.step(render=True)
    UsdPhysics.RigidBodyAPI(battery_prim).CreateKinematicEnabledAttr().Set(False)
    for _ in range(60):
        world.step(render=True)

    rgba = camera.get_rgba()
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
        raise RuntimeError("Table camera returned no usable RGB image")
    rgb = rgba[:, :, :3].astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / "sim_table_rgb.png"), bgr)

    camera_prim = stage.GetPrimAtPath(cell.TABLE_CAM_PATH)
    usd_camera = UsdGeom.Camera(camera_prim)
    transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
    width, height = C.TABLE_CAM["resolution"]
    camera_json = {
        "camera": "table",
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
    (out / "camera.json").write_text(
        json.dumps(camera_json, indent=2) + "\n", encoding="utf-8"
    )
    calibrated_camera = CameraModel(camera_json)

    workspace_roi = None
    if not args.disable_workspace_roi:
        workspace_roi = projected_workspace_roi(
            calibrated_camera,
            x_bounds=(0.18, 0.55),
            y_bounds=(-0.18, 0.30),
            height_bounds=(0.0, 0.07),
            margin_px=48,
        )

    response = request_instances(
        bgr,
        args.ipc_dir,
        timeout_seconds=args.timeout_seconds,
        roi_xyxy=workspace_roi,
    )
    (out / "ipc_response.json").write_text(
        json.dumps(response, indent=2) + "\n", encoding="utf-8"
    )

    accepted_batteries = [
        item
        for item in response.get("detections", [])
        if item.get("class_id") == 0 and item.get("accepted")
    ]
    accepted_zones = [
        item
        for item in response.get("detections", [])
        if item.get("class_id") == 1 and item.get("accepted")
    ]

    overlay = bgr.copy()
    if workspace_roi is not None:
        x1, y1, x2, y2 = workspace_roi
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            overlay,
            "calibrated workspace ROI",
            (x1 + 8, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    battery_records = []
    for detection in accepted_batteries:
        mask = polygon_mask((height, width), detection["polygon_px"])
        try:
            pose = estimate_pose_from_mask(
                mask,
                calibrated_camera,
                plane_z=calibrated_camera.table_z + C.BATTERY_RADIUS,
                expected_length_m=C.BATTERY_BODY_LENGTH,
                expected_width_m=C.BATTERY_DIAMETER,
                workspace_xy=((0.18, 0.70), (-0.47, 0.47)),
            )
            battery_records.append(
                {
                    "confidence": detection["confidence"],
                    "pose": pose.to_dict(),
                    "geometry_valid": True,
                }
            )
        except Exception as exc:
            battery_records.append(
                {
                    "confidence": detection["confidence"],
                    "geometry_valid": False,
                    "geometry_error": f"{type(exc).__name__}: {exc}",
                }
            )
        contour = np.rint(np.asarray(detection["polygon_px"])).astype(np.int32)
        cv2.polylines(overlay, [contour], True, (0, 220, 0), 3, cv2.LINE_AA)

    zone_records = []
    for detection in accepted_zones:
        polygon = np.asarray(detection["polygon_px"], dtype=np.float32)
        try:
            zone_pose = estimate_drop_zone_from_polygon(
                polygon,
                calibrated_camera,
                rim_height_above_table_m=bowl.BOWL_HEIGHT,
                expected_outer_radius_m=bowl.BOWL_OUTER_RADIUS,
                workspace_xy=((0.10, 0.70), (-0.47, 0.47)),
            )
            zone_records.append(
                {
                    "confidence": detection["confidence"],
                    **zone_pose.to_dict(),
                    "geometry_valid": True,
                }
            )
        except Exception as exc:
            zone_records.append(
                {
                    "confidence": detection["confidence"],
                    "center_xy_m": None,
                    "geometry_valid": False,
                    "geometry_error": f"{type(exc).__name__}: {exc}",
                }
            )
        cv2.polylines(
            overlay,
            [np.rint(polygon).astype(np.int32)],
            True,
            (255, 220, 0),
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out / "sim_yolo_overlay.png"), overlay)

    # Evaluation-only simulator pose.  It is read after perception and never
    # changes the detected pose or any future robot command.
    battery = SingleXFormPrim(battery_path)
    truth_position, truth_quaternion = battery.get_world_pose()
    wq, xq, yq, zq = truth_quaternion
    rotation = np.array(
        [
            [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * wq), 2 * (xq * zq + yq * wq)],
            [2 * (xq * yq + zq * wq), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * wq)],
            [2 * (xq * zq - yq * wq), 2 * (yq * zq + xq * wq), 1 - 2 * (xq * xq + yq * yq)],
        ]
    )
    truth_axis = rotation[:2, 0]
    truth_axis /= np.linalg.norm(truth_axis)

    valid_batteries = [item for item in battery_records if item["geometry_valid"]]
    valid_zones = [
        item
        for item in zone_records
        if item.get("geometry_valid") and item.get("center_xy_m") is not None
    ]
    best_battery = max(valid_batteries, key=lambda item: item["confidence"], default=None)
    best_zone = max(valid_zones, key=lambda item: item["confidence"], default=None)
    centre_error = None
    yaw_error = None
    if best_battery is not None:
        detected_xy = np.asarray(best_battery["pose"]["center_xy_m"], float)
        detected_axis = np.asarray(best_battery["pose"]["axis_xy"], float)
        centre_error = float(np.linalg.norm(detected_xy - truth_position[:2]))
        dot = float(abs(np.dot(detected_axis, truth_axis)))
        yaw_error = math.degrees(math.acos(max(0.0, min(1.0, dot))))
    bowl_error = None
    if best_zone is not None and best_zone["center_xy_m"] is not None:
        bowl_error = float(
            np.linalg.norm(np.asarray(best_zone["center_xy_m"]) - bowl.BOWL_XY)
        )

    task_complete = bool(
        best_battery is not None
        and best_zone is not None
        and best_zone["center_xy_m"] is not None
    )
    report = {
        "task": "aaa_yolo_ipc_perception_test",
        "task_complete": task_complete,
        "requested_scene": {
            "battery_x_m": x,
            "battery_y_m": y,
            "battery_yaw_deg": yaw_deg,
            "battery_brand": resolved_brand,
            "battery_visual_roll_deg": visual_roll_deg,
            "bowl_xy_m": list(bowl.BOWL_XY),
        },
        "control_source": "external_yolo_instance_masks_plus_calibrated_geometry",
        "simulator_pose_used_for_control": False,
        "battery_candidates": battery_records,
        "drop_zone_candidates": zone_records,
        "selected_battery": best_battery,
        "selected_drop_zone": best_zone,
        "evaluation": {
            "battery_truth_position_world_m": truth_position.tolist(),
            "battery_truth_axis_xy": truth_axis.tolist(),
            "battery_center_error_mm": None if centre_error is None else 1000 * centre_error,
            "battery_yaw_error_deg": yaw_error,
            "bowl_center_error_mm": None if bowl_error is None else 1000 * bowl_error,
        },
        "ipc": {
            "directory": args.ipc_dir,
            "workspace_roi_xyxy": workspace_roi,
            "inference_seconds": response.get("inference_seconds"),
            "model": response.get("model"),
            "settings": response.get("settings"),
        },
    }
    report_path = out / "sim_yolo_perception_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[sim-yolo] complete={task_complete}; "
        f"batteries={len(valid_batteries)}; zones={len(zone_records)}; "
        f"report={report_path}",
        flush=True,
    )
    if not task_complete:
        raise SystemExit(2)
finally:
    app.close()

