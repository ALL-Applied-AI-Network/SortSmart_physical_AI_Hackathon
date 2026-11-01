"""Vision-only AAA pick, carry and bowl-drop test in Isaac Sim.

Control inputs come exclusively from external YOLO instance masks and calibrated
camera geometry.  Simulator poses are read only after planning, for evaluation
and success verification.
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
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--video-camera", choices=("table", "side"), default="side")
parser.add_argument("--no-video-frames", action="store_true")
parser.add_argument(
    "--complex-clutter-demo",
    action="store_true",
    help="use five mixed-brand batteries plus physical-scene distractors",
)
parser.add_argument(
    "--perception-attempts",
    type=int,
    default=5,
    help="maximum table-camera frames evaluated before safe abstention",
)
parser.add_argument(
    "--perception-stride-steps",
    type=int,
    default=4,
    help="rendered physics steps between perception attempts",
)
parser.add_argument(
    "--disable-wrist-fallback",
    action="store_true",
    help="disable the fixed safe wrist-camera scan after a table-camera miss",
)
args = parser.parse_args()
if args.fps != 30:
    parser.error("Use 30 fps; each frame advances two 1/60 s physics steps")
if args.perception_attempts < 1:
    parser.error("--perception-attempts must be at least 1")
if args.perception_stride_steps < 1:
    parser.error("--perception-stride-steps must be at least 1")
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
    import complex_battery_scene
    import fk
    import ik
    import scene_config as C
    from backproject import CameraModel
    from battery_2d_perception import estimate_pose_from_mask
    from battery_sort_planning import make_battery_sort_plan
    from drop_zone_geometry import estimate_drop_zone_from_polygon
    from perception_burst import response_summary, select_best_response
    from wrist_scan_planning import scan_tip_targets_m
    from workspace_roi import projected_workspace_roi
    from yolo_ipc_client import polygon_mask, request_instances

    out = Path(args.out)
    frames = out / "frames"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Use an empty output directory: {out}")
    frames.mkdir(parents=True, exist_ok=True)

    test_x = C.BATTERY_TEST_XY[0] if args.x_m is None else args.x_m
    test_y = C.BATTERY_TEST_XY[1] if args.y_m is None else args.y_m
    yaw_deg = C.BATTERY_TEST_YAW_DEG if args.yaw_deg is None else args.yaw_deg
    if not (0.18 <= test_x <= 0.55 and -0.30 <= test_y <= 0.30):
        raise ValueError("Requested battery XY is outside the validated workspace")
    if math.hypot(test_x, test_y) > 0.52:
        raise ValueError("Requested battery XY is too close to the reach limit")

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    if args.complex_clutter_demo:
        battery_paths = complex_battery_scene.add_to_stage(stage)
        requested_scene = complex_battery_scene.requested_scene_description()
    else:
        battery_paths = cell.add_batteries(
            stage,
            [(test_x, test_y, yaw_deg)],
            brand=args.battery_brand,
            visual_roll_deg=args.battery_visual_roll_deg,
        )
    bowl.add_battery_drop_bowl(stage)
    battery_prims = [stage.GetPrimAtPath(path) for path in battery_paths]
    if not args.complex_clutter_demo:
        battery_prim = battery_prims[0]
        resolved_brand = battery_prim.GetAttribute("batteryBrand").Get()
        visual_roll_deg = battery_prim.GetAttribute("batteryVisualRollDeg").Get()
        requested_scene = {
            "mode": "single_battery",
            "battery_x_m": test_x,
            "battery_y_m": test_y,
            "battery_yaw_deg": yaw_deg,
            "battery_brand": resolved_brand,
            "battery_visual_roll_deg": visual_roll_deg,
        }
    for battery_prim in battery_prims:
        UsdPhysics.RigidBodyAPI(battery_prim).CreateKinematicEnabledAttr().Set(True)

    table_camera = Camera(
        prim_path=cell.TABLE_CAM_PATH,
        resolution=C.TABLE_CAM["resolution"],
    )
    wrist_camera = Camera(
        prim_path=cell.WRIST_CAM_PATH,
        resolution=C.WRIST_CAM["resolution"],
    )
    if args.video_camera == "table":
        video_camera = table_camera
    else:
        video_camera = Camera(prim_path=cell.SIDE_CAM_PATH, resolution=(960, 540))

    cell.enable_ccd(world)
    world.reset()
    table_camera.initialize()
    wrist_camera.initialize()
    if video_camera is not table_camera:
        video_camera.initialize()

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
    configuration = np.deg2rad(
        [C.CAPTURE_POSE_DEG[f"J{i}"] for i in range(1, 7)]
    )
    if np.any(configuration < limits[:, 0]) or np.any(configuration > limits[:, 1]):
        raise RuntimeError("Capture pose exceeds joint limits")

    def command(q6, jaw):
        complete = arm.get_joint_positions().copy()
        complete[arm_indices] = q6
        for name, value in cell.jaw_targets(jaw).items():
            complete[joint_index[name]] = value
        arm.apply_action(ArticulationAction(joint_positions=complete))

    complete = arm.get_joint_positions().copy()
    complete[arm_indices] = configuration
    for name, value in cell.jaw_targets(C.JAW_OPEN).items():
        complete[joint_index[name]] = value
    arm.set_joint_positions(complete)
    # Match the already validated perception-only capture sequence exactly:
    # issue the parked-arm target once, then advance 90 + 60 physics steps.
    # Reissuing the action every step and taking an extra rendered step changes
    # the appearance of a 15 x 53 px cylindrical target enough to move YOLO's
    # confidence across the safety gate.
    arm.apply_action(ArticulationAction(joint_positions=complete))

    def step_frame(record=True):
        for _ in range(2):
            world.step(render=True)
        if record and not args.no_video_frames:
            rgba = video_camera.get_rgba()
            if rgba is not None and rgba.ndim == 3 and rgba.shape[2] >= 3:
                image = cv2.cvtColor(
                    rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(str(frames / f"f{step_frame.number:05d}.png"), image)
                step_frame.number += 1

    step_frame.number = 0

    # Initialise robot drives, materials and cameras while the battery is fixed.
    for _ in range(90):
        world.step(render=True)
    for battery_prim in battery_prims:
        UsdPhysics.RigidBodyAPI(battery_prim).CreateKinematicEnabledAttr().Set(False)
    for _ in range(60):
        world.step(render=True)

    # Capture immediately at the same simulation step as the validated
    # perception-only script.  With 150 rendered warm-up/settling steps the
    # table camera must already have produced a frame.
    rgba = table_camera.get_rgba()
    if rgba is None:
        raise RuntimeError("Table camera returned no usable RGB image")
    bgr = cv2.cvtColor(
        rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(str(out / "planning_rgb.png"), bgr)

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
    workspace_roi = projected_workspace_roi(
        calibrated_camera,
        x_bounds=(0.18, 0.55),
        y_bounds=(-0.18, 0.30),
        height_bounds=(0.0, 0.07),
        margin_px=48,
    )
    # A 15x53-pixel AAA can change confidence substantially after a one-pixel
    # render/shading change.  Evaluate a short burst, exactly as a real camera
    # stream would, and use the best complete observation.  Thresholds remain
    # unchanged; if every frame fails, motion is still safely inhibited.
    observations = []
    for attempt in range(1, args.perception_attempts + 1):
        if attempt > 1:
            for _ in range(args.perception_stride_steps):
                command(configuration, C.JAW_OPEN)
                world.step(render=True)
            candidate = table_camera.get_rgba()
            if (
                candidate is None
                or candidate.ndim != 3
                or candidate.shape[2] < 3
            ):
                raise RuntimeError(
                    f"Table camera returned no usable RGB image on attempt {attempt}"
                )
            bgr = cv2.cvtColor(
                candidate[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
            )
        response = request_instances(
            bgr,
            args.ipc_dir,
            timeout_seconds=args.timeout_seconds,
            roi_xyxy=workspace_roi,
        )
        cv2.imwrite(str(out / f"planning_rgb_attempt_{attempt:02d}.png"), bgr)
        (out / f"ipc_response_attempt_{attempt:02d}.json").write_text(
            json.dumps(response, indent=2) + "\n", encoding="utf-8"
        )
        summary = response_summary(response)
        observations.append(
            {
                "attempt": attempt,
                "image": bgr.copy(),
                "response": response,
                "summary": summary,
            }
        )
        if summary["battery_accepted"] and summary["drop_zone_accepted"]:
            break

    best_index = select_best_response(
        [observation["response"] for observation in observations]
    )
    selected_observation = observations[best_index]
    selected_attempt = selected_observation["attempt"]
    bgr = selected_observation["image"]
    response = selected_observation["response"]
    cv2.imwrite(str(out / "planning_rgb.png"), bgr)
    (out / "ipc_response.json").write_text(
        json.dumps(response, indent=2) + "\n", encoding="utf-8"
    )
    attempt_report = {
        "maximum_attempts": args.perception_attempts,
        "attempts_run": len(observations),
        "selected_attempt": selected_attempt,
        "attempts": [
            {"attempt": item["attempt"], **item["summary"]}
            for item in observations
        ],
    }
    (out / "perception_attempts.json").write_text(
        json.dumps(attempt_report, indent=2) + "\n", encoding="utf-8"
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
    x1, y1, x2, y2 = workspace_roi
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)

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
                    "source_camera": "table",
                }
            )
        except Exception as exc:
            battery_records.append(
                {
                    "confidence": detection["confidence"],
                    "geometry_valid": False,
                    "source_camera": "table",
                    "geometry_error": f"{type(exc).__name__}: {exc}",
                }
            )
        contour = np.rint(np.asarray(detection["polygon_px"])).astype(np.int32)
        cv2.polylines(overlay, [contour], True, (0, 220, 0), 3, cv2.LINE_AA)

    zone_records = []
    for detection in accepted_zones:
        polygon = np.asarray(detection["polygon_px"], dtype=np.float32)
        try:
            pose = estimate_drop_zone_from_polygon(
                polygon,
                calibrated_camera,
                rim_height_above_table_m=bowl.BOWL_HEIGHT,
                expected_outer_radius_m=bowl.BOWL_OUTER_RADIUS,
                workspace_xy=((0.10, 0.70), (-0.47, 0.47)),
            )
            zone_records.append(
                {
                    "confidence": detection["confidence"],
                    **pose.to_dict(),
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
    cv2.imwrite(str(out / "planning_overlay.png"), overlay)

    valid_batteries = [item for item in battery_records if item["geometry_valid"]]
    valid_zones = [
        item
        for item in zone_records
        if item.get("geometry_valid") and item.get("center_xy_m") is not None
    ]
    selected_battery = max(
        valid_batteries, key=lambda item: item["confidence"], default=None
    )
    selected_zone = max(
        valid_zones, key=lambda item: item["confidence"], default=None
    )

    # If the table camera has a valid bowl but misses the tiny battery, move
    # through a fixed high-clearance wrist-camera scan.  The grid is independent
    # of simulator object truth.  Jaws remain open, the tool tip stays 100 mm
    # above the table, and grasp motion is still inhibited until wrist YOLO
    # passes the unchanged 0.60 threshold plus calibrated AAA geometry checks.
    motion_start_configuration = configuration.copy()
    observation_motion_started = False
    wrist_reobservation = {
        "enabled": not args.disable_wrist_fallback,
        "triggered": False,
        "scan_attempts": [],
        "selected_scan": None,
        "selected_battery": None,
    }

    if (
        selected_battery is None
        and selected_zone is not None
        and not args.disable_wrist_fallback
    ):
        wrist_reobservation["triggered"] = True
        scan_configuration = configuration.copy()
        _, scan_tip = fk.tool_pose(scan_configuration, tool_offset)

        def observation_spread(q6):
            rotation, _ = fk.fk(q6)
            value = rotation @ np.asarray(C.FINGER_SPREAD_AXIS, dtype=float)
            value[2] = 0.0
            value /= np.linalg.norm(value)
            return value

        def observation_ease(value):
            return value * value * (3.0 - 2.0 * value)

        def observation_turn(start, finish, progress):
            angle = math.atan2(
                start[0] * finish[1] - start[1] * finish[0],
                float(np.dot(start[:2], finish[:2])),
            )
            c = math.cos(angle * progress)
            s = math.sin(angle * progress)
            return np.array(
                [
                    start[0] * c - start[1] * s,
                    start[0] * s + start[1] * c,
                    0.0,
                ]
            )

        scan_spread = observation_spread(scan_configuration)
        desired_scan_spread = np.array([0.0, 1.0, 0.0])

        def move_observation(q0, start_tip, start_spread, finish_tip):
            local_q = np.asarray(q0, dtype=float).copy()
            start_tip = np.asarray(start_tip, dtype=float)
            finish_tip = np.asarray(finish_tip, dtype=float)
            start_spread = np.asarray(start_spread, dtype=float)
            frame_count = max(1, round(1.2 * args.fps))
            for frame_index in range(frame_count):
                progress = observation_ease((frame_index + 1) / frame_count)
                target = start_tip + (finish_tip - start_tip) * progress
                direction = observation_turn(
                    start_spread, desired_scan_spread, progress
                )
                local_q, solved = ik.solve(
                    local_q,
                    target,
                    downward,
                    limits,
                    tool_offset=tool_offset,
                    spread_dir=direction,
                    iters=500,
                    damping=0.02,
                )
                rotation, actual = fk.tool_pose(local_q, tool_offset)
                actual_spread = rotation @ np.asarray(
                    C.FINGER_SPREAD_AXIS, dtype=float
                )
                aligned = abs(float(np.dot(actual_spread, direction))) > 0.999
                if (
                    not solved
                    or np.linalg.norm(actual - target) > 0.001
                    or not aligned
                ):
                    raise RuntimeError(
                        "IK failed during wrist observation "
                        f"at frame {frame_index}"
                    )
                command(local_q, C.JAW_OPEN)
                step_frame(record=True)
            for _ in range(4):
                command(local_q, C.JAW_OPEN)
                step_frame(record=True)
            return local_q, finish_tip, desired_scan_spread.copy()

        wrist_width, wrist_height = C.WRIST_CAM["resolution"]
        for scan_index, target in enumerate(scan_tip_targets_m(), start=1):
            scan_configuration, scan_tip, scan_spread = move_observation(
                scan_configuration, scan_tip, scan_spread, target
            )
            observation_motion_started = True
            motion_start_configuration = scan_configuration.copy()

            wrist_rgba = wrist_camera.get_rgba()
            if (
                wrist_rgba is None
                or wrist_rgba.ndim != 3
                or wrist_rgba.shape[2] < 3
            ):
                raise RuntimeError("Wrist camera returned no usable RGB image")
            wrist_bgr = cv2.cvtColor(
                wrist_rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(
                str(out / f"wrist_rgb_scan_{scan_index:02d}.png"), wrist_bgr
            )

            wrist_prim = stage.GetPrimAtPath(cell.WRIST_CAM_PATH)
            wrist_usd_camera = UsdGeom.Camera(wrist_prim)
            wrist_transform = UsdGeom.Xformable(
                wrist_prim
            ).ComputeLocalToWorldTransform(0)
            wrist_camera_json = {
                "camera": "wrist",
                "resolution": [wrist_width, wrist_height],
                "focal_length_mm": float(
                    wrist_usd_camera.GetFocalLengthAttr().Get()
                ),
                "horizontal_aperture_mm": float(
                    wrist_usd_camera.GetHorizontalApertureAttr().Get()
                ),
                "vertical_aperture_mm": float(
                    wrist_usd_camera.GetVerticalApertureAttr().Get()
                ),
                "camera_to_world": [
                    [
                        float(wrist_transform[row][column])
                        for column in range(4)
                    ]
                    for row in range(4)
                ],
                "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
                "robot_base_xy": [0.0, 0.0],
            }
            (out / f"wrist_camera_scan_{scan_index:02d}.json").write_text(
                json.dumps(wrist_camera_json, indent=2) + "\n",
                encoding="utf-8",
            )
            calibrated_wrist = CameraModel(wrist_camera_json)
            wrist_response = request_instances(
                wrist_bgr,
                args.ipc_dir,
                timeout_seconds=args.timeout_seconds,
                roi_xyxy=None,
            )
            (
                out / f"wrist_ipc_response_scan_{scan_index:02d}.json"
            ).write_text(
                json.dumps(wrist_response, indent=2) + "\n",
                encoding="utf-8",
            )

            wrist_overlay = wrist_bgr.copy()
            raw_wrist_batteries = [
                item
                for item in wrist_response.get("detections", [])
                if item.get("class_id") == 0
            ]
            wrist_candidates = []
            for detection in raw_wrist_batteries:
                contour = np.rint(
                    np.asarray(detection["polygon_px"])
                ).astype(np.int32)
                colour = (
                    (0, 220, 0)
                    if detection.get("accepted")
                    else (0, 165, 255)
                )
                cv2.polylines(
                    wrist_overlay, [contour], True, colour, 3, cv2.LINE_AA
                )
                if not detection.get("accepted"):
                    continue
                mask = polygon_mask(
                    (wrist_height, wrist_width), detection["polygon_px"]
                )
                try:
                    pose = estimate_pose_from_mask(
                        mask,
                        calibrated_wrist,
                        plane_z=calibrated_wrist.table_z + C.BATTERY_RADIUS,
                        expected_length_m=C.BATTERY_BODY_LENGTH,
                        expected_width_m=C.BATTERY_DIAMETER,
                        workspace_xy=((0.18, 0.55), (-0.30, 0.30)),
                    )
                    wrist_candidates.append(
                        {
                            "confidence": float(detection["confidence"]),
                            "pose": pose.to_dict(),
                            "geometry_valid": True,
                            "source_camera": "wrist",
                            "source_scan": scan_index,
                        }
                    )
                except Exception as exc:
                    wrist_candidates.append(
                        {
                            "confidence": float(detection["confidence"]),
                            "geometry_valid": False,
                            "source_camera": "wrist",
                            "source_scan": scan_index,
                            "geometry_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            cv2.imwrite(
                str(out / f"wrist_overlay_scan_{scan_index:02d}.png"),
                wrist_overlay,
            )
            valid_wrist = [
                item for item in wrist_candidates if item.get("geometry_valid")
            ]
            best_wrist = max(
                valid_wrist,
                key=lambda item: item["confidence"],
                default=None,
            )
            raw_confidence = max(
                (
                    float(item["confidence"])
                    for item in raw_wrist_batteries
                ),
                default=None,
            )
            wrist_reobservation["scan_attempts"].append(
                {
                    "scan": scan_index,
                    "commanded_tip_robot_m": list(target),
                    "battery_raw_confidence": raw_confidence,
                    "battery_accepted_by_yolo": any(
                        bool(item.get("accepted"))
                        for item in raw_wrist_batteries
                    ),
                    "geometry_valid_candidates": len(valid_wrist),
                    "selected_battery": best_wrist,
                    "inference_seconds": wrist_response.get(
                        "inference_seconds"
                    ),
                }
            )
            if best_wrist is not None:
                selected_battery = best_wrist
                battery_records.append(best_wrist)
                wrist_reobservation["selected_scan"] = scan_index
                wrist_reobservation["selected_battery"] = best_wrist
                break

    perception = {
        "battery_candidates": battery_records,
        "drop_zone_candidates": zone_records,
        "selected_battery": selected_battery,
        "selected_drop_zone": selected_zone,
        "workspace_roi_xyxy": workspace_roi,
        "selected_attempt": selected_attempt,
        "perception_attempts": attempt_report,
        "wrist_reobservation": wrist_reobservation,
        "inference_seconds": response.get("inference_seconds"),
        "settings": response.get("settings"),
    }
    (out / "planning_perception.json").write_text(
        json.dumps(perception, indent=2) + "\n", encoding="utf-8"
    )

    if selected_battery is None or selected_zone is None:
        if selected_zone is None:
            abstention_reason = "drop zone was not confidently detected"
        elif wrist_reobservation["triggered"]:
            abstention_reason = (
                "battery was not confidently detected by table or wrist cameras"
            )
        else:
            abstention_reason = "battery was not confidently detected"
        report = {
            "task": "aaa_yolo_pick_place",
            "task_complete": False,
            "safely_abstained": True,
            "reason": abstention_reason,
            "control_source": (
                "external_yolo_instance_masks_plus_calibrated_geometry_"
                "with_fixed_wrist_fallback"
            ),
            "simulator_pose_used_for_control": False,
            "arm_motion_started": observation_motion_started,
            "observation_motion_started": observation_motion_started,
            "grasp_motion_started": False,
            "jaws_closed": False,
            "requested_scene": requested_scene,
            "perception": perception,
        }
        report_path = out / "battery_yolo_pick_place_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[pick-place] safe abstention; report={report_path}", flush=True)
        raise SystemExit(2)

    # All command geometry below is derived before any simulator object pose is
    # read.  The later truth reads are evaluation-only.
    detected_pose = selected_battery["pose"]
    plan = make_battery_sort_plan(
        detected_pose["center_xy_m"],
        detected_pose["axis_xy"],
        selected_zone["center_xy_m"],
        surface_dz_m=C.SURFACE_DZ,
        tip_clearance_m=C.BATTERY_TIP_CLEARANCE,
        bowl_height_m=bowl.BOWL_HEIGHT,
        bowl_floor_height_m=bowl.BOWL_FLOOR_THICKNESS,
        bowl_place_clearance_m=bowl.BOWL_PLACE_CLEARANCE,
    )

    if len(battery_paths) == 1:
        battery_path = battery_paths[0]
    else:
        detected_xy = np.asarray(detected_pose["center_xy_m"], dtype=float)

        def evaluation_distance(path):
            position, _ = SingleXFormPrim(path).get_world_pose()
            return float(np.linalg.norm(np.asarray(position[:2]) - detected_xy))

        # Evaluation-only association after all command geometry is fixed.
        battery_path = min(battery_paths, key=evaluation_distance)
    battery = SingleXFormPrim(battery_path)

    def battery_pose():
        position, quaternion = battery.get_world_pose()
        w, x, y, z = quaternion
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )
        axis = rotation[:, 0]
        axis[2] = 0.0
        axis /= np.linalg.norm(axis)
        return np.asarray(position, float), np.asarray(quaternion, float), axis

    initial_position, initial_quaternion, truth_axis = battery_pose()
    raw_spread = np.array([*plan.raw_spread_xy, 0.0])

    def current_spread(q6):
        rotation, _ = fk.fk(q6)
        value = rotation @ np.asarray(C.FINGER_SPREAD_AXIS, dtype=float)
        value[2] = 0.0
        value /= np.linalg.norm(value)
        return value

    start_spread = current_spread(motion_start_configuration)
    pick_spread = raw_spread if np.dot(raw_spread, start_spread) >= 0 else -raw_spread

    def ease(value):
        return value * value * (3.0 - 2.0 * value)

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

    events = []

    def move(name, start, finish, seconds, jaw_start, jaw_finish, spread_start, spread_finish):
        local_q = move.configuration
        start = np.asarray(start, dtype=float)
        finish = np.asarray(finish, dtype=float)
        frame_count = max(1, round(seconds * args.fps))
        for frame_index in range(frame_count):
            progress = ease((frame_index + 1) / frame_count)
            target = start + (finish - start) * progress
            direction = turn(spread_start, spread_finish, progress)
            local_q, solved = ik.solve(
                local_q,
                target,
                downward,
                limits,
                tool_offset=tool_offset,
                spread_dir=direction,
            )
            if not solved:
                local_q, solved = ik.solve(
                    local_q,
                    target,
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
            if not solved or np.linalg.norm(actual - target) > 0.001 or not aligned:
                raise RuntimeError(f"IK failed during {name} at frame {frame_index}")
            jaw = jaw_start + (jaw_finish - jaw_start) * progress
            command(local_q, jaw)
            step_frame(record=True)
        move.configuration = local_q
        position, _, _ = battery_pose()
        events.append(
            {
                "phase": name,
                "commanded_tip_robot_m": finish.tolist(),
                "battery_center_world_m_evaluation_only": position.tolist(),
            }
        )

    move.configuration = motion_start_configuration
    _, current_tip = fk.tool_pose(motion_start_configuration, tool_offset)
    pick_tip = np.asarray(plan.pick_tip_m)
    pick_above = np.asarray(plan.pick_above_m)
    pick_lift = np.asarray(plan.pick_lift_m)
    bowl_above = np.asarray(plan.bowl_above_m)
    bowl_release = np.asarray(plan.bowl_release_m)

    move("approach", current_tip, pick_above, 1.2, C.JAW_OPEN, C.JAW_OPEN, start_spread, pick_spread)
    move("descend", pick_above, pick_tip, 0.8, C.JAW_OPEN, C.JAW_OPEN, pick_spread, pick_spread)
    move("close", pick_tip, pick_tip, 0.6, C.JAW_OPEN, C.BATTERY_JAW_HALF_TARGET, pick_spread, pick_spread)
    move("grasp_dwell", pick_tip, pick_tip, 0.5, C.BATTERY_JAW_HALF_TARGET, C.BATTERY_JAW_HALF_TARGET, pick_spread, pick_spread)
    position_after_close, _, _ = battery_pose()
    move("lift", pick_tip, pick_lift, 1.0, C.BATTERY_JAW_HALF_TARGET, C.BATTERY_JAW_HALF_TARGET, pick_spread, pick_spread)
    move("lift_hold", pick_lift, pick_lift, 0.5, C.BATTERY_JAW_HALF_TARGET, C.BATTERY_JAW_HALF_TARGET, pick_spread, pick_spread)
    position_after_lift, _, _ = battery_pose()
    move("carry_to_bowl", pick_lift, bowl_above, 1.2, C.BATTERY_JAW_HALF_TARGET, C.BATTERY_JAW_HALF_TARGET, pick_spread, pick_spread)
    position_over_bowl, _, _ = battery_pose()
    move("lower_into_bowl", bowl_above, bowl_release, 1.0, C.BATTERY_JAW_HALF_TARGET, C.BATTERY_JAW_HALF_TARGET, pick_spread, pick_spread)
    position_before_release, _, _ = battery_pose()
    move("release", bowl_release, bowl_release, 0.9, C.BATTERY_JAW_HALF_TARGET, C.JAW_OPEN, pick_spread, pick_spread)
    move("release_dwell", bowl_release, bowl_release, 0.7, C.JAW_OPEN, C.JAW_OPEN, pick_spread, pick_spread)
    move("retreat", bowl_release, bowl_above, 0.7, C.JAW_OPEN, C.JAW_OPEN, pick_spread, pick_spread)

    # Let the released battery settle in the bowl without changing the plan.
    for _ in range(90):
        command(move.configuration, C.JAW_OPEN)
        step_frame(record=True)

    final_position, final_quaternion, final_axis = battery_pose()
    lift_delta = float(position_after_lift[2] - initial_position[2])
    pickup_verified = bool(lift_delta > 0.020)
    detected_bowl_xy = np.asarray(plan.bowl_center_xy_m)
    carried_to_detected_bowl = bool(
        np.linalg.norm(position_over_bowl[:2] - detected_bowl_xy) < 0.035
        and position_over_bowl[2] - initial_position[2] > 0.020
    )
    truth_bowl_xy = np.asarray(bowl.BOWL_XY, dtype=float)
    final_bowl_radius = float(np.linalg.norm(final_position[:2] - truth_bowl_xy))
    surface_world_z = C.TABLE_TOP_Z + C.SURFACE_DZ
    inside_bowl = bool(
        final_bowl_radius < bowl.BOWL_INNER_RADIUS - C.BATTERY_LENGTH / 2.0
        and surface_world_z < final_position[2] < surface_world_z + bowl.BOWL_HEIGHT + 0.030
    )
    task_complete = bool(pickup_verified and carried_to_detected_bowl and inside_bowl)

    report = {
        "task": "aaa_yolo_pick_place",
        "task_complete": task_complete,
        "safely_abstained": False,
        "control_source": (
            "external_yolo_instance_masks_plus_calibrated_geometry_"
            "with_fixed_wrist_fallback"
        ),
        "simulator_pose_used_for_control": False,
        "arm_motion_started": True,
        "observation_motion_started": observation_motion_started,
        "grasp_motion_started": True,
        "video_camera": args.video_camera,
        "frames": step_frame.number,
        "requested_scene": requested_scene,
        "perception": perception,
        "plan": plan.to_dict(),
        "grasp": {
            "jaw_half_target_m": C.BATTERY_JAW_HALF_TARGET,
            "finger_spread_world": pick_spread.tolist(),
            "pickup_verified": pickup_verified,
            "lift_delta_m": lift_delta,
        },
        "placement": {
            "release_strategy": "low_energy_place_on_bowl_floor",
            "battery_surface_clearance_above_floor_m": bowl.BOWL_PLACE_CLEARANCE,
            "carried_to_detected_bowl": carried_to_detected_bowl,
            "inside_bowl": inside_bowl,
            "final_center_distance_to_truth_bowl_m_evaluation_only": final_bowl_radius,
        },
        "evaluation_only": {
            "selected_battery_prim_path": battery_path,
            "initial_position_world_m": initial_position.tolist(),
            "initial_orientation_wxyz": initial_quaternion.tolist(),
            "initial_axis_world": truth_axis.tolist(),
            "position_after_close_world_m": position_after_close.tolist(),
            "position_after_lift_world_m": position_after_lift.tolist(),
            "position_over_bowl_world_m": position_over_bowl.tolist(),
            "position_before_release_world_m": position_before_release.tolist(),
            "final_position_world_m": final_position.tolist(),
            "final_orientation_wxyz": final_quaternion.tolist(),
            "final_axis_world": final_axis.tolist(),
            "truth_bowl_xy_m": list(bowl.BOWL_XY),
        },
        "events": events,
    }
    report_path = out / "battery_yolo_pick_place_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[pick-place] complete={task_complete}; pickup={pickup_verified}; "
        f"carried={carried_to_detected_bowl}; inside_bowl={inside_bowl}; "
        f"report={report_path}",
        flush=True,
    )
    if not task_complete:
        raise SystemExit(2)
finally:
    app.close()
