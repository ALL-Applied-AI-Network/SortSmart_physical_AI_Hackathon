"""Repeated external-YOLO AAA pick/place test in one Isaac Sim scene.

Every grasp target is re-observed from camera images. Simulator object poses
are read only after command geometry is fixed, for evaluation and success
verification. The five complex-scene batteries are placed into deterministic
bowl-relative slots and the complete sequence is recorded as one video.
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
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--video-camera", choices=("table", "side"), default="side")
parser.add_argument("--no-video-frames", action="store_true")
parser.add_argument("--max-batteries", type=int, default=5)
parser.add_argument("--perception-attempts", type=int, default=5)
parser.add_argument("--perception-stride-steps", type=int, default=4)
parser.add_argument("--disable-wrist-fallback", action="store_true")
args = parser.parse_args()
if args.fps != 30:
    parser.error("Use 30 fps; each frame advances four 1/120 s physics steps")
if not 1 <= args.max_batteries <= 5:
    parser.error("--max-batteries must be between 1 and 5")
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
    from multi_battery_planning import (
        PLACEMENT_SLOT_Y_OFFSETS_M,
        placement_slot_xy,
        placement_slot_spec,
        release_tip_robot_z,
        select_pick_candidate,
    )
    from side_rgb_pile_height import estimate_pile_top_from_masks
    from wrist_scan_planning import scan_tip_targets_m
    from workspace_roi import projected_workspace_roi
    from yolo_ipc_client import polygon_mask, request_instances

    out = Path(args.out)
    frames = out / "frames"
    report_path = out / "battery_yolo_multi_pick_place_report.json"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Use an empty output directory: {out}")
    frames.mkdir(parents=True, exist_ok=True)

    physics_hz = 120
    simulation_steps_per_frame = physics_hz // args.fps
    if simulation_steps_per_frame * args.fps != physics_hz:
        raise RuntimeError("physics rate must be an integer multiple of video FPS")
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1 / physics_hz,
        rendering_dt=1 / args.fps,
    )
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    battery_paths = complex_battery_scene.add_to_stage(stage)
    bowl.add_battery_drop_bowl(stage)
    requested_scene = complex_battery_scene.requested_scene_description()
    target_count = min(args.max_batteries, len(battery_paths))

    battery_prims = [stage.GetPrimAtPath(path) for path in battery_paths]
    for prim in battery_prims:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)

    table_camera = Camera(
        prim_path=cell.TABLE_CAM_PATH,
        resolution=C.TABLE_CAM["resolution"],
    )
    wrist_camera = Camera(
        prim_path=cell.WRIST_CAM_PATH,
        resolution=C.WRIST_CAM["resolution"],
    )
    side_resolution = (960, 540)
    side_camera = Camera(
        prim_path=cell.SIDE_CAM_PATH,
        resolution=side_resolution,
    )
    if args.video_camera == "table":
        video_camera = table_camera
    else:
        video_camera = side_camera

    cell.enable_ccd(world)
    world.reset()
    table_camera.initialize()
    wrist_camera.initialize()
    side_camera.initialize()

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
    capture_configuration = np.deg2rad(
        [C.CAPTURE_POSE_DEG[f"J{i}"] for i in range(1, 7)]
    )
    if np.any(capture_configuration < limits[:, 0]) or np.any(
        capture_configuration > limits[:, 1]
    ):
        raise RuntimeError("Capture pose exceeds joint limits")

    def command(q6, jaw):
        complete = arm.get_joint_positions().copy()
        complete[arm_indices] = q6
        for name, value in cell.jaw_targets(jaw).items():
            complete[joint_index[name]] = value
        arm.apply_action(ArticulationAction(joint_positions=complete))

    complete = arm.get_joint_positions().copy()
    complete[arm_indices] = capture_configuration
    for name, value in cell.jaw_targets(C.JAW_OPEN).items():
        complete[joint_index[name]] = value
    arm.set_joint_positions(complete)
    arm.apply_action(ArticulationAction(joint_positions=complete))

    def step_frame(record=True):
        for substep in range(simulation_steps_per_frame):
            world.step(render=substep == simulation_steps_per_frame - 1)
        if record and not args.no_video_frames:
            rgba = video_camera.get_rgba()
            if rgba is not None and rgba.ndim == 3 and rgba.shape[2] >= 3:
                image = cv2.cvtColor(
                    rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                cv2.putText(
                    image,
                    step_frame.status,
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(frames / f"f{step_frame.number:05d}.png"), image)
                step_frame.number += 1

    step_frame.number = 0
    step_frame.status = "initial scene"
    step_frame.cycle = 0

    report = {
        "task": "aaa_yolo_multi_pick_place",
        "status": "initializing",
        "task_complete": False,
        "safely_abstained": False,
        "reason": None,
        "control_source": (
            "repeated_external_yolo_instance_masks_plus_calibrated_geometry_"
            "with_fixed_wrist_fallback"
        ),
        "simulator_pose_used_for_control": False,
        "arm_motion_started": False,
        "grasp_motion_started": False,
        "video_camera": args.video_camera,
        "frames": 0,
        "physics_hz": physics_hz,
        "range_sensing": (
            "calibrated_table_rgb_xy_plus_calibrated_side_rgb_pile_height"
        ),
        "requested_scene": requested_scene,
        "requested_pick_count": target_count,
        "completed_pick_count": 0,
        "placement_slot_y_offsets_m": list(PLACEMENT_SLOT_Y_OFFSETS_M[:target_count]),
        "placement_slots": [
            {
                "offset_xy_m": list(placement_slot_spec(index).offset_xy_m),
                "layer": placement_slot_spec(index).layer,
                "battery_axis": placement_slot_spec(index).battery_axis,
            }
            for index in range(target_count)
        ],
        "detected_bowl_center_xy_m": None,
        "cycles": [],
    }

    def write_report():
        report["frames"] = step_frame.number
        report["completed_pick_count"] = sum(
            bool(cycle.get("task_complete")) for cycle in report["cycles"]
        )
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    write_report()

    # Warm up the robot, physics and cameras while all batteries are fixed.
    for step in range(round(1.5 * physics_hz)):
        world.step(render=(step + 1) % simulation_steps_per_frame == 0)
    for prim in battery_prims:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(False)
    for step in range(round(1.0 * physics_hz)):
        world.step(render=(step + 1) % simulation_steps_per_frame == 0)

    motion = {"configuration": capture_configuration.copy()}
    for _ in range(args.fps):
        command(motion["configuration"], C.JAW_OPEN)
        step_frame(record=True)

    def camera_description(prim_path, cfg, name):
        prim = stage.GetPrimAtPath(prim_path)
        usd_camera = UsdGeom.Camera(prim)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        width, height = cfg["resolution"]
        return {
            "camera": name,
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

    table_camera_json = camera_description(cell.TABLE_CAM_PATH, C.TABLE_CAM, "table")
    (out / "table_camera.json").write_text(
        json.dumps(table_camera_json, indent=2) + "\n", encoding="utf-8"
    )
    calibrated_table = CameraModel(table_camera_json)
    table_width, table_height = C.TABLE_CAM["resolution"]
    workspace_roi = projected_workspace_roi(
        calibrated_table,
        x_bounds=(0.18, 0.55),
        y_bounds=(-0.18, 0.30),
        height_bounds=(0.0, 0.07),
        margin_px=48,
    )
    side_cfg = dict(C.SIDE_CAM)
    side_cfg["resolution"] = side_resolution
    side_camera_json = camera_description(cell.SIDE_CAM_PATH, side_cfg, "side")
    (out / "side_camera.json").write_text(
        json.dumps(side_camera_json, indent=2) + "\n", encoding="utf-8"
    )
    calibrated_side = CameraModel(side_camera_json)

    def camera_bgr(camera, label):
        rgba = camera.get_rgba()
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
            raise RuntimeError(f"{label} camera returned no usable RGB image")
        return cv2.cvtColor(
            rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
        )

    def battery_records(response, camera_model, image_shape, source_camera, source_id):
        height, width = image_shape[:2]
        records = []
        for detection in response.get("detections", []):
            if detection.get("class_id") != 0 or not detection.get("accepted"):
                continue
            mask = polygon_mask((height, width), detection["polygon_px"])
            try:
                pose = estimate_pose_from_mask(
                    mask,
                    camera_model,
                    plane_z=camera_model.table_z + C.BATTERY_RADIUS,
                    expected_length_m=C.BATTERY_BODY_LENGTH,
                    expected_width_m=C.BATTERY_DIAMETER,
                    workspace_xy=((0.18, 0.55), (-0.30, 0.30)),
                )
                records.append(
                    {
                        "confidence": float(detection["confidence"]),
                        "pose": pose.to_dict(),
                        "geometry_valid": True,
                        "source_camera": source_camera,
                        "source_id": source_id,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "confidence": float(detection["confidence"]),
                        "geometry_valid": False,
                        "source_camera": source_camera,
                        "source_id": source_id,
                        "geometry_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return records

    def zone_records(response, camera_model):
        records = []
        for detection in response.get("detections", []):
            if detection.get("class_id") != 1 or not detection.get("accepted"):
                continue
            polygon = np.asarray(detection["polygon_px"], dtype=np.float32)
            try:
                pose = estimate_drop_zone_from_polygon(
                    polygon,
                    camera_model,
                    rim_height_above_table_m=bowl.BOWL_HEIGHT,
                    expected_outer_radius_m=bowl.BOWL_OUTER_RADIUS,
                    workspace_xy=((0.10, 0.70), (-0.47, 0.47)),
                )
                records.append(
                    {
                        "confidence": float(detection["confidence"]),
                        **pose.to_dict(),
                        "geometry_valid": True,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "confidence": float(detection["confidence"]),
                        "center_xy_m": None,
                        "geometry_valid": False,
                        "geometry_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return records

    def draw_overlay(image, response, roi=None):
        overlay = image.copy()
        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
        for detection in response.get("detections", []):
            polygon = detection.get("polygon_px")
            if not polygon:
                continue
            contour = np.rint(np.asarray(polygon)).astype(np.int32)
            if detection.get("class_id") == 0:
                colour = (0, 220, 0) if detection.get("accepted") else (0, 165, 255)
            else:
                colour = (255, 220, 0) if detection.get("accepted") else (0, 165, 255)
            cv2.polylines(overlay, [contour], True, colour, 3, cv2.LINE_AA)
        return overlay

    def observe_side_pile_height(cycle_index, bowl_center_xy):
        """Estimate pile top from calibrated side RGB before a stacked drop."""

        step_frame.status = (
            f"cycle {cycle_index + 1}/{target_count} | side RGB pile height"
        )
        cycle_dir = out / f"cycle_{cycle_index + 1:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        image = camera_bgr(side_camera, "side")
        response = request_instances(
            image,
            args.ipc_dir,
            timeout_seconds=args.timeout_seconds,
            roi_xyxy=None,
        )
        masks = []
        mask_detections = []
        for detection in response.get("detections", []):
            if detection.get("class_id") != 0:
                continue
            confidence = float(detection.get("confidence", 0.0))
            polygon = detection.get("polygon_px")
            # A low-confidence silhouette can safely raise the release height;
            # it can never lower the count-based fallback height.
            if confidence < 0.25 or not polygon:
                continue
            masks.append(polygon_mask(image.shape[:2], polygon))
            mask_detections.append(
                {
                    "confidence": confidence,
                    "accepted_by_yolo": bool(detection.get("accepted")),
                }
            )

        floor_top_world_z = (
            C.TABLE_TOP_Z + C.SURFACE_DZ + bowl.BOWL_FLOOR_THICKNESS
        )
        camera_side = 1.0 if calibrated_side.eye[1] >= bowl_center_xy[1] else -1.0
        raw = estimate_pile_top_from_masks(
            calibrated_side,
            masks,
            bowl_center_xy,
            floor_top_world_z=floor_top_world_z,
            bowl_radius_m=bowl.BOWL_INNER_RADIUS,
            maximum_height_above_floor_m=0.060,
            assumed_y_offset_toward_camera_m=camera_side * 0.035,
            percentile=99.0,
        )
        # Four successful first-layer placements imply at least one battery
        # diameter above the floor. RGB refines upward from that prior, while a
        # 12 mm clamp prevents an unrelated side-view false positive from
        # commanding an unnecessarily high drop.
        planned_first_layer_top = floor_top_world_z + C.BATTERY_DIAMETER
        if raw["valid"]:
            bounded_visual_top = float(
                np.clip(
                    raw["pile_top_world_z"],
                    floor_top_world_z,
                    planned_first_layer_top + 0.012,
                )
            )
            fused_top = max(planned_first_layer_top, bounded_visual_top)
            source = "max(planned_first_layer, calibrated_side_rgb_upper_bound)"
        else:
            bounded_visual_top = None
            fused_top = planned_first_layer_top + 0.004
            source = "conservative_count_based_fallback_after_side_rgb_miss"

        result = {
            "sensor_mode": "calibrated_side_camera_rgb_only",
            "simulator_depth_used": False,
            "simulator_object_pose_used": False,
            "raw_rgb_geometry": raw,
            "mask_detections": mask_detections,
            "floor_top_world_z": floor_top_world_z,
            "planned_first_layer_top_world_z": planned_first_layer_top,
            "bounded_visual_top_world_z": bounded_visual_top,
            "fused_pile_top_world_z": fused_top,
            "fusion_source": source,
        }
        cv2.imwrite(str(cycle_dir / "side_pile_rgb.png"), image)
        cv2.imwrite(
            str(cycle_dir / "side_pile_overlay.png"),
            draw_overlay(image, response),
        )
        (cycle_dir / "side_pile_response.json").write_text(
            json.dumps(response, indent=2) + "\n", encoding="utf-8"
        )
        (cycle_dir / "side_pile_height.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result

    def observe_table(cycle_index, known_bowl_xy):
        step_frame.status = f"cycle {cycle_index + 1}/{target_count} | table perception"
        cycle_dir = out / f"cycle_{cycle_index + 1:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        observations = []
        for attempt in range(1, args.perception_attempts + 1):
            if attempt > 1:
                for _ in range(args.perception_stride_steps):
                    command(motion["configuration"], C.JAW_OPEN)
                    step_frame(record=True)
            image = camera_bgr(table_camera, "table")
            response = request_instances(
                image,
                args.ipc_dir,
                timeout_seconds=args.timeout_seconds,
                roi_xyxy=workspace_roi,
            )
            batteries = battery_records(
                response,
                calibrated_table,
                image.shape,
                "table",
                attempt,
            )
            zones = zone_records(response, calibrated_table)
            selected_zone = max(
                (
                    item
                    for item in zones
                    if item.get("geometry_valid")
                    and item.get("center_xy_m") is not None
                ),
                key=lambda item: item["confidence"],
                default=None,
            )
            candidate_bowl = (
                known_bowl_xy
                if known_bowl_xy is not None
                else None if selected_zone is None else selected_zone["center_xy_m"]
            )
            selected_battery = None
            if candidate_bowl is not None:
                selected_battery = select_pick_candidate(
                    batteries,
                    candidate_bowl,
                    exclusion_radius_m=complex_battery_scene.BOWL_EXCLUSION_RADIUS_M,
                )
            overlay = draw_overlay(image, response, workspace_roi)
            cv2.imwrite(
                str(cycle_dir / f"table_rgb_attempt_{attempt:02d}.png"), image
            )
            cv2.imwrite(
                str(cycle_dir / f"table_overlay_attempt_{attempt:02d}.png"),
                overlay,
            )
            (cycle_dir / f"table_response_attempt_{attempt:02d}.json").write_text(
                json.dumps(response, indent=2) + "\n", encoding="utf-8"
            )
            rank = (
                int(selected_battery is not None and candidate_bowl is not None),
                0.0 if selected_battery is None else selected_battery["confidence"],
                0.0 if selected_zone is None else selected_zone["confidence"],
            )
            observations.append(
                {
                    "attempt": attempt,
                    "response": response,
                    "battery_candidates": batteries,
                    "drop_zone_candidates": zones,
                    "selected_battery": selected_battery,
                    "selected_drop_zone": selected_zone,
                    "candidate_bowl_xy_m": candidate_bowl,
                    "rank": rank,
                }
            )
            if rank[0]:
                break
        selected = max(observations, key=lambda item: item["rank"])
        summary = {
            "maximum_attempts": args.perception_attempts,
            "attempts_run": len(observations),
            "selected_attempt": selected["attempt"],
            "attempts": [
                {
                    "attempt": item["attempt"],
                    "battery_candidates": len(item["battery_candidates"]),
                    "selected_battery_confidence": (
                        None
                        if item["selected_battery"] is None
                        else item["selected_battery"]["confidence"]
                    ),
                    "drop_zone_candidates": len(item["drop_zone_candidates"]),
                    "selected_drop_zone_confidence": (
                        None
                        if item["selected_drop_zone"] is None
                        else item["selected_drop_zone"]["confidence"]
                    ),
                }
                for item in observations
            ],
        }
        (cycle_dir / "table_perception_attempts.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        selected["attempt_summary"] = summary
        return selected

    def ease(value):
        return value * value * (3.0 - 2.0 * value)

    def current_spread(q6):
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

    def move_tool(
        name,
        start,
        finish,
        seconds,
        jaw_start,
        jaw_finish,
        spread_start,
        spread_finish,
        *,
        event_log=None,
        battery_pose_fn=None,
    ):
        step_frame.status = f"cycle {step_frame.cycle}/{target_count} | {name}"
        local_q = motion["configuration"]
        start = np.asarray(start, dtype=float)
        finish = np.asarray(finish, dtype=float)
        spread_start = np.asarray(spread_start, dtype=float)
        spread_finish = np.asarray(spread_finish, dtype=float)
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
        motion["configuration"] = local_q
        if event_log is not None:
            event = {
                "phase": name,
                "commanded_tip_robot_m": finish.tolist(),
            }
            if battery_pose_fn is not None:
                position, _, _ = battery_pose_fn()
                event["battery_center_world_m_evaluation_only"] = position.tolist()
            event_log.append(event)

    def move_joints(name, target_q, seconds, jaw, event_log=None):
        step_frame.status = f"cycle {step_frame.cycle}/{target_count} | {name}"
        start_q = motion["configuration"].copy()
        target_q = np.asarray(target_q, dtype=float)
        frame_count = max(1, round(seconds * args.fps))
        for frame_index in range(frame_count):
            progress = ease((frame_index + 1) / frame_count)
            q = start_q + (target_q - start_q) * progress
            command(q, jaw)
            step_frame(record=True)
        motion["configuration"] = target_q.copy()
        if event_log is not None:
            event_log.append({"phase": name})

    def observe_wrist(cycle_index, bowl_center_xy):
        step_frame.status = f"cycle {cycle_index + 1}/{target_count} | wrist reobserve"
        cycle_dir = out / f"cycle_{cycle_index + 1:02d}"
        scan_report = {
            "enabled": not args.disable_wrist_fallback,
            "triggered": True,
            "scan_attempts": [],
            "selected_scan": None,
        }
        desired_spread = np.array([0.0, 1.0, 0.0])
        start_spread = current_spread(motion["configuration"])
        if np.dot(start_spread, desired_spread) < 0:
            desired_spread = -desired_spread
        for scan_index, target in enumerate(scan_tip_targets_m(), start=1):
            _, start_tip = fk.tool_pose(motion["configuration"], tool_offset)
            move_tool(
                f"wrist_scan_{scan_index}",
                start_tip,
                target,
                1.2,
                C.JAW_OPEN,
                C.JAW_OPEN,
                start_spread,
                desired_spread,
            )
            for _ in range(4):
                command(motion["configuration"], C.JAW_OPEN)
                step_frame(record=True)
            start_spread = desired_spread.copy()
            image = camera_bgr(wrist_camera, "wrist")
            camera_json = camera_description(
                cell.WRIST_CAM_PATH, C.WRIST_CAM, "wrist"
            )
            camera_model = CameraModel(camera_json)
            response = request_instances(
                image,
                args.ipc_dir,
                timeout_seconds=args.timeout_seconds,
                roi_xyxy=None,
            )
            candidates = battery_records(
                response,
                camera_model,
                image.shape,
                "wrist",
                scan_index,
            )
            selected = select_pick_candidate(
                candidates,
                bowl_center_xy,
                exclusion_radius_m=complex_battery_scene.BOWL_EXCLUSION_RADIUS_M,
            )
            cv2.imwrite(
                str(cycle_dir / f"wrist_rgb_scan_{scan_index:02d}.png"), image
            )
            cv2.imwrite(
                str(cycle_dir / f"wrist_overlay_scan_{scan_index:02d}.png"),
                draw_overlay(image, response),
            )
            (cycle_dir / f"wrist_camera_scan_{scan_index:02d}.json").write_text(
                json.dumps(camera_json, indent=2) + "\n", encoding="utf-8"
            )
            (cycle_dir / f"wrist_response_scan_{scan_index:02d}.json").write_text(
                json.dumps(response, indent=2) + "\n", encoding="utf-8"
            )
            scan_report["scan_attempts"].append(
                {
                    "scan": scan_index,
                    "commanded_tip_robot_m": list(target),
                    "geometry_valid_candidates": sum(
                        bool(item.get("geometry_valid")) for item in candidates
                    ),
                    "selected_battery": selected,
                    "inference_seconds": response.get("inference_seconds"),
                }
            )
            if selected is not None:
                scan_report["selected_scan"] = scan_index
                return selected, scan_report
        return None, scan_report

    remaining_paths = list(battery_paths)
    picked_paths = []
    detected_bowl_xy = None
    arm_motion_started = False
    grasp_motion_started = False
    abstention_reason = None

    for cycle_index in range(target_count):
        step_frame.cycle = cycle_index + 1
        print(
            f"[multi-pick] cycle={cycle_index + 1}/{target_count}: observing",
            flush=True,
        )
        table_observation = observe_table(cycle_index, detected_bowl_xy)
        selected_zone = table_observation["selected_drop_zone"]
        if detected_bowl_xy is None and selected_zone is not None:
            detected_bowl_xy = [float(v) for v in selected_zone["center_xy_m"]]
            report["detected_bowl_center_xy_m"] = detected_bowl_xy

        selected_battery = table_observation["selected_battery"]
        wrist_report = {
            "enabled": not args.disable_wrist_fallback,
            "triggered": False,
            "scan_attempts": [],
            "selected_scan": None,
        }
        if detected_bowl_xy is None:
            abstention_reason = "drop zone was not confidently detected on cycle 1"
        elif selected_battery is None and not args.disable_wrist_fallback:
            selected_battery, wrist_report = observe_wrist(
                cycle_index, detected_bowl_xy
            )
            arm_motion_started = True
        if selected_battery is None and abstention_reason is None:
            abstention_reason = (
                f"no battery outside the bowl passed vision gates on cycle "
                f"{cycle_index + 1}"
            )

        perception_report = {
            "table": {
                "selected_attempt": table_observation["attempt"],
                "attempt_summary": table_observation["attempt_summary"],
                "battery_candidates": table_observation["battery_candidates"],
                "drop_zone_candidates": table_observation["drop_zone_candidates"],
                "selected_battery": table_observation["selected_battery"],
                "selected_drop_zone": table_observation["selected_drop_zone"],
                "workspace_roi_xyxy": workspace_roi,
            },
            "wrist_reobservation": wrist_report,
            "selected_battery": selected_battery,
            "locked_drop_zone_center_xy_m": detected_bowl_xy,
        }

        if abstention_reason is not None:
            report["cycles"].append(
                {
                    "cycle": cycle_index + 1,
                    "task_complete": False,
                    "safely_abstained": True,
                    "reason": abstention_reason,
                    "grasp_motion_started": False,
                    "perception": perception_report,
                }
            )
            report["status"] = "safely_abstained"
            report["safely_abstained"] = True
            report["reason"] = abstention_reason
            report["arm_motion_started"] = arm_motion_started
            report["grasp_motion_started"] = grasp_motion_started
            write_report()
            break

        slot_xy = placement_slot_xy(detected_bowl_xy, cycle_index)
        slot_spec = placement_slot_spec(cycle_index)
        pile_height_observation = None
        if slot_spec.layer > 0:
            pile_height_observation = observe_side_pile_height(
                cycle_index, detected_bowl_xy
            )
        detected_pose = selected_battery["pose"]
        plan = make_battery_sort_plan(
            detected_pose["center_xy_m"],
            detected_pose["axis_xy"],
            slot_xy,
            surface_dz_m=C.SURFACE_DZ,
            tip_clearance_m=C.BATTERY_TIP_CLEARANCE,
            bowl_height_m=bowl.BOWL_HEIGHT,
            bowl_floor_height_m=bowl.BOWL_FLOOR_THICKNESS,
            bowl_place_clearance_m=bowl.BOWL_PLACE_CLEARANCE,
        )

        # Evaluation-only association after every command coordinate is fixed.
        detected_xy = np.asarray(detected_pose["center_xy_m"], dtype=float)

        def evaluation_distance(path):
            position, _ = SingleXFormPrim(path).get_world_pose()
            return float(np.linalg.norm(np.asarray(position[:2]) - detected_xy))

        evaluation_pool = remaining_paths or battery_paths
        battery_path = min(evaluation_pool, key=evaluation_distance)
        battery = SingleXFormPrim(battery_path)

        def selected_battery_pose():
            position, quaternion = battery.get_world_pose()
            w, x, y, z = quaternion
            rotation = np.array(
                [
                    [
                        1 - 2 * (y * y + z * z),
                        2 * (x * y - z * w),
                        2 * (x * z + y * w),
                    ],
                    [
                        2 * (x * y + z * w),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w),
                    ],
                    [
                        2 * (x * z - y * w),
                        2 * (y * z + x * w),
                        1 - 2 * (x * x + y * y),
                    ],
                ],
                dtype=float,
            )
            axis = rotation[:, 0]
            axis[2] = 0.0
            axis /= np.linalg.norm(axis)
            return np.asarray(position, float), np.asarray(quaternion, float), axis

        initial_position, initial_quaternion, truth_axis = selected_battery_pose()
        raw_spread = np.array([*plan.raw_spread_xy, 0.0])
        start_spread = current_spread(motion["configuration"])
        pick_spread = (
            raw_spread if np.dot(raw_spread, start_spread) >= 0 else -raw_spread
        )
        release_spread = (
            np.array([0.0, 1.0, 0.0])
            if slot_spec.battery_axis == "x"
            else np.array([1.0, 0.0, 0.0])
        )
        if np.dot(release_spread, pick_spread) < 0:
            release_spread = -release_spread

        _, current_tip = fk.tool_pose(motion["configuration"], tool_offset)
        pick_tip = np.asarray(plan.pick_tip_m)
        pick_above = np.asarray(plan.pick_above_m)
        pick_lift = np.asarray(plan.pick_lift_m)
        bowl_above = np.asarray(plan.bowl_above_m)
        bowl_release = np.asarray(plan.bowl_release_m)
        commanded_bottom_clearance_m = bowl.BOWL_PLACE_CLEARANCE
        if pile_height_observation is not None:
            bowl_release[2] = release_tip_robot_z(
                pile_height_observation["fused_pile_top_world_z"],
                robot_base_world_z=C.TABLE_TOP_Z,
                tip_clearance_m=C.BATTERY_TIP_CLEARANCE,
                release_clearance_m=0.003,
            )
            commanded_bottom_clearance_m = float(
                C.TABLE_TOP_Z
                + bowl_release[2]
                - C.BATTERY_TIP_CLEARANCE
                - pile_height_observation["fused_pile_top_world_z"]
            )
            if commanded_bottom_clearance_m < 0.0025:
                raise RuntimeError(
                    "stacked release violates the visual pile-clearance gate"
                )
        events = []
        arm_motion_started = True
        grasp_motion_started = True

        move_tool(
            "approach",
            current_tip,
            pick_above,
            1.2,
            C.JAW_OPEN,
            C.JAW_OPEN,
            start_spread,
            pick_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        move_tool(
            "descend",
            pick_above,
            pick_tip,
            0.8,
            C.JAW_OPEN,
            C.JAW_OPEN,
            pick_spread,
            pick_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        move_tool(
            "close",
            pick_tip,
            pick_tip,
            0.6,
            C.JAW_OPEN,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            pick_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        move_tool(
            "grasp_dwell",
            pick_tip,
            pick_tip,
            0.5,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            pick_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        position_after_close, _, _ = selected_battery_pose()
        move_tool(
            "lift",
            pick_tip,
            pick_lift,
            1.0,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            pick_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        move_tool(
            "lift_hold",
            pick_lift,
            pick_lift,
            0.4,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            pick_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        position_after_lift, _, _ = selected_battery_pose()
        move_tool(
            "carry_and_align",
            pick_lift,
            bowl_above,
            1.3,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            release_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        position_over_bowl, _, _ = selected_battery_pose()
        move_tool(
            "lower_into_bowl",
            bowl_above,
            bowl_release,
            1.8 if slot_spec.layer > 0 else 1.3,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            release_spread,
            release_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        position_before_release, _, _ = selected_battery_pose()
        move_tool(
            "release",
            bowl_release,
            bowl_release,
            0.9,
            C.BATTERY_JAW_HALF_TARGET,
            C.JAW_OPEN,
            release_spread,
            release_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        move_tool(
            "release_dwell",
            bowl_release,
            bowl_release,
            0.5,
            C.JAW_OPEN,
            C.JAW_OPEN,
            release_spread,
            release_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )
        move_tool(
            "retreat",
            bowl_release,
            bowl_above,
            0.7,
            C.JAW_OPEN,
            C.JAW_OPEN,
            release_spread,
            release_spread,
            event_log=events,
            battery_pose_fn=selected_battery_pose,
        )

        for _ in range(60):
            command(motion["configuration"], C.JAW_OPEN)
            step_frame(record=True)

        final_position, final_quaternion, final_axis = selected_battery_pose()
        lift_delta = float(position_after_lift[2] - initial_position[2])
        pickup_verified = bool(lift_delta > 0.020)
        slot_xy_array = np.asarray(slot_xy, dtype=float)
        carried_to_slot = bool(
            np.linalg.norm(position_over_bowl[:2] - slot_xy_array) < 0.035
            and position_over_bowl[2] - initial_position[2] > 0.020
        )
        truth_bowl_xy = np.asarray(bowl.BOWL_XY, dtype=float)
        final_bowl_radius = float(
            np.linalg.norm(final_position[:2] - truth_bowl_xy)
        )
        surface_world_z = C.TABLE_TOP_Z + C.SURFACE_DZ
        inside_bowl = bool(
            final_bowl_radius < bowl.BOWL_INNER_RADIUS - C.BATTERY_LENGTH / 2.0
            and surface_world_z
            < final_position[2]
            < surface_world_z + bowl.BOWL_HEIGHT + 0.050
        )
        cycle_complete = bool(pickup_verified and carried_to_slot and inside_bowl)

        cycle_report = {
            "cycle": cycle_index + 1,
            "task_complete": cycle_complete,
            "safely_abstained": False,
            "grasp_motion_started": True,
            "perception": perception_report,
            "placement_slot_index": cycle_index,
            "placement_slot_xy_m": list(slot_xy),
            "placement_layer": slot_spec.layer,
            "release_battery_axis": slot_spec.battery_axis,
            "pile_height_observation": pile_height_observation,
            "commanded_release_tip_robot_m": bowl_release.tolist(),
            "commanded_battery_bottom_clearance_m": (
                commanded_bottom_clearance_m
            ),
            "plan": plan.to_dict(),
            "grasp": {
                "jaw_half_target_m": C.BATTERY_JAW_HALF_TARGET,
                "pick_finger_spread_world": pick_spread.tolist(),
                "release_finger_spread_world": release_spread.tolist(),
                "pickup_verified": pickup_verified,
                "lift_delta_m": lift_delta,
            },
            "placement": {
                "release_strategy": (
                    "rgb_height_aware_cross_layer_place"
                    if slot_spec.layer > 0
                    else "spaced_first_layer_low_energy_place"
                ),
                "carried_to_slot": carried_to_slot,
                "inside_bowl": inside_bowl,
                "final_center_distance_to_truth_bowl_m_evaluation_only": (
                    final_bowl_radius
                ),
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
        report["cycles"].append(cycle_report)
        if cycle_complete and battery_path in remaining_paths:
            remaining_paths.remove(battery_path)
            picked_paths.append(battery_path)
        report["arm_motion_started"] = arm_motion_started
        report["grasp_motion_started"] = grasp_motion_started
        report["status"] = "running" if cycle_complete else "failed_quality_gate"
        if not cycle_complete:
            report["reason"] = f"cycle {cycle_index + 1} failed a motion quality gate"
        write_report()

        # Return the empty gripper to the validated capture posture before the
        # next table-camera observation.  This is a fixed, truth-independent
        # joint target and is recorded as part of the continuous video.
        move_joints(
            "return_to_capture",
            capture_configuration,
            1.4,
            C.JAW_OPEN,
            event_log=events,
        )
        for _ in range(round(0.4 * args.fps)):
            command(motion["configuration"], C.JAW_OPEN)
            step_frame(record=True)
        write_report()

        if not cycle_complete:
            break

    successful_cycles = sum(
        bool(cycle.get("task_complete")) for cycle in report["cycles"]
    )
    if successful_cycles == target_count:
        step_frame.status = "final bowl inventory check"
        for _ in range(60):
            command(motion["configuration"], C.JAW_OPEN)
            step_frame(record=True)

    surface_world_z = C.TABLE_TOP_Z + C.SURFACE_DZ
    truth_bowl_xy = np.asarray(bowl.BOWL_XY, dtype=float)
    final_inventory = []
    for path in picked_paths:
        position, _ = SingleXFormPrim(path).get_world_pose()
        position = np.asarray(position, dtype=float)
        radius = float(np.linalg.norm(position[:2] - truth_bowl_xy))
        still_inside = bool(
            radius < bowl.BOWL_INNER_RADIUS - C.BATTERY_LENGTH / 2.0
            and surface_world_z
            < position[2]
            < surface_world_z + bowl.BOWL_HEIGHT + 0.050
        )
        final_inventory.append(
            {
                "battery_prim_path": path,
                "final_position_world_m": position.tolist(),
                "center_distance_to_truth_bowl_m": radius,
                "inside_bowl": still_inside,
            }
        )
    all_picked_still_inside = bool(
        len(final_inventory) == target_count
        and all(item["inside_bowl"] for item in final_inventory)
    )
    report["task_complete"] = bool(
        successful_cycles == target_count and all_picked_still_inside
    )
    report["completed_pick_count"] = successful_cycles
    report["all_quality_gates_passed"] = report["task_complete"]
    report["evaluation_only"] = {
        "remaining_unassociated_battery_prim_paths": remaining_paths,
        "final_inventory": final_inventory,
        "all_picked_batteries_still_inside_bowl": all_picked_still_inside,
    }
    if report["task_complete"]:
        report["status"] = "complete"
        report["safely_abstained"] = False
        report["reason"] = None
    elif successful_cycles == target_count and not all_picked_still_inside:
        report["status"] = "failed_final_inventory_gate"
        report["reason"] = (
            "one or more previously placed batteries were outside the bowl "
            "during the final inventory check"
        )
    elif report["status"] == "running":
        report["status"] = "incomplete"
    write_report()

    print(
        f"[multi-pick] complete={report['task_complete']}; "
        f"successful={successful_cycles}/{target_count}; "
        f"frames={step_frame.number}; report={report_path}",
        flush=True,
    )
    if not report["task_complete"]:
        raise SystemExit(2)
finally:
    app.close()
