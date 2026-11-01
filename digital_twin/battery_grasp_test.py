"""Isolated grasp/lift trial for one horizontal AAA battery.

The default remains the validated mechanics baseline and uses the simulated
battery pose after settling.  ``--use-2d-vision --camera table`` instead commands
XY and yaw solely from the calibrated RGB image, matching a flat-table FANUC
iRVision 2D deployment.  Simulator object poses remain evaluation-only there.

Example:
    /isaacsim/python.sh battery_grasp_test.py --out runs/battery_01
    /isaacsim/python.sh battery_grasp_test.py --out runs/battery_physics \
        --physics-only --tip-clearance-mm 2.5 --jaw-half-mm 2.28
    /isaacsim/python.sh battery_grasp_test.py --out runs/battery_2d \
        --use-2d-vision --camera table
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True)
parser.add_argument("--yaw-deg", type=float, default=None)
parser.add_argument("--x-m", type=float, default=None)
parser.add_argument("--y-m", type=float, default=None)
parser.add_argument(
    "--tip-clearance-mm",
    type=float,
    default=None,
    help="physical fingertip height above the work surface",
)
parser.add_argument(
    "--jaw-half-mm",
    type=float,
    default=None,
    help="commanded per-side straight-face opening after closure",
)
parser.add_argument("--lift-mm", type=float, default=None)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--camera", choices=("table", "side", "wrist"), default="side")
parser.add_argument("--physics-only", action="store_true")
parser.add_argument(
    "--no-video-frames",
    action="store_true",
    help="keep perception images and JSON reports but do not save trajectory PNGs",
)
parser.add_argument(
    "--video-motion-only",
    action="store_true",
    help="record only the grasp trajectory, omitting the 5 s initialization hold",
)
parser.add_argument(
    "--use-2d-vision",
    action="store_true",
    help=(
        "command the grasp from the calibrated table RGB image instead of the "
        "simulator object pose; requires --camera table and rendering"
    ),
)
args = parser.parse_args()
if args.fps != 30:
    parser.error("Use 30 fps: the test advances two 1/60 s physics steps per frame")
if args.use_2d_vision and args.physics_only:
    parser.error("--use-2d-vision requires rendering; remove --physics-only")
if args.use_2d_vision and args.camera != "table":
    parser.error("The first calibrated 2D baseline requires --camera table")
args.out = os.path.abspath(args.out)

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
    import cell
    import fk
    import ik
    import scene_config as C
    from backproject import CameraModel
    from battery_2d_perception import detect_synthetic_battery, draw_detection

    out = Path(args.out)
    frames = out / "frames"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Use an empty output directory: {out}")
    frames.mkdir(parents=True, exist_ok=True)

    yaw_deg = C.BATTERY_TEST_YAW_DEG if args.yaw_deg is None else args.yaw_deg
    test_x = C.BATTERY_TEST_XY[0] if args.x_m is None else args.x_m
    test_y = C.BATTERY_TEST_XY[1] if args.y_m is None else args.y_m
    if not (0.18 <= test_x <= 0.55 and -0.30 <= test_y <= 0.30):
        raise ValueError("Requested battery XY is outside the validated table work area")
    if math.hypot(test_x, test_y) > 0.52:
        raise ValueError("Requested battery XY is too close to the ER-4iA reach limit")
    tip_clearance = (
        C.BATTERY_TIP_CLEARANCE
        if args.tip_clearance_mm is None
        else args.tip_clearance_mm / 1000.0
    )
    jaw_target = (
        C.BATTERY_JAW_HALF_TARGET
        if args.jaw_half_mm is None
        else args.jaw_half_mm / 1000.0
    )
    lift_height = (
        C.BATTERY_LIFT_HEIGHT
        if args.lift_mm is None
        else args.lift_mm / 1000.0
    )
    if not 0 < tip_clearance < C.BATTERY_DIAMETER:
        raise ValueError("Tip clearance must be between 0 and one battery diameter")
    if not C.JAW_CLOSED <= jaw_target <= C.JAW_OPEN:
        raise ValueError("Jaw target is outside the calibrated per-jaw travel")
    if lift_height <= 0:
        raise ValueError("Lift height must be positive")

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    battery_path = cell.add_batteries(stage, [(test_x, test_y, yaw_deg)])[0]
    battery_prim = stage.GetPrimAtPath(battery_path)
    UsdPhysics.RigidBodyAPI(battery_prim).CreateKinematicEnabledAttr().Set(True)

    cam_paths = {
        "table": cell.TABLE_CAM_PATH,
        "side": cell.SIDE_CAM_PATH,
        "wrist": cell.WRIST_CAM_PATH,
    }
    cam_cfgs = {"table": C.TABLE_CAM, "side": C.SIDE_CAM, "wrist": C.WRIST_CAM}
    camera = None
    if not args.physics_only:
        camera = Camera(
            prim_path=cam_paths[args.camera], resolution=cam_cfgs[args.camera]["resolution"]
        )

    cell.enable_ccd(world)
    world.reset()
    if camera is not None:
        camera.initialize()

    arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    names = list(arm.dof_names)
    arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
    joint_idx = {name: i for i, name in enumerate(names)}
    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    tool = np.array(C.TOOL_AXIS, float) * cell.tip_offset()
    down = np.array([0.0, 0.0, -1.0])
    if args.use_2d_vision:
        # Park outside the table-camera work area before taking the image, just
        # as the cube pipeline does.  Starting the camera observation directly
        # over the battery would turn gripper occlusion into a fake detector
        # failure that the real deployment also needs to avoid.
        q = np.deg2rad([C.CAPTURE_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise RuntimeError("Calibrated camera capture pose exceeds joint limits")
        _, home = fk.tool_pose(q, tool)
    else:
        home = np.array([0.30, 0.0, 0.24])
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q, ok = ik.solve(q, home, down, limits, tool_offset=tool)
        if not ok:
            raise RuntimeError("Home IK failed")

    def command(q6, jaw):
        full = arm.get_joint_positions().copy()
        full[arm_idx] = q6
        for joint, value in cell.jaw_targets(jaw).items():
            full[joint_idx[joint]] = value
        arm.apply_action(ArticulationAction(joint_positions=full))

    full = arm.get_joint_positions().copy()
    full[arm_idx] = q
    for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
        full[joint_idx[joint]] = value
    arm.set_joint_positions(full)

    def step_frame(record=True):
        for _ in range(2):
            world.step(render=not args.physics_only)
        if camera is not None:
            rgba = camera.get_rgba()
            # RTX annotators commonly return None during their first rendered
            # frames.  Treat that as camera warm-up, not a failed experiment.
            if (
                record
                and not args.no_video_frames
                and rgba is not None
                and rgba.ndim == 3
                and rgba.shape[2] >= 3
            ):
                cv2.imwrite(
                    str(frames / f"f{step_frame.number:05d}.png"),
                    cv2.cvtColor(
                        rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                    ),
                )
        step_frame.number += 1

    step_frame.number = 0

    # Hold the battery fixed while the arm and MDL materials initialise.
    for _ in range(90):
        command(q, C.JAW_OPEN)
        step_frame(record=not args.video_motion_only)

    UsdPhysics.RigidBodyAPI(battery_prim).CreateKinematicEnabledAttr().Set(False)
    for _ in range(60):
        command(q, C.JAW_OPEN)
        step_frame(record=not args.video_motion_only)

    battery = SingleXFormPrim(battery_path)

    vision_pose = None
    if args.use_2d_vision:
        # Require an actual RGB frame for control.  This second bounded warm-up
        # protects headless runs where material compilation delayed the table
        # camera beyond the ordinary arm/object settling period.
        rgba = None
        for _ in range(60):
            world.step(render=True)
            candidate = camera.get_rgba()
            if (
                candidate is not None
                and candidate.ndim == 3
                and candidate.shape[2] >= 3
            ):
                rgba = candidate
                break
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
            raise RuntimeError(
                "Table camera returned no usable RGB image after 60 warm-up frames"
            )
        rgb = rgba[:, :, :3].astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        prim = stage.GetPrimAtPath(cell.TABLE_CAM_PATH)
        gcam = UsdGeom.Camera(prim)
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        width, height = C.TABLE_CAM["resolution"]
        camera_json = {
            "camera": "table",
            "resolution": [width, height],
            "focal_length_mm": float(gcam.GetFocalLengthAttr().Get()),
            "horizontal_aperture_mm": float(gcam.GetHorizontalApertureAttr().Get()),
            "vertical_aperture_mm": float(gcam.GetVerticalApertureAttr().Get()),
            "camera_to_world": [
                [float(xf[row][column]) for column in range(4)] for row in range(4)
            ],
            "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
            "robot_base_xy": [0.0, 0.0],
        }
        calibrated_camera = CameraModel(camera_json)
        vision_pose, battery_mask = detect_synthetic_battery(
            bgr,
            calibrated_camera,
            plane_z=calibrated_camera.table_z + C.BATTERY_RADIUS,
            expected_length_m=C.BATTERY_BODY_LENGTH,
            expected_width_m=C.BATTERY_DIAMETER,
            workspace_xy=((0.18, 0.70), (-0.47, 0.47)),
        )
        cv2.imwrite(str(out / "battery_perception_rgb.png"), bgr)
        cv2.imwrite(str(out / "battery_perception_mask.png"), battery_mask)
        cv2.imwrite(
            str(out / "battery_perception_overlay.png"),
            draw_detection(bgr, vision_pose, battery_mask),
        )
        (out / "battery_perception.json").write_text(
            json.dumps(
                {
                    "control_source": "calibrated_2d_rgb",
                    "detector": "synthetic_orange_hsv_baseline",
                    "pose": vision_pose.to_dict(),
                    "camera": camera_json,
                },
                indent=2,
            )
            + "\n"
        )

    def battery_pose():
        position, quat = battery.get_world_pose()
        w, x, y, z = quat
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        axis = rotation[:, 0]
        axis[2] = 0.0
        axis /= np.linalg.norm(axis)
        return np.array(position, float), np.array(quat, float), axis

    # In vision mode, these are the only quantities allowed to command XY/yaw.
    # The simulator pose is read afterwards strictly for evaluation and pickup
    # verification; it never changes the commanded grasp.
    if vision_pose is not None:
        control_xy = np.asarray(vision_pose.center_xy_m, float)
        battery_axis = np.array([*vision_pose.axis_xy, 0.0], float)
        control_source = "calibrated_2d_rgb"
    else:
        truth_position_for_control, _, truth_axis_for_control = battery_pose()
        control_xy = truth_position_for_control[:2]
        battery_axis = truth_axis_for_control
        control_source = "simulator_pose_mechanics_baseline"

    initial_position, initial_quat, evaluation_axis = battery_pose()
    spread = np.array([-battery_axis[1], battery_axis[0], 0.0])
    pick = np.array(
        [control_xy[0], control_xy[1], C.SURFACE_DZ + tip_clearance]
    )
    above = pick + [0.0, 0.0, 0.10]
    lifted = pick + [0.0, 0.0, lift_height]

    events = []
    trajectory_ik_failures = 0

    def turn(a, b, t):
        angle = math.atan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a[:2], b[:2])))
        angle *= t
        c, s = math.cos(angle), math.sin(angle)
        return np.array([a[0] * c - a[1] * s, a[0] * s + a[1] * c, 0.0])

    def ease(t):
        return t * t * (3 - 2 * t)

    def move(name, start, finish, seconds, jaw0, jaw1, orient, rotate_from_current=False):
        local_q = move.q
        start_spread = None
        if rotate_from_current:
            rotation, _ = fk.fk(local_q)
            start_spread = rotation @ np.array(C.FINGER_SPREAD_AXIS)
            start_spread[2] = 0.0
            start_spread /= np.linalg.norm(start_spread)
            if np.dot(start_spread, orient) < 0:
                orient = -orient
        count = max(1, round(seconds * args.fps))
        for index in range(count):
            t = ease((index + 1) / count)
            target = start + (finish - start) * t
            direction = turn(start_spread, orient, t) if start_spread is not None else orient
            local_q, solved = ik.solve(
                local_q,
                target,
                down,
                limits,
                tool_offset=tool,
                spread_dir=direction,
            )
            if not solved:
                local_q, solved = ik.solve(
                    local_q,
                    target,
                    down,
                    limits,
                    tool_offset=tool,
                    spread_dir=direction,
                    iters=500,
                    damping=0.02,
                )
            rotation, actual = fk.tool_pose(local_q, tool)
            aligned = abs(np.dot(rotation @ np.array(C.FINGER_SPREAD_AXIS), direction)) > 0.999
            if not solved or np.linalg.norm(actual - target) > 0.001 or not aligned:
                raise RuntimeError(f"IK failed during {name} at frame {index}")
            command(local_q, jaw0 + (jaw1 - jaw0) * t)
            step_frame()
        move.q = local_q
        position, _, _ = battery_pose()
        events.append(
            {
                "phase": name,
                "battery_center_world_m": position.tolist(),
                "commanded_tip_robot_m": finish.tolist(),
            }
        )

    move.q = q
    move("approach", home, above, 1.2, C.JAW_OPEN, C.JAW_OPEN, spread, True)
    move("descend", above, pick, 0.8, C.JAW_OPEN, C.JAW_OPEN, spread)
    move("close", pick, pick, 0.6, C.JAW_OPEN, jaw_target, spread)
    move("dwell", pick, pick, 0.5, jaw_target, jaw_target, spread)
    position_after_close, _, _ = battery_pose()
    move("lift", pick, lifted, 1.0, jaw_target, jaw_target, spread)
    move("hold", lifted, lifted, 1.0, jaw_target, jaw_target, spread)
    final_position, final_quat, final_axis = battery_pose()

    lift_delta = float(final_position[2] - initial_position[2])
    xy_drift = float(np.linalg.norm(final_position[:2] - initial_position[:2]))
    pickup_verified = bool(lift_delta > max(0.020, 0.60 * lift_height))
    retained = bool(pickup_verified and xy_drift < 0.020)
    measured = arm.get_joint_positions()
    straight_gap = float(
        2 * C.JAW_OPEN
        - abs(measured[joint_idx["jaw_l"]])
        - abs(measured[joint_idx["jaw_r"]])
    )
    report = {
        "task": "aaa_battery_grasp_test",
        "task_complete": retained,
        "pickup_verified": pickup_verified,
        "retained_after_hold": retained,
        "physics_only": args.physics_only,
        "control_source": control_source,
        "simulator_pose_used_for_control": not args.use_2d_vision,
        "frames": step_frame.number,
        "video_motion_only": args.video_motion_only,
        "battery": {
            "diameter_m": C.BATTERY_DIAMETER,
            "length_m": C.BATTERY_LENGTH,
            "mass_kg": C.BATTERY_MASS,
            "positive_terminal_diameter_m": C.BATTERY_POS_TERMINAL_DIAMETER,
            "negative_terminal_diameter_m": C.BATTERY_NEG_TERMINAL_DIAMETER,
            "requested_xy_m": [test_x, test_y],
            "requested_yaw_deg": yaw_deg,
        },
        "grasp": {
            "tip_clearance_m": tip_clearance,
            "jaw_half_target_m": jaw_target,
            "measured_straight_gap_m": straight_gap,
            "battery_axis_world": battery_axis.tolist(),
            "finger_spread_world": spread.tolist(),
            "grasp_offset_along_axis_m": float(
                np.dot(pick[:2] - initial_position[:2], battery_axis[:2])
            ),
        },
        "initial_position_world_m": initial_position.tolist(),
        "initial_axis_world_evaluation": evaluation_axis.tolist(),
        "perception": (
            None
            if vision_pose is None
            else {
                **vision_pose.to_dict(),
                "center_error_m": float(
                    np.linalg.norm(np.asarray(vision_pose.center_xy_m) - initial_position[:2])
                ),
                "axis_alignment_abs_dot": float(
                    abs(np.dot(np.asarray(vision_pose.axis_xy), evaluation_axis[:2]))
                ),
            }
        ),
        "position_after_close_world_m": position_after_close.tolist(),
        "final_position_world_m": final_position.tolist(),
        "final_orientation_wxyz": final_quat.tolist(),
        "final_axis_world": final_axis.tolist(),
        "lift_delta_m": lift_delta,
        "xy_drift_m": xy_drift,
        "trajectory_ik_failures": trajectory_ik_failures,
        "events": events,
    }
    (out / "battery_grasp_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[battery] pickup={pickup_verified}; retained={retained}; "
        f"lift={lift_delta * 1000:.1f} mm; drift={xy_drift * 1000:.1f} mm; "
        f"report={out / 'battery_grasp_report.json'}",
        flush=True,
    )
    if not retained:
        raise SystemExit(2)
finally:
    app.close()
