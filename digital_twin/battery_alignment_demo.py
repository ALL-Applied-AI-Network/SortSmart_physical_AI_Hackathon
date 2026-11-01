"""Pick six horizontal AAA batteries and place them in one aligned row."""

import argparse
import json
import math
import os
import sys
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--camera",
    choices=("side", "table"),
    default="side",
)
args = parser.parse_args()

if args.fps != 30:
    parser.error("This demo expects 30 fps")

args.out = os.path.abspath(args.out)


from isaacsim import SimulationApp

app = SimulationApp({"headless": True})


try:
    import cv2
    import numpy as np
    import omni.usd

    from pxr import UsdPhysics

    from isaacsim.core.api import World
    from isaacsim.core.prims import (
        SingleArticulation,
        SingleXFormPrim,
    )
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera

    sys.path.insert(
        0,
        os.path.dirname(os.path.abspath(__file__)),
    )

    import cell
    import fk
    import ik
    import scene_config as C

    # ------------------------------------------------------------ output

    out = Path(args.out)

    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"Use an empty output directory: {out}"
        )

    frames = out / "frames"

    frames.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------ layout

    starts = list(C.BATTERY_DEMO_STARTS)
    targets = list(C.BATTERY_DEMO_TARGETS)

    if len(starts) != len(targets) or not starts:
        raise ValueError(
            "BATTERY_DEMO_STARTS and "
            "BATTERY_DEMO_TARGETS must have "
            "equal non-zero length"
        )

    # ------------------------------------------------------------ world

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1 / 60,
        rendering_dt=1 / 30,
    )

    stage = omni.usd.get_context().get_stage()

    cell.build(
        stage,
        with_objects=False,
    )

    battery_paths = cell.add_batteries(
        stage,
        starts,
    )

    # Hold all batteries still while initializing.
    for path in battery_paths:
        UsdPhysics.RigidBodyAPI(
            stage.GetPrimAtPath(path)
        ).CreateKinematicEnabledAttr().Set(True)

    camera_path = (
        cell.SIDE_CAM_PATH
        if args.camera == "side"
        else cell.TABLE_CAM_PATH
    )

    camera = Camera(
        prim_path=camera_path,
        # resolution=(960, 540),
        resolution = (1280, 720),
    )

    cell.enable_ccd(world)

    world.reset()
    camera.initialize()

    # ------------------------------------------------------------ robot

    robot = SingleArticulation(
        prim_path=C.ROBOT_PRIM,
        name="er4ia",
    )

    robot.initialize()

    joint_names = list(robot.dof_names)

    arm_indices = [
        joint_names.index(f"J{i}")
        for i in range(1, 7)
    ]

    joint_index = {
        name: index
        for index, name in enumerate(joint_names)
    }

    joint_limits = np.deg2rad(
        [
            C.JOINT_LIMITS_DEG[f"J{i}"]
            for i in range(1, 7)
        ]
    )

    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)

    tool_offset = (
        np.array(C.TOOL_AXIS, dtype=float)
        * cell.tip_offset()
    )

    downward = np.array(
        [0.0, 0.0, -1.0],
        dtype=float,
    )

    home = np.array(
        [0.30, 0.0, 0.24],
        dtype=float,
    )

    configuration = np.deg2rad(
        [
            C.HOME_POSE_DEG[f"J{i}"]
            for i in range(1, 7)
        ]
    )

    configuration, solved = ik.solve(
        configuration,
        home,
        downward,
        joint_limits,
        tool_offset=tool_offset,
    )

    if not solved:
        raise RuntimeError("Home IK failed")

    def command(q6, jaw):
        """Command arm joints and both gripper jaws."""

        complete = robot.get_joint_positions().copy()
        complete[arm_indices] = q6

        for name, value in cell.jaw_targets(
            jaw
        ).items():
            complete[
                joint_index[name]
            ] = value

        robot.apply_action(
            ArticulationAction(
                joint_positions=complete
            )
        )

    # Place robot directly at the starting pose.
    complete = robot.get_joint_positions().copy()
    complete[arm_indices] = configuration

    for name, value in cell.jaw_targets(
        C.JAW_OPEN
    ).items():
        complete[
            joint_index[name]
        ] = value

    robot.set_joint_positions(complete)

    # ------------------------------------------------------------ rendering

    def step(capture=True):
        """Advance two physics steps and optionally save one frame."""

        for _ in range(2):
            world.step(render=capture)

        if capture:
            rgba = camera.get_rgba()

            # if rgba is None:
            #     raise RuntimeError(
            #         "Camera returned no image"
            #     )

            if rgba is None:
                print(
                    "[camera] RGB frame not ready; skipping frame",
                    flush=True,
                )
                return

            rgba = np.asarray(rgba)

            if (
                rgba.ndim != 3
                or rgba.shape[2] < 3
            ):
                print(
                    f"[camera] Unexpected frame shape: {rgba.shape}; "
                    "skipping frame",
                    flush=True,
                )
                return
            

            image = cv2.cvtColor(
                rgba[:, :, :3].astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )

            frame_path = (
                frames
                / f"f{step.number:05d}.png"
            )

            cv2.imwrite(
                str(frame_path),
                image,
            )

            step.number += 1

    step.number = 0

    # Warm up robot drives and materials without recording.
    for _ in range(90):
        command(
            configuration,
            C.JAW_OPEN,
        )
        step(capture=False)

    # Warm up RTX and the camera before requesting RGB data.
    # These frames are rendered but not written to disk.
    for _ in range(30):
        
        command(
            configuration,
            C.JAW_OPEN,
        )

        world.step(render=True)
        world.step(render=True)

    # Release all batteries and allow them to settle.
    for path in battery_paths:
        UsdPhysics.RigidBodyAPI(
            stage.GetPrimAtPath(path)
        ).CreateKinematicEnabledAttr().Set(False)

    for _ in range(60):
        command(
            configuration,
            C.JAW_OPEN,
        )
        step(capture=False)

    wrappers = {
        path: SingleXFormPrim(path)
        for path in battery_paths
    }

    # ------------------------------------------------------------ pose helpers

    def battery_pose(path):
        """Return centre, quaternion and horizontal long axis."""

        position, quaternion = (
            wrappers[path].get_world_pose()
        )

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

        # Cylinder local X is the battery long axis.
        axis = rotation[:, 0]
        axis[2] = 0.0

        axis_length = np.linalg.norm(axis)

        if axis_length < 1e-8:
            raise RuntimeError(
                f"Battery is not horizontal: {path}"
            )

        axis /= axis_length

        return (
            np.array(position, dtype=float),
            np.array(quaternion, dtype=float),
            axis,
        )

    def ease(value):
        """Smooth trajectory interpolation."""

        return (
            value
            * value
            * (3.0 - 2.0 * value)
        )

    def turn(start, finish, progress):
        """Rotate one horizontal direction toward another."""

        angle = math.atan2(
            start[0] * finish[1]
            - start[1] * finish[0],
            float(
                np.dot(
                    start[:2],
                    finish[:2],
                )
            ),
        )

        angle *= progress

        cosine = math.cos(angle)
        sine = math.sin(angle)

        return np.array(
            [
                start[0] * cosine
                - start[1] * sine,

                start[0] * sine
                + start[1] * cosine,

                0.0,
            ],
            dtype=float,
        )

    def current_spread(q6):
        """Return the current horizontal jaw movement direction."""

        rotation, _ = fk.fk(q6)

        spread = (
            rotation
            @ np.array(
                C.FINGER_SPREAD_AXIS,
                dtype=float,
            )
        )

        spread[2] = 0.0
        spread /= np.linalg.norm(spread)

        return spread

    # ------------------------------------------------------------ movement

    def move(
        name,
        start,
        finish,
        seconds,
        jaw_start,
        jaw_finish,
        spread_start,
        spread_finish,
    ):
        """Move the arm and rotate the wrist smoothly."""

        local_configuration = move.configuration

        frame_count = max(
            1,
            round(seconds * args.fps),
        )

        for frame_index in range(frame_count):
            progress = ease(
                (frame_index + 1)
                / frame_count
            )

            target = (
                start
                + (finish - start)
                * progress
            )

            spread_direction = turn(
                spread_start,
                spread_finish,
                progress,
            )

            local_configuration, solved = ik.solve(
                local_configuration,
                target,
                downward,
                joint_limits,
                tool_offset=tool_offset,
                spread_dir=spread_direction,
            )

            if not solved:
                local_configuration, solved = ik.solve(
                    local_configuration,
                    target,
                    downward,
                    joint_limits,
                    tool_offset=tool_offset,
                    spread_dir=spread_direction,
                    iters=500,
                    damping=0.02,
                )

            rotation, actual_position = fk.tool_pose(
                local_configuration,
                tool_offset,
            )

            actual_spread = (
                rotation
                @ np.array(
                    C.FINGER_SPREAD_AXIS,
                    dtype=float,
                )
            )

            aligned = (
                abs(
                    np.dot(
                        actual_spread,
                        spread_direction,
                    )
                )
                > 0.999
            )

            position_error = np.linalg.norm(
                actual_position - target
            )

            if (
                not solved
                or position_error > 0.001
                or not aligned
            ):
                raise RuntimeError(
                    f"IK failed during {name}, "
                    f"frame {frame_index}"
                )

            jaw_opening = (
                jaw_start
                + (
                    jaw_finish
                    - jaw_start
                )
                * progress
            )

            command(
                local_configuration,
                jaw_opening,
            )

            step(capture=True)

        move.configuration = (
            local_configuration
        )

    move.configuration = configuration

    # ------------------------------------------------------------ task

    results = []

    surface_tip_z = (
        C.SURFACE_DZ
        + C.BATTERY_TIP_CLEARANCE
    )

    for index, (
        battery_path,
        target_entry,
    ) in enumerate(
        zip(
            battery_paths,
            targets,
        )
    ):
        position, _, battery_axis = (
            battery_pose(battery_path)
        )

        # Jaw closing direction is perpendicular to the battery.
        raw_pick_spread = np.array(
            [
                -battery_axis[1],
                battery_axis[0],
                0.0,
            ],
            dtype=float,
        )

        starting_spread = current_spread(
            move.configuration
        )

        # The gripper is symmetric. Select the equivalent direction
        # requiring the smallest wrist rotation.
        direction_sign = (
            -1.0
            if np.dot(
                starting_spread,
                raw_pick_spread,
            ) < 0
            else 1.0
        )

        pick_spread = (
            direction_sign
            * raw_pick_spread
        )

        target_x, target_y, target_yaw_deg = (
            target_entry
        )

        target_yaw = math.radians(
            target_yaw_deg
        )

        target_axis = np.array(
            [
                math.cos(target_yaw),
                math.sin(target_yaw),
                0.0,
            ],
            dtype=float,
        )

        place_spread = (
            direction_sign
            * np.array(
                [
                    -target_axis[1],
                    target_axis[0],
                    0.0,
                ],
                dtype=float,
            )
        )

        pick_position = np.array(
            [
                position[0],
                position[1],
                surface_tip_z,
            ],
            dtype=float,
        )

        pick_above = (
            pick_position
            + np.array(
                [
                    0.0,
                    0.0,
                    C.BATTERY_DEMO_APPROACH_HEIGHT,
                ]
            )
        )

        pick_lift = (
            pick_position
            + np.array(
                [
                    0.0,
                    0.0,
                    C.BATTERY_DEMO_CARRY_HEIGHT,
                ]
            )
        )

        place_position = np.array(
            [
                target_x,
                target_y,
                surface_tip_z,
            ],
            dtype=float,
        )

        place_above = (
            place_position
            + np.array(
                [
                    0.0,
                    0.0,
                    C.BATTERY_DEMO_CARRY_HEIGHT,
                ]
            )
        )

        _, current_tip = fk.tool_pose(
            move.configuration,
            tool_offset,
        )

        # Approach while rotating wrist toward the battery.
        move(
            f"battery_{index}_approach",
            current_tip,
            pick_above,
            1.0,
            C.JAW_OPEN,
            C.JAW_OPEN,
            starting_spread,
            pick_spread,
        )

        # Descend with open jaws.
        move(
            f"battery_{index}_descend",
            pick_above,
            pick_position,
            0.6,
            C.JAW_OPEN,
            C.JAW_OPEN,
            pick_spread,
            pick_spread,
        )

        # Close around the centre of the cylindrical body.
        move(
            f"battery_{index}_close",
            pick_position,
            pick_position,
            0.5,
            C.JAW_OPEN,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            pick_spread,
        )

        # Allow contact to settle.
        move(
            f"battery_{index}_dwell",
            pick_position,
            pick_position,
            0.3,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            pick_spread,
        )

        # Lift vertically.
        move(
            f"battery_{index}_lift",
            pick_position,
            pick_lift,
            0.7,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            pick_spread,
        )

        # Carry to target and rotate battery into final orientation.
        move(
            f"battery_{index}_carry",
            pick_lift,
            place_above,
            1.2,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            pick_spread,
            place_spread,
        )

        # Lower to the table.
        move(
            f"battery_{index}_lower",
            place_above,
            place_position,
            0.7,
            C.BATTERY_JAW_HALF_TARGET,
            C.BATTERY_JAW_HALF_TARGET,
            place_spread,
            place_spread,
        )

        # Open the jaws.
        move(
            f"battery_{index}_release",
            place_position,
            place_position,
            0.4,
            C.BATTERY_JAW_HALF_TARGET,
            C.JAW_OPEN,
            place_spread,
            place_spread,
        )

        # Wait for the battery to settle.
        move(
            f"battery_{index}_release_dwell",
            place_position,
            place_position,
            0.3,
            C.JAW_OPEN,
            C.JAW_OPEN,
            place_spread,
            place_spread,
        )

        # Move safely upward.
        move(
            f"battery_{index}_retreat",
            place_position,
            place_above,
            0.6,
            C.JAW_OPEN,
            C.JAW_OPEN,
            place_spread,
            place_spread,
        )

        # Additional recorded settling frames.
        for _ in range(15):
            command(
                move.configuration,
                C.JAW_OPEN,
            )
            step(capture=True)

        final_position, _, final_axis = (
            battery_pose(battery_path)
        )

        position_error = float(
            np.linalg.norm(
                final_position[:2]
                - [target_x, target_y]
            )
        )

        axis_alignment = float(
            np.dot(
                final_axis,
                target_axis,
            )
        )

        placed = bool(
            position_error < 0.015
            and axis_alignment
            > math.cos(
                math.radians(10)
            )
        )

        results.append(
            {
                "battery": index,
                "initial_yaw_deg": (
                    starts[index][2]
                ),
                "target_xy_m": [
                    target_x,
                    target_y,
                ],
                "target_yaw_deg": (
                    target_yaw_deg
                ),
                "final_position_world_m": (
                    final_position.tolist()
                ),
                "final_axis_world": (
                    final_axis.tolist()
                ),
                "position_error_m": (
                    position_error
                ),
                "axis_alignment": (
                    axis_alignment
                ),
                "placed": placed,
            }
        )

        print(
            f"[align] battery={index}; "
            f"placed={placed}; "
            f"error={position_error * 1000:.1f} mm; "
            f"alignment={axis_alignment:.4f}",
            flush=True,
        )

    # Hold final aligned arrangement on screen.
    for _ in range(60):
        command(
            move.configuration,
            C.JAW_OPEN,
        )
        step(capture=True)

    # ------------------------------------------------------------ report

    report = {
        "task": "aaa_battery_alignment_demo",
        "task_complete": all(
            item["placed"]
            for item in results
        ),
        "camera": args.camera,
        "fps": args.fps,
        "frames": step.number,
        "batteries": results,
    }

    report_path = (
        out
        / "battery_alignment_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    print(
        f"[align] report={report_path}; "
        f"complete={report['task_complete']}",
        flush=True,
    )

    if not report["task_complete"]:
        raise SystemExit(2)

finally:
    app.close()