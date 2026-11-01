
import json
from pathlib import Path
import numpy as np

import scene_config as C
import ik
import fk

# 读取之前生成的抓取计划
run = Path("runs/capture_01")
plan = json.loads(
    (run / "sort_plan_baseline.json").read_text()
)

assert plan["source"] == "OpenCV geometry baseline"
assert plan["planning_only"] is True
assert plan["stack_ready"] and len(plan["picks"]) == 6
assert [p["colour"] for p in plan["picks"]] == list(
    C.CUBE_STACK_COLOURS
)

# 初始化机器人运动学
ik.set_tool_axis(C.TOOL_AXIS)
ik.set_spread_axis(C.FINGER_SPREAD_AXIS)

limits = np.deg2rad([
    C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)
])

tool = np.array(C.TOOL_AXIS) * C.TOOL_LENGTH
down = [0.0, 0.0, -1.0]
home = np.array([0.30, 0.0, 0.24])

q = np.deg2rad([
    C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)
])


def check(q, position, direction, phase):
    q, ok = ik.solve(
        q, position, down, limits,
        tool_offset=tool,
        spread_dir=direction
    )

    actual = fk.tool_pose(q, tool)[1]
    error_mm = np.linalg.norm(
        actual - position
    ) * 1000

    if not ok or error_mm > 1.0:
        raise RuntimeError(
            f"{phase}: IK failed, error={error_mm:.2f} mm"
        )

    return q


# 验证初始位置
q = check(q, home, None, "home")

# 依次检查六个方块的抓取及放置位置
for i, pick in enumerate(plan["picks"], 1):
    spread = np.array(pick["spread_dir"], float)

    rotation, _ = fk.fk(q)
    current = rotation @ np.array(C.FINGER_SPREAD_AXIS)

    if np.dot(current, spread) < 0:
        spread = -spread

    dest_spread = np.array(
        pick["place_spread_dir"], float
    )

    if np.dot(spread, dest_spread) < 0:
        dest_spread = -dest_spread

    x, y = pick["xy"]
    dx, dy = pick["place_xy"]
    z, dz = pick["pick_z"], pick["place_z"]

    checkpoints = [
        ("approach", [x, y, z + 0.13], spread),
        ("pick", [x, y, z], spread),
        ("lift", [x, y, z + 0.13], spread),
        ("carry", [dx, dy, dz + 0.13], dest_spread),
        ("place", [dx, dy, dz], dest_spread),
        ("retreat", [dx, dy, dz + 0.13], dest_spread),
        ("home", home, None),
    ]

    for phase, position, direction in checkpoints:
        q = check(q, position, direction, phase)

    print(
        f"Cube {i} ({pick['colour']}): "
        "IK checkpoints passed"
    )

print("\nAll 6 cubes passed sampled IK checkpoints!")
