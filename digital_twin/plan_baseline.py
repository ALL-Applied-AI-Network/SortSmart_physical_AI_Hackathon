
import json
from pathlib import Path

import numpy as np

from backproject import CameraModel
from plan_sort import build_plan
import scene_config as C

run = Path("runs/capture_01")
camera = CameraModel(str(run / "camera.json"))

poses = json.loads(
    (run / "cube_poses_baseline.json").read_text()
)

# 只接受通过视觉质量检查的六个方块
if len(poses) != 6 or not all(
    p["pose_accepted"] for p in poses
):
    raise RuntimeError("Six reliable cube poses required")

observations = []

for pose in poses:
    xy = np.array(pose["xy_m"], dtype=float)
    yaw = float(pose["yaw_deg"])
    angle = np.radians(yaw)

    direction = np.array([
        np.cos(angle), np.sin(angle)
    ])

    # 物体中心的归一化图像坐标
    center = camera.project([
        *xy, camera.table_z + C.CUBE_EDGE / 2
    ]) / [camera.W, camera.H]

    # 根据视觉估计姿态，生成顶面边缘端点
    endpoints = []

    for sign in (-1, 1):
        pt = xy + sign * C.CUBE_EDGE / 2 * direction
        uv = camera.project([
            *pt, camera.table_z + C.CUBE_EDGE
        ]) / [camera.W, camera.H]
        endpoints.append(uv.tolist())

    observations.append({
        "category": "cube",
        "colour": pose["colour"],
        "label": f'{pose["colour"]} cube',
        "center": center.tolist(),
        "top_edge_endpoints": endpoints,
        "pose_confidence": float(pose["silhouette_iou"])
    })

# 调用学校原有的抓取规划逻辑
plan = build_plan(
    {
        "coordinate_convention": "unit",
        "objects": observations
    },
    camera,
    img=None,  # 之前已经完成图像姿态优化
    task="cube_stack"
)

# 记录真实算法来源，避免被误认为 Cosmos 输出
plan["source"] = "OpenCV geometry baseline"
plan["planning_only"] = True

output = run / "sort_plan_baseline.json"
output.write_text(json.dumps(plan, indent=2))

print("Stack ready:", plan["stack_ready"])
print("Planned picks:", len(plan["picks"]))

for i, pick in enumerate(plan["picks"], 1):
    print(
        f'{i}. {pick["colour"]:7s} '
        f'pick={np.round(pick["xy"], 4)} '
        f'yaw={pick["yaw_deg"]:.2f}° '
        f'level={pick["stack_level"]}'
    )

if not plan["stack_ready"] or len(plan["picks"]) != 6:
    print("Skipped:", [
        (o.get("colour"), o.get("skip_reason"))
        for o in plan["skipped"]
    ])
    raise RuntimeError("Incomplete grasp plan")

print("Saved:", output)
