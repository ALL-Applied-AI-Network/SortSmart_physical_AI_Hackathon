
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from backproject import CameraModel
import scene_config as C

run = Path("runs/capture_01")

# 1. 读取相机标定和 OpenCV 检测结果
camera = CameraModel(str(run / "camera.json"))

detections = json.loads(
    (run / "detections_baseline.json").read_text()
)["objects"]

# 2. 图像坐标 -> 机器人基座坐标
candidates = []

for obj in detections:
    u = float(obj["center"][0]) * camera.W
    v = float(obj["center"][1]) * camera.H

    # 方块中心位于桌面上方半个边长
    point = camera.to_plane(
        u, v,
        z=camera.table_z + C.CUBE_EDGE / 2
    )

    if point is None:
        print("Invalid projection:", obj["label"])
        continue

    # 学校场景中机器人基座 XY 与世界 XY 对齐
    xyz = [
        float(point[0]),
        float(point[1]),
        float(C.CUBE_EDGE / 2)
    ]

    candidates.append({
        "label": obj["label"],
        "colour": obj["colour"],
        "center_robot_m": xyz,
        "yaw_deg": None,
        "ready_for_execution": False
    })

    print(
        f'{obj["label"]:15s} '
        f'X={xyz[0]:.4f} m  '
        f'Y={xyz[1]:.4f} m  '
        f'Z={xyz[2]:.4f} m'
    )

# 3. 保存候选抓取位置
output = run / "grasp_candidates.json"
output.write_text(json.dumps({
    "coordinate_frame": "robot_base",
    "objects": candidates
}, indent=2))

print(f"\nFound {len(candidates)} candidate positions")
print("Saved:", output)

# 4. 与仿真真值比较，仅用于评估
gt_path = run / "ground_truth.json"

if gt_path.exists():
    gt = json.loads(gt_path.read_text())["objects"]

    print("\n=== Position error evaluation ===")

    errors = []
    for colour in ("yellow", "blue", "red"):
        predicted = [
            p for p in candidates if p["colour"] == colour
        ]
        actual = [
            g for g in gt
            if g["category"] == "cube"
            and g["colour"] == colour
        ]

        if not predicted or not actual:
            print(colour, ": missing detection or ground truth")
            continue

        distances = np.array([
            [
                np.linalg.norm(
                    np.array(p["center_robot_m"][:2])
                    - np.array(g["xyz"][:2])
                )
                for g in actual
            ]
            for p in predicted
        ])

        rows, cols = linear_sum_assignment(distances)

        for i, j in zip(rows, cols):
            err_mm = distances[i, j] * 1000
            errors.append(err_mm)
            print(
                f"{colour:8s} -> {actual[j]['name']}: "
                f"{err_mm:.2f} mm"
            )

    if errors:
        print(
            f"\nMean XY error: {np.mean(errors):.2f} mm"
        )
