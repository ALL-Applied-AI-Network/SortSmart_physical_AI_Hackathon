
import json
from pathlib import Path

import cv2
import numpy as np

from backproject import CameraModel
from cube_pose import refine_pose, square_yaw

run = Path("runs/capture_01")

# 加载图像、相机参数和之前计算的坐标
image = cv2.imread(str(run / "world_camera.png"))
if image is None:
    raise FileNotFoundError("world_camera.png")

camera = CameraModel(str(run / "camera.json"))

objects = json.loads(
    (run / "grasp_candidates.json").read_text()
)["objects"]

# Ground truth 只用于最后评估，不参与姿态预测
ground_truth = json.loads(
    (run / "ground_truth.json").read_text()
)["objects"]

results = []

print("=== Cube Orientation Estimation ===")

for obj in objects:
    xy = np.array(obj["center_robot_m"][:2], dtype=float)

    # 直接利用图片轮廓拟合方块旋转角度
    fitted_xy, yaw, quality = refine_pose(
        image,
        camera,
        {"colour": obj["colour"]},
        xy,
        yaw=0,
        search_half_width=45
    )

    # 根据颜色与距离匹配 GT，仅用于计算误差
    matches = [
        g for g in ground_truth
        if g["category"] == "cube"
        and g["colour"] == obj["colour"]
    ]
    gt = min(
        matches,
        key=lambda g: np.linalg.norm(
            fitted_xy - np.array(g["xyz"][:2])
        )
    )

    error = abs(square_yaw(yaw - gt["yaw_deg"]))

    result = {
        "colour": obj["colour"],
        "xy_m": fitted_xy.tolist(),
        "yaw_deg": float(yaw),
        "silhouette_iou": float(quality),
        "yaw_error_deg_mod90": float(error),
        "pose_accepted": bool(quality >= 0.85)
    }
    results.append(result)

    print(
        f'{obj["colour"]:7s} '
        f'yaw={yaw:7.2f}°  '
        f'error={error:6.2f}°  '
        f'IoU={quality:.3f}'
    )

out = run / "cube_poses_baseline.json"
out.write_text(json.dumps(results, indent=2))

print("\nReliable poses:",
      sum(r["pose_accepted"] for r in results),
      "/", len(results))
print("Saved:", out)
