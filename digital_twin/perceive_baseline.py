

import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from backproject import CameraModel
import scene_config as C

# 1. 读取 Isaac Sim 生成的相机图片
run = Path("runs/capture_01")
image = cv2.imread(str(run / "world_camera.png"))

if image is None:
    raise FileNotFoundError("world_camera.png not found")

H, W = image.shape[:2]
camera = CameraModel(str(run / "camera.json"))
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 2. HSV 颜色范围（OpenCV: H=0..179）
ranges = {
    "yellow": [((12, 35, 45), (43, 255, 255))],
    "blue": [((90, 45, 30), (140, 255, 255))],
    "red": [
        ((0, 65, 40), (12, 255, 255)),
        ((165, 65, 40), (179, 255, 255)),
    ],
}

draw_colors = {
    "yellow": (0, 220, 255),
    "blue": (255, 100, 0),
    "red": (0, 0, 255),
}

detections = []
overlay = image.copy()

# 3. 分别检测三种颜色
for colour, thresholds in ranges.items():
    mask = np.zeros((H, W), dtype=np.uint8)

    for lower, upper in thresholds:
        mask |= cv2.inRange(
            hsv, np.array(lower), np.array(upper)
        )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # 过滤太小或明显细长的色块
        scale = W / 1920
        if area < 20 * scale**2:
            continue
        if max(w / max(h, 1), h / max(w, 1)) > 2.0:
            continue

        # 4. 根据相机参数估计候选物体的尺寸
        p = camera.to_plane(
            cx, cy, z=camera.table_z + C.CUBE_EDGE / 2
        )
        if p is None:
            continue

        # 排除工作台之外的物体
        if not (
            C.BACKDROP_X[0] <= p[0] <= C.BACKDROP_X[1]
            and abs(p[1]) <= C.BACKDROP_WIDTH / 2
        ):
            continue

        # 用已知的 10mm 方块尺寸过滤较大的 caps
        half = C.CUBE_EDGE / 2
        px1 = camera.project(p + [half, 0, 0])
        px2 = camera.project(p - [half, 0, 0])
        py1 = camera.project(p + [0, half, 0])
        py2 = camera.project(p - [0, half, 0])

        if any(v is None for v in (px1, px2, py1, py2)):
            continue

        expected = max(
            np.linalg.norm(px1 - px2),
            np.linalg.norm(py1 - py2)
        )
        if not 0.3 * expected <= max(w, h) <= 2.6 * expected:
            continue

        # 5. 记录检测结果
        obj = {
            "label": f"{colour} cube",
            "category": "cube",
            "colour": colour,
            "center": [round(cx / W, 5), round(cy / H, 5)],
            "bbox": [
                round(x / W, 5), round(y / H, 5),
                round((x + w) / W, 5),
                round((y + h) / H, 5)
            ]
        }
        detections.append(obj)

        # 在图片上画检测框
        color = draw_colors[colour]
        cv2.rectangle(
            overlay, (x, y), (x + w, y + h), color, 3
        )
        cv2.putText(
            overlay, f"{colour} cube",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 * scale, color, 2
        )

# 6. 保存结果（不覆盖学校原始 Cosmos 输出）
result = {
    "model": "opencv_color_baseline",
    "task": "cube_stack",
    "coordinate_convention": "unit",
    "objects": detections
}

(run / "detections_baseline.json").write_text(
    json.dumps(result, indent=2)
)
cv2.imwrite(str(run / "detections_baseline.png"), overlay)

counts = Counter(obj["colour"] for obj in detections)
print("Detection counts:", dict(counts))
print("Total detected:", len(detections))
print("Saved detections_baseline.png")
print("Saved detections_baseline.json")
