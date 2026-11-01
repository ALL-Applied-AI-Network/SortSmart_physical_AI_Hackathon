# 物理机械臂 v08.3：安全下降 + YOLO 掩膜深度融合

## 本版做什么

- 完整保留 v07.2 已通过实机验证的四段垂直下降、固定 table-depth 安全门、wrist-depth 质量门和 robot-self mask。
- v08.3 使用稳健桌面主平面在线校正 table-depth 的稳定整体偏移，并要求预检偏移连续稳定。它还先用机械臂几何模型验证高度至少 100 mm 的 robot-self 点，再从局部高度图中排除这些点，修复夹爪点恰好落在 120 mm 边界下方造成的误报。低于 100 mm 的电池、盒子和真实障碍物不会被排除。30 mm 最小净空、10 mm 深度不确定性和 25 mm wrist 防护余量保持不变。
- 在最低观测位对 `table` 与 `wrist` RGB 运行 YOLO 分割。
- 只保留投影到 `aaa_battery` 实例掩膜内部的深度点，再计算电池在 base 坐标系中的中心、尺寸、高度以及相对 TCP 的水平/垂直/三维距离。
- 连续观测 10 帧，至少 3 帧融合有效且中心最大偏差不超过 20 mm，才令 `perception_validation_passed=true`。
- 本版不横移、不闭爪、不抓取；YOLO 结果不参与运动控制。识别或融合异常时，机械臂沿原垂直路径返回。

## 安装

在 `PhysicalAI-Hackathon-2026` 根目录执行：

```bash
cp policy.py policy_before_v08.py
tar -xzf /你的路径/physical_guarded_yolo_depth_fusion_v08_3.tar.gz
uv run python -m py_compile policy.py
uv run python -m unittest -v test_guarded_yolo_depth_v08.py
```

确认模型存在：

```bash
ls -lh models/aaa_yolo26s_best.pt
```

v08.3 与 v07.2/v08 的 `CAMERAS`、`DEPTH` 和 `ACTION` 接口完全相同；若当前提交仍有效，通常直接运行即可，不必重新 `--submit`：

```bash
uv run run.py
```

## 场景

先保持 v07.2 实机测试时的场景和机械臂起始位不变。保留 5.5–5.6 cm 参考盒及现有杂物，只确认一节真实 AAA 电池同时能被 table/wrist 相机看到；不要把物体重新移入已验证的垂直下降安全圆柱中。

## 查看结果

```bash
RUN=$(ls -dt guarded_yolo_depth_v08_3_runs/rollout_* | head -1)
cat "$RUN/guarded_yolo_depth_v08_3_report.json"
open "$RUN/overlays"
```

重点字段：

- `safe_motion_complete`: 安全下降和原路返回是否完成。
- `perception_validation_passed`: YOLO + 深度融合是否通过多帧稳定性门。
- `frames_with_valid_battery_fusion`: 有效融合帧数，目标至少 3。
- `battery_fusion_aggregate.median_center_xy_base_m`: 电池中心的 base 坐标。
- `median_*_distance_*_mm`: 电池相对 TCP 的距离。
- `selected_camera_counts`: 最终每帧采用 table 或 wrist 融合的次数。

若 `safe_motion_complete=true` 但 `perception_validation_passed=false`，说明运动安全链路正常，只是本轮没有得到稳定电池深度；不要据此开启抓取。
