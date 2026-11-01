# wrist depth 与机器人自身点云验证 v05

v04 证明 table depth 能可靠检测低矮桌面障碍，但固定相机也会看到机械臂
本体。v05 保持机械臂静止，同时完成两项验证：

1. 使用当前关节和 TCP 构造机械臂骨架胶囊，解释 table depth 中高于
   120 mm 的机械臂自身点云；
2. 使用 wrist depth 近距离测量同一个 55--56 mm 盒子和周围桌面。

## 安全行为

- `ACTION = "joints"`，原样保持当前关节和夹爪状态；
- 不扫描、不下降、不平移、不改变夹爪；
- 同步申请 table depth 与 wrist depth；
- 采集 20 帧并输出重复性报告；
- 任一深度格式、标定、盒高或自身点云检查失败时判定失败。

## 安装

```bash
cd /path/to/PhysicalAI-Hackathon-2026
cp policy.py policy_before_wrist_selfmask_v05.py
tar -xzf ../physical_wrist_selfmask_validation_v05.tar.gz
```

## 先运行 mock

```bash
uv run run.py --mock --trials 1 --trial-s 8
```

## 必须重新提交请求

v05 将深度请求从 `["table"]` 改为 `["table", "wrist"]`，因此旧 request
不能直接复用：

```bash
uv run run.py --submit
```

操作员批准后运行：

```bash
uv run run.py
```

盒子保持在 v03/v04 的中央位置，机械臂保持当前初始姿态。

## 查看结果

```bash
RUN=$(ls -dt wrist_selfmask_diagnostics/rollout_* | head -1)
cat "$RUN/wrist_selfmask_summary.json"
```

重点字段：

```text
minimum_table_robot_self_explained_fraction
maximum_unexplained_table_high_points
mean_wrist_valid_depth_fraction
wrist_calibration_valid_frames
wrist_box_detection_fraction
wrist_box_height_median_mm
wrist_box_height_std_mm
wrist_depth_and_self_mask_validation_passed
motion_enable_recommended
next_stage
```

打包结果：

```bash
tar -czf wrist_selfmask_v05_result.tar.gz \
  -C "$RUN" \
  wrist_selfmask_summary.json first_sample_report.json \
  samples.jsonl first_depth_frames_mm.npz
```

即使本阶段通过，`motion_enable_recommended` 仍为 `false`。通过后才生成只
开放高位横移和单点受保护下降的有限运动版本。
