# 真实机械臂桌面高度图安全审计 v04

v03 已确认 table depth 能稳定测量 55--56 mm 盒子。v04 将相同深度数据
用于审计原扫描路线的净空，但仍然 **不移动机械臂**。

## v04 检查什么

- 八个低位扫描点：TCP 目标 `z = -235 mm`；
- 每个点周围半径 60 mm 的最高桌面障碍；
- 未来高位横移路线：`z = -165 mm`，走廊半径 70 mm；
- 深度高度额外增加 10 mm 不确定性余量；
- 低位扫描至少保留 25 mm TCP 净空；
- 高位横移至少保留 50 mm TCP 净空；
- 深度缺失、标定异常或净空不足时判定失败。

程序仍使用 `ACTION = "joints"` 原样保持当前关节和夹爪状态。报告中的目标
只是被审计，不会发送给机械臂。

## 重要限制

固定 table camera 会同时看到 FANUC 机械臂本体。因此 v04 只审计受控的
电池分拣桌面物体层：外部物体必须不高于 120 mm。高于 120 mm 的点会单独
计数，不参与低矮桌面障碍高度计算；运行前必须人工确认场景内没有超过
120 mm 的外部杂物。

即使 v04 通过，报告也会保持：

```text
motion_enable_recommended = false
```

下一阶段必须加入 wrist depth 与机器人自身点云过滤，才能开放受保护运动。

## 安装

```bash
cd /path/to/PhysicalAI-Hackathon-2026
cp policy.py policy_before_heightmap_v04.py
tar -xzf ../physical_heightmap_safety_audit_v04.tar.gz
```

## 运行

v04 与 v03 的接口相同：table RGB、table depth、joint action。如果已批准的
request 仍有 rollout，可以直接继续；否则重新提交。

```bash
uv run run.py --mock --trials 1 --trial-s 8
uv run run.py
```

盒子可以保持在 v03 的位置。终端应显示类似：

```text
[heightmap] sample 1/20: vertical_min=34.8 mm, corridor_min=104.8 mm, passed=True
```

## 查看和打包结果

```bash
RUN=$(ls -dt heightmap_safety_audits/rollout_* | head -1)
cat "$RUN/heightmap_safety_summary.json"

tar -czf heightmap_safety_v04_result.tar.gz \
  -C "$RUN" \
  heightmap_safety_summary.json first_route_checks.json \
  samples.jsonl first_table_depth_mm.npz
```

重点字段：

```text
route_pass_fraction
minimum_vertical_clearance_across_frames_mm
minimum_corridor_clearance_across_frames_mm
queries_with_points_above_task_envelope
tabletop_heightmap_audit_passed
motion_enable_recommended
next_stage
```

使用刚才上传的真实盒子第一帧离线复算得到：

```text
minimum_vertical_clearance = 34.77 mm
minimum_corridor_clearance = 104.81 mm
all_route_checks_passed = true
```

这些数值已经包含 10 mm 深度不确定性余量。
