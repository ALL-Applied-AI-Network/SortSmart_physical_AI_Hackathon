# 物理机械臂 v10：完整电池抓取与投放

v10 在 v09 已通过的多帧感知和非接触路径检查后，执行一次完整流程：

1. 从低位观察点抬升到安全搬运高度；
2. 高位横移至电池上方；
3. 分段下降至根据电池高度计算的抓取位置；
4. 闭合夹爪并垂直抬升；
5. 用 table RGB + depth 检查原位置是否仍有电池；
6. 验证抓取后，高位搬运到检测到的收纳碗；
7. 在测得碗沿上方 35 mm 释放；
8. 检查碗内电池、撤离并回到起点。

任一抓取前安全门或抓取后验证失败，程序都会保持当前夹爪状态、先垂直抬升，
再经高位返回起点。

## 本版本的约束

- 仅允许一节平躺 AAA 电池和一个空收纳碗；不要放标定盒或其他杂物。
- 电池长轴必须大致沿 robot base X 方向，允许误差不超过 10°。
- 不执行 FANUC R 角旋转。
- 起始 TCP 和相机标定必须与 v09 成功实验相同。
- 操作人员必须留在急停附近，机械臂运行时不得进入桌面工作区。

## 安装与运行

在 `PhysicalAI-Hackathon-2026` 根目录：

```bash
cp policy.py policy_before_v10.py
tar -xzf /path/to/physical_guarded_pick_place_v10.tar.gz

uv run run.py --mock --trials 1 --trial-s 8
uv run run.py
```

终端应显示：

```text
[v10] guarded physical pick/place started
```

## 查看结果

```bash
RUN=$(ls -dt physical_pick_place_v10_runs/rollout_* | head -1)
cat "$RUN/physical_pick_place_v10_report.json"
open "$RUN/overlays"
```

完整成功应包含：

```text
task_complete: true
safe_motion_complete: true
execution_started: true
pickup_verified: true
placement_verified: true
horizontal_motion_commanded: true
gripper_transition_commanded: true
```

如果 `task_complete` 为 false，不要立即重复运行；先查看 `reason`、
`pickup_verification`、`placement_verification` 和最后一个 `events` 条目。
