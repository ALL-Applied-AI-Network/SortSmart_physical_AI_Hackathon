# 物理机械臂 v09：完整抓取路径的影子规划

v09 继承已经通过真机验证的 v08.3 安全下降、桌面深度校正、robot-self
过滤以及 YOLO 电池掩膜/深度融合。机械臂实际执行的动作仍然只有：原位分段下降、
低位观察、沿原路径返回。

新增内容：

- 固定 table RGB camera 同时识别 `battery_drop_zone`；
- 将电池中心、轴向和收纳碗中心转换到 robot base / UF0；
- 用 table depth height map 离线检查电池接近点、搬运走廊和碗上方接近点；
- 输出完整的抓取、抬升、搬运、低位投放和撤离 waypoint；
- `execution_enabled` 恒为 `false`，不会横移、不会闭合或张开夹爪。

## 安装

在 `PhysicalAI-Hackathon-2026` 根目录：

```bash
cp policy.py policy_before_v09.py
tar -xzf /path/to/physical_shadow_pick_place_v09.tar.gz
uv run run.py --mock --trials 1 --trial-s 8
uv run run.py
```

如果当前实验环境已经提交并且直接运行 `uv run run.py` 能获取新 rollout，通常不必
再次执行 `--submit`。

## 查看结果

```bash
RUN=$(ls -dt shadow_pick_place_v09_runs/rollout_* | head -1)
cat "$RUN/shadow_pick_place_v09_report.json"
open "$RUN/overlays"
```

重点检查：

```text
task_complete: true
safe_motion_complete: true
battery_validation_passed: true
drop_zone_validation_passed: true
non_contact_route_validation_passed: true
shadow_plan_ready: true
execution_enabled: false
horizontal_motion_commanded: false
gripper_transition_commanded: false
```

`shadow_pick_place_plan.waypoints` 是下一阶段供审核的完整候选路径。接触高度和 FANUC
R 角转换仍标记为 provisional，不应手工改成可执行动作。

## 首次桌面布置

1. 保持机械臂起始 TCP 与 v08.3 成功实验相同。
2. 移走 5.5 cm 标定盒、杂物和其他电池。
3. 只放一节平躺 AAA 电池，优先沿用已验证位置，约为 base
   `x=417 mm, y=-12 mm`；电池周围至少留 100 mm 空间。
4. 放一个空的、较宽、较浅的训练样式收纳碗，建议中心约为 base
   `x=280 mm, y=180 mm`，碗沿不高于 60 mm。
5. 电池和碗中心至少相距 120 mm；两者都要完整出现在 table camera 中，并远离桌边。
6. start TCP 垂直投影约为 `x=300 mm, y=0 mm`；其 75 mm 半径内保持完全空旷。
7. 首轮不要在碗中预放电池，也不要加入螺丝或方块。

运行时人员不得进入工作区，并保持急停可立即触达。
