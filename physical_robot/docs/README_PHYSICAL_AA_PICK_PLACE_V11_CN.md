# 物理机械臂 v11-AA：Amazon Basics AA 抓取与投放

本版本针对一节平躺的 **Amazon Basics AA/LR6 1.5 V** 电池。它沿用已验证的
table/wrist depth 安全门、YOLO 实例掩膜和碗定位，但在执行前额外要求深度点云满足
AA 几何范围。现有模型仍是用 AAA 数据训练的，因此 YOLO 只负责提出候选；AA 尺寸
验证不通过时，机械臂不会开始横移或闭合夹爪。

相较 v10 的关键调整：

- AA 深度门限：长 42–60 mm、宽 9–22 mm、台面上方最高点 9.5–20 mm；
- 夹取中心高度改为测得高度的 70%，并限制在台面上方 8.5–11.0 mm；
- 闭合等待从 5 帧增加到 8 帧；
- 闭合后先垂直试抬 35 mm，再抬到安全搬运高度，降低起步滑脱概率；
- 抬升后仍用 table RGB + depth 验证原位置，验证失败会保持夹爪并安全回撤。

## 桌面布置

1. 只放 **一节 Amazon Basics AA 电池**和此前已识别成功的空碗；移走盒子、螺丝和其他杂物。
2. 保持相机标定、桌面和碗的位置与最近成功的 v10 实验一致。
3. 电池必须平躺，长轴沿 robot base X 方向，角度误差不超过 10°。
4. 电池周围至少留 100 mm 空白；电池与碗中心至少相距 120 mm。
5. 操作员全程在急停旁；首次只运行一次，不要连续提交多个 rollout。

## 安装与运行

在 `PhysicalAI-Hackathon-2026` 根目录：

```bash
cp policy.py policy_before_aa_v11.py
tar -xzf /path/to/physical_guarded_aa_pick_place_v11.tar.gz

uv run run.py --mock --trials 1 --trial-s 8
uv run run.py
```

终端应显示：

```text
[v11-aa] guarded physical AA pick/place started
```

## 查看结果

```bash
RUN=$(ls -dt physical_pick_place_aa_v11_runs/rollout_* | head -1)
cat "$RUN/physical_pick_place_aa_v11_report.json"
open "$RUN/overlays"
```

先确认以下字段，再判断是否完整成功：

```text
execution_plan.aa_geometry_valid: true
execution_plan.target_battery_format: AA/LR6
pickup_verified: true
placement_verified: true
task_complete: true
```

如果 `task_complete` 为 false，不要立即重跑；先查看 `reason`、
`battery_fusion_aggregate`、`execution_plan`、`pickup_verification` 和最后一个 `events` 条目。

