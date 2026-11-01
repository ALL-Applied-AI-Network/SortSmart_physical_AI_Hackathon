# Stage 1.1：碗中心修正与随机姿态测试

本阶段仍然只做视觉测试，不移动机械臂。

## 改动

- 在真实碗口高度反投影，不再错误地投影到桌面；
- 对 YOLO 碗轮廓进行稳健圆拟合；
- 拟合不可靠时自动回退到高度修正后的轮廓质心；
- 使用 12 组不同位置和非固定倍数 yaw 测试 AAA 电池；
- 输出 JSON 与 CSV 汇总结果；
- 全程禁止使用仿真真值生成控制结果，真值只在最后计算误差。

## 安装

在 Isaac 容器中：

```bash
cd /workspace/fanuc_er4ia_sim

cp sim_yolo_perception_test.py sim_yolo_perception_test_before_stage11.py

tar -xzf /workspace/battery_sim_yolo_stage11_patch_v01.tar.gz
chmod +x run_sim_yolo_sweep.sh
```

## 运行

终端 A 的 `yolo_ipc_worker.py` 必须保持运行。

在 Isaac 容器的终端 B：

```bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim \
IPC_DIR=/workspace/fanuc_er4ia_sim/runs/yolo_ipc \
./run_sim_yolo_sweep.sh runs/sim_yolo_sweep_01
```

该测试会启动 12 次 Isaac Sim，需要等待几分钟。单次失败不会中止后续试验。

## 查看汇总

```bash
cat runs/sim_yolo_sweep_01/sim_yolo_sweep_report.json
```

重点指标：

- `success_rate` 至少为 `0.90`；
- `battery_center_error_mm.p95` 不超过 `5.0`；
- `battery_yaw_error_deg.p95` 不超过 `5.0`；
- `bowl_center_error_mm.p95` 不超过 `10.0`；
- `all_quality_gates_passed` 为 `true`。

如果终端最终返回状态 2，但汇总文件已经生成，说明测试完成但至少一项质量门槛未通过，不代表程序崩溃。
