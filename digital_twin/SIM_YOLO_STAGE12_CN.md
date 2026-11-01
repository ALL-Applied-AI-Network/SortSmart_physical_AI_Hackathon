# Stage 1.2：标定工作区 ROI 小目标增强

## 原因

12 组测试中，碗中心和电池几何定位已经准确，但 4 组电池漏检。失败画面没有遮挡，其中 3 组电池置信度为 0.522、0.526 和 0.577，略低于 0.60 门槛；另 1 组低于模型原始 0.30 输出门槛。

本补丁不降低置信度阈值，而是利用相机标定把机器人工作区域投影成图像 ROI。YOLO 只处理包含电池和碗的区域，使 AAA 电池在模型输入中占据更多像素，并减少机器人与空白背景干扰。

多边形输出会自动加回 ROI 偏移，因此后续反投影仍使用完整相机坐标，不会破坏标定。

## 安装

```bash
cd /workspace/fanuc_er4ia_sim

cp yolo_ipc_client.py yolo_ipc_client_before_stage12.py
cp yolo_ipc_worker.py yolo_ipc_worker_before_stage12.py
cp sim_yolo_perception_test.py sim_yolo_perception_test_before_stage12.py

tar -xzf /workspace/battery_sim_yolo_stage12_patch_v01.tar.gz
```

## 必须重启终端 A 的 YOLO worker

旧 worker 没有 ROI 协议支持。先按 `Ctrl+C` 停止，再用原命令重新启动，仍保持：

- `--imgsz 1024`
- `--battery-conf 0.60`
- `--drop-conf 0.30`

不要降低阈值，以便与 Stage 1.1 公平比较。

## 单场景检查

终端 B：

```bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim \
IPC_DIR=/workspace/fanuc_er4ia_sim/runs/yolo_ipc \
./run_sim_yolo_perception.sh runs/sim_yolo_stage12_single_01 \
  --x-m 0.34 --y-m -0.12 --yaw-deg 7.0
```

这正是之前完全漏检的 `trial_01`。检查：

```bash
cat runs/sim_yolo_stage12_single_01/sim_yolo_perception_report.json
```

若 `task_complete` 为 `true`，再执行完整 sweep：

```bash
ISAAC=/isaac-sim \
IPC_DIR=/workspace/fanuc_er4ia_sim/runs/yolo_ipc \
./run_sim_yolo_sweep.sh runs/sim_yolo_sweep_02
```

## A/B 回退

单场景命令末尾添加 `--disable-workspace-roi` 可以恢复整幅图推理，用于对比。该开关仅用于诊断。
