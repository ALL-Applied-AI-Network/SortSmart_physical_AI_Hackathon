# Stage 1.2b：双尺度分类推理修正

Stage 1.2 的 ROI 成功恢复了最困难的 AAA 电池：置信度从低于 0.30 提升到 0.683，中心误差 1.13 mm，yaw 误差 1.13°。但是碗在裁剪图中丢失。

本修正仅替换 `yolo_ipc_worker.py`：

- 完整相机图像只检测 `battery_drop_zone`；
- 标定工作区 ROI 只检测 `aaa_battery`；
- ROI 多边形自动换算回完整图像坐标；
- 两次推理结果合并后再返回 Isaac；
- 保持电池阈值 0.60、碗阈值 0.30 不变。

安装：

```bash
cd /workspace/fanuc_er4ia_sim
cp yolo_ipc_worker.py yolo_ipc_worker_before_stage12b.py
tar -xzf /workspace/battery_sim_yolo_stage12b_patch_v01.tar.gz
```

必须在终端 A 按 `Ctrl+C` 停止旧 worker，然后用原命令重新启动。

终端 B 使用新的输出目录复测相同场景：

```bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim \
IPC_DIR=/workspace/fanuc_er4ia_sim/runs/yolo_ipc \
./run_sim_yolo_perception.sh runs/sim_yolo_stage12b_single_01 \
  --x-m 0.34 --y-m -0.12 --yaw-deg 7.0
```

报告中应同时出现：

- `battery_candidates`；
- `drop_zone_candidates`；
- `task_complete: true`；
- `ipc.settings.inference_mode: split_scale_by_class`。
