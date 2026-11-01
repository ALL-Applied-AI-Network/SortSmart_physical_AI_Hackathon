# Isaac Sim + 真实 YOLO：第一阶段

这一阶段只验证感知接口，不驱动机械臂抓取：

1. Isaac 构建一枚 AAA 电池和白灰色碗；
2. Table camera 渲染 RGB；
3. 外层训练环境中的 YOLO26s-seg 推理；
4. 通过共享 `/workspace` 返回实例掩膜多边形；
5. Isaac 使用已有相机标定计算电池 XY/yaw 和碗中心；
6. 模拟器真实位姿仅用于误差评估。

## 终端 A：外层 workspace 容器

不要进入 `isaac-fanuc`。运行：

```bash
cd /workspace/fanuc_er4ia_sim

YOLO_PY=/workspace/fanuc_er4ia_sim/aaa_yolo_seg_training_v01/.venv/bin/python
MODEL=/workspace/fanuc_er4ia_sim/aaa_battery_seg_combined_v01/runs/aaa_yolo26s_seg/train_v01/weights/best.pt

"$YOLO_PY" yolo_ipc_worker.py \
  --model "$MODEL" \
  --ipc-dir /workspace/fanuc_er4ia_sim/runs/yolo_ipc \
  --device 0
```

看到 `[yolo-worker] ready` 后保持终端运行。

## 终端 B：Isaac 容器

```bash
docker exec -it isaac-fanuc bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim \
IPC_DIR=/workspace/fanuc_er4ia_sim/runs/yolo_ipc \
./run_sim_yolo_perception.sh runs/sim_yolo_stage1_01
```

## 检查

```bash
cat runs/sim_yolo_stage1_01/sim_yolo_perception_report.json
```

必须检查 `task_complete`，不能只看 Isaac 退出码。查看：

- `sim_table_rgb.png`
- `sim_yolo_overlay.png`
- `ipc_response.json`
- `sim_yolo_perception_report.json`

如果模型没有识别合成电池或碗，这不是 IPC 失败，而是 sim-to-real
外观域差异。下一步应调整合成材质/纹理或用少量合成数据再训练，不能用
模拟器真实坐标代替检测结果。

