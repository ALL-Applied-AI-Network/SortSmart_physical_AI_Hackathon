# 实物碗外观修正 v02

本补丁只替换 `battery_drop_bowl.py`，不会修改相机标定、YOLO 服务或电池定位代码。

主要变化：

- 碗口直径由 110 mm 增加到约 192 mm；
- 由直壁圆筒改为大号浅碗的倾斜曲面；
- 改用暖灰/米白色哑光塑料材质；
- 隐藏 PhysX 碰撞代理，避免它们出现在相机画面中；
- 保留碗底和环形墙碰撞，后续可用于真实的投放仿真。

在 Isaac 容器内安装：

```bash
cd /workspace/fanuc_er4ia_sim
cp battery_drop_bowl.py battery_drop_bowl_before_v02.py
tar -xzf /workspace/battery_sim_bowl_v02_patch.tar.gz
```

终端 A 的 YOLO worker 可以继续运行。终端 B 使用新的输出目录测试：

```bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim \
IPC_DIR=/workspace/fanuc_er4ia_sim/runs/yolo_ipc \
./run_sim_yolo_perception.sh runs/sim_yolo_stage1_02
```

检查：

```bash
cat runs/sim_yolo_stage1_02/sim_yolo_perception_report.json
```

重点查看：

- `drop_zone_candidates` 不再为空；
- `selected_drop_zone` 不为 `null`；
- `task_complete` 为 `true`；
- `bowl_center_error_mm` 应有数值。

如果仍未识别，不要先降低阈值。请检查：

- `runs/sim_yolo_stage1_02/sim_table_rgb.png`
- `runs/sim_yolo_stage1_02/sim_yolo_overlay.png`
- `runs/sim_yolo_stage1_02/ipc_response.json`

然后再根据实物碗的准确外径、高度和颜色继续校准。
