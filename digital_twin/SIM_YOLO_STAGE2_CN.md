# Stage 2：外部 YOLO 驱动的完整电池抓取与投放

## 流程

1. 机械臂停在标定观察姿态；
2. table camera 拍摄一次规划图像；
3. 完整图检测碗，标定 ROI 检测 AAA 电池；
4. 由分割掩膜计算电池中心、yaw 和碗中心；
5. 夹爪旋转到电池侧面抓取方向；
6. 下降、闭合、抬升；
7. 搬运到检测出的碗中心上方；
8. 松开电池并退回安全高度；
9. 使用仿真真值进行事后评分，但不影响任何控制命令。

如果电池或碗没有通过置信度与几何检查，程序会在机械臂运动前安全退出。

## 安装

```bash
cd /workspace/fanuc_er4ia_sim
tar -xzf /workspace/battery_sim_yolo_stage2_patch_v01.tar.gz
chmod +x run_battery_yolo_pick_place.sh
```

## YOLO worker

终端 A 必须恢复 `--imgsz 1024`。Stage 2 不使用失败的 1536 放大方案。

## 第一次测试

终端 B：

```bash
cd /workspace/fanuc_er4ia_sim

ISAAC=/isaac-sim \
IPC_DIR=/workspace/fanuc_er4ia_sim/runs/yolo_ipc \
./run_battery_yolo_pick_place.sh runs/battery_yolo_pick_place_01 \
  --x-m 0.34 --y-m -0.12 --yaw-deg 7.0 \
  --no-video-frames
```

先使用已经验证可识别的 `trial_01`，并关闭视频以降低第一次机械测试的负担。

查看：

```bash
cat runs/battery_yolo_pick_place_01/battery_yolo_pick_place_report.json
```

必须满足：

- `task_complete: true`；
- `simulator_pose_used_for_control: false`；
- `grasp.pickup_verified: true`；
- `placement.carried_to_detected_bowl: true`；
- `placement.inside_bowl: true`。

通过后可去掉 `--no-video-frames`，使用新的输出目录重新运行并生成轨迹帧。
