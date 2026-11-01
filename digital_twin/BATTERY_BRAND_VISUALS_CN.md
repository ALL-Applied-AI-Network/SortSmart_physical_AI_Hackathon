# AAA 双品牌 Digital Twin 外观补丁

本补丁根据实拍照片为仿真 AAA 电池增加两套外观：

- `amazon`：Amazon Basics 橙色标签、蓝黑端带；
- `seven_eleven`：24/7 Life 黑金标签；
- `random`：根据位置、yaw 和固定 seed 可复现地选择品牌。

两种外观仍然属于同一个 YOLO 类别 `aaa_battery`。补丁只增加 UV
渲染外壳，原来的 AAA 尺寸、质量、摩擦、碰撞圆柱和抓取参数保持不变。

## 安装

在 `/workspace/fanuc_er4ia_sim` 中解压补丁。纹理 PNG 已经包含在补丁中，
通常不需要重新生成。

```bash
cd /workspace/fanuc_er4ia_sim
tar -xzf /workspace/battery_brand_visuals_stage2_burst_v03.tar.gz
chmod +x run_sim_yolo_perception.sh run_sim_yolo_sweep.sh \
  run_battery_yolo_pick_place.sh
python3 -m py_compile \
  battery_brand_visuals.py \
  make_battery_brand_textures.py \
  sim_yolo_perception_test.py \
  battery_yolo_pick_place.py
python3 -m unittest test_battery_brand_textures
```

若普通环境没有 `python3`，语法检查可以在 Isaac 容器中使用：

```bash
/isaac-sim/python.sh -m py_compile \
  battery_brand_visuals.py \
  sim_yolo_perception_test.py \
  battery_yolo_pick_place.py
```

## 单张感知 A/B 测试

YOLO worker 继续使用 `imgsz=1024`，不需要重新训练或改成 1536。

Amazon：

```bash
ISAAC=/isaac-sim IPC_DIR="$PWD/runs/yolo_ipc" \
./run_sim_yolo_perception.sh runs/brand_amazon_01 \
  --x-m 0.34 --y-m -0.12 --yaw-deg 7.0 \
  --battery-brand amazon \
  --battery-visual-roll-deg 0
```

24/7 Life：

```bash
ISAAC=/isaac-sim IPC_DIR="$PWD/runs/yolo_ipc" \
./run_sim_yolo_perception.sh runs/brand_247_01 \
  --x-m 0.34 --y-m -0.12 --yaw-deg 7.0 \
  --battery-brand seven_eleven \
  --battery-visual-roll-deg 0
```

检查：

```bash
cat runs/brand_amazon_01/sim_yolo_perception_report.json
cat runs/brand_247_01/sim_yolo_perception_report.json
```

同时查看每个目录中的：

- `sim_table_rgb.png`：Isaac 原始相机图；
- `sim_yolo_overlay.png`：YOLO 掩膜叠加；
- `sim_yolo_perception_report.json`：识别置信度和定位误差。

## 复测原来失败的 trial 12

```bash
ISAAC=/isaac-sim IPC_DIR="$PWD/runs/yolo_ipc" \
./run_sim_yolo_perception.sh runs/trial12_247_texture_01 \
  --x-m 0.37 --y-m 0.02 --yaw-deg 142.3 \
  --battery-brand seven_eleven
```

## 双品牌 12 次 sweep

`run_sim_yolo_sweep.sh` 已经改为 Amazon 与 24/7 Life 各 6 次，并随机化
标签绕电池轴的角度：

```bash
ISAAC=/isaac-sim IPC_DIR="$PWD/runs/yolo_ipc" \
./run_sim_yolo_sweep.sh runs/sim_yolo_brands_sweep_01
```

汇总报告中的 `brand_breakdown` 会分别列出两种外观的检测次数、成功次数和
置信度。

## 整体抓取并放入碗中

```bash
ISAAC=/isaac-sim IPC_DIR="$PWD/runs/yolo_ipc" \
./run_battery_yolo_pick_place.sh runs/pick_place_amazon_01 \
  --x-m 0.34 --y-m -0.12 --yaw-deg 7.0 \
  --battery-brand amazon \
  --no-video-frames
```

第一次建议先运行 `--no-video-frames`。确认报告中的以下字段均为 `true`：

- `task_complete`；
- `grasp.pickup_verified`；
- `placement.carried_to_detected_bowl`；
- `placement.inside_bowl`。

抓取脚本默认最多检查 5 个连续相机帧，并选择同时检测到电池和碗、且电池
置信度最高的一帧。阈值不会降低；若 5 帧全部失败，机械臂仍会在运动前安全
停止。每次尝试记录在 `perception_attempts.json`，对应图片保存为
`planning_rgb_attempt_*.png`。

外观补丁不使用模拟器物体真值来控制机械臂；控制输入仍然只有外部 YOLO
实例掩膜和标定后的相机几何。
