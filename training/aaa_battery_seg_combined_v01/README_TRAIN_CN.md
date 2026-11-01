# AAA 电池实例分割训练包

本数据集用于训练 `yolo26s-seg`，类别定义如下：

- `0`: `aaa_battery`
- `1`: `battery_drop_zone`

## 数据规模

| Split | 图片 | AAA 电池实例 | 收纳区实例 | 空负样本 |
| --- | ---: | ---: | ---: | ---: |
| train | 105 | 350 | 43 | 17 |
| val | 14 | 36 | 14 | 0 |
| test | 5 | 40 | 5 | 0 |

手机照片仅进入 train。机械臂相机的 episode 保持独立，并继续用于 val/test。

## 开始训练

进入本目录并激活已经安装 Ultralytics 的 Python 环境，然后执行：

```bash
python train_yolo26s_seg.py \
  --data dataset.yaml \
  --model yolo26s-seg.pt \
  --epochs 150 \
  --imgsz 1024 \
  --batch 8 \
  --device 0 \
  --project runs/aaa_yolo26s_seg \
  --name train_v01
```

如果显存不足，先把 `--batch 8` 改为 `--batch 4`；仍然不足时再把 `--imgsz 1024` 改为 `--imgsz 768`。

脚本完成训练后会自动使用 `best.pt` 在 test split 上评估。

## 说明

当前 test split 只有一个独立机械臂场景的多个视角，因此模型指标只能用于开发阶段比较。最终报告还应使用多个未参与训练的真实场景进行抓取测试。
