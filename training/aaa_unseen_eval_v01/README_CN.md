# AAA 电池未见图片推理与像素姿态评估

这个包包含：

- 训练完成的 `best.pt`；
- 24 张未进入 56 张人工标注训练子集的 sanity 图片；
- `evaluate_unseen.py` 推理与像素姿态脚本；
- 图片来源清单和原始图片联系表。

注意：这些 sanity 图片仍来自相同拍摄批次，可能与训练图片相似，只能验证推理流程。最终性能必须使用重新布置的机械臂真实场景测试。

## 运行

把压缩包上传到 `/workspace/fanuc_er4ia_sim`，然后：

```bash
cd /workspace/fanuc_er4ia_sim
source aaa_yolo_seg_training_v01/.venv/bin/activate
python -m zipfile -e aaa_unseen_eval_v01.zip .
cd aaa_unseen_eval_v01

python evaluate_unseen.py \
  --model best.pt \
  --source sanity_images \
  --output runs/sanity_eval_v01 \
  --imgsz 1024 \
  --device 0 \
  --battery-conf 0.60 \
  --drop-conf 0.30 \
  --iou 0.50
```

主要输出：

- `runs/sanity_eval_v01/overlay_contact_sheet.jpg`：所有预测叠加图；
- `runs/sanity_eval_v01/unseen_eval_report.json`：置信度、中心、像素 yaw、长短轴、面积与抓取候选状态；
- `runs/sanity_eval_v01/overlays/`：逐张预测图。

## 颜色

- 绿色：可抓取的电池候选；
- 橙色：低置信度或已经位于收纳区内的电池；
- 青色：收纳区。

## 当前输出的限制

脚本输出的是图片像素坐标和图片平面 yaw，还不是 FANUC 机器人坐标。把它部署到真实机械臂之前，必须完成相机标定以及像素到机器人工作台坐标的转换。
