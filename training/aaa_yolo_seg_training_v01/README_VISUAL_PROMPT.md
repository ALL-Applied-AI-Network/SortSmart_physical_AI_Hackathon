# AAA battery visual-prompt pre-labeling

This patch does not overwrite the previous text-prompt labels.

Files to copy into `aaa_yolo_seg_training_v01/`:

- `01b_visual_prompt_pseudolabel.py`
- `visual_prompt_reference.png`
- `visual_prompt_reference_boxes.jpg` (preview only)

First run a 30-image diagnostic sample:

```bash
python 01b_visual_prompt_pseudolabel.py \
  --images raw_images \
  --labels draft_labels_visual_test \
  --overlays qa/visual_prompt_test_overlays \
  --reference visual_prompt_reference.png \
  --model yoloe-26s-seg.pt \
  --conf 0.08 \
  --imgsz 1280 \
  --device 0 \
  --sample-size 30
```

Review `qa/visual_contact_sheet_001.jpg` and individual files in
`qa/visual_prompt_test_overlays/`.

If the diagnostic sample is substantially better, process all 210 images:

```bash
python 01b_visual_prompt_pseudolabel.py \
  --images raw_images \
  --labels draft_labels_visual \
  --overlays qa/visual_prompt_overlays \
  --reference visual_prompt_reference.png \
  --model yoloe-26s-seg.pt \
  --conf 0.08 \
  --imgsz 1280 \
  --device 0
```

The two battery brands are both assigned to class `0` (`aaa_battery`). These
outputs are still draft labels: every image must be reviewed before training.
