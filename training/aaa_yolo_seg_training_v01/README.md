# AAA Battery Instance-Segmentation Training v0.1

This workspace bootstraps a single-class AAA-battery segmenter for the FANUC
table camera.

## Included data

- 106 original phone images
- 60 additional phone images, including many no-battery negative scenes
- 44 stable FANUC table-camera images
- 210 images total

The first model intentionally excludes the side and wrist cameras because the
runtime detector will initially use the table camera. Camera domains can be
added later as separate experiments.

## Model plan

1. `YOLOE-26n-seg` zero-shot inference with the prompt `AAA battery` provides
   the un-fine-tuned baseline and draft masks.
2. A human checks and corrects every draft mask.
3. `YOLO26n-seg` is fine-tuned as a fast, closed-set model with one class:
   `aaa_battery`.
4. Robot episodes 000–002 are used for training. Episodes 003–007 remain in
   validation. The final test set must be newly recorded physical scenes.

## Important: labels are required

The supplied images are not ground-truth labeled. Do not train immediately on
unchecked pseudo-labels. Every visible battery must have a polygon, including
partially visible batteries. A no-battery image must have an empty `.txt` file.
Any missed battery would otherwise be treated as background during training.

YOLO segmentation labels use one row per instance:

```text
0 x1 y1 x2 y2 x3 y3 ...
```

Coordinates are normalized to `[0, 1]`; class `0` means `aaa_battery`.

## Run location

Run training in the DGX GPU Python environment, not inside the Isaac Sim
container. A typical prompt is:

```text
(base) root@sortsmarta6f3:/workspace/fanuc_er4ia_sim#
```

## Installation

From the FANUC project directory:

```bash
cd /workspace/fanuc_er4ia_sim
unzip -o /workspace/aaa_yolo_seg_training_v01.zip -d .
cd aaa_yolo_seg_training_v01

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python 00_check_environment.py
```

`cuda_available` must be `True`. The first YOLOE run downloads its checkpoint,
CLIP package, and text encoder, so it needs network access once.

## Step 1: zero-shot baseline and draft masks

```bash
python 01_zero_shot_and_pseudolabel.py \
  --images raw_images \
  --labels draft_labels \
  --overlays qa/pseudo_overlays \
  --model yoloe-26n-seg.pt \
  --prompt "AAA battery" \
  --conf 0.12 \
  --device 0
```

Outputs:

- `draft_labels/*.txt`
- `qa/pseudo_overlays/*.jpg`
- `qa/zero_shot_report.json`

The zero-shot report is also the M0 baseline. Record how many batteries were
missed and how many distractors were predicted.

## Step 2: human correction

Inspect every image in `qa/pseudo_overlays`. Correct masks using an annotation
tool that can import/export YOLO segmentation polygons, such as CVAT or the
Ultralytics annotation editor.

Check all of the following:

- both 7/11 and Amazon batteries use class `aaa_battery`;
- all visible batteries are labeled;
- distractors are not labeled;
- polygons follow the cylindrical body closely;
- negative images have an empty label file;
- partially visible batteries are still labeled.

After correction, put the final YOLO polygon files in `draft_labels/`, then run:

```bash
python 02_review_labels.py \
  --images raw_images \
  --labels draft_labels
```

Inspect `qa/review_contact_sheet.jpg`. The script checks label syntax and
missing files, but human inspection is still required for semantic correctness.

## Step 3: build leakage-aware splits

```bash
python 03_build_dataset.py
```

Expected split before any manual removals:

- train: 168 images
- validation: 42 images
- test: deliberately empty; use future unseen robot scenes

Do not move images from the future ten-scene physical test into training.

## Step 4: fine-tune YOLO nano segmentation

```bash
python 04_train.py \
  --model yolo26n-seg.pt \
  --device 0 \
  --batch -1
```

Training uses two stages:

- 30 epochs with the first ten layers frozen;
- up to 70 epochs fully unfrozen with early stopping.

The script uses full 180-degree rotation augmentation because a battery may have
any yaw on the table.

## Step 5: evaluate and inspect overlays

```bash
python 05_evaluate.py \
  --weights runs/aaa_seg/stage2_unfrozen/weights/best.pt \
  --source dataset/images/val
```

Primary offline metrics are mask mAP, detection recall, false positives, and
visual inspection of overlays. The final project metrics must also include
real-robot grasp and placement success.

## Step 6: produce center and yaw predictions

```bash
python 06_predict_pose.py \
  --weights runs/aaa_seg/stage2_unfrozen/weights/best.pt \
  --source dataset/images/val
```

For each predicted mask, the script applies PCA to its polygon and writes:

- confidence;
- center in image pixels;
- battery yaw in `[0, 180)` degrees.

Camera calibration must then convert the image center to robot `x, y`. The
gripper-frame convention determines the fixed 90-degree offset needed for the
side grasp.

## Collection box

`configs/box_config.yaml` proposes an internal size of:

```text
100 × 80 × 35 mm
```

Fourteen AAA batteries have a combined geometric volume of about 54 cubic
centimeters; the proposed box is about 280 cubic centimeters, leaving enough
room for random drops and a safe gripper release opening.

The box is assumed fixed near the upper-left of the table-camera view. Do not
copy that visual location directly into robot coordinates. Install the box,
verify reachability and collisions, calibrate one fixed drop pose, and fill in
`drop_pose_robot`. Also fill in `no_pick_roi_pixels` so batteries already inside
the box are never selected again.

