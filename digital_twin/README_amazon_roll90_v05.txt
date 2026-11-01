Amazon roll-90 visual coverage patch v05

Diagnosis
---------
The failed Isaac frames render the AAA battery at only about 14 x 52 pixels.
At a 90-degree wrapper roll, the old texture exposed an almost featureless
orange strip plus a narrow specular highlight.  The real reference photos show
large black model text and regulatory pictograms around all wrapper sides.

Changes
-------
- Adds bold high-contrast label structure to every circumference quarter.
- Adds simplified regulatory pictograms referenced from the supplied photos.
- Raises Amazon wrapper roughness from 0.30 to 0.40 to broaden the highlight.
- Does not change collision, mass, friction, grasp planning, motion, or YOLO
  thresholds.

Install from /workspace/fanuc_er4ia_sim
---------------------------------------
cp battery_brand_visuals.py battery_brand_visuals_before_roll90_v05.py
cp make_battery_brand_textures.py make_battery_brand_textures_before_roll90_v05.py
cp assets/battery_textures/amazon_basics_aaa.png \
  assets/battery_textures/amazon_basics_aaa_before_roll90_v05.png

tar -xzf /workspace/amazon_roll90_visual_patch_v05.tar.gz

/isaac-sim/python.sh -m py_compile \
  battery_brand_visuals.py \
  make_battery_brand_textures.py

Optional host-side texture tests
--------------------------------
python3 -m unittest -v test_battery_brand_textures.py

Validation run
--------------
ISAAC=/isaac-sim \
IPC_DIR="$PWD/runs/yolo_ipc" \
./run_battery_yolo_pick_place.sh runs/pick_place_amazon_roll90_02 \
  --x-m 0.34 \
  --y-m -0.12 \
  --yaw-deg 7.0 \
  --battery-brand amazon \
  --battery-visual-roll-deg 90 \
  --no-video-frames

Inspect
-------
cat runs/pick_place_amazon_roll90_02/perception_attempts.json
cat runs/pick_place_amazon_roll90_02/battery_yolo_pick_place_report.json

The safety gate remains battery_conf=0.60.  If the new visual still does not
reach that threshold, the robot must safely abstain; the next step is an active
wrist-camera re-observation rather than lowering the threshold.
