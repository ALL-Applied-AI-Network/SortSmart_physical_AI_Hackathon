# v07.2 guarded vertical descent and depth ranging

## Scope

This experiment does only the following:

1. Hold the observed start pose and require five consecutive safe depth frames.
2. Descend vertically at the same X/Y through Z = `-150, -190, -220, -235 mm`.
   At every reached height, require three consecutive safe frames (up to five
   attempts) before continuing.
   While the robot is moving, the fixed table camera and robot-self mask are
   the fail-closed guards; wrist calibration is diagnostic until the target is
   reached and stationary.
3. At the low pose, collect ten table/wrist depth observations.
4. Report connected tabletop components, possible battery-sized components,
   and their distances from the TCP.
5. Retreat upward through the same vertical waypoints.

It commands no horizontal motion and no gripper transition.

## Scene layout

- Keep a clear cylinder of at least **75 mm radius directly below the gripper**.
- Put one AAA battery near the previous box location, approximately base-frame
  `(X=380 mm, Y=20 mm)`. This is visible to the wrist camera but outside the
  guarded descent cylinder around the current TCP `(X≈300 mm, Y≈0 mm)`.
- Put the 55–56 mm reference box at least **120 mm away from the descent line**,
  for example near `(X=430 mm, Y=-120 mm)`.
- Optionally add one or two matte, stable objects of height **10–40 mm**, each
  at least 100 mm from the descent line and separated by 50 mm.
- No bowl, loose screws, reflective metal, stacked objects, hands, cables, or
  external object taller than 80 mm in this first run.
- Keep the gripper open and keep the emergency stop accessible.

If base-frame marks are unavailable, use the simpler physical rule: leave an
8 cm radius circle under the gripper empty, place the battery where the box was
during v06.1, then move the box roughly 15 cm sideways.

## Install and run

```bash
cp policy.py policy_before_guarded_descent_v07.py
tar -xzf physical_guarded_descent_ranging_v07_2.tar.gz
uv run python -m unittest -v test_guarded_descent_v07.py
uv run run.py --mock --trials 1 --trial-s 8
```

v07 changes the action declaration from joint hold to TCP motion, so it needs a
new approved request:

```bash
uv run run.py --submit
```

Watch the first real rollout. The expected progression is five preflight
messages, four descent steps, ten low observations, and an upward retreat.
Stop immediately if the physical motion is not purely vertical.

## Result files

```text
guarded_descent_v07_runs/rollout_*/guarded_descent_v07_report.json
guarded_descent_v07_runs/rollout_*/preflight.jsonl
guarded_descent_v07_runs/rollout_*/descent_safety.jsonl
guarded_descent_v07_runs/rollout_*/low_observations.jsonl
```

The component label `battery_sized` is only a depth-geometry hint, not YOLO
semantic identification. The report records measured length, width, height,
horizontal distance, vertical TCP clearance, and Euclidean TCP distance. YOLO
fusion should be added only after this guarded ranging experiment succeeds.
