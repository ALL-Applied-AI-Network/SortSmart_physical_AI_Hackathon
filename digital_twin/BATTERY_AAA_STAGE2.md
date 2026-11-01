# AAA Stage 2: calibrated 2D pose sweep

This stage remains AAA-only. It does not add AA geometry, classification or
grasp parameters.

It evaluates the same calibrated RGB-only controller at six table positions
and long-axis yaws: 0, 30, 60, 90, 120 and 150 degrees. Each trial must detect
the cell, command the side grasp without simulator pose input, lift it and hold
it. Trajectory frames are disabled to keep the sweep small; each trial retains
its perception overlay and JSON reports.

Copy the files in this patch into the project root, then run inside the Isaac
Sim container:

```bash
cd /workspace/fanuc_er4ia_sim
chmod +x run_battery_grasp.sh run_battery_2d_sweep.sh

ISAAC=/isaac-sim \
./run_battery_2d_sweep.sh runs/aaa_stage2_01
```

Primary outputs:

- `runs/aaa_stage2_01/battery_2d_sweep_report.json`
- `runs/aaa_stage2_01/battery_2d_sweep_trials.csv`
- `runs/aaa_stage2_01/trial_*/battery_perception_overlay.png`
- `runs/aaa_stage2_01/trial_*/battery_grasp_report.json`

Require:

```json
"reports_found": 6,
"vision_control_trials": 6,
"successes": 6,
"all_trials_successful": true
```

If a trial fails, the sweep stops at that trial so its terminal traceback is not
lost. Use a new empty root directory after correcting the cause.
