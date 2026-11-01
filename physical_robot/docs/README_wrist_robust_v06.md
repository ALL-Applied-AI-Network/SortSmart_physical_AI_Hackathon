# Physical wrist-depth validation v06.1

This remains a **stationary diagnostic**. It holds the observed joints and
never commands a Cartesian pose, grasp, or scan motion.

The v06.1 hotfix also handles the local mock's `tcp_pose=None` correctly. The
mock cannot perform the TCP-dependent self-mask diagnostic, so the policy logs
that limitation once, holds the measured joints, and lets the mock verify only
the action interface. It does not fabricate a TCP or send a Cartesian command.

## Why v05 failed

The supplied 18-frame capture showed two different facts:

- The wrist table-plane median was accurate (`-0.45 mm` median across frames).
- A stable spatial error band made the absolute p95 about `18.2 mm` and the
  untrimmed local table-plane RMSE about `9.0 mm`.

Removing depth-discontinuity pixels did not remove the p95 error, so it is not
primarily a box-edge artifact. At the same time, five-frame median consensus
reduced the measured box-height variation to about `0.74 mm`.

## v06 decision logic

- Table-camera robot/self masking must still pass.
- Wrist table median must be within `2 mm`.
- Wrist table absolute p95 must be at most `20 mm`.
- The known-height box must be detected in at least 90% of frames.
- At least 90% of frames must pass their individual gates.
- Box-height median must be `50–61 mm`; raw standard deviation at most `2 mm`.
- Every five-frame rolling median must stay within `2.5 mm` of the overall
  median.
- Any later wrist-depth motion guard must reserve at least `25 mm`, larger
  than the observed p95 envelope.

Even after a pass, `motion_enable_recommended` remains `false`. A pass only
permits development of the next limited, depth-guarded scan test.

## Install

From the `PhysicalAI-Hackathon-2026` folder:

```bash
cp policy.py policy_before_wrist_robust_v06.py
tar -xzf physical_wrist_robust_validation_v06_1.tar.gz
```

The archive installs `policy.py`, `robot_self_filter.py`, calibration files,
the test, and this README.

Because v06 keeps the same action, camera, depth, and image declarations as
v05, an existing approved request with unused rollouts normally does not need
another `--submit`. If that request has expired or has no rollouts left, submit
again.

```bash
uv run run.py --mock --trials 1 --trial-s 8
uv run run.py
```

Keep the 55–56 mm box in the same measured ROI and do not add other tall
objects. The key output is:

```text
wrist_robust_diagnostics/rollout_*/wrist_robust_summary.json
```

The desired stationary result is:

```json
{
  "wrist_depth_and_self_mask_validation_passed": true,
  "motion_enable_recommended": false,
  "next_stage": "limited_depth_guarded_scan_motion_with_25mm_wrist_margin"
}
```

## Local tests

```bash
uv run python -m unittest -v test_wrist_robust_v06.py
```
