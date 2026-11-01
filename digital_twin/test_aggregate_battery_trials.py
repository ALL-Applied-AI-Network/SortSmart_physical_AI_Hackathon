"""Host-only regression for AAA sweep metric aggregation."""

import unittest
from pathlib import Path

from aggregate_battery_trials import summarize


class BatterySweepAggregationTests(unittest.TestCase):
    def test_success_and_pose_metrics(self):
        report = {
            "task_complete": True,
            "control_source": "calibrated_2d_rgb",
            "simulator_pose_used_for_control": False,
            "battery": {"requested_xy_m": [0.36, 0.0], "requested_yaw_deg": 30.0},
            "perception": {
                "confidence": 0.9,
                "center_error_m": 0.001,
                "axis_alignment_abs_dot": math_cos_deg(2.0),
            },
            "lift_delta_m": 0.05,
            "xy_drift_m": 0.0003,
        }
        summary, rows = summarize(
            [(Path("trial_01/battery_grasp_report.json"), report)], expected_trials=1
        )
        self.assertTrue(summary["all_trials_successful"])
        self.assertEqual(summary["vision_control_trials"], 1)
        self.assertAlmostEqual(summary["center_error_mm"]["mean"], 1.0)
        self.assertAlmostEqual(summary["yaw_error_deg"]["mean"], 2.0)
        self.assertTrue(rows[0]["success"])


def math_cos_deg(degrees):
    import math

    return math.cos(math.radians(degrees))


if __name__ == "__main__":
    unittest.main()
