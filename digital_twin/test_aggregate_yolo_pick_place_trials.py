"""Host regression for full pick/place sweep aggregation."""

import math
from pathlib import Path
import unittest

from aggregate_yolo_pick_place_trials import summarize


def report(brand, source, roll):
    return {
        "task_complete": True,
        "safely_abstained": False,
        "control_source": (
            "external_yolo_instance_masks_plus_calibrated_geometry_"
            "with_fixed_wrist_fallback"
        ),
        "simulator_pose_used_for_control": False,
        "requested_scene": {
            "battery_brand": brand,
            "battery_visual_roll_deg": roll,
            "battery_x_m": 0.34,
            "battery_y_m": -0.12,
            "battery_yaw_deg": 7.0,
        },
        "perception": {
            "selected_battery": {
                "confidence": 0.8,
                "source_camera": source,
                "pose": {
                    "center_xy_m": [0.341, -0.12],
                    "axis_xy": [math.cos(math.radians(8)), math.sin(math.radians(8))],
                },
            },
            "selected_drop_zone": {"center_xy_m": [0.282, 0.18]},
            "wrist_reobservation": {
                "triggered": source == "wrist",
                "selected_scan": 1 if source == "wrist" else None,
            },
        },
        "grasp": {"pickup_verified": True, "lift_delta_m": 0.12},
        "placement": {
            "carried_to_detected_bowl": True,
            "inside_bowl": True,
            "final_center_distance_to_truth_bowl_m_evaluation_only": 0.0025,
        },
        "evaluation_only": {
            "initial_position_world_m": [0.34, -0.12, 0.90],
            "initial_axis_world": [
                math.cos(math.radians(7)),
                math.sin(math.radians(7)),
            ],
            "truth_bowl_xy_m": [0.28, 0.18],
        },
    }


class PickPlaceSweepAggregationTests(unittest.TestCase):
    def test_balanced_success_summary_and_errors(self):
        reports = [
            (Path("trial_01/battery_yolo_pick_place_report.json"), report("amazon", "wrist", 90)),
            (Path("trial_02/battery_yolo_pick_place_report.json"), report("seven_eleven", "table", 0)),
        ]
        summary, rows = summarize(reports, expected_trials=2, min_success_rate=0.90)
        self.assertTrue(summary["all_quality_gates_passed"])
        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["wrist_camera_selections"], 1)
        self.assertEqual(summary["table_camera_selections"], 1)
        self.assertAlmostEqual(summary["battery_center_error_mm"]["mean"], 1.0)
        self.assertAlmostEqual(summary["battery_yaw_error_deg"]["mean"], 1.0)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
