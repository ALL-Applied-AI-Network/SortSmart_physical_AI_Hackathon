import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose, load
try:
    from policy_wrist_robust_v06 import Policy
except ModuleNotFoundError:  # installed archive exposes the policy as policy.py
    from policy import Policy
from robot_self_filter import classify_robot_self_points, robot_skeleton_segments


HOME = np.array(
    [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0],
    dtype=np.float32,
)
TCP = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]


def synthetic_depth(camera_name: str, box_height_mm: float = 55.0) -> np.ndarray:
    cam = load()[camera_name]
    transform = cam["T"]
    if cam["frame"] == "flange":
        transform = flange_pose(HOME) @ transform
    width, height = cam["size"]
    v, u = np.mgrid[0:height, 0:width]
    camera_rays = np.stack(
        [
            (u - cam["cx"]) / cam["fx"],
            (v - cam["cy"]) / cam["fy"],
            np.ones((height, width)),
        ],
        -1,
    )
    base_rays = camera_rays @ transform[:3, :3].T
    origin = transform[:3, 3]
    table_scale = (SURFACE_Z - origin[2]) / base_rays[..., 2]
    top_scale = (
        SURFACE_Z + box_height_mm / 1000.0 - origin[2]
    ) / base_rays[..., 2]
    top_points = origin + top_scale[..., None] * base_rays
    on_box = (
        (top_points[..., 0] >= 0.26)
        & (top_points[..., 0] <= 0.42)
        & (top_points[..., 1] >= -0.12)
        & (top_points[..., 1] <= 0.07)
    )
    scale = np.where(on_box, top_scale, table_scale)
    return np.where((scale > 0) & (scale < 3), scale * 1000, 0).astype(
        np.uint16
    )


class WristRobustTests(unittest.TestCase):
    def test_missing_mock_tcp_is_held_without_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy.OUTPUT_ROOT = Path(directory)
            policy.reset()
            obs = {
                "depth": {
                    "table": synthetic_depth("table_depth"),
                    "wrist": synthetic_depth("wrist_depth"),
                },
                "tcp_pose": None,
                "joints": HOME,
            }
            for _ in range(3):
                action = policy.act(obs)
                np.testing.assert_allclose(action, HOME, atol=1e-5)
            self.assertEqual(policy._sample_number, 0)
            self.assertEqual(policy._samples, [])

    def test_points_on_robot_skeleton_are_classified(self):
        segments = robot_skeleton_segments(HOME, TCP)
        points = np.asarray([(a + b) / 2.0 for a, b in segments])
        result = classify_robot_self_points(points, HOME, TCP)
        self.assertTrue(np.all(result["self_mask"]))

    def test_stationary_combined_validation_passes_synthetic_box(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy.OUTPUT_ROOT = Path(directory)
            policy.reset()
            obs = {
                "depth": {
                    "table": synthetic_depth("table_depth"),
                    "wrist": synthetic_depth("wrist_depth"),
                },
                "tcp_pose": TCP,
                "joints": HOME,
            }
            for _ in range(10):
                action = policy.act(obs)
                np.testing.assert_allclose(action, HOME, atol=1e-5)
            summary = json.loads(
                (policy._rollout_dir / "wrist_robust_summary.json").read_text()
            )
            self.assertEqual(summary["samples"], 10)
            self.assertEqual(summary["wrist_box_detection_fraction"], 1.0)
            self.assertAlmostEqual(
                summary["wrist_box_height_median_mm"], 55.0, delta=1.0
            )
            self.assertTrue(
                summary["wrist_depth_and_self_mask_validation_passed"]
            )
            self.assertFalse(summary["motion_enable_recommended"])

    @staticmethod
    def captured_metric(height, p95=18.2, passed=True):
        return {
            "sample_passed": passed,
            "table_robot_self_explained_fraction": 0.968,
            "table_high_points_unexplained": 100,
            "wrist_valid_depth_fraction": 0.92,
            "wrist_calibration_valid": True,
            "wrist_box_height_mm": float(height),
            "wrist_table_p95_absolute_error_mm": float(p95),
        }

    def test_v05_capture_advances_only_with_25mm_guard_margin(self):
        # The 18 measured v05 box heights: noisy per-frame, stable under the
        # five-frame consensus used by v06.
        heights = [
            54.004, 53.703, 58.777, 54.627, 55.344, 53.006,
            55.550, 56.103, 57.978, 57.814, 54.648, 53.074,
            54.103, 54.292, 53.326, 54.342, 54.665, 52.463,
        ]
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy._rollout_dir = Path(directory)
            policy._samples = [self.captured_metric(value) for value in heights]
            policy._write_summary()
            summary = json.loads(
                (Path(directory) / "wrist_robust_summary.json").read_text()
            )
            self.assertTrue(
                summary["wrist_depth_and_self_mask_validation_passed"]
            )
            self.assertEqual(
                summary["required_wrist_depth_guard_margin_mm"], 25.0
            )
            self.assertFalse(summary["motion_enable_recommended"])

    def test_temporally_unstable_height_fails_closed(self):
        heights = [48.0, 63.0] * 6
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy._rollout_dir = Path(directory)
            policy._samples = [self.captured_metric(value) for value in heights]
            policy._write_summary()
            summary = json.loads(
                (Path(directory) / "wrist_robust_summary.json").read_text()
            )
            self.assertFalse(
                summary["wrist_depth_and_self_mask_validation_passed"]
            )


if __name__ == "__main__":
    unittest.main()
