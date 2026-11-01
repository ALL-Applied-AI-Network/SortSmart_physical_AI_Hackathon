import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose, load
from policy_wrist_selfmask_v05 import Policy
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


class WristSelfMaskTests(unittest.TestCase):
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
                (policy._rollout_dir / "wrist_selfmask_summary.json").read_text()
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


if __name__ == "__main__":
    unittest.main()
