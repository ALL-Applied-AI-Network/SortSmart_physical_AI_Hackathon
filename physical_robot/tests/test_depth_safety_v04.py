import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration.geometry import SURFACE_Z, load
from depth_safety import TableDepthSafetyGuard
from policy_heightmap_safety_v04 import Policy


HOME = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0]


def synthetic_depth(box_height_mm: float | None = None) -> np.ndarray:
    cam = load()["table_depth"]
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
    base_rays = camera_rays @ cam["T"][:3, :3].T
    origin = cam["T"][:3, 3]
    table_scale = (SURFACE_Z - origin[2]) / base_rays[..., 2]
    scale = table_scale
    if box_height_mm is not None:
        top_z = SURFACE_Z + box_height_mm / 1000.0
        top_scale = (top_z - origin[2]) / base_rays[..., 2]
        top_points = origin + top_scale[..., None] * base_rays
        on_box = (
            (top_points[..., 0] >= 0.33)
            & (top_points[..., 0] <= 0.43)
            & (top_points[..., 1] >= -0.10)
            & (top_points[..., 1] <= 0.08)
        )
        scale = np.where(on_box, top_scale, table_scale)
    return np.where((scale > 0) & (scale < 3), scale * 1000, 0).astype(
        np.uint16
    )


class DepthSafetyTests(unittest.TestCase):
    def test_55mm_box_passes_scan_clearance(self):
        snapshot = TableDepthSafetyGuard().snapshot(synthetic_depth(55.0))
        check = snapshot.evaluate_vertical_target(
            x_mm=395.0,
            y_mm=0.0,
            target_z_uf0_mm=-235.0,
            footprint_radius_mm=60.0,
            minimum_clearance_mm=25.0,
        )
        self.assertTrue(check["passed"])
        self.assertGreater(check["predicted_tcp_clearance_mm"], 30.0)

    def test_85mm_box_blocks_scan_descent(self):
        snapshot = TableDepthSafetyGuard().snapshot(synthetic_depth(85.0))
        check = snapshot.evaluate_vertical_target(
            x_mm=395.0,
            y_mm=0.0,
            target_z_uf0_mm=-235.0,
            footprint_radius_mm=60.0,
            minimum_clearance_mm=25.0,
        )
        self.assertFalse(check["passed"])
        self.assertLess(check["predicted_tcp_clearance_mm"], 25.0)

    def test_stationary_policy_audits_route_and_holds_joints(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy.OUTPUT_ROOT = Path(directory)
            policy.reset()
            obs = {
                "depth": {"table": synthetic_depth(55.0)},
                "tcp_pose": [300.0, 0.0, -90.0, -180.0, 0.0, 0.0],
                "joints": np.asarray(HOME, dtype=np.float32),
            }
            for _ in range(10):
                action = policy.act(obs)
                np.testing.assert_allclose(action, HOME, atol=1e-5)
            summary = json.loads(
                (
                    policy._rollout_dir / "heightmap_safety_summary.json"
                ).read_text()
            )
            self.assertEqual(summary["samples"], 10)
            self.assertTrue(summary["robot_motion_commanded"] is False)
            self.assertEqual(summary["route_pass_fraction"], 1.0)
            self.assertTrue(summary["heightmap_safety_audit_passed"])
            self.assertFalse(summary["motion_enable_recommended"])
            self.assertEqual(
                summary["next_stage"],
                "wrist_depth_and_robot_self_mask_validation",
            )


if __name__ == "__main__":
    unittest.main()
