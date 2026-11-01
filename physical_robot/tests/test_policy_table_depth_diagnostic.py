import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLORMAP_TURBO = 0
    cv2_stub.COLOR_RGB2BGR = 1
    cv2_stub.FONT_HERSHEY_SIMPLEX = 0
    cv2_stub.LINE_AA = 0
    cv2_stub.applyColorMap = lambda image, _: np.repeat(image[..., None], 3, 2)
    cv2_stub.cvtColor = lambda image, _: image[..., ::-1].copy()
    cv2_stub.putText = lambda image, *args, **kwargs: image
    cv2_stub.imwrite = lambda *args, **kwargs: True
    sys.modules["cv2"] = cv2_stub

from calibration.geometry import SURFACE_Z, load
from policy import Policy


HOME = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0]


def mock_table_depth() -> np.ndarray:
    cam = load()["table_depth"]
    width, height = cam["size"]
    v, u = np.mgrid[0:height, 0:width]
    rays = np.stack(
        [
            (u - cam["cx"]) / cam["fx"],
            (v - cam["cy"]) / cam["fy"],
            np.ones((height, width)),
        ],
        -1,
    ) @ cam["T"][:3, :3].T
    scale = (-0.0051 - cam["T"][2, 3]) / rays[..., 2]
    return np.where((scale > 0) & (scale < 3), scale * 1000, 0).astype(
        np.uint16
    )


def mock_box_depth(height_m: float = 0.055) -> np.ndarray:
    """Ideal table plus a horizontal central box top (no rendered sides)."""

    cam = load()["table_depth"]
    width, image_height = cam["size"]
    v, u = np.mgrid[0:image_height, 0:width]
    camera_rays = np.stack(
        [
            (u - cam["cx"]) / cam["fx"],
            (v - cam["cy"]) / cam["fy"],
            np.ones((image_height, width)),
        ],
        -1,
    )
    base_rays = camera_rays @ cam["T"][:3, :3].T
    camera_origin = cam["T"][:3, 3]
    table_scale = (SURFACE_Z - camera_origin[2]) / base_rays[..., 2]
    top_scale = (
        SURFACE_Z + height_m - camera_origin[2]
    ) / base_rays[..., 2]
    top_points = camera_origin + top_scale[..., None] * base_rays
    on_top = (
        (top_points[..., 0] >= 0.26)
        & (top_points[..., 0] <= 0.42)
        & (top_points[..., 1] >= -0.12)
        & (top_points[..., 1] <= 0.07)
    )
    scale = np.where(on_top, top_scale, table_scale)
    return np.where((scale > 0) & (scale < 3), scale * 1000, 0).astype(
        np.uint16
    )


class TableDepthDiagnosticPolicyTests(unittest.TestCase):
    def test_mock_depth_passes_and_policy_holds_observed_joints(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy.OUTPUT_ROOT = Path(directory)
            policy.reset()
            tcp = [450.0, 0.0, -165.0, -180.0, 0.0, 0.0]
            obs = {
                "depth": {"table": mock_table_depth()},
                "images": {
                    "table": np.zeros((360, 640, 3), dtype=np.uint8)
                },
                "tcp_pose": tcp,
                "joints": np.asarray(HOME, dtype=np.float32),
            }
            for _ in range(10):
                action = policy.act(obs)
                np.testing.assert_allclose(action, HOME, atol=1e-5)

            summary = json.loads(
                (policy._rollout_dir / "table_depth_summary.json").read_text()
            )
            self.assertEqual(summary["samples"], 10)
            self.assertTrue(summary["robot_motion_commanded"] is False)
            self.assertTrue(summary["ready_for_object_height_validation"])
            self.assertFalse(summary["object_height_validation_passed"])

    def test_known_height_box_passes_repeatability_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy.OUTPUT_ROOT = Path(directory)
            policy.reset()
            obs = {
                "depth": {"table": mock_box_depth()},
                "images": {
                    "table": np.zeros((360, 640, 3), dtype=np.uint8)
                },
                "tcp_pose": [450.0, 0.0, -165.0, -180.0, 0.0, 0.0],
                "joints": np.asarray(HOME, dtype=np.float32),
            }
            for _ in range(10):
                action = policy.act(obs)
                np.testing.assert_allclose(action, HOME, atol=1e-5)

            summary = json.loads(
                (policy._rollout_dir / "box_height_summary.json").read_text()
            )
            self.assertEqual(summary["box_height_frames"], 10)
            self.assertAlmostEqual(
                summary["box_detection_fraction"], 1.0, places=6
            )
            self.assertAlmostEqual(
                summary["box_height_median_mm"], 55.0, delta=0.5
            )
            self.assertLessEqual(summary["box_height_std_mm"], 0.1)
            self.assertTrue(summary["object_height_validation_passed"])


if __name__ == "__main__":
    unittest.main()
