"""Host-only tests for calibrated workspace cropping."""

import unittest

import numpy as np

from backproject import CameraModel
from workspace_roi import projected_workspace_roi


def _camera_model():
    eye = np.array([0.78, -0.04, 1.70], dtype=float)
    target = np.array([0.36, 0.04, 0.90], dtype=float)
    camera_z = eye - target
    camera_z /= np.linalg.norm(camera_z)
    world_up = np.array([0.0, 0.0, 1.0])
    camera_x = np.cross(world_up, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.vstack((camera_x, camera_y, camera_z))
    matrix[3, :3] = eye
    return CameraModel(
        {
            "resolution": [1920, 1080],
            "focal_length_mm": 11.2158,
            "horizontal_aperture_mm": 20.955,
            "vertical_aperture_mm": 20.955 * 1080 / 1920,
            "camera_to_world": matrix.tolist(),
            "table_top_z": 0.90,
        }
    )


class WorkspaceRoiTests(unittest.TestCase):
    def test_roi_contains_projected_workspace_and_is_smaller_than_frame(self):
        camera = _camera_model()
        roi = projected_workspace_roi(
            camera,
            x_bounds=(0.18, 0.55),
            y_bounds=(-0.18, 0.30),
            height_bounds=(0.0, 0.07),
            margin_px=32,
        )
        x1, y1, x2, y2 = roi
        self.assertGreater(x1, 0)
        self.assertGreater(y1, 0)
        self.assertLess(x2, camera.W)
        self.assertLess(y2, camera.H)
        self.assertLess((x2 - x1) * (y2 - y1), camera.W * camera.H)
        for x in (0.18, 0.55):
            for y in (-0.18, 0.30):
                for height in (0.0, 0.07):
                    u, v = camera.project([x, y, camera.table_z + height])
                    self.assertTrue(x1 <= u <= x2)
                    self.assertTrue(y1 <= v <= y2)


if __name__ == "__main__":
    unittest.main()
