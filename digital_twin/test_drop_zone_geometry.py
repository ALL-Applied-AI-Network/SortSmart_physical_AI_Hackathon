"""Host-only tests for calibrated bowl-rim geometry."""

import math
import unittest

import numpy as np

from backproject import CameraModel
from drop_zone_geometry import estimate_drop_zone_from_polygon


def _camera_model():
    eye = np.array([0.78, -0.04, 1.70], dtype=float)
    target = np.array([0.28, 0.18, 0.90], dtype=float)
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


class DropZoneGeometryTests(unittest.TestCase):
    def test_recovers_oblique_circular_rim_center(self):
        camera = _camera_model()
        expected_center = np.array([0.28, 0.18])
        radius = 0.096
        rim_z = camera.table_z + 0.036
        polygon = []
        for angle in np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False):
            world = [
                expected_center[0] + radius * math.cos(angle),
                expected_center[1] + radius * math.sin(angle),
                rim_z,
            ]
            polygon.append(camera.project(world))

        pose = estimate_drop_zone_from_polygon(
            polygon,
            camera,
            rim_height_above_table_m=0.036,
            expected_outer_radius_m=radius,
        )
        self.assertEqual(pose.method, "rim_plane_robust_circle_fit")
        self.assertLess(np.linalg.norm(np.asarray(pose.center_xy_m) - expected_center), 0.001)
        self.assertAlmostEqual(pose.fitted_radius_m, radius, places=3)

    def test_rejects_center_outside_workspace(self):
        camera = _camera_model()
        polygon = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], dtype=float)
        with self.assertRaises(RuntimeError):
            estimate_drop_zone_from_polygon(
                polygon,
                camera,
                rim_height_above_table_m=0.036,
                expected_outer_radius_m=0.096,
            )


if __name__ == "__main__":
    unittest.main()
