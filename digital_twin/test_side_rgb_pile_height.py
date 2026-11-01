import unittest

import numpy as np

from backproject import CameraModel
from side_rgb_pile_height import estimate_pile_top_from_masks


class SideRgbPileHeightTests(unittest.TestCase):
    def _camera(self):
        # Synthetic calibrated camera at y=1, z=1 looking horizontally toward -Y.
        return CameraModel(
            {
                "resolution": [640, 480],
                "focal_length_mm": 10.0,
                "horizontal_aperture_mm": 10.0,
                "vertical_aperture_mm": 7.5,
                "camera_to_world": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0],
                ],
                "table_top_z": 0.9,
            }
        )

    def test_recovers_top_height_from_calibrated_rgb_pixels(self):
        camera = self._camera()
        pixel = camera.project((0.0, 0.0, 0.920))
        self.assertIsNotNone(pixel)
        u, v = np.rint(pixel).astype(int)
        mask = np.zeros((camera.H, camera.W), dtype=bool)
        mask[max(0, v - 1) : v + 2, max(0, u - 1) : u + 2] = True
        result = estimate_pile_top_from_masks(
            camera,
            [mask],
            (0.0, 0.0),
            floor_top_world_z=0.900,
            bowl_radius_m=0.052,
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["pile_top_world_z"], 0.920, delta=0.003)

    def test_empty_masks_fail_closed(self):
        camera = self._camera()
        result = estimate_pile_top_from_masks(
            camera,
            [np.zeros((camera.H, camera.W), dtype=bool)],
            (0.0, 0.0),
            floor_top_world_z=0.900,
            bowl_radius_m=0.052,
        )
        self.assertFalse(result["valid"])
        self.assertIsNone(result["pile_top_world_z"])


if __name__ == "__main__":
    unittest.main()
