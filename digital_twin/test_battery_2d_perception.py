"""Host-only tests for the shared sim/real calibrated battery pose estimator."""

import math
import unittest

import cv2
import numpy as np

from battery_2d_perception import estimate_pose_from_mask


class _LinearCamera:
    W = 1000
    H = 700

    def to_plane(self, u, v, z=None):
        # 1000 px/m, with image v pointing opposite robot-base +Y.
        return np.array([(u - 500.0) / 1000.0, (350.0 - v) / 1000.0, z])


def _axis_error_degrees(a, b):
    delta = (a - b + 90.0) % 180.0 - 90.0
    return abs(delta)


class Battery2DPerceptionTests(unittest.TestCase):
    def test_recovers_centre_and_long_axis_from_mask(self):
        mask = np.zeros((_LinearCamera.H, _LinearCamera.W), np.uint8)
        expected_xy = np.array([0.12, -0.08])
        yaw_world = 31.0
        # Image y points downward, so the image-space rectangle angle is -yaw.
        center_px = (500 + expected_xy[0] * 1000, 350 - expected_xy[1] * 1000)
        rect = (center_px, (43.7, 10.5), -yaw_world)
        cv2.fillConvexPoly(mask, cv2.boxPoints(rect).round().astype(np.int32), 255)

        pose = estimate_pose_from_mask(
            mask,
            _LinearCamera(),
            plane_z=0.00525,
            expected_length_m=0.0437,
            expected_width_m=0.0105,
        )
        self.assertLess(np.linalg.norm(np.asarray(pose.center_xy_m) - expected_xy), 0.002)
        self.assertLess(_axis_error_degrees(pose.yaw_deg, yaw_world), 2.0)
        self.assertGreater(pose.confidence, 0.75)

    def test_rejects_non_aaa_sized_component(self):
        mask = np.zeros((_LinearCamera.H, _LinearCamera.W), np.uint8)
        cv2.rectangle(mask, (450, 300), (550, 400), 255, -1)
        with self.assertRaises(RuntimeError):
            estimate_pose_from_mask(mask, _LinearCamera(), plane_z=0.00525)


if __name__ == "__main__":
    unittest.main()
