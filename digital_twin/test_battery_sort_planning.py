"""Host-only planning and endpoint IK tests for AAA-to-bowl sorting."""

import math
import unittest

import numpy as np

import fk
import ik
import scene_config as C
from battery_sort_planning import make_battery_sort_plan


class BatterySortPlanningTests(unittest.TestCase):
    def _plan(self, xy=(0.34, -0.12), yaw_deg=7.0):
        yaw = math.radians(yaw_deg)
        return make_battery_sort_plan(
            xy,
            (math.cos(yaw), math.sin(yaw)),
            (0.282, 0.180),
            surface_dz_m=C.SURFACE_DZ,
            tip_clearance_m=C.BATTERY_TIP_CLEARANCE,
            bowl_height_m=0.036,
            bowl_floor_height_m=0.006,
            bowl_place_clearance_m=0.001,
        )

    def test_spread_is_perpendicular_and_release_is_near_floor(self):
        plan = self._plan()
        axis = np.asarray(plan.battery_axis_xy)
        spread = np.asarray(plan.raw_spread_xy)
        self.assertAlmostEqual(float(np.dot(axis, spread)), 0.0, places=7)
        self.assertAlmostEqual(float(np.linalg.norm(spread)), 1.0, places=7)
        expected_z = C.SURFACE_DZ + 0.006 + C.BATTERY_TIP_CLEARANCE + 0.001
        self.assertAlmostEqual(plan.bowl_release_m[2], expected_z, places=7)
        self.assertGreater(plan.bowl_release_m[2], C.SURFACE_DZ + 0.006)
        self.assertLess(plan.bowl_release_m[2], C.SURFACE_DZ + 0.036)
        self.assertEqual(plan.bowl_release_m[:2], plan.bowl_center_xy_m)

    def test_all_sweep_endpoint_poses_are_reachable(self):
        cases = [
            (0.34, -0.12, 7.0),
            (0.39, -0.10, 19.0),
            (0.44, -0.08, 38.0),
            (0.32, -0.05, 57.0),
            (0.37, -0.03, 76.0),
            (0.43, 0.00, 94.0),
            (0.33, 0.03, 113.0),
            (0.39, 0.05, 132.0),
            (0.45, 0.07, 151.0),
            (0.35, -0.09, 169.0),
            (0.42, -0.04, 27.5),
            (0.37, 0.02, 142.3),
        ]
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        tool = np.asarray(C.TOOL_AXIS, dtype=float) * C.TOOL_LENGTH
        down = np.array([0.0, 0.0, -1.0])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)

        for x, y, yaw in cases:
            with self.subTest(x=x, y=y, yaw=yaw):
                plan = self._plan((x, y), yaw)
                q = np.deg2rad([C.CAPTURE_POSE_DEG[f"J{i}"] for i in range(1, 7)])
                rotation, _ = fk.tool_pose(q, tool)
                current_spread = rotation @ np.asarray(C.FINGER_SPREAD_AXIS, dtype=float)
                current_spread[2] = 0.0
                current_spread /= np.linalg.norm(current_spread)
                spread = np.array([*plan.raw_spread_xy, 0.0])
                if np.dot(spread, current_spread) < 0.0:
                    spread = -spread
                _, current_tip = fk.tool_pose(q, tool)
                segments = (
                    (current_tip, plan.pick_above_m, current_spread, spread),
                    (plan.pick_above_m, plan.pick_tip_m, spread, spread),
                    (plan.pick_tip_m, plan.pick_lift_m, spread, spread),
                    (plan.pick_lift_m, plan.bowl_above_m, spread, spread),
                    (plan.bowl_above_m, plan.bowl_release_m, spread, spread),
                    (plan.bowl_release_m, plan.bowl_retreat_m, spread, spread),
                )
                for start, finish, spread_start, spread_finish in segments:
                    start = np.asarray(start, dtype=float)
                    finish = np.asarray(finish, dtype=float)
                    spread_start = np.asarray(spread_start, dtype=float)
                    spread_finish = np.asarray(spread_finish, dtype=float)
                    angle = math.atan2(
                        spread_start[0] * spread_finish[1]
                        - spread_start[1] * spread_finish[0],
                        float(np.dot(spread_start[:2], spread_finish[:2])),
                    )
                    for index in range(30):
                        linear = (index + 1) / 30.0
                        progress = linear * linear * (3.0 - 2.0 * linear)
                        target = start + (finish - start) * progress
                        c = math.cos(angle * progress)
                        s = math.sin(angle * progress)
                        direction = np.array(
                            [
                                spread_start[0] * c - spread_start[1] * s,
                                spread_start[0] * s + spread_start[1] * c,
                                0.0,
                            ]
                        )
                        q, solved = ik.solve(
                            q,
                            target,
                            down,
                            limits,
                            tool_offset=tool,
                            spread_dir=direction,
                            iters=500,
                            damping=0.02,
                        )
                        self.assertTrue(solved, msg=f"unreachable target {target}")


if __name__ == "__main__":
    unittest.main()
