import math
import unittest

import numpy as np

import battery_sort_planning
import complex_battery_scene
import fk
import ik
import multi_battery_planning as planning
import scene_config as C


class MultiBatteryPlanningTests(unittest.TestCase):
    def test_four_floor_slots_fit_bowl_floor_with_parallel_aaa(self):
        floor_radius = 0.052
        battery_half_length = 0.0445 / 2
        battery_half_width = 0.0105 / 2
        for index in range(4):
            x, y = planning.placement_slot_xy((0.28, 0.18), index)
            offset_y = abs(y - 0.18) + battery_half_width
            corner_radius = math.hypot(battery_half_length, offset_y)
            self.assertLess(corner_radius, floor_radius)
            self.assertAlmostEqual(x, 0.28)

    def test_slot_spacing_exceeds_battery_diameter(self):
        offsets = planning.PLACEMENT_SLOT_Y_OFFSETS_M[:4]
        gaps = [b - a for a, b in zip(offsets, offsets[1:])]
        self.assertTrue(all(gap > 0.0105 for gap in gaps))

    def test_fifth_slot_is_crosswise_second_layer(self):
        slot = planning.placement_slot_spec(4)
        self.assertEqual(slot.layer, 1)
        self.assertEqual(slot.battery_axis, "y")
        self.assertEqual(slot.offset_xy_m, (0.0, 0.0))

    def test_release_tip_height_tracks_measured_pile_top(self):
        z = planning.release_tip_robot_z(
            0.9114,
            robot_base_world_z=0.9000,
            tip_clearance_m=0.0026,
            release_clearance_m=0.0030,
        )
        self.assertAlmostEqual(z, 0.0170)

    def test_candidate_in_bowl_is_never_reselected(self):
        records = [
            {
                "confidence": 0.99,
                "geometry_valid": True,
                "pose": {"center_xy_m": [0.28, 0.18]},
            },
            {
                "confidence": 0.71,
                "geometry_valid": True,
                "pose": {"center_xy_m": [0.40, -0.10]},
            },
        ]
        chosen = planning.select_pick_candidate(
            records,
            (0.28, 0.18),
            exclusion_radius_m=0.115,
        )
        self.assertEqual(chosen["confidence"], 0.71)

    def test_invalid_geometry_is_not_selected(self):
        records = [
            {
                "confidence": 0.99,
                "geometry_valid": False,
                "pose": {"center_xy_m": [0.40, -0.10]},
            }
        ]
        self.assertIsNone(
            planning.select_pick_candidate(
                records,
                (0.28, 0.18),
                exclusion_radius_m=0.115,
            )
        )

    def test_all_five_pick_and_aligned_release_paths_are_ik_reachable(self):
        limits = np.deg2rad(
            [C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]
        )
        tool = np.asarray(C.TOOL_AXIS, dtype=float) * C.TOOL_LENGTH
        down = np.array([0.0, 0.0, -1.0])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        capture_q = np.deg2rad(
            [C.CAPTURE_POSE_DEG[f"J{i}"] for i in range(1, 7)]
        )

        for index, entry in enumerate(complex_battery_scene.BATTERIES):
            with self.subTest(index=index):
                x, y, yaw_deg = entry[:3]
                yaw = math.radians(yaw_deg)
                slot = planning.placement_slot_xy((0.282, 0.180), index)
                plan = battery_sort_planning.make_battery_sort_plan(
                    (x, y),
                    (math.cos(yaw), math.sin(yaw)),
                    slot,
                    surface_dz_m=C.SURFACE_DZ,
                    tip_clearance_m=C.BATTERY_TIP_CLEARANCE,
                    bowl_height_m=0.036,
                    bowl_floor_height_m=0.006,
                    bowl_place_clearance_m=0.001,
                )
                q = capture_q.copy()
                rotation, current_tip = fk.tool_pose(q, tool)
                start_spread = rotation @ np.asarray(
                    C.FINGER_SPREAD_AXIS, dtype=float
                )
                start_spread[2] = 0.0
                start_spread /= np.linalg.norm(start_spread)
                pick_spread = np.array([*plan.raw_spread_xy, 0.0])
                if np.dot(pick_spread, start_spread) < 0:
                    pick_spread = -pick_spread
                slot_spec = planning.placement_slot_spec(index)
                release_spread = (
                    np.array([0.0, 1.0, 0.0])
                    if slot_spec.battery_axis == "x"
                    else np.array([1.0, 0.0, 0.0])
                )
                if np.dot(release_spread, pick_spread) < 0:
                    release_spread = -release_spread
                bowl_release = np.asarray(plan.bowl_release_m, dtype=float)
                if slot_spec.layer > 0:
                    planned_pile_top = 0.9000 + C.SURFACE_DZ + 0.006 + 0.0105
                    bowl_release[2] = planning.release_tip_robot_z(
                        planned_pile_top,
                        robot_base_world_z=0.9000,
                        tip_clearance_m=C.BATTERY_TIP_CLEARANCE,
                        release_clearance_m=0.003,
                    )
                segments = (
                    (current_tip, plan.pick_above_m, start_spread, pick_spread),
                    (plan.pick_above_m, plan.pick_tip_m, pick_spread, pick_spread),
                    (plan.pick_tip_m, plan.pick_lift_m, pick_spread, pick_spread),
                    (
                        plan.pick_lift_m,
                        plan.bowl_above_m,
                        pick_spread,
                        release_spread,
                    ),
                    (
                        plan.bowl_above_m,
                        bowl_release,
                        release_spread,
                        release_spread,
                    ),
                    (
                        bowl_release,
                        plan.bowl_retreat_m,
                        release_spread,
                        release_spread,
                    ),
                )
                for start, finish, spread_start, spread_finish in segments:
                    start = np.asarray(start, dtype=float)
                    finish = np.asarray(finish, dtype=float)
                    angle = math.atan2(
                        spread_start[0] * spread_finish[1]
                        - spread_start[1] * spread_finish[0],
                        float(np.dot(spread_start[:2], spread_finish[:2])),
                    )
                    for step in range(30):
                        linear = (step + 1) / 30.0
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
                        self.assertTrue(solved, msg=f"unreachable {target}")


if __name__ == "__main__":
    unittest.main()
