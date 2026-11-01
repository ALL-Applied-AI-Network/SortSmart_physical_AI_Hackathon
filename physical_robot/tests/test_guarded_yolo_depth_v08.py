import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose, load, project
from depth_safety import HeightMapSnapshot, SafetyConfig
try:
    from policy_guarded_descent_v07 import Policy
except ModuleNotFoundError:  # installed archive exposes the policy as policy.py
    from policy import Policy


HOME = np.array(
    [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0],
    dtype=np.float32,
)
TCP = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]
BATTERY_X = (0.34, 0.391)
BATTERY_Y = (-0.006, 0.006)


def synthetic_depth(camera_name: str, object_height_mm: float = 11.0) -> np.ndarray:
    camera = load()[camera_name]
    transform = camera["T"]
    if camera["frame"] == "flange":
        transform = flange_pose(HOME) @ transform
    width, height = camera["size"]
    v, u = np.mgrid[0:height, 0:width]
    rays = np.stack(
        [
            (u - camera["cx"]) / camera["fx"],
            (v - camera["cy"]) / camera["fy"],
            np.ones((height, width)),
        ],
        -1,
    )
    base_rays = rays @ transform[:3, :3].T
    origin = transform[:3, 3]
    table_scale = (SURFACE_Z - origin[2]) / base_rays[..., 2]
    top_scale = (
        SURFACE_Z + object_height_mm / 1000.0 - origin[2]
    ) / base_rays[..., 2]
    top = origin + top_scale[..., None] * base_rays
    on_object = (
        (top[..., 0] >= BATTERY_X[0])
        & (top[..., 0] <= BATTERY_X[1])
        & (top[..., 1] >= BATTERY_Y[0])
        & (top[..., 1] <= BATTERY_Y[1])
    )
    scale = np.where(on_object, top_scale, table_scale)
    return np.where((scale > 0) & (scale < 3), scale * 1000, 0).astype(np.uint16)


def projected_polygon(camera_name: str) -> np.ndarray:
    cameras = load()
    camera = cameras[camera_name]
    z = SURFACE_Z + 0.011
    points = np.array(
        [
            [BATTERY_X[0], BATTERY_Y[0], z],
            [BATTERY_X[1], BATTERY_Y[0], z],
            [BATTERY_X[1], BATTERY_Y[1], z],
            [BATTERY_X[0], BATTERY_Y[1], z],
        ]
    )
    if camera_name == "wrist":
        inverse = np.linalg.inv(flange_pose(HOME))
        points = points @ inverse[:3, :3].T + inverse[:3, 3]
    uv = project(camera, points)
    uv[:, 0] *= 640 / camera["size"][0]
    uv[:, 1] *= 360 / camera["size"][1]
    return uv.astype(np.float32)


def projected_drop_zone_polygon() -> np.ndarray:
    camera = load()["table"]
    points = np.array(
        [
            [0.25, 0.15, SURFACE_Z],
            [0.31, 0.15, SURFACE_Z],
            [0.31, 0.21, SURFACE_Z],
            [0.25, 0.21, SURFACE_Z],
        ]
    )
    uv = project(camera, points)
    uv[:, 0] *= 640 / camera["size"][0]
    uv[:, 1] *= 360 / camera["size"][1]
    return uv.astype(np.float32)


def polygon_area(polygon: np.ndarray) -> float:
    shifted = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1]
    return abs(float(np.sum(cross))) * 0.5


def observation() -> dict:
    blank = np.full((360, 640, 3), 160, dtype=np.uint8)
    return {
        "images": {"table": blank.copy(), "wrist": blank.copy()},
        "depth": {
            "table": synthetic_depth("table_depth"),
            "wrist": synthetic_depth("wrist_depth"),
        },
        "tcp_pose": TCP.copy(),
        "joints": HOME.copy(),
    }


class GuardedYoloDepthFusionTests(unittest.TestCase):
    def test_shallow_task_objects_do_not_poison_table_calibration_gate(self):
        rng = np.random.default_rng(7)
        table_z = rng.normal(0.0, 1.2, 9000)
        shallow_objects = rng.normal(12.0, 1.0, 1000)
        heights_mm = np.concatenate([table_z, shallow_objects])
        points = np.zeros((len(heights_mm), 3), dtype=float)
        points[:, 0] = 0.30
        points[:, 1] = 0.00
        points[:, 2] = SURFACE_Z + heights_mm / 1000.0
        snapshot = HeightMapSnapshot(points, 0.92, SafetyConfig())
        self.assertTrue(snapshot.calibration_valid)
        self.assertLess(snapshot.table_p95_absolute_error_mm, 10.0)
        self.assertLess(snapshot.table_inlier_fraction, 0.95)

    def test_stable_negative_table_plane_offset_is_corrected(self):
        rng = np.random.default_rng(11)
        table_z = rng.normal(-8.5, 1.2, 9000)
        shallow_objects = rng.normal(3.5, 1.0, 1000)
        heights_mm = np.concatenate([table_z, shallow_objects])
        points = np.zeros((len(heights_mm), 3), dtype=float)
        points[:, 0] = 0.30
        points[:, 1] = 0.00
        points[:, 2] = SURFACE_Z + heights_mm / 1000.0
        snapshot = HeightMapSnapshot(points, 0.90, SafetyConfig())
        self.assertTrue(snapshot.calibration_valid)
        self.assertAlmostEqual(snapshot.table_plane_offset_mm, -8.5, delta=0.2)
        self.assertLess(snapshot.table_p95_absolute_error_mm, 10.0)
        self.assertAlmostEqual(
            float(np.percentile(snapshot.height_mm, 99.0)), 12.0, delta=1.5
        )

    def test_verified_high_robot_points_can_be_excluded_but_85mm_obstacle_cannot(self):
        table = np.zeros((1000, 3), dtype=float)
        table[:, 0] = 0.30
        table[:, 2] = SURFACE_Z
        robot = np.repeat([[0.30, 0.0, SURFACE_Z + 0.119]], 9, axis=0)
        snapshot = HeightMapSnapshot(
            np.concatenate([table, robot]), 0.90, SafetyConfig()
        )
        without_filter = snapshot.evaluate_vertical_target(
            300.0, 0.0, -235.0, 75.0, 30.0
        )
        self.assertFalse(without_filter["passed"])
        excluded = np.zeros(len(snapshot.points), dtype=bool)
        excluded[-9:] = True
        with_filter = snapshot.evaluate_vertical_target(
            300.0, 0.0, -235.0, 75.0, 30.0, excluded
        )
        self.assertTrue(with_filter["passed"])
        self.assertEqual(with_filter["robot_self_points_excluded_locally"], 9)

        obstacle = np.repeat([[0.30, 0.0, SURFACE_Z + 0.085]], 20, axis=0)
        obstacle_snapshot = HeightMapSnapshot(
            np.concatenate([table, obstacle]), 0.90, SafetyConfig()
        )
        blocked = obstacle_snapshot.evaluate_vertical_target(
            300.0, 0.0, -235.0, 75.0, 30.0
        )
        self.assertFalse(blocked["passed"])

    def test_safe_descent_fuses_stable_battery_and_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy.OUTPUT_ROOT = Path(directory)

            def fake_infer(camera, _rgb, observation_index):
                polygon = projected_polygon(camera)
                result = [
                    {
                        "camera": camera,
                        "class_id": 0,
                        "class_name": "aaa_battery",
                        "confidence": 0.9,
                        "accepted": True,
                        "center_px": polygon.mean(axis=0).tolist(),
                        "mask_area_px": polygon_area(polygon),
                        "polygon_px": polygon.tolist(),
                        "_polygon": polygon,
                        "observation_index": observation_index,
                    }
                ]
                if camera == "table":
                    drop = projected_drop_zone_polygon()
                    result.append(
                        {
                            "camera": camera,
                            "class_id": 1,
                            "class_name": "battery_drop_zone",
                            "confidence": 0.85,
                            "accepted": True,
                            "center_px": drop.mean(axis=0).tolist(),
                            "mask_area_px": polygon_area(drop),
                            "polygon_px": drop.tolist(),
                            "_polygon": drop,
                            "observation_index": observation_index,
                        }
                    )
                return result

            policy._infer_battery_masks = fake_infer

            def fake_verification(_obs, phase):
                return {
                    "phase": phase,
                    "accepted_table_height_batteries": 1,
                    "minimum_distance_to_pick_mm": 100.0 if phase == "pickup" else 180.0,
                    "minimum_distance_to_drop_mm": None if phase == "pickup" else 5.0,
                    "observed_gripper": 1.0 if phase == "pickup" else 0.0,
                    "passed": True,
                }

            policy._table_battery_verification = fake_verification
            policy.reset()
            obs = observation()
            for _ in range(180):
                action = policy.act(obs)
                obs["tcp_pose"] = action[:6]
                obs["joints"][6] = action[6]
                if policy._state in {"complete", "aborted"}:
                    break

            report_path = (
                policy._rollout_dir / "physical_pick_place_aa_v11_report.json"
            )
            report = json.loads(report_path.read_text())
            self.assertEqual(policy._state, "complete")
            self.assertTrue(report["safe_motion_complete"])
            self.assertTrue(report["perception_validation_passed"])
            self.assertEqual(report["frames_with_valid_battery_fusion"], 10)
            self.assertTrue(report["drop_zone_validation_passed"])
            self.assertTrue(report["non_contact_route_validation_passed"])
            self.assertTrue(report["shadow_plan_ready"])
            self.assertTrue(report["execution_enabled"])
            self.assertTrue(report["pickup_verified"])
            self.assertTrue(report["placement_verified"])
            self.assertTrue(report["horizontal_motion_commanded"])
            self.assertTrue(report["gripper_transition_commanded"])
            self.assertEqual(
                len(report["execution_plan"]["steps"]), 17
            )


if __name__ == "__main__":
    unittest.main()
