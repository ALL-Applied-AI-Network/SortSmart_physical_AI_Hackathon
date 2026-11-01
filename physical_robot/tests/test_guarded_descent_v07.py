import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration.geometry import SURFACE_Z, flange_pose, load
try:
    from policy_guarded_descent_v07 import Policy
except ModuleNotFoundError:  # installed archive exposes the policy as policy.py
    from policy import Policy


HOME = np.array(
    [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0],
    dtype=np.float32,
)
TCP = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]


def synthetic_depth(camera_name: str, object_height_mm: float) -> np.ndarray:
    cam = load()[camera_name]
    transform = cam["T"]
    if cam["frame"] == "flange":
        transform = flange_pose(HOME) @ transform
    width, height = cam["size"]
    v, u = np.mgrid[0:height, 0:width]
    rays = np.stack(
        [
            (u - cam["cx"]) / cam["fx"],
            (v - cam["cy"]) / cam["fy"],
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
        (top[..., 0] >= 0.26)
        & (top[..., 0] <= 0.42)
        & (top[..., 1] >= -0.12)
        & (top[..., 1] <= 0.07)
    )
    scale = np.where(on_object, top_scale, table_scale)
    return np.where((scale > 0) & (scale < 3), scale * 1000, 0).astype(
        np.uint16
    )


def observation(height_mm: float) -> dict:
    return {
        "depth": {
            "table": synthetic_depth("table_depth", height_mm),
            "wrist": synthetic_depth("wrist_depth", height_mm),
        },
        "tcp_pose": TCP.copy(),
        "joints": HOME.copy(),
    }


class GuardedDescentTests(unittest.TestCase):
    def _run(self, height_mm: float):
        directory = tempfile.TemporaryDirectory()
        policy = Policy()
        policy.OUTPUT_ROOT = Path(directory.name)
        policy.reset()
        obs = observation(height_mm)
        for _ in range(80):
            action = policy.act(obs)
            obs["tcp_pose"] = action[:6]
            if policy._state in {"complete", "aborted"}:
                break
        report = json.loads(
            (policy._rollout_dir / "guarded_descent_v07_report.json").read_text()
        )
        return directory, policy, report

    def test_55mm_reference_allows_guarded_descent_and_retreat(self):
        directory, policy, report = self._run(55.0)
        self.addCleanup(directory.cleanup)
        self.assertEqual(policy._state, "complete")
        self.assertTrue(report["task_complete"])
        self.assertEqual(report["lowest_commanded_z_uf0_mm"], -235.0)
        self.assertEqual(report["low_observation_frames"], 10)
        self.assertFalse(report["horizontal_motion_commanded"])
        self.assertFalse(report["gripper_transition_commanded"])

    def test_90mm_obstacle_abstains_without_descent(self):
        directory, policy, report = self._run(90.0)
        self.addCleanup(directory.cleanup)
        self.assertEqual(policy._state, "aborted")
        self.assertTrue(report["safely_abstained"])
        self.assertFalse(report["motion_started"])
        self.assertEqual(report["low_observation_frames"], 0)

    def test_transient_wrist_shift_while_moving_does_not_false_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            policy.OUTPUT_ROOT = Path(directory)
            policy.reset()
            original_wrist_quality = policy._wrist_quality

            def moving_wrist_quality(depth, obs):
                result = original_wrist_quality(depth, obs)
                z = float(obs["tcp_pose"][2])
                if -145.0 < z < -100.0:
                    result["valid"] = False
                    result["median_error_mm"] = -3.2
                return result

            policy._wrist_quality = moving_wrist_quality
            obs = observation(55.0)
            for _ in range(120):
                action = policy.act(obs)
                current = np.asarray(obs["tcp_pose"], dtype=float)
                target = np.asarray(action[:6], dtype=float)
                delta = target - current
                if abs(delta[2]) > 15.0:
                    target[2] = current[2] + np.sign(delta[2]) * 15.0
                obs["tcp_pose"] = target.tolist()
                if policy._state in {"complete", "aborted"}:
                    break
            self.assertEqual(policy._state, "complete")
            report = json.loads(
                (policy._rollout_dir / "guarded_descent_v07_report.json").read_text()
            )
            self.assertTrue(report["task_complete"])
            self.assertGreater(report["descent_safety_failed_frames"], 0)


if __name__ == "__main__":
    unittest.main()
