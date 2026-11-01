"""Conservative geometric classification of high table-depth robot points."""

from __future__ import annotations

import numpy as np

from calibration.geometry import CHAIN, UF0_Z_MM, _rot


def robot_skeleton_segments(
    joints_deg, tcp_pose_uf0_mm
) -> list[tuple[np.ndarray, np.ndarray]]:
    joints = np.radians(np.asarray(joints_deg, dtype=float).reshape(-1)[:6])
    if joints.size < 6 or not np.all(np.isfinite(joints)):
        raise RuntimeError("six finite robot joints are required")
    tcp = np.asarray(tcp_pose_uf0_mm, dtype=float).reshape(-1)
    if tcp.size < 3 or not np.all(np.isfinite(tcp[:3])):
        raise RuntimeError("finite TCP xyz is required for robot self filtering")

    rotation = np.eye(3)
    position = np.zeros(3)
    origins = [position.copy()]
    for (offset, axis, sign), q in zip(CHAIN, joints):
        position = position + rotation @ np.asarray(offset, dtype=float)
        origins.append(position.copy())
        rotation = rotation @ _rot(axis, sign * q)

    segments = [
        (start, end)
        for start, end in zip(origins[:-1], origins[1:])
        if np.linalg.norm(end - start) > 1e-6
    ]
    tcp_base_m = np.array(
        [
            tcp[0] / 1000.0,
            tcp[1] / 1000.0,
            (tcp[2] - UF0_Z_MM) / 1000.0,
        ]
    )
    if np.linalg.norm(tcp_base_m - origins[-1]) > 1e-6:
        segments.append((origins[-1], tcp_base_m))
    return segments


def _distance_to_segment(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator <= 1e-12:
        return np.linalg.norm(points - start, axis=1)
    fraction = np.clip(
        ((points - start) @ segment) / denominator, 0.0, 1.0
    )
    closest = start + fraction[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def classify_robot_self_points(
    points_base_m: np.ndarray,
    joints_deg,
    tcp_pose_uf0_mm,
    capsule_radius_m: float = 0.125,
) -> dict:
    """Classify points near the current robot skeleton as robot/self.

    This is only used on points above the controlled tabletop-object envelope;
    it must not be used to erase ordinary low obstacles.
    """

    points = np.asarray(points_base_m, dtype=float).reshape(-1, 3)
    segments = robot_skeleton_segments(joints_deg, tcp_pose_uf0_mm)
    if not len(points):
        return {
            "self_mask": np.zeros(0, dtype=bool),
            "minimum_distance_m": np.zeros(0, dtype=float),
            "segments": segments,
        }
    distances = np.stack(
        [_distance_to_segment(points, a, b) for a, b in segments], axis=0
    )
    minimum = np.min(distances, axis=0)
    return {
        "self_mask": minimum <= float(capsule_radius_m),
        "minimum_distance_m": minimum,
        "segments": segments,
    }
