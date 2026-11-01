"""Calibrated drop-zone centre estimation from a YOLO instance polygon.

The collection bowl is not flat.  Projecting its image centroid onto the table
plane creates a systematic offset for an oblique camera.  This module projects
the visible outer contour onto the known rim plane and robustly fits the known
circular rim.  It falls back to the height-corrected polygon centroid if the
circle fit is not geometrically plausible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DropZonePose:
    center_xy_m: tuple[float, float]
    center_px: tuple[float, float]
    method: str
    rim_plane_z_m: float
    polygon_area_px: float
    fitted_radius_m: float | None
    fit_rmse_mm: float | None
    fit_inlier_fraction: float | None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["center_xy_m"] = list(self.center_xy_m)
        value["center_px"] = list(self.center_px)
        return value


def _polygon_centroid(polygon: np.ndarray) -> tuple[np.ndarray, float]:
    x = polygon[:, 0]
    y = polygon[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) < 1e-6:
        return polygon.mean(axis=0), 0.0
    centroid = np.array(
        [
            np.sum((x + x_next) * cross) / (6.0 * signed_area),
            np.sum((y + y_next) * cross) / (6.0 * signed_area),
        ],
        dtype=float,
    )
    return centroid, abs(signed_area)


def _densify_closed_polygon(polygon: np.ndarray, step_px: float = 4.0) -> np.ndarray:
    samples = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(distance / step_px)))
        for fraction in np.linspace(0.0, 1.0, count, endpoint=False):
            samples.append(start + fraction * (end - start))
    return np.asarray(samples, dtype=float)


def _circle_from_three(points: np.ndarray):
    p1, p2, p3 = points
    matrix = 2.0 * np.array([p2 - p1, p3 - p1], dtype=float)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-10:
        return None
    rhs = np.array(
        [np.dot(p2, p2) - np.dot(p1, p1), np.dot(p3, p3) - np.dot(p1, p1)],
        dtype=float,
    )
    center = np.linalg.solve(matrix, rhs)
    radius = float(np.linalg.norm(p1 - center))
    return center, radius


def _least_squares_circle(points: np.ndarray):
    if len(points) < 3:
        return None
    design = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    rhs = np.sum(points * points, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3:
        return None
    center = solution[:2]
    radius_squared = float(solution[2] + np.dot(center, center))
    if radius_squared <= 0.0:
        return None
    return center, math.sqrt(radius_squared)


def _robust_circle(points: np.ndarray, expected_radius_m: float):
    if len(points) < 3:
        return None

    residual_threshold = max(0.006, 0.12 * expected_radius_m)
    radius_min = 0.65 * expected_radius_m
    radius_max = 1.35 * expected_radius_m
    rng = np.random.default_rng(20261030)
    best = None
    iterations = min(700, max(120, 8 * len(points)))

    for _ in range(iterations):
        indices = rng.choice(len(points), size=3, replace=False)
        candidate = _circle_from_three(points[indices])
        if candidate is None:
            continue
        center, radius = candidate
        if not radius_min <= radius <= radius_max:
            continue
        residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inliers = residuals <= residual_threshold
        count = int(np.count_nonzero(inliers))
        if count < 3:
            continue
        median = float(np.median(residuals[inliers]))
        radius_penalty = abs(radius - expected_radius_m) / expected_radius_m
        score = (count, -median, -radius_penalty)
        if best is None or score > best[0]:
            best = (score, inliers)

    if best is None:
        return None

    inliers = best[1]
    fitted = _least_squares_circle(points[inliers])
    if fitted is None:
        return None
    center, radius = fitted

    # Recompute once after refinement.
    residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    inliers = residuals <= residual_threshold
    if np.count_nonzero(inliers) >= 3:
        refined = _least_squares_circle(points[inliers])
        if refined is not None:
            center, radius = refined
            residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
            inliers = residuals <= residual_threshold

    rmse = float(np.sqrt(np.mean(np.square(residuals[inliers]))))
    fraction = float(np.mean(inliers))
    return center, float(radius), rmse, fraction


def estimate_drop_zone_from_polygon(
    polygon_px,
    camera,
    *,
    rim_height_above_table_m: float,
    expected_outer_radius_m: float,
    workspace_xy=((0.10, 0.70), (-0.47, 0.47)),
) -> DropZonePose:
    """Estimate the bowl centre without using simulator object pose."""

    polygon = np.asarray(polygon_px, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("drop-zone polygon must have shape Nx2 with N >= 3")
    if not np.isfinite(polygon).all():
        raise ValueError("drop-zone polygon contains non-finite coordinates")
    if rim_height_above_table_m <= 0.0 or expected_outer_radius_m <= 0.0:
        raise ValueError("rim height and outer radius must be positive")

    centroid_px, area_px = _polygon_centroid(polygon)
    rim_z = float(camera.table_z + rim_height_above_table_m)
    centroid_world = camera.to_plane(float(centroid_px[0]), float(centroid_px[1]), z=rim_z)
    if centroid_world is None:
        raise RuntimeError("drop-zone centroid ray does not intersect the rim plane")

    dense_px = _densify_closed_polygon(polygon)
    world_points = []
    for u, v in dense_px:
        point = camera.to_plane(float(u), float(v), z=rim_z)
        if point is not None and np.isfinite(point[:2]).all():
            world_points.append(point[:2])
    world_points = np.asarray(world_points, dtype=float)

    fitted = _robust_circle(world_points, expected_outer_radius_m)
    method = "rim_plane_polygon_centroid"
    selected = np.asarray(centroid_world[:2], dtype=float)
    fitted_radius = None
    rmse_mm = None
    inlier_fraction = None
    if fitted is not None:
        fit_center, fitted_radius, fit_rmse, inlier_fraction = fitted
        fit_is_valid = bool(
            0.75 * expected_outer_radius_m <= fitted_radius <= 1.25 * expected_outer_radius_m
            and fit_rmse <= 0.015
            and inlier_fraction >= 0.45
        )
        rmse_mm = 1000.0 * fit_rmse
        if fit_is_valid:
            selected = fit_center
            method = "rim_plane_robust_circle_fit"

    (x_min, x_max), (y_min, y_max) = workspace_xy
    if not (x_min <= selected[0] <= x_max and y_min <= selected[1] <= y_max):
        raise RuntimeError(f"drop-zone centre outside workspace: {selected.tolist()}")

    return DropZonePose(
        center_xy_m=(float(selected[0]), float(selected[1])),
        center_px=(float(centroid_px[0]), float(centroid_px[1])),
        method=method,
        rim_plane_z_m=rim_z,
        polygon_area_px=float(area_px),
        fitted_radius_m=None if fitted_radius is None else float(fitted_radius),
        fit_rmse_mm=rmse_mm,
        fit_inlier_fraction=inlier_fraction,
    )
