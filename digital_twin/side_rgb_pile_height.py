"""Estimate bowl pile height from calibrated side-camera RGB masks.

The estimator uses only camera calibration, battery segmentation pixels and the
vision-derived bowl centre.  It does not read simulator depth or object poses.
"""

from __future__ import annotations

import numpy as np


def estimate_pile_top_from_masks(
    camera,
    masks,
    bowl_center_xy_m,
    *,
    floor_top_world_z: float,
    bowl_radius_m: float,
    maximum_height_above_floor_m: float = 0.060,
    assumed_y_offset_toward_camera_m: float = 0.0,
    percentile: float = 99.0,
):
    """Return a conservative pile-top estimate from side RGB masks.

    Rays are intersected with a vertical plane through the bowl.  Moving that
    plane slightly toward the side camera produces an upper height bound for
    batteries distributed across the bowl's depth.  Pixel samples outside the
    bowl's X span or plausible Z range are rejected.
    """

    bx, by = (float(value) for value in bowl_center_xy_m)
    plane_y = by + float(assumed_y_offset_toward_camera_m)
    all_z = []
    masks_used = 0
    for mask in masks:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or not np.any(mask):
            continue
        rows, columns = np.nonzero(mask)
        # Bound runtime for large masks without biasing one image region.
        if len(rows) > 8000:
            sample = np.linspace(0, len(rows) - 1, 8000, dtype=int)
            rows = rows[sample]
            columns = columns[sample]

        x_ndc = columns.astype(float) / camera.W * 2.0 - 1.0
        y_ndc = 1.0 - rows.astype(float) / camera.H * 2.0
        directions_camera = np.column_stack(
            (
                x_ndc * camera.tan_h,
                y_ndc * camera.tan_v,
                -np.ones_like(x_ndc),
            )
        )
        directions_world = directions_camera @ camera.R
        dy = directions_world[:, 1]
        usable = np.abs(dy) > 1e-9
        t = np.full_like(dy, np.nan, dtype=float)
        t[usable] = (plane_y - camera.eye[1]) / dy[usable]
        points = camera.eye[None, :] + t[:, None] * directions_world
        z_min = float(floor_top_world_z) - 0.003
        z_max = float(floor_top_world_z) + float(maximum_height_above_floor_m)
        valid = (
            usable
            & np.isfinite(points).all(axis=1)
            & (t > 0.0)
            & (np.abs(points[:, 0] - bx) <= float(bowl_radius_m))
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )
        if np.any(valid):
            all_z.append(points[valid, 2])
            masks_used += 1

    if not all_z:
        return {
            "valid": False,
            "pile_top_world_z": None,
            "height_above_floor_m": None,
            "sample_count": 0,
            "masks_used": 0,
            "assumed_vertical_plane_y_m": plane_y,
        }

    z_values = np.concatenate(all_z)
    top = float(np.percentile(z_values, float(percentile)))
    return {
        "valid": True,
        "pile_top_world_z": top,
        "height_above_floor_m": top - float(floor_top_world_z),
        "sample_count": int(len(z_values)),
        "masks_used": masks_used,
        "assumed_vertical_plane_y_m": plane_y,
        "percentile": float(percentile),
    }

