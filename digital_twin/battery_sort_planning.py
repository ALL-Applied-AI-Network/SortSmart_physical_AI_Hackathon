"""Pure geometry for vision-driven AAA pick and bowl placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BatterySortPlan:
    battery_center_xy_m: tuple[float, float]
    battery_axis_xy: tuple[float, float]
    raw_spread_xy: tuple[float, float]
    bowl_center_xy_m: tuple[float, float]
    pick_tip_m: tuple[float, float, float]
    pick_above_m: tuple[float, float, float]
    pick_lift_m: tuple[float, float, float]
    bowl_above_m: tuple[float, float, float]
    bowl_release_m: tuple[float, float, float]
    bowl_retreat_m: tuple[float, float, float]

    def to_dict(self) -> dict:
        value = asdict(self)
        for key, item in tuple(value.items()):
            if isinstance(item, tuple):
                value[key] = list(item)
        return value


def make_battery_sort_plan(
    battery_center_xy,
    battery_axis_xy,
    bowl_center_xy,
    *,
    surface_dz_m,
    tip_clearance_m,
    approach_height_m=0.10,
    carry_height_m=0.12,
    bowl_height_m,
    bowl_floor_height_m,
    bowl_place_clearance_m,
) -> BatterySortPlan:
    battery_xy = np.asarray(battery_center_xy, dtype=float)
    axis = np.asarray(battery_axis_xy, dtype=float)
    bowl_xy = np.asarray(bowl_center_xy, dtype=float)
    if battery_xy.shape != (2,) or bowl_xy.shape != (2,) or axis.shape != (2,):
        raise ValueError("battery centre, axis and bowl centre must be 2D vectors")
    if not np.isfinite(np.concatenate((battery_xy, axis, bowl_xy))).all():
        raise ValueError("planning inputs must be finite")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        raise ValueError("battery axis has zero length")
    axis /= axis_norm
    spread = np.array([-axis[1], axis[0]], dtype=float)

    pick_z = float(surface_dz_m + tip_clearance_m)
    # The held battery centre sits (radius - tip_clearance) above the TCP.
    # Reusing the pick tip clearance over the bowl floor therefore leaves the
    # cylindrical surface exactly bowl_place_clearance_m above that floor.
    release_z = float(
        surface_dz_m
        + bowl_floor_height_m
        + tip_clearance_m
        + bowl_place_clearance_m
    )
    travel_z = max(pick_z + carry_height_m, release_z + 0.060)
    pick = np.array([battery_xy[0], battery_xy[1], pick_z])
    pick_above = pick + np.array([0.0, 0.0, approach_height_m])
    pick_lift = np.array([battery_xy[0], battery_xy[1], travel_z])
    bowl_above = np.array([bowl_xy[0], bowl_xy[1], travel_z])
    bowl_release = np.array([bowl_xy[0], bowl_xy[1], release_z])

    def triple(value):
        return tuple(float(component) for component in value)

    return BatterySortPlan(
        battery_center_xy_m=tuple(float(v) for v in battery_xy),
        battery_axis_xy=tuple(float(v) for v in axis),
        raw_spread_xy=tuple(float(v) for v in spread),
        bowl_center_xy_m=tuple(float(v) for v in bowl_xy),
        pick_tip_m=triple(pick),
        pick_above_m=triple(pick_above),
        pick_lift_m=triple(pick_lift),
        bowl_above_m=triple(bowl_above),
        bowl_release_m=triple(bowl_release),
        bowl_retreat_m=triple(bowl_above),
    )
