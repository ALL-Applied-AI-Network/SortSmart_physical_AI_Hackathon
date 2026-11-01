"""Pure helpers for repeated vision-guided AAA placement into one bowl."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementSlot:
    offset_xy_m: tuple[float, float]
    layer: int
    battery_axis: str


# The first four cells form a loose floor layer with 18 mm centre spacing.
# The fifth is released crosswise above the centre instead of being driven into
# a crowded floor slot.  This mirrors a real bowl, where later cells naturally
# stack, while keeping the commanded gripper trajectory contact-free.
PLACEMENT_SLOTS = (
    PlacementSlot((0.0, -0.027), 0, "x"),
    PlacementSlot((0.0, -0.009), 0, "x"),
    PlacementSlot((0.0, 0.009), 0, "x"),
    PlacementSlot((0.0, 0.027), 0, "x"),
    PlacementSlot((0.0, 0.0), 1, "y"),
)
PLACEMENT_SLOT_Y_OFFSETS_M = tuple(slot.offset_xy_m[1] for slot in PLACEMENT_SLOTS)


def placement_slot_xy(bowl_center_xy_m, index: int):
    """Return one deterministic, bowl-relative placement slot."""

    if not 0 <= index < len(PLACEMENT_SLOTS):
        raise IndexError("placement slot index is outside the five-cell layout")
    x, y = (float(value) for value in bowl_center_xy_m)
    dx, dy = PLACEMENT_SLOTS[index].offset_xy_m
    return (x + dx, y + dy)


def placement_slot_spec(index: int):
    if not 0 <= index < len(PLACEMENT_SLOTS):
        raise IndexError("placement slot index is outside the five-cell layout")
    return PLACEMENT_SLOTS[index]


def release_tip_robot_z(
    pile_top_world_z: float,
    *,
    robot_base_world_z: float,
    tip_clearance_m: float,
    release_clearance_m: float,
):
    """TCP Z that leaves the held cell just above a measured pile top.

    The battery centre sits `(radius - tip_clearance)` above the TCP, so the
    radius cancels when converting a desired bottom-surface clearance to TCP Z.
    """

    values = (
        pile_top_world_z,
        robot_base_world_z,
        tip_clearance_m,
        release_clearance_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("release height inputs must be finite")
    if tip_clearance_m <= 0 or release_clearance_m < 0:
        raise ValueError("release clearances must be non-negative")
    return float(
        pile_top_world_z
        - robot_base_world_z
        + tip_clearance_m
        + release_clearance_m
    )


def outside_bowl_exclusion(
    center_xy_m,
    bowl_center_xy_m,
    *,
    exclusion_radius_m: float,
):
    """True when a candidate is outside the bowl and can still be picked."""

    x, y = (float(value) for value in center_xy_m)
    bx, by = (float(value) for value in bowl_center_xy_m)
    return math.hypot(x - bx, y - by) > float(exclusion_radius_m)


def select_pick_candidate(records, bowl_center_xy_m, *, exclusion_radius_m):
    """Select the highest-confidence valid battery outside the bowl."""

    eligible = []
    for record in records:
        pose = record.get("pose") or {}
        center = pose.get("center_xy_m")
        if not record.get("geometry_valid") or center is None:
            continue
        if outside_bowl_exclusion(
            center,
            bowl_center_xy_m,
            exclusion_radius_m=exclusion_radius_m,
        ):
            eligible.append(record)
    return max(
        eligible,
        key=lambda item: float(item.get("confidence", 0.0)),
        default=None,
    )

