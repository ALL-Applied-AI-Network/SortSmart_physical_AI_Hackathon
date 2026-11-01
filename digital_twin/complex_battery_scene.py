"""Deterministic multi-battery clutter layout for the video demonstration."""

from __future__ import annotations

import math


# (x, y, yaw_deg, brand, visual_roll_deg)
BATTERIES = (
    (0.34, -0.12, 7.0, "amazon", 90.0),
    (0.39, -0.10, 38.0, "seven_eleven", 0.0),
    (0.44, -0.08, 76.0, "amazon", 180.0),
    (0.33, 0.03, 113.0, "seven_eleven", 270.0),
    (0.43, 0.00, 151.0, "amazon", 0.0),
)

# Small coloured cubes, two green industrial vials, two cap-like parts and
# three dark screws reproduce the categories visible in the physical camera
# clutter images while staying clear of the ivory collection bowl.
CUBES = (
    (0.27, -0.12, "red"),
    (0.31, -0.05, "yellow"),
    (0.38, -0.16, "blue"),
    (0.47, -0.15, "red"),
    (0.49, 0.07, "yellow"),
    (0.41, 0.12, "blue"),
)
CUBE_YAWS_DEG = (12.0, -18.0, 31.0, -27.0, 8.0, 24.0)
VIALS = (
    (0.45, 0.15, 90.0),
    (0.51, -0.20, 20.0),
)
CAPS = (
    (0.25, -0.02),
    (0.50, -0.04),
)
SCREWS = (
    (0.29, -0.19, 18.0),
    (0.50, -0.11, 122.0),
    (0.42, 0.20, 64.0),
)
BOWL_XY = (0.28, 0.18)
BOWL_EXCLUSION_RADIUS_M = 0.115

# Static procedural M6-like screw distractors. Keep these dimensions local:
# the legacy sorting configuration intentionally disables its screw constants.
SCREW_LENGTH_M = 0.030
SCREW_SHANK_DIAMETER_M = 0.006
SCREW_HEAD_DIAMETER_M = 0.012
SCREW_HEAD_HEIGHT_M = 0.006


def validate_layout():
    """Fail early if edits create overlapping batteries or bowl intrusions."""
    centres = [(entry[0], entry[1]) for entry in BATTERIES]
    for index, a in enumerate(centres):
        if math.dist(a, BOWL_XY) <= BOWL_EXCLUSION_RADIUS_M:
            raise ValueError(f"battery {index} is too close to the bowl")
        for other_index, b in enumerate(centres[index + 1 :], start=index + 1):
            if math.dist(a, b) < 0.050:
                raise ValueError(
                    f"batteries {index} and {other_index} are too close"
                )
    if {entry[3] for entry in BATTERIES} != {"amazon", "seven_eleven"}:
        raise ValueError("complex scene must contain both battery brands")
    if len(CUBES) != len(CUBE_YAWS_DEG):
        raise ValueError("every cube needs one yaw angle")
    return True


def requested_scene_description():
    return {
        "mode": "complex_clutter_single_pick_demo",
        "battery_count": len(BATTERIES),
        "batteries": [
            {
                "x_m": entry[0],
                "y_m": entry[1],
                "yaw_deg": entry[2],
                "brand": entry[3],
                "visual_roll_deg": entry[4],
            }
            for entry in BATTERIES
        ],
        "clutter": {
            "cubes": len(CUBES),
            "vials": len(VIALS),
            "caps": len(CAPS),
            "screws": len(SCREWS),
        },
    }


def add_to_stage(stage):
    """Add the validated clutter and return the five battery prim paths."""
    validate_layout()
    import cell
    import scene_config as C
    from pxr import Gf, UsdGeom, UsdPhysics

    surface_z = C.TABLE_TOP_Z + C.SURFACE_DZ
    colour = {
        "red": (0.82, 0.08, 0.05),
        "yellow": (0.95, 0.72, 0.03),
        "blue": (0.04, 0.28, 0.82),
    }
    material = {
        "red": "plastic_red",
        "yellow": "bin_yellow",
        "blue": "plastic_blue",
    }

    # Procedural primitives avoid the metrics-layer edit-target problem caused
    # by loading the legacy referenced vial/cube USD files into an anonymous
    # Isaac stage. These are static distractors: they look like the physical
    # clutter but cannot roll into the planned grasp path during warm-up.
    for index, ((x, y, name), yaw) in enumerate(zip(CUBES, CUBE_YAWS_DEG)):
        prim = cell.box(
            stage,
            f"/World/ClutterCube{index}",
            (C.CUBE_EDGE, C.CUBE_EDGE, C.CUBE_EDGE),
            (x, y, surface_z + C.CUBE_EDGE / 2),
            colour[name],
            rot=(0.0, 0.0, yaw),
            mat=material[name],
        )
        cell.bind_physics_material(prim)
        cell.small_contact_offsets(prim)

    for index, (x, y, yaw) in enumerate(VIALS):
        root = UsdGeom.Xform.Define(stage, f"/World/ClutterVial{index}")
        transform = UsdGeom.Xformable(root.GetPrim())
        transform.AddTranslateOp().Set(
            Gf.Vec3d(x, y, surface_z + C.VIAL_RADIUS)
        )
        transform.AddRotateZOp().Set(float(yaw))
        transform.AddRotateYOp().Set(90.0)
        body = UsdGeom.Cylinder.Define(stage, f"/World/ClutterVial{index}/Body")
        body.CreateAxisAttr("Z")
        body.CreateRadiusAttr(C.VIAL_RADIUS)
        body.CreateHeightAttr(C.VIAL_LENGTH)
        body.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.74, 0.24)])
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        cell.bind_physics_material(body.GetPrim())
        cell.small_contact_offsets(body.GetPrim())

    for index, (x, y) in enumerate(CAPS):
        cap = cell.cyl(
            stage,
            f"/World/ClutterCap{index}",
            C.CAP_RADIUS,
            C.CAP_HEIGHT,
            (x, y, surface_z + C.CAP_HEIGHT / 2),
            (0.10, 0.10, 0.12),
            collision=True,
            mat="black_plastic",
        )
        cell.bind_physics_material(cap)
        cell.small_contact_offsets(cap)

    for index, (x, y, yaw) in enumerate(SCREWS):
        yaw_rad = math.radians(yaw)
        axis = (math.cos(yaw_rad), math.sin(yaw_rad))
        shank = cell.box(
            stage,
            f"/World/ClutterScrew{index}/Shank",
            (
                SCREW_LENGTH_M,
                SCREW_SHANK_DIAMETER_M,
                SCREW_SHANK_DIAMETER_M,
            ),
            (x, y, surface_z + SCREW_SHANK_DIAMETER_M / 2),
            (0.04, 0.04, 0.05),
            rot=(0.0, 0.0, yaw),
            mat="black_oxide",
        )
        head_x = x - axis[0] * (
            SCREW_LENGTH_M + SCREW_HEAD_DIAMETER_M
        ) / 2
        head_y = y - axis[1] * (
            SCREW_LENGTH_M + SCREW_HEAD_DIAMETER_M
        ) / 2
        head = cell.box(
            stage,
            f"/World/ClutterScrew{index}/Head",
            (
                SCREW_HEAD_DIAMETER_M,
                SCREW_HEAD_DIAMETER_M,
                SCREW_HEAD_HEIGHT_M,
            ),
            (head_x, head_y, surface_z + SCREW_HEAD_HEIGHT_M / 2),
            (0.035, 0.035, 0.045),
            rot=(0.0, 0.0, yaw),
            mat="black_oxide",
        )
        for prim in (shank, head):
            cell.bind_physics_material(prim)
            cell.small_contact_offsets(prim)

    return cell.add_batteries(stage, entries=BATTERIES)
