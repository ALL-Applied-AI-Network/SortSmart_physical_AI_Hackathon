"""Procedural white drop bowl used by the AAA sorting twin."""

from __future__ import annotations

import math

import numpy as np

from pxr import Gf, UsdGeom, UsdShade

import cell
import scene_config as C


BOWL_XY = (0.28, 0.18)
BOWL_OUTER_RADIUS = 0.096
BOWL_INNER_RADIUS = 0.087
BOWL_BASE_RADIUS = 0.056
BOWL_FLOOR_RADIUS = 0.052
BOWL_HEIGHT = 0.036
BOWL_FLOOR_THICKNESS = 0.006
# Place the held battery just above the internal floor before opening the jaws.
# The old 55 mm clearance above the rim produced an 82 mm free fall; occasional
# asymmetric jaw contact then gave the cell enough energy to bounce out.
BOWL_PLACE_CLEARANCE = 0.001
BOWL_PATH = "/World/BatteryDropBowl"


def _ensure_bowl_material(stage):
    """Warm ivory, matte plastic matching the physical collection bowl."""

    name = "bowl_ivory"
    if name not in cell.MATS:
        from isaacsim.core.api.materials import OmniPBR

        pbr = OmniPBR(
            prim_path="/World/Looks/bowl_ivory",
            name=name,
            color=np.array((0.76, 0.72, 0.62), dtype=float),
        )
        pbr.set_metallic_constant(0.0)
        pbr.set_reflection_roughness(0.68)
        cell.MATS[name] = pbr.material
    return name


def add_battery_drop_bowl(stage):
    """Create the large, shallow ivory bowl used in the physical cell.

    The visible mesh and the collision proxies are deliberately separate.
    Proxies remain active in PhysX but are hidden from the camera so YOLO sees
    one smooth bowl rather than a ring of collision boxes.
    """

    surface_z = C.TABLE_TOP_Z + C.SURFACE_DZ
    root = UsdGeom.Xform.Define(stage, BOWL_PATH)
    UsdGeom.Xformable(root.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(BOWL_XY[0], BOWL_XY[1], surface_z)
    )

    material_name = _ensure_bowl_material(stage)

    segments = 96
    points = []
    rings = []
    # Cross-section ordered from the outside bottom, over the rounded lip and
    # down the inside to the shallow floor.  Revolving it gives the tapered,
    # wide silhouette visible in the real table/side/wrist camera data.
    profile = (
        (BOWL_BASE_RADIUS, 0.000),
        (0.063, 0.004),
        (0.070, 0.010),
        (0.083, 0.026),
        (0.095, 0.034),
        (BOWL_OUTER_RADIUS, BOWL_HEIGHT),
        (BOWL_INNER_RADIUS, BOWL_HEIGHT),
        (0.084, 0.033),
        (0.075, 0.023),
        (0.060, 0.010),
        (BOWL_FLOOR_RADIUS, BOWL_FLOOR_THICKNESS),
    )
    for radius, z in profile:
        start = len(points)
        points.extend(
            Gf.Vec3f(
                radius * math.cos(2 * math.pi * i / segments),
                radius * math.sin(2 * math.pi * i / segments),
                z,
            )
            for i in range(segments)
        )
        rings.append(start)
    floor_center = len(points)
    points.append(Gf.Vec3f(0, 0, BOWL_FLOOR_THICKNESS))
    underside_center = len(points)
    points.append(Gf.Vec3f(0, 0, 0))

    counts = []
    indices = []
    for i in range(segments):
        j = (i + 1) % segments
        for ring_index in range(len(rings) - 1):
            ring_a = rings[ring_index]
            ring_b = rings[ring_index + 1]
            face = (ring_a + i, ring_a + j, ring_b + j, ring_b + i)
            counts.append(4)
            indices.extend(face)
        counts.append(3)
        indices.extend((floor_center, rings[-1] + i, rings[-1] + j))
        counts.append(3)
        indices.extend((underside_center, rings[0] + j, rings[0] + i))

    mesh = UsdGeom.Mesh.Define(stage, BOWL_PATH + "/Visual")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    cell.bind_mat(mesh.GetPrim(), material_name)

    # Static collision floor.
    floor = cell.cyl(
        stage,
        BOWL_PATH + "Colliders/Floor",
        BOWL_FLOOR_RADIUS,
        BOWL_FLOOR_THICKNESS,
        (
            BOWL_XY[0],
            BOWL_XY[1],
            surface_z + BOWL_FLOOR_THICKNESS / 2,
        ),
        (0.76, 0.72, 0.62),
        collision=True,
        mat=None,
    )
    cell.bind_physics_material(floor)
    cell.small_contact_offsets(floor)
    UsdGeom.Imageable(floor).MakeInvisible()

    # Hidden tangent boxes form a stable circular collision wall.
    proxy_count = 24
    radius = (BOWL_BASE_RADIUS + BOWL_INNER_RADIUS) / 2
    radial_thickness = BOWL_INNER_RADIUS - BOWL_BASE_RADIUS
    tangent_length = 2 * math.pi * radius / proxy_count * 1.08
    for index in range(proxy_count):
        angle = 2 * math.pi * index / proxy_count
        proxy = cell.box(
            stage,
            f"{BOWL_PATH}Colliders/Wall{index:02d}",
            (radial_thickness, tangent_length, BOWL_HEIGHT),
            (
                BOWL_XY[0] + radius * math.cos(angle),
                BOWL_XY[1] + radius * math.sin(angle),
                surface_z + BOWL_HEIGHT / 2,
            ),
            (0.76, 0.72, 0.62),
            rot=(0.0, 0.0, math.degrees(angle)),
            mat=None,
        )
        cell.bind_physics_material(proxy)
        cell.small_contact_offsets(proxy)
        UsdGeom.Imageable(proxy).MakeInvisible()

    return BOWL_PATH


def bowl_release_tip_position():
    """Robot-base-frame TCP position for a low-energy floor placement."""

    return (
        BOWL_XY[0],
        BOWL_XY[1],
        C.SURFACE_DZ
        + BOWL_FLOOR_THICKNESS
        + C.BATTERY_TIP_CLEARANCE
        + BOWL_PLACE_CLEARANCE,
    )

