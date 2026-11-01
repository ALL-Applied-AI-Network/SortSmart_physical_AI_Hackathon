"""Photo-referenced visual wrappers for simulated AAA batteries.

This module intentionally owns visuals only.  The collision cylinder, mass,
friction and terminal collision geometry remain in :mod:`cell` so introducing
brand appearance cannot change grasp physics.
"""

from __future__ import annotations

import math
import os

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade


BRANDS = ("amazon", "seven_eleven")
_HERE = os.path.dirname(os.path.abspath(__file__))
_TEXTURE_DIR = os.path.join(_HERE, "assets", "battery_textures")
_TEXTURES = {
    "amazon": os.path.join(_TEXTURE_DIR, "amazon_basics_aaa.png"),
    "seven_eleven": os.path.join(_TEXTURE_DIR, "seven_eleven_aaa.png"),
}
_BASE_COLOURS = {
    "amazon": (0.918, 0.624, 0.075),
    "seven_eleven": (0.025, 0.070, 0.105),
}
_NEGATIVE_CAP_COLOURS = {
    "amazon": (0.110, 0.180, 0.220),
    "seven_eleven": (0.018, 0.025, 0.030),
}
_POSITIVE_CAP_COLOURS = {
    "amazon": (0.110, 0.180, 0.220),
    "seven_eleven": (0.760, 0.480, 0.085),
}


def resolve_brand(requested, index=0, x=0.0, y=0.0, yaw_deg=0.0, seed=20261030):
    """Resolve ``random`` deterministically so an experiment is reproducible."""
    requested = str(requested).strip().lower()
    aliases = {
        "24/7": "seven_eleven",
        "24_7": "seven_eleven",
        "7eleven": "seven_eleven",
        "7_eleven": "seven_eleven",
        "amazon_basics": "amazon",
    }
    requested = aliases.get(requested, requested)
    if requested in BRANDS:
        return requested
    if requested != "random":
        raise ValueError(f"Unsupported battery brand: {requested!r}")
    # Integer arithmetic avoids Python's process-randomised hash().
    key = (
        int(seed)
        + 73856093 * int(index)
        + 19349663 * int(round(float(x) * 1000.0))
        + 83492791 * int(round(float(y) * 1000.0))
        + 2654435761 * int(round(float(yaw_deg) * 10.0))
    )
    return BRANDS[key & 1]


def deterministic_roll_deg(index, x, y, yaw_deg, seed=20261030):
    """Return a stable 0..360-degree wrapper roll for domain variation."""
    value = (
        2246822519 * (int(index) + 1)
        + 3266489917 * int(round(float(x) * 1000.0))
        + 668265263 * int(round(float(y) * 1000.0))
        + 374761393 * int(round(float(yaw_deg) * 10.0))
        + int(seed)
    )
    return float(value % 3600) / 10.0


def _textured_material(stage, brand):
    path = f"/World/Looks/battery_wrapper_{brand}"
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        return UsdShade.Material(prim)
    texture = _TEXTURES[brand]
    if not os.path.isfile(texture):
        raise FileNotFoundError(
            f"Battery texture missing: {texture}. Run make_battery_brand_textures.py"
        )
    from isaacsim.core.api.materials import OmniPBR

    pbr = OmniPBR(
        prim_path=path,
        name=f"battery_wrapper_{brand}",
        color=np.ones(3, dtype=float),
    )
    pbr.set_metallic_constant(0.0)
    # The orange Amazon wrapper showed a near-white stripe at table-camera
    # scale that erased most label pixels at a 90-degree visual roll.  The
    # slightly rougher value matches the broad highlight in the reference
    # photos while retaining a glossy consumer-battery appearance.
    pbr.set_reflection_roughness(0.40 if brand == "amazon" else 0.32)
    pbr.set_project_uvw(False)
    pbr.shaders_list[0].CreateInput(
        "diffuse_texture", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(texture))
    return pbr.material


def _solid_material(stage, name, colour, metallic=0.0, roughness=0.34):
    path = f"/World/Looks/{name}"
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        return UsdShade.Material(prim)
    from isaacsim.core.api.materials import OmniPBR

    pbr = OmniPBR(
        prim_path=path,
        name=name,
        color=np.asarray(colour, dtype=float),
    )
    pbr.set_metallic_constant(float(metallic))
    pbr.set_reflection_roughness(float(roughness))
    return pbr.material


def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _wrapper_mesh(stage, path, body_length, radius, roll_deg, segments=96):
    """Create a UV cylinder sidewall whose local long axis is X."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    half = float(body_length) / 2.0
    radius = float(radius) + 0.000025  # avoid z-fighting with physics cylinder
    roll = math.radians(float(roll_deg))
    points = []
    texcoords = []
    for index in range(segments + 1):
        t = index / segments
        theta = -math.pi + 2.0 * math.pi * t + roll
        y = radius * math.sin(theta)
        z = radius * math.cos(theta)
        points.extend((Gf.Vec3f(-half, y, z), Gf.Vec3f(half, y, z)))
        # s follows battery length; t follows circumference.  Duplicating the
        # seam vertices prevents interpolation across the label boundary.
        texcoords.extend((Gf.Vec2f(0.0, t), Gf.Vec2f(1.0, t)))
    indices = []
    for index in range(segments):
        left = 2 * index
        right = left + 1
        next_left = left + 2
        next_right = left + 3
        indices.extend((left, right, next_right, next_left))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * segments)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)
    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(texcoords)
    return mesh.GetPrim()


def _cap(stage, path, x, radius, thickness, material):
    cap = UsdGeom.Cylinder.Define(stage, path)
    cap.CreateAxisAttr("X")
    cap.CreateRadiusAttr(float(radius) + 0.000035)
    cap.CreateHeightAttr(float(thickness))
    UsdGeom.Xformable(cap.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x), 0, 0))
    _bind(cap.GetPrim(), material)
    return cap.GetPrim()


def add_brand_visuals(
    stage,
    root_path,
    brand,
    body_length,
    radius,
    roll_deg=0.0,
):
    """Attach a wrapper and end-band caps under an existing battery root."""
    if brand not in BRANDS:
        raise ValueError(f"Unsupported resolved brand: {brand!r}")
    wrapper_material = _textured_material(stage, brand)
    body_material = _solid_material(
        stage, f"battery_body_{brand}", _BASE_COLOURS[brand]
    )
    negative_material = _solid_material(
        stage, f"battery_negative_cap_{brand}", _NEGATIVE_CAP_COLOURS[brand]
    )
    positive_material = _solid_material(
        stage, f"battery_positive_cap_{brand}", _POSITIVE_CAP_COLOURS[brand]
    )
    terminal_material = _solid_material(
        stage,
        "battery_brand_terminal_metal",
        (0.650, 0.665, 0.690),
        metallic=0.82,
        roughness=0.30,
    )
    body_prim = stage.GetPrimAtPath(root_path + "/Body")
    if body_prim.IsValid():
        _bind(body_prim, body_material)
    wrapper = _wrapper_mesh(
        stage,
        root_path + "/BrandVisual/Wrapper",
        body_length,
        radius,
        roll_deg,
    )
    _bind(wrapper, wrapper_material)
    cap_thickness = 0.00022
    half = float(body_length) / 2.0
    _cap(
        stage,
        root_path + "/BrandVisual/NegativeBand",
        -half - cap_thickness / 2.0,
        radius,
        cap_thickness,
        negative_material,
    )
    _cap(
        stage,
        root_path + "/BrandVisual/PositiveBand",
        half + cap_thickness / 2.0,
        radius,
        cap_thickness,
        positive_material,
    )
    # The physics body's original negative disk sits just inside the body end.
    # Repeat it on the outside of the decorative band so it remains visible.
    terminal_thickness = 0.00010
    _cap(
        stage,
        root_path + "/BrandVisual/NegativeContact",
        -half - cap_thickness - terminal_thickness / 2.0,
        0.0043 / 2.0,
        terminal_thickness,
        terminal_material,
    )

