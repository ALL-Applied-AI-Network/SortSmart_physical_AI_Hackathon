"""Project a calibrated robot workspace into a conservative image ROI."""

from __future__ import annotations

import itertools
import math


def projected_workspace_roi(
    camera,
    *,
    x_bounds,
    y_bounds,
    height_bounds=(0.0, 0.07),
    margin_px=48,
):
    """Return an integer ``[x1, y1, x2, y2]`` ROI in full-image pixels.

    Both the table plane and the top of expected objects are projected so the
    crop remains valid for the bowl, batteries and future low clutter.  The
    ROI is computed only from calibrated camera geometry and configured work
    limits; simulator object pose is not read.
    """

    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError("workspace bounds must be increasing")
    if height_bounds[0] > height_bounds[1]:
        raise ValueError("height bounds must be increasing")
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")

    pixels = []
    for x, y, height in itertools.product(x_bounds, y_bounds, height_bounds):
        pixel = camera.project([x, y, camera.table_z + height])
        if pixel is not None:
            pixels.append(pixel)
    if len(pixels) < 4:
        raise RuntimeError("workspace does not project to enough visible image points")

    u_values = [float(pixel[0]) for pixel in pixels]
    v_values = [float(pixel[1]) for pixel in pixels]
    x1 = max(0, int(math.floor(min(u_values) - margin_px)))
    y1 = max(0, int(math.floor(min(v_values) - margin_px)))
    x2 = min(int(camera.W), int(math.ceil(max(u_values) + margin_px)))
    y2 = min(int(camera.H), int(math.ceil(max(v_values) + margin_px)))
    if x2 - x1 < 128 or y2 - y1 < 128:
        raise RuntimeError(f"projected workspace ROI is unexpectedly small: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]
