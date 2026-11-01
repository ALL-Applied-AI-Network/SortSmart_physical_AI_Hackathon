"""Calibrated 2D pose estimation for a horizontal AAA battery.

The geometry in this module is deliberately simulator-independent.  Both the
Isaac test and a future real-cell adapter provide a BGR image plus a calibrated
``backproject.CameraModel``.  A detector supplies a binary battery-body mask;
the common code below converts that mask into robot-base XY and long-axis yaw.

The built-in HSV detector is only the synthetic-scene baseline.  For the real
24/7 Life battery, replace ``synthetic_orange_mask`` with an instance-segmentation
model (or an iRVision result) and keep ``estimate_pose_from_mask`` unchanged.
"""

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class Battery2DPose:
    center_xy_m: tuple[float, float]
    axis_xy: tuple[float, float]
    yaw_deg: float
    observed_length_m: float
    observed_width_m: float
    area_px: int
    confidence: float
    bbox_px: tuple[int, int, int, int]

    def to_dict(self):
        return asdict(self)


def _odd(value):
    value = max(3, int(round(value)))
    return value if value % 2 else value + 1


def synthetic_orange_mask(image_bgr):
    """Return the body mask used by the current orange Isaac AAA asset.

    This is intentionally named ``synthetic`` so it is not accidentally
    described as a production detector for the branded physical cell.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected a BGR image with shape HxWx3")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 75, 45)), np.array((28, 255, 255)))
    scale = image_bgr.shape[1] / 1920.0
    kernel = np.ones((_odd(3 * scale), _odd(3 * scale)), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _canonical_axis(axis):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    # A cylinder long axis is equivalent after a 180 degree turn.  Keep a
    # deterministic sign so reports and wrist commands do not jump by 180 deg.
    if axis[0] < 0 or (abs(axis[0]) < 1e-12 and axis[1] < 0):
        axis = -axis
    return axis


def _component_pose(
    component,
    camera,
    plane_z,
    expected_length_m,
    expected_width_m,
    min_area_px,
    workspace_xy,
):
    area = int(np.count_nonzero(component))
    if area < min_area_px:
        return None

    vv, uu = np.nonzero(component)
    # Deproject a bounded sample of body pixels to the known battery-centre
    # plane.  This handles a tilted/oblique calibrated 2D camera more cleanly
    # than applying cv2.minAreaRect directly in pixel space.
    stride = max(1, len(uu) // 5000)
    points = []
    for u, v in zip(uu[::stride], vv[::stride]):
        p = camera.to_plane(float(u) + 0.5, float(v) + 0.5, z=plane_z)
        if p is not None and np.all(np.isfinite(p[:2])):
            points.append(p[:2])
    if len(points) < 20:
        return None
    points = np.asarray(points, float)

    seed = points.mean(axis=0)
    covariance = np.cov((points - seed).T)
    values, vectors = np.linalg.eigh(covariance)
    axis = _canonical_axis(vectors[:, int(np.argmax(values))])
    normal = np.array([-axis[1], axis[0]])
    along = points @ axis
    across = points @ normal
    lo_a, hi_a = np.percentile(along, (1.0, 99.0))
    lo_n, hi_n = np.percentile(across, (1.0, 99.0))
    length = float(hi_a - lo_a)
    width = float(hi_n - lo_n)
    center = axis * ((lo_a + hi_a) / 2.0) + normal * ((lo_n + hi_n) / 2.0)

    if not (0.55 * expected_length_m <= length <= 1.45 * expected_length_m):
        return None
    if not (0.45 * expected_width_m <= width <= 1.80 * expected_width_m):
        return None
    if workspace_xy is not None:
        (xmin, xmax), (ymin, ymax) = workspace_xy
        if not (xmin <= center[0] <= xmax and ymin <= center[1] <= ymax):
            return None

    length_error = abs(length / expected_length_m - 1.0)
    width_error = abs(width / expected_width_m - 1.0)
    shape_score = math.exp(-2.5 * length_error - 2.0 * width_error)
    confidence = float(np.clip(shape_score, 0.0, 1.0))
    x0, x1 = int(uu.min()), int(uu.max()) + 1
    y0, y1 = int(vv.min()), int(vv.max()) + 1
    yaw = math.degrees(math.atan2(axis[1], axis[0]))
    return Battery2DPose(
        center_xy_m=(float(center[0]), float(center[1])),
        axis_xy=(float(axis[0]), float(axis[1])),
        yaw_deg=float(yaw),
        observed_length_m=length,
        observed_width_m=width,
        area_px=area,
        confidence=confidence,
        bbox_px=(x0, y0, x1, y1),
    )


def estimate_pose_from_mask(
    mask,
    camera,
    plane_z,
    expected_length_m=0.0437,
    expected_width_m=0.0105,
    workspace_xy=None,
    min_area_px=None,
):
    """Estimate the best horizontal battery pose from a binary body mask.

    ``plane_z`` is the battery centre plane in the same world frame as the
    camera calibration.  For a known horizontal AAA cell it is the calibrated
    table surface plus 5.25 mm.  No simulator object pose is read here.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("battery mask must be a 2D array")
    binary = (mask > 0).astype(np.uint8)
    if min_area_px is None:
        min_area_px = max(20, round(45 * (camera.W / 1920.0) ** 2))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area_px:
            continue
        pose = _component_pose(
            labels == label,
            camera,
            plane_z,
            expected_length_m,
            expected_width_m,
            min_area_px,
            workspace_xy,
        )
        if pose is not None:
            candidates.append(pose)
    if not candidates:
        raise RuntimeError("no calibrated AAA-sized battery body found in the image")
    return max(candidates, key=lambda pose: (pose.confidence, pose.area_px))


def detect_synthetic_battery(
    image_bgr,
    camera,
    plane_z,
    expected_length_m=0.0437,
    expected_width_m=0.0105,
    workspace_xy=None,
):
    mask = synthetic_orange_mask(image_bgr)
    pose = estimate_pose_from_mask(
        mask,
        camera,
        plane_z,
        expected_length_m=expected_length_m,
        expected_width_m=expected_width_m,
        workspace_xy=workspace_xy,
    )
    return pose, mask


def draw_detection(image_bgr, pose, mask=None):
    overlay = image_bgr.copy()
    if mask is not None:
        tint = np.zeros_like(overlay)
        tint[:, :, 1] = (mask > 0).astype(np.uint8) * 150
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.28, 0.0)
    x0, y0, x1, y1 = pose.bbox_px
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.putText(
        overlay,
        f"AAA yaw={pose.yaw_deg:.1f} conf={pose.confidence:.2f}",
        (x0, max(25, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * image_bgr.shape[1] / 1920.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay
