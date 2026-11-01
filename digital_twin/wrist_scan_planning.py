"""Fixed, truth-independent observation grid for wrist-camera reinspection."""

from __future__ import annotations


# Tool-tip poses in robot-base metres.  With the calibrated wrist-camera mount,
# these four poses place the lens about 230 mm above the table and cover the
# validated battery workspace with overlapping views.  The 100 mm tip height
# remains well above the battery and the shallow bowl.
WRIST_SCAN_TIP_TARGETS_M = (
    (0.213, -0.150, 0.100),
    (0.213, 0.150, 0.100),
    (0.343, 0.150, 0.100),
    (0.343, -0.150, 0.100),
)


def scan_tip_targets_m():
    """Return a mutable copy for callers without exposing module state."""
    return [tuple(float(value) for value in target) for target in WRIST_SCAN_TIP_TARGETS_M]
