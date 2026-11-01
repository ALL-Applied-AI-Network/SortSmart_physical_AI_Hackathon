"""Filesystem IPC client shared by Isaac and the external YOLO worker.

Isaac Sim intentionally does not import Torch or Ultralytics.  It writes one
RGB observation to a shared workspace directory, atomically publishes a JSON
request, and waits for an atomically written response containing instance-mask
polygons.  The external worker runs in the already validated training venv.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

import cv2
import numpy as np


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def request_instances(
    image_bgr: np.ndarray,
    ipc_directory: str | Path,
    *,
    timeout_seconds: float = 90.0,
    roi_xyxy=None,
) -> dict:
    """Submit one BGR image and return the worker response.

    Request and response files are retained for audit/debugging.  A unique ID
    prevents stale results from an earlier simulator run being reused.
    """

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected BGR image with shape HxWx3")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    roi = None
    if roi_xyxy is not None:
        if len(roi_xyxy) != 4:
            raise ValueError("roi_xyxy must contain x1, y1, x2, y2")
        roi = [int(value) for value in roi_xyxy]
        x1, y1, x2, y2 = roi
        height, width = image.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"roi_xyxy is outside the image: {roi}")
        if x2 - x1 < 32 or y2 - y1 < 32:
            raise ValueError(f"roi_xyxy is too small: {roi}")

    root = Path(ipc_directory).expanduser().resolve()
    request_dir = root / "requests"
    response_dir = root / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"{time.time_ns()}_{uuid.uuid4().hex[:10]}"
    image_path = request_dir / f"{request_id}.png"
    request_path = request_dir / f"{request_id}.json"
    response_path = response_dir / f"{request_id}.json"

    temporary_image = request_dir / f"{request_id}.tmp.png"
    if not cv2.imwrite(str(temporary_image), image):
        raise RuntimeError(f"could not write IPC image: {temporary_image}")
    os.replace(temporary_image, image_path)

    _atomic_json(
        request_path,
        {
            "protocol": "fanuc_aaa_yolo_ipc_v1",
            "request_id": request_id,
            "image_path": str(image_path),
            "image_shape": list(image.shape),
            "roi_xyxy": roi,
            "created_unix_s": time.time(),
        },
    )

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if response_path.is_file():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if response.get("request_id") != request_id:
                raise RuntimeError("YOLO IPC response ID does not match request")
            if response.get("error"):
                raise RuntimeError(f"YOLO worker failed: {response['error']}")
            return response
        time.sleep(0.05)

    raise TimeoutError(
        f"No YOLO response after {timeout_seconds:.1f} s: {response_path}. "
        "Start yolo_ipc_worker.py outside the Isaac container."
    )


def polygon_mask(shape: tuple[int, int], polygon_px) -> np.ndarray:
    """Rasterize one response polygon into a uint8 mask."""

    height, width = map(int, shape)
    polygon = np.asarray(polygon_px, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("instance polygon must have shape Nx2 with N >= 3")
    contour = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    return mask

