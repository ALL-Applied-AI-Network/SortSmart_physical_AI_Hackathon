"""External Ultralytics worker for Isaac Sim AAA perception.

Run this script outside the Isaac container with the same venv used to train or
evaluate ``best.pt``.  The worker loads the model once, watches the shared IPC
directory, and returns instance-mask polygons as JSON.  It never commands the
robot or reads simulator ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--ipc-dir", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--battery-conf", type=float, default=0.60)
    parser.add_argument("--drop-conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    parser.add_argument(
        "--once",
        action="store_true",
        help="process one request and exit (useful for diagnostics)",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _predict_pass(model, source, args, *, classes, confidence):
    started = time.perf_counter()
    result = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=confidence,
        iou=args.iou,
        device=args.device,
        classes=list(classes),
        retina_masks=True,
        verbose=False,
        save=False,
    )[0]
    return result, time.perf_counter() - started


def _append_detections(detections, result, args, *, allowed_classes, offset):
    if result.boxes is None or result.masks is None:
        return
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
    for class_id, confidence, polygon in zip(classes, confidences, result.masks.xy):
        class_id = int(class_id)
        if class_id not in allowed_classes:
            continue
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.ndim != 2 or len(polygon) < 3:
            continue
        polygon = polygon + offset
        threshold = args.battery_conf if class_id == 0 else args.drop_conf
        detections.append(
            {
                "class_id": class_id,
                "class_name": str(result.names.get(class_id, class_id)),
                "confidence": float(confidence),
                "accepted": bool(float(confidence) >= threshold),
                "polygon_px": polygon.round(2).tolist(),
            }
        )


def predict(model: YOLO, request: dict, args: argparse.Namespace) -> dict:
    image_path = Path(request["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"request image is missing: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not decode request image: {image_path}")
    full_height, full_width = image.shape[:2]
    roi = request.get("roi_xyxy")
    crop = None
    crop_offset = np.zeros(2, dtype=np.float32)
    if roi is not None:
        if len(roi) != 4:
            raise ValueError("request roi_xyxy must contain four integers")
        x1, y1, x2, y2 = map(int, roi)
        if not (0 <= x1 < x2 <= full_width and 0 <= y1 < y2 <= full_height):
            raise ValueError(f"request ROI is outside the image: {roi}")
        crop = image[y1:y2, x1:x2]
        crop_offset[:] = (x1, y1)

    started = time.perf_counter()
    detections = []
    passes = []
    if crop is None:
        result, duration = _predict_pass(
            model,
            image,
            args,
            classes=(0, 1),
            confidence=min(args.battery_conf, args.drop_conf),
        )
        _append_detections(
            detections,
            result,
            args,
            allowed_classes={0, 1},
            offset=np.zeros(2, dtype=np.float32),
        )
        passes.append({"name": "full_frame_all_classes", "seconds": duration})
        inference_mode = "full_frame"
    else:
        # Large, context-dependent bowl: infer on the complete table frame.
        bowl_result, bowl_duration = _predict_pass(
            model,
            image,
            args,
            classes=(1,),
            confidence=args.drop_conf,
        )
        _append_detections(
            detections,
            bowl_result,
            args,
            allowed_classes={1},
            offset=np.zeros(2, dtype=np.float32),
        )
        passes.append({"name": "full_frame_drop_zone", "seconds": bowl_duration})

        # Tiny AAA battery: use the calibrated crop while preserving full-frame
        # output coordinates through crop_offset.
        battery_result, battery_duration = _predict_pass(
            model,
            crop,
            args,
            classes=(0,),
            confidence=min(args.battery_conf, 0.30),
        )
        _append_detections(
            detections,
            battery_result,
            args,
            allowed_classes={0},
            offset=crop_offset,
        )
        passes.append({"name": "workspace_roi_aaa_battery", "seconds": battery_duration})
        inference_mode = "split_scale_by_class"

    return {
        "protocol": "fanuc_aaa_yolo_ipc_v1",
        "request_id": request["request_id"],
        "image_shape": [full_height, full_width, 3],
        "roi_xyxy": roi,
        "model": str(args.model.resolve()),
        "settings": {
            "imgsz": args.imgsz,
            "device": str(args.device),
            "battery_conf": args.battery_conf,
            "drop_conf": args.drop_conf,
            "iou": args.iou,
            "inference_mode": inference_mode,
        },
        "inference_seconds": time.perf_counter() - started,
        "inference_passes": passes,
        "detections": detections,
        "error": None,
    }


def main() -> None:
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    args.ipc_dir = args.ipc_dir.expanduser().resolve()
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")

    request_dir = args.ipc_dir / "requests"
    response_dir = args.ipc_dir / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    print(f"[yolo-worker] loading {args.model} on {args.device}", flush=True)
    model = YOLO(str(args.model))
    print(f"[yolo-worker] ready; IPC={args.ipc_dir}", flush=True)

    processed = 0
    try:
        while True:
            did_work = False
            for request_path in sorted(request_dir.glob("*.json")):
                response_path = response_dir / request_path.name
                if response_path.exists():
                    continue
                did_work = True
                request = None
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                    response = predict(model, request, args)
                    print(
                        f"[yolo-worker] {request['request_id']}: "
                        f"detections={len(response['detections'])} "
                        f"time={response['inference_seconds']:.3f}s",
                        flush=True,
                    )
                except Exception as exc:
                    request_id = request_path.stem
                    if request is not None:
                        request_id = request.get("request_id", request_id)
                    response = {
                        "protocol": "fanuc_aaa_yolo_ipc_v1",
                        "request_id": request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "detections": [],
                    }
                    print(f"[yolo-worker] ERROR: {response['error']}", flush=True)
                atomic_json(response_path, response)
                processed += 1
                if args.once:
                    return
            if not did_work:
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print(f"[yolo-worker] stopped after {processed} request(s)", flush=True)


if __name__ == "__main__":
    main()

