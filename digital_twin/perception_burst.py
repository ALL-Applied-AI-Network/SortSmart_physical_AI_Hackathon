"""Selection helpers for auditable multi-frame perception bursts."""

from __future__ import annotations


def _best_detection(response, class_id):
    candidates = [
        item
        for item in response.get("detections", [])
        if item.get("class_id") == class_id
    ]
    return max(candidates, key=lambda item: float(item.get("confidence", 0.0)), default=None)


def response_summary(response):
    battery = _best_detection(response, 0)
    zone = _best_detection(response, 1)
    return {
        "battery_confidence": None if battery is None else float(battery["confidence"]),
        "battery_accepted": bool(battery and battery.get("accepted")),
        "drop_zone_confidence": None if zone is None else float(zone["confidence"]),
        "drop_zone_accepted": bool(zone and zone.get("accepted")),
    }


def response_rank(response):
    """Prefer a complete observation, then battery acceptance/confidence."""
    summary = response_summary(response)
    battery_confidence = summary["battery_confidence"] or 0.0
    drop_zone_confidence = summary["drop_zone_confidence"] or 0.0
    complete = summary["battery_accepted"] and summary["drop_zone_accepted"]
    return (
        int(complete),
        int(summary["battery_accepted"]),
        battery_confidence,
        int(summary["drop_zone_accepted"]),
        drop_zone_confidence,
    )


def select_best_response(responses):
    if not responses:
        raise ValueError("responses must not be empty")
    return max(range(len(responses)), key=lambda index: response_rank(responses[index]))

