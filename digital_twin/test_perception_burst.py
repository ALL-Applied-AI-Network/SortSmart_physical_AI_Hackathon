import unittest

from perception_burst import response_summary, select_best_response


def response(battery=None, battery_accepted=False, zone=0.82, zone_accepted=True):
    detections = []
    if zone is not None:
        detections.append(
            {"class_id": 1, "confidence": zone, "accepted": zone_accepted}
        )
    if battery is not None:
        detections.append(
            {
                "class_id": 0,
                "confidence": battery,
                "accepted": battery_accepted,
            }
        )
    return {"detections": detections}


class PerceptionBurstTests(unittest.TestCase):
    def test_summary_preserves_rejected_candidate(self):
        summary = response_summary(response(0.551, False))
        self.assertEqual(summary["battery_confidence"], 0.551)
        self.assertFalse(summary["battery_accepted"])

    def test_complete_observation_beats_higher_rejected_confidence(self):
        responses = [response(0.59, False), response(0.62, True)]
        self.assertEqual(select_best_response(responses), 1)

    def test_highest_complete_battery_confidence_wins(self):
        responses = [response(0.66, True), response(0.81, True), response(0.74, True)]
        self.assertEqual(select_best_response(responses), 1)

    def test_no_battery_uses_highest_diagnostic_candidate(self):
        responses = [response(0.42, False), response(0.55, False), response(None)]
        self.assertEqual(select_best_response(responses), 1)


if __name__ == "__main__":
    unittest.main()
