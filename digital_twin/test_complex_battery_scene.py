from pathlib import Path
import unittest

import complex_battery_scene as scene


class ComplexBatterySceneTests(unittest.TestCase):
    def test_layout_is_valid_and_contains_expected_categories(self):
        self.assertTrue(scene.validate_layout())
        self.assertEqual(len(scene.BATTERIES), 5)
        self.assertEqual({item[3] for item in scene.BATTERIES}, {"amazon", "seven_eleven"})
        self.assertGreaterEqual(len(scene.CUBES), 6)
        self.assertGreaterEqual(len(scene.VIALS), 2)
        self.assertGreaterEqual(len(scene.SCREWS), 3)

    def test_description_is_json_friendly(self):
        description = scene.requested_scene_description()
        self.assertEqual(description["battery_count"], 5)
        self.assertEqual(description["mode"], "complex_clutter_single_pick_demo")

    def test_complex_clutter_uses_no_legacy_usd_references(self):
        source = Path(scene.__file__).read_text(encoding="utf-8")
        self.assertNotIn("add_sort_objects", source)
        self.assertNotIn("add_screws(stage)", source)
        self.assertNotIn("add_reference_to_stage", source)

    def test_procedural_screw_dimensions_are_positive(self):
        self.assertGreater(scene.SCREW_LENGTH_M, 0.0)
        self.assertGreater(scene.SCREW_SHANK_DIAMETER_M, 0.0)
        self.assertGreater(scene.SCREW_HEAD_DIAMETER_M, 0.0)
        self.assertGreater(scene.SCREW_HEAD_HEIGHT_M, 0.0)


if __name__ == "__main__":
    unittest.main()
