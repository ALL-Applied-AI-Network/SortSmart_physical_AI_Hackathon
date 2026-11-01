import unittest

from wrist_scan_planning import scan_tip_targets_m


class WristScanPlanningTests(unittest.TestCase):
    def test_grid_is_fixed_safe_and_covers_both_workspace_sides(self):
        targets = scan_tip_targets_m()
        self.assertEqual(len(targets), 4)
        self.assertEqual(len(set(targets)), 4)
        self.assertTrue(all(target[2] >= 0.100 for target in targets))
        self.assertLess(min(target[1] for target in targets), -0.10)
        self.assertGreater(max(target[1] for target in targets), 0.10)
        self.assertLess(min(target[0] for target in targets), 0.25)
        self.assertGreater(max(target[0] for target in targets), 0.30)


if __name__ == "__main__":
    unittest.main()
