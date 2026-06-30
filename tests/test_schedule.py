import unittest

from gs_assisted.schedule import (
    default_gs_start_iter, percent_save_iterations, resolve_save_iterations,
)


class TestSchedule(unittest.TestCase):
    def test_30k_default_grid(self):
        pts = percent_save_iterations(30000, 10)
        self.assertEqual(pts, [3000, 6000, 9000, 12000, 15000,
                               18000, 21000, 24000, 27000, 30000])

    def test_last_point_is_total(self):
        pts = percent_save_iterations(333, 10)
        self.assertEqual(pts[-1], 333)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            percent_save_iterations(0, 10)
        with self.assertRaises(ValueError):
            percent_save_iterations(100, 0)
        with self.assertRaises(ValueError):
            percent_save_iterations(100, 101)

    def test_default_gs_start(self):
        self.assertEqual(default_gs_start_iter(), 5000)
        self.assertEqual(default_gs_start_iter(start_pruning=6000), 7000)

    def test_resolve_adds_offgrid_gs_start(self):
        pts = resolve_save_iterations(30000, 10, gs_start_iter=5000)
        self.assertIn(5000, pts)
        self.assertEqual(sorted(pts), pts)

    def test_resolve_no_duplicate_when_ongrid(self):
        pts = resolve_save_iterations(30000, 10, gs_start_iter=6000)
        self.assertEqual(pts.count(6000), 1)

    def test_resolve_ignores_out_of_range(self):
        pts = resolve_save_iterations(30000, 10, gs_start_iter=40000)
        self.assertNotIn(40000, pts)


if __name__ == "__main__":
    unittest.main()
