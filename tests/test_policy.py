import unittest

from gs_assisted.residual_gs.policy import (
    ResidualGsPolicy, can_insert_more, clamp_insertion_count, remaining_capacity,
)


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = ResidualGsPolicy(max_insert_per_event=5000, max_total_gs=100000)

    def test_defaults_match_experiment_spec(self):
        p = ResidualGsPolicy()
        self.assertEqual(p.residual_top_percent, 10.0)
        self.assertEqual(p.max_triangle_contribution, 0.35)
        self.assertEqual(p.min_checkpoint_repeats, 2)
        self.assertEqual(p.min_view_repeats, 3)
        self.assertEqual(p.max_insert_per_event, 5000)
        self.assertEqual(p.max_total_gs, 100000)

    def test_remaining_capacity(self):
        self.assertEqual(remaining_capacity(0, self.policy), 100000)
        self.assertEqual(remaining_capacity(99000, self.policy), 1000)
        self.assertEqual(remaining_capacity(100000, self.policy), 0)
        self.assertEqual(remaining_capacity(200000, self.policy), 0)  # never negative

    def test_can_insert_more(self):
        self.assertTrue(can_insert_more(0, self.policy))
        self.assertFalse(can_insert_more(100000, self.policy))

    def test_clamp_per_event_cap(self):
        self.assertEqual(clamp_insertion_count(9000, 0, self.policy), 5000)

    def test_clamp_global_cap(self):
        # 800 left globally, request 5000 -> 800
        self.assertEqual(clamp_insertion_count(5000, 99200, self.policy), 800)

    def test_clamp_zero_at_cap(self):
        self.assertEqual(clamp_insertion_count(5000, 100000, self.policy), 0)

    def test_clamp_rejects_negative(self):
        with self.assertRaises(ValueError):
            clamp_insertion_count(-1, 0, self.policy)


if __name__ == "__main__":
    unittest.main()
