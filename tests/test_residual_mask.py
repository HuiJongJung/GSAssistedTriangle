import unittest

import numpy as np

from gs_assisted.residual_gs import residual_mask as rm


class TestResidualMask(unittest.TestCase):
    def test_photometric_residual(self):
        render = np.zeros((3, 2, 2))
        gt = np.ones((3, 2, 2))
        res = rm.photometric_residual(render, gt, xp=np)
        self.assertEqual(res.shape, (1, 2, 2))
        np.testing.assert_allclose(res, 1.0)

    def test_top_percent_selects_correct_count(self):
        vals = np.arange(100, dtype=np.float64).reshape(1, 10, 10)
        mask = rm.top_percent_mask(vals, 10.0, xp=np)
        self.assertEqual(int(mask.sum()), 10)        # top 10% of 100 pixels
        self.assertTrue(mask[0, 9, 9])               # largest value selected
        self.assertFalse(mask[0, 0, 0])              # smallest not selected

    def test_top_percent_validates_range(self):
        vals = np.ones((1, 2, 2))
        with self.assertRaises(ValueError):
            rm.top_percent_mask(vals, 0.0, xp=np)
        with self.assertRaises(ValueError):
            rm.top_percent_mask(vals, 150.0, xp=np)

    def test_low_contribution_mask(self):
        alpha = np.array([[[0.1, 0.5], [0.9, 0.34]]])
        mask = rm.low_contribution_mask(alpha, 0.35, xp=np)
        np.testing.assert_array_equal(mask, [[[True, False], [False, True]]])

    def test_candidate_mask_is_intersection(self):
        # pixel (0,0): big residual + low triangle coverage -> candidate
        render = np.zeros((3, 1, 4))
        gt = np.zeros((3, 1, 4))
        gt[:, 0, 0] = 1.0          # large residual only at col 0
        gt[:, 0, 1] = 1.0          # large residual at col 1 too
        tri_alpha = np.array([[[0.1, 0.9, 0.1, 0.1]]])  # low coverage except col 1
        cand = rm.candidate_mask(render, gt, tri_alpha, residual_top_percent=50.0,
                                 max_triangle_contribution=0.35, xp=np)
        self.assertTrue(cand[0, 0, 0])    # high residual AND low coverage
        self.assertFalse(cand[0, 0, 1])   # high residual but high coverage

    def test_accept_region_checkpoint_or_view(self):
        a = np.array([[[True, False, False]]])
        b = np.array([[[True, True, False]]])
        # checkpoint path: needs 2 repeats -> only col 0
        ck = rm.accept_region(checkpoint_masks=[a, b], min_checkpoint_repeats=2,
                              min_view_repeats=3, xp=np)
        np.testing.assert_array_equal(ck, [[[True, False, False]]])
        # OR with view path that accepts col 1 (1 view, threshold 1)
        c = np.array([[[False, True, False]]])
        both = rm.accept_region(checkpoint_masks=[a, b], view_masks=[c],
                                min_checkpoint_repeats=2, min_view_repeats=1, xp=np)
        np.testing.assert_array_equal(both, [[[True, True, False]]])

    def test_accept_region_requires_input(self):
        with self.assertRaises(ValueError):
            rm.accept_region(min_checkpoint_repeats=2, min_view_repeats=3, xp=np)


if __name__ == "__main__":
    unittest.main()
