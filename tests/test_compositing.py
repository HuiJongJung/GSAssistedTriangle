import unittest

import numpy as np

from gs_assisted.compositing import (
    OVER, DEPTH_AWARE, composite, composite_over, gs_contribution_ratio,
)


def _img(value, shape=(3, 2, 2)):
    return np.full(shape, value, dtype=np.float64)


class TestCompositeOver(unittest.TestCase):
    def test_opaque_background_formula(self):
        t = _img(0.2)
        g = _img(0.8)
        a = np.full((1, 2, 2), 0.5)
        out, out_a = composite_over(g, a, t, None, xp=np)
        np.testing.assert_allclose(out, 0.8 * 0.5 + 0.2 * 0.5)
        np.testing.assert_allclose(out_a, 1.0)

    def test_two_layer_alpha(self):
        top = _img(1.0)
        bot = _img(0.0)
        a_top = np.full((1, 2, 2), 0.25)
        a_bot = np.full((1, 2, 2), 1.0)
        out, out_a = composite_over(top, a_top, bot, a_bot, xp=np)
        np.testing.assert_allclose(out_a, 1.0)
        np.testing.assert_allclose(out, 0.25)  # (1*.25 + 0*1*.75)/1


class TestComposite(unittest.TestCase):
    def test_over_mode_matches_proposed_formula(self):
        t, g = _img(0.2), _img(0.9)
        ag = np.full((1, 2, 2), 0.3)
        res = composite(t, np.ones((1, 2, 2)), g, ag, xp=np, mode=OVER)
        np.testing.assert_allclose(res["mixed"], 0.9 * 0.3 + 0.2 * 0.7)

    def test_depth_aware_gs_fully_in_front(self):
        t, g = _img(0.2), _img(0.9)
        at = np.ones((1, 2, 2))
        ag = np.ones((1, 2, 2))
        td = np.full((1, 2, 2), 5.0)
        gd = np.full((1, 2, 2), 1.0)  # gaussian closer
        res = composite(t, at, g, ag, xp=np, mode=DEPTH_AWARE, t_depth=td, g_depth=gd)
        np.testing.assert_allclose(res["mixed"], 0.9)
        np.testing.assert_allclose(res["gs_front"], 1.0)

    def test_depth_aware_gs_occluded_behind_opaque_triangle(self):
        t, g = _img(0.2), _img(0.9)
        at = np.ones((1, 2, 2))           # opaque triangle
        ag = np.ones((1, 2, 2))
        td = np.full((1, 2, 2), 1.0)      # triangle closer
        gd = np.full((1, 2, 2), 5.0)
        res = composite(t, at, g, ag, xp=np, mode=DEPTH_AWARE, t_depth=td, g_depth=gd)
        np.testing.assert_allclose(res["mixed"], 0.2)  # gaussian hidden
        np.testing.assert_allclose(res["gs_front"], 0.0)

    def test_depth_aware_partial_alpha(self):
        t, g = _img(0.0), _img(1.0)
        at = np.ones((1, 2, 2))
        ag = np.full((1, 2, 2), 0.5)
        td = np.full((1, 2, 2), 5.0)
        gd = np.full((1, 2, 2), 1.0)
        res = composite(t, at, g, ag, xp=np, mode=DEPTH_AWARE, t_depth=td, g_depth=gd)
        np.testing.assert_allclose(res["mixed"], 0.5)  # 0.5*g + 0.5*t

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            composite(_img(0.1), np.ones((1, 2, 2)), _img(0.1), np.ones((1, 2, 2)),
                      xp=np, mode="bogus")

    def test_depth_aware_requires_depth(self):
        with self.assertRaises(ValueError):
            composite(_img(0.1), np.ones((1, 2, 2)), _img(0.1), np.ones((1, 2, 2)),
                      xp=np, mode=DEPTH_AWARE)


class TestContributionRatio(unittest.TestCase):
    def test_zero_when_gaussian_matches_triangle(self):
        t = _img(0.4)
        g = _img(0.4)
        ag = np.ones((1, 2, 2))
        mixed = composite(t, np.ones((1, 2, 2)), g, ag, xp=np, mode=OVER)["mixed"]
        r = gs_contribution_ratio(t, g, ag, xp=np, mixed=mixed)
        self.assertAlmostEqual(float(r), 0.0, places=7)

    def test_positive_when_gaussian_differs(self):
        t = _img(0.1)
        g = _img(0.9)
        ag = np.ones((1, 2, 2))
        mixed = composite(t, np.ones((1, 2, 2)), g, ag, xp=np, mode=OVER)["mixed"]
        r = gs_contribution_ratio(t, g, ag, xp=np, mixed=mixed)
        self.assertGreater(float(r), 0.0)


if __name__ == "__main__":
    unittest.main()
