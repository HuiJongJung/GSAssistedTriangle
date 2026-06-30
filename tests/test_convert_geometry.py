import unittest

import numpy as np

from gs_assisted.convert import geometry as geo


class TestQuaternion(unittest.TestCase):
    def test_identity_quat_is_identity_rotation(self):
        R = geo.quat_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0]), xp=np)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)

    def test_covariance_from_scale_identity_quat(self):
        scale = np.array([2.0, 1.0, 0.5])
        cov = geo.covariance_from_scale_quat(scale, np.array([1.0, 0, 0, 0]), xp=np)
        np.testing.assert_allclose(cov, np.diag(scale ** 2), atol=1e-7)


class TestTangentFrame(unittest.TestCase):
    def test_principal_axes_and_sigmas(self):
        cov = np.diag([4.0, 1.0, 0.25])  # variances along x, y, z
        e1, e2, normal, s1, s2 = geo.tangent_frame(cov, xp=np)
        self.assertAlmostEqual(float(s1), 2.0, places=6)   # sqrt(4)
        self.assertAlmostEqual(float(s2), 1.0, places=6)   # sqrt(1)
        # e1 along x, e2 along y, normal along z (up to sign)
        self.assertAlmostEqual(abs(float(e1[0])), 1.0, places=6)
        self.assertAlmostEqual(abs(float(e2[1])), 1.0, places=6)
        self.assertAlmostEqual(abs(float(normal[2])), 1.0, places=6)


class TestQuad(unittest.TestCase):
    def test_axis_aligned_corners(self):
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.diag([4.0, 1.0, 0.0001])
        quad = geo.gaussian_to_quad(mean, cov, size_factor=1.0, xp=np)
        self.assertEqual(quad.shape, (4, 3))
        # corners should be mean +/- s1*x +/- s2*y, s1=2, s2=1
        offsets = quad - mean
        # z component negligible (tangent plane is xy)
        np.testing.assert_allclose(offsets[:, 2], 0.0, atol=1e-2)
        # set of (|x|,|y|) offsets all equal (2,1)
        np.testing.assert_allclose(np.abs(offsets[:, 0]), 2.0, atol=1e-6)
        np.testing.assert_allclose(np.abs(offsets[:, 1]), 1.0, atol=1e-6)


class TestGaussianToTriangles(unittest.TestCase):
    def test_single_patch_fields(self):
        mean = np.zeros(3)
        cov = np.diag([1.0, 1.0, 0.01])
        rgb = np.array([0.5, 0.5, 0.5])
        patch = geo.gaussian_to_triangles(mean, cov, rgb, np.array(0.7), xp=np)
        self.assertEqual(patch["vertices"].shape, (4, 3))
        self.assertEqual(patch["triangles"].shape, (2, 3))
        self.assertEqual(patch["sh_dc"].shape, (4, 3))
        self.assertEqual(patch["opacity"].shape, (4,))
        # rgb 0.5 -> SH DC 0
        np.testing.assert_allclose(patch["sh_dc"], 0.0, atol=1e-7)
        np.testing.assert_allclose(patch["opacity"], 0.7)
        # triangles reference the 4 quad vertices and share the 0-2 diagonal
        np.testing.assert_array_equal(patch["triangles"], [[0, 1, 2], [0, 2, 3]])

    def test_batch_offsets_indices(self):
        means = np.zeros((3, 3))
        covs = np.stack([np.diag([1.0, 1.0, 0.01])] * 3)
        rgbs = np.full((3, 3), 0.5)
        ops = np.full((3,), 0.9)
        out = geo.gaussians_to_triangles(means, covs, rgbs, ops, xp=np)
        self.assertEqual(out["vertices"].shape, (12, 3))
        self.assertEqual(out["triangles"].shape, (6, 3))
        # second gaussian's triangles offset by +4, third by +8
        np.testing.assert_array_equal(out["triangles"][2], [4, 5, 6])
        np.testing.assert_array_equal(out["triangles"][4], [8, 9, 10])

    def test_empty_batch_raises(self):
        with self.assertRaises(ValueError):
            geo.gaussians_to_triangles(np.zeros((0, 3)), np.zeros((0, 3, 3)),
                                       np.zeros((0, 3)), np.zeros((0,)), xp=np)


if __name__ == "__main__":
    unittest.main()
