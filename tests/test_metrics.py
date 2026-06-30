import unittest

import numpy as np

from gs_assisted.diagnostics.metrics import (
    build_diagnostic_record, psnr_np, ssim_np,
)


class TestMetrics(unittest.TestCase):
    def test_psnr_identical_is_inf(self):
        img = np.random.rand(3, 8, 8)
        self.assertEqual(psnr_np(img, img), float("inf"))

    def test_psnr_known_value(self):
        a = np.zeros((1, 4, 4))
        b = np.full((1, 4, 4), 0.1)
        # mse = 0.01 -> psnr = -10*log10(0.01) = 20 dB
        self.assertAlmostEqual(psnr_np(a, b), 20.0, places=4)

    def test_ssim_identical_near_one(self):
        img = np.random.rand(3, 16, 16)
        self.assertAlmostEqual(ssim_np(img, img), 1.0, places=4)

    def test_ssim_lower_for_different(self):
        a = np.zeros((3, 16, 16))
        b = np.ones((3, 16, 16))
        self.assertLess(ssim_np(a, b), 0.5)

    def test_diagnostic_record_schema(self):
        rec = build_diagnostic_record(
            iteration=6000, triangle_count=1000, gs_count=200,
            gs_contribution_ratio=0.12, wall_clock_s=3.5,
            psnr=28.1, ssim=0.91, lpips=0.08,
        )
        self.assertEqual(rec["iteration"], 6000)
        self.assertEqual(rec["gs_count"], 200)
        self.assertEqual(rec["metrics"]["psnr"], 28.1)
        self.assertIsNone(
            build_diagnostic_record(
                iteration=1, triangle_count=1, gs_count=0,
                gs_contribution_ratio=0.0, wall_clock_s=0.0,
            )["metrics"]["psnr"]
        )


if __name__ == "__main__":
    unittest.main()
