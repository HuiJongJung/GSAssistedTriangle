"""Numpy image metrics and diagnostic-record assembly.

These are used by the harness for CPU unit tests and for any non-GPU bookkeeping.
The real training/eval loop reuses the upstream torch ``psnr``/``ssim`` and
``lpips`` implementations on GPU; the numpy SSIM here matches their windowed
formulation (11x11 Gaussian window, sigma 1.5) closely enough for diagnostics
but is not bit-identical.

Image arrays are ``[3, H, W]`` (or ``[1, H, W]``) floats in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np

VARIANTS = (
    "A_baseline_triangle_only",
    "B_ours_mixed_triangle_gs",
    "C_ours_converted_triangle_only",
)


def psnr_np(img, gt, max_val=1.0):
    """Peak signal-to-noise ratio in dB."""
    mse = np.mean((img.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def _gaussian_kernel1d(window_size=11, sigma=1.5):
    half = (window_size - 1) / 2.0
    x = np.arange(window_size) - half
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _blur(img, kernel):
    """Separable reflect-padded convolution over the last two axes."""
    pad = len(kernel) // 2
    out = np.empty_like(img, dtype=np.float64)
    for c in range(img.shape[0]):
        ch = img[c].astype(np.float64)
        ch = np.pad(ch, ((pad, pad), (0, 0)), mode="reflect")
        ch = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, ch)
        ch = np.pad(ch, ((0, 0), (pad, pad)), mode="reflect")
        ch = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, ch)
        out[c] = ch
    return out


def ssim_np(img, gt, window_size=11, sigma=1.5, max_val=1.0):
    """Mean structural similarity over a Gaussian window (numpy)."""
    k = _gaussian_kernel1d(window_size, sigma)
    img = img.astype(np.float64)
    gt = gt.astype(np.float64)
    mu1, mu2 = _blur(img, k), _blur(gt, k)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = _blur(img * img, k) - mu1_sq
    sigma2_sq = _blur(gt * gt, k) - mu2_sq
    sigma12 = _blur(img * gt, k) - mu12
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def build_diagnostic_record(*, iteration, triangle_count, gs_count,
                            gs_contribution_ratio, wall_clock_s,
                            psnr=None, ssim=None, lpips=None, extra=None):
    """Assemble one checkpoint's diagnostic dict in a stable schema.

    Matches the "Diagnostics" list in ``EXPERIMENT.md`` (scalar fields; the image
    maps are written alongside as files by the render driver).
    """
    record = {
        "iteration": int(iteration),
        "triangle_count": int(triangle_count),
        "gs_count": int(gs_count),
        "gs_contribution_ratio": float(gs_contribution_ratio),
        "wall_clock_s": float(wall_clock_s),
        "metrics": {
            "psnr": None if psnr is None else float(psnr),
            "ssim": None if ssim is None else float(ssim),
            "lpips": None if lpips is None else float(lpips),
        },
    }
    if extra:
        record.update(extra)
    return record
