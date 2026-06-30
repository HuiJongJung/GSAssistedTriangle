"""Residual-region detection for residual-Gaussian insertion.

Implements the candidate-region rules from ``EXPERIMENT.md``:

* photometric residual in the top ``residual_top_percent`` of pixels;
* triangle alpha/contribution below ``max_triangle_contribution``;
* repeated evidence: a region must recur in at least ``min_checkpoint_repeats``
  recent checkpoints *or* in at least ``min_view_repeats`` views before it is
  accepted for insertion.

All functions are array-library agnostic (pass ``xp=numpy`` for the CPU tests or
``xp=torch`` in training) and operate on maps shaped ``[1, H, W]`` (alpha/
residual) or ``[3, H, W]`` (rgb). Booleans are returned as native boolean arrays
of the chosen library.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def photometric_residual(render_rgb, gt_rgb, *, xp=np):
    """Per-pixel L1 photometric residual, shape ``[1, H, W]``."""
    res = xp.abs(render_rgb - gt_rgb)
    # mean over the colour channel (axis 0), keep a leading singleton dim.
    return res.sum(0, keepdims=True) / render_rgb.shape[0]


def top_percent_mask(value_map, percent, *, xp=np):
    """Boolean mask selecting the highest ``percent`` percent of ``value_map``.

    Threshold is found by sorting (works identically in numpy and torch), so this
    avoids the ``percentile`` vs ``quantile`` API mismatch between the libraries.
    """
    if not (0.0 < percent <= 100.0):
        raise ValueError("percent must be in (0, 100]")
    flat = value_map.reshape(-1)
    n = flat.shape[0]
    ordered = xp.sort(flat)
    k = int(np.floor((1.0 - percent / 100.0) * n))
    k = max(0, min(k, n - 1))
    threshold = ordered[k]
    return value_map >= threshold


def low_contribution_mask(triangle_alpha, max_triangle_contribution, *, xp=np):
    """Pixels where the triangle branch is under-covering, shape ``[1, H, W]``."""
    return triangle_alpha < max_triangle_contribution


def candidate_mask(render_rgb, gt_rgb, triangle_alpha, *, residual_top_percent,
                   max_triangle_contribution, xp=np):
    """A single view/checkpoint's candidate residual region.

    Intersection of "high photometric residual" and "low triangle contribution".
    """
    res = photometric_residual(render_rgb, gt_rgb, xp=xp)
    hi_res = top_percent_mask(res, residual_top_percent, xp=xp)
    low_tri = low_contribution_mask(triangle_alpha, max_triangle_contribution, xp=xp)
    return hi_res & low_tri


def accumulate(masks, *, xp=np):
    """Sum a list of boolean ``[1, H, W]`` masks into an integer hit-count map."""
    if len(masks) == 0:
        raise ValueError("need at least one mask to accumulate")
    total = masks[0] * 0
    # Convert via arithmetic so it works for numpy bool and torch bool alike.
    for m in masks:
        total = total + xp.where(m, m * 0 + 1, m * 0)
    return total


def repeated_region_mask(masks, min_repeats, *, xp=np):
    """Regions present in at least ``min_repeats`` of the supplied masks.

    Use with a list of per-checkpoint masks (``min_repeats=min_checkpoint_repeats``)
    or per-view masks (``min_repeats=min_view_repeats``).
    """
    counts = accumulate(masks, xp=xp)
    return counts >= min_repeats


def accept_region(checkpoint_masks=None, view_masks=None, *, min_checkpoint_repeats,
                  min_view_repeats, xp=np):
    """Final acceptance mask: recurs across enough checkpoints OR enough views.

    At least one of ``checkpoint_masks`` / ``view_masks`` must be provided.
    """
    accepted = None
    if checkpoint_masks:
        accepted = repeated_region_mask(checkpoint_masks, min_checkpoint_repeats, xp=xp)
    if view_masks:
        by_view = repeated_region_mask(view_masks, min_view_repeats, xp=xp)
        accepted = by_view if accepted is None else (accepted | by_view)
    if accepted is None:
        raise ValueError("provide checkpoint_masks and/or view_masks")
    return accepted
