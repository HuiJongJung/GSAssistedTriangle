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
    ordered = getattr(ordered, "values", ordered)
    k = int(np.floor((1.0 - percent / 100.0) * n))
    k = max(0, min(k, n - 1))
    threshold = ordered[k]
    return value_map >= threshold


def low_contribution_mask(triangle_alpha, max_triangle_contribution, *, xp=np):
    """Pixels where the triangle branch is under-covering, shape ``[1, H, W]``."""
    return triangle_alpha < max_triangle_contribution


def candidate_mask(render_rgb, gt_rgb, triangle_alpha, *, residual_top_percent,
                   max_triangle_contribution, xp=np):
    """A single view/checkpoint's candidate residual region (legacy alpha gate).

    Intersection of "high photometric residual" and "low triangle contribution".
    Kept for reference/tests; the training loop now uses
    :func:`geometry_candidate_mask`, because the ``alpha < t`` gate goes dead once
    Triangle Splatting+ saturates alpha ~ 1 everywhere (the residual Gaussians end
    up recruited only transiently, in slow-converging background that triangles
    later cover -- the "temporal recruitment mismatch").
    """
    res = photometric_residual(render_rgb, gt_rgb, xp=xp)
    hi_res = top_percent_mask(res, residual_top_percent, xp=xp)
    low_tri = low_contribution_mask(triangle_alpha, max_triangle_contribution, xp=xp)
    return hi_res & low_tri


def normal_disagreement(rend_normal, gt_normal, *, xp=np):
    """Per-pixel normal mismatch in ``[0, 1]``, shape ``[1, H, W]``.

    ``rend_normal`` / ``gt_normal`` are ``[3, H, W]`` (assumed ~unit length).
    ``0.5 * (1 - cos)`` is high where the rendered surface normal disagrees with
    the monocular (metric3d) prior -- i.e. where the triangle geometry is *wrong*,
    not merely textured. Unlike triangle alpha, this stays informative after the
    opaque triangles have saturated alpha ~ 1, so it is a geometry-failure signal
    rather than a coverage signal.
    """
    dot = (rend_normal * gt_normal).sum(0, keepdims=True)
    dis = (1.0 - dot) * 0.5
    return dis.clip(0.0, 1.0) if hasattr(dis, "clip") else xp.clip(dis, 0.0, 1.0)


def geometry_candidate_mask(render_rgb, gt_rgb, rend_normal, gt_normal, *,
                            residual_top_percent, normal_top_percent,
                            depth_instability=None, depth_top_percent=None, xp=np):
    """Candidate residual region driven by *geometry-failure* evidence.

    Intersection of:
      * high photometric residual (top ``residual_top_percent``), and
      * high normal disagreement vs the metric3d prior (top ``normal_top_percent``),
      * optionally high local depth instability (top ``depth_top_percent``), a
        precomputed ``[1, H, W]`` variance map supplied by the training loop.

    All signals are top-percent thresholded, hence scale-free. This replaces the
    ``residual ∩ (alpha < t)`` gate so recruitment targets genuine geometry/topology
    failure instead of fuzzy appearance residual (foliage) that should stay Gaussian
    and must *not* be promoted to triangles.
    """
    res = photometric_residual(render_rgb, gt_rgb, xp=xp)
    mask = top_percent_mask(res, residual_top_percent, xp=xp)
    dis = normal_disagreement(rend_normal, gt_normal, xp=xp)
    mask = mask & top_percent_mask(dis, normal_top_percent, xp=xp)
    if depth_instability is not None and depth_top_percent is not None:
        mask = mask & top_percent_mask(depth_instability, depth_top_percent, xp=xp)
    return mask


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
