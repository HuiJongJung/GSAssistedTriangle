"""Triangle + residual-Gaussian image-space compositing.

This module is deliberately *array-library agnostic*: every function takes an
``xp`` argument that is either :mod:`numpy` (used by the unit tests, which run on
CPU without torch) or :mod:`torch` (used by the real training loop on GPU). Only
operations that exist with identical semantics in both libraries are used
(arithmetic, broadcasting, ``xp.where`` and the ``.clip`` method), so the exact
same code path is exercised in tests and in training.

Conventions
-----------
* ``rgb`` tensors have shape ``[3, H, W]`` and hold *straight* (non
  premultiplied) colour in ``[0, 1]``.
* ``alpha`` and ``depth`` tensors have shape ``[1, H, W]`` so they broadcast over
  the colour channel.
* ``depth`` is camera-space distance; *smaller means closer to the camera*.

Two compositing modes are supported:

``"over"``
    The original proposal ``mixed = G*aG + T*(1 - aG)``. The Gaussian branch is
    painted unconditionally on top of an opaque triangle background. Cheap, but
    ignores depth ordering (a Gaussian behind a triangle still shows through).

``"depth_aware"`` (default, recommended)
    Per-pixel two-layer *over* compositing where the front layer is chosen by
    comparing triangle and Gaussian depth. Respects occlusion at negligible
    extra cost. Non-differentiable only on the measure-zero depth-tie set, so
    gradients flow normally during joint optimisation.
"""

from __future__ import annotations

OVER = "over"
DEPTH_AWARE = "depth_aware"
MODES = (OVER, DEPTH_AWARE)

_EPS = 1e-8


def composite_over(top_rgb, top_alpha, bottom_rgb, bottom_alpha=None, *, xp, eps=_EPS):
    """Straight-alpha *over* of ``top`` on ``bottom``.

    If ``bottom_alpha`` is ``None`` the bottom layer is treated as fully opaque
    (alpha == 1), which is the common case for the opaque triangle background and
    reduces exactly to ``top_rgb*top_alpha + bottom_rgb*(1 - top_alpha)``.

    Returns ``(out_rgb, out_alpha)``.
    """
    if bottom_alpha is None:
        out_rgb = top_rgb * top_alpha + bottom_rgb * (1.0 - top_alpha)
        out_alpha = top_alpha * 0.0 + 1.0  # ones, same shape/type as top_alpha
        return out_rgb, out_alpha

    out_alpha = top_alpha + bottom_alpha * (1.0 - top_alpha)
    denom = out_alpha.clip(eps, None) if hasattr(out_alpha, "clip") else xp.clip(out_alpha, eps, None)
    out_rgb = (top_rgb * top_alpha + bottom_rgb * bottom_alpha * (1.0 - top_alpha)) / denom
    return out_rgb, out_alpha


def composite(
    t_rgb,
    t_alpha,
    g_rgb,
    g_alpha,
    *,
    xp,
    mode=DEPTH_AWARE,
    t_depth=None,
    g_depth=None,
    eps=_EPS,
):
    """Composite a triangle render ``T`` and a residual-Gaussian render ``G``.

    Parameters
    ----------
    t_rgb, g_rgb : array ``[3, H, W]``
    t_alpha, g_alpha : array ``[1, H, W]``
    mode : {"over", "depth_aware"}
    t_depth, g_depth : array ``[1, H, W]``
        Required for ``depth_aware``; ignored for ``over``.

    Returns
    -------
    dict with keys ``mixed`` (``[3,H,W]``), ``alpha`` (``[1,H,W]``) and
    ``gs_front`` (``[1,H,W]`` float mask, all ones in ``over`` mode) describing
    where the Gaussian layer ended up in front of the triangle layer.
    """
    if mode not in MODES:
        raise ValueError(f"unknown compositing mode {mode!r}; expected one of {MODES}")

    if mode == OVER:
        mixed, _ = composite_over(g_rgb, g_alpha, t_rgb, None, xp=xp, eps=eps)
        ones = g_alpha * 0.0 + 1.0
        return {"mixed": mixed, "alpha": ones, "gs_front": ones}

    if t_depth is None or g_depth is None:
        raise ValueError("depth_aware compositing requires t_depth and g_depth")

    # A Gaussian pixel is in front when it is closer (smaller depth). Where the
    # Gaussian branch is fully transparent it can never be the front layer.
    gs_front = (g_depth <= t_depth) & (g_alpha > eps)

    top_rgb = xp.where(gs_front, g_rgb, t_rgb)
    top_alpha = xp.where(gs_front, g_alpha, t_alpha)
    bot_rgb = xp.where(gs_front, t_rgb, g_rgb)
    bot_alpha = xp.where(gs_front, t_alpha, g_alpha)

    mixed, out_alpha = composite_over(top_rgb, top_alpha, bot_rgb, bot_alpha, xp=xp, eps=eps)
    # Cast boolean mask to float in an xp-agnostic way.
    gs_front_f = xp.where(gs_front, g_alpha * 0.0 + 1.0, g_alpha * 0.0)
    return {"mixed": mixed, "alpha": out_alpha, "gs_front": gs_front_f}


def gs_contribution_ratio(t_rgb, g_rgb, g_alpha, *, xp, mixed=None, mode=DEPTH_AWARE,
                          t_depth=None, g_depth=None, eps=_EPS):
    """Fraction of the composited image's radiance contributed by the Gaussian
    branch. A scalar in ``[0, 1]`` used as a diagnostic in variant ``B``.

    Defined as ``sum(|mixed - T_only|) / sum(|mixed| + eps)`` where ``T_only`` is
    the triangle-only render. This captures "how much did the Gaussians change
    the picture" without needing per-pixel ownership labels.
    """
    if mixed is None:
        mixed = composite(t_rgb, _ones_like(t_rgb, xp)[:1], g_rgb, g_alpha, xp=xp,
                          mode=mode, t_depth=t_depth, g_depth=g_depth, eps=eps)["mixed"]
    num = xp.abs(mixed - t_rgb).sum()
    den = xp.abs(mixed).sum() + eps
    return num / den


def _ones_like(arr, xp):
    return arr * 0.0 + 1.0
