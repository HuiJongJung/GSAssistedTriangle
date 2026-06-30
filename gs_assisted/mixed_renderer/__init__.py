"""Mixed Triangle + residual-Gaussian renderer (GPU / server only).

Combines the upstream Triangle Splatting+ render dict with a
:class:`gs_assisted.gs_backend.GaussianBranch` render using the array-agnostic
compositing core (:mod:`gs_assisted.compositing`), driven here with ``xp=torch``
so training uses the exact code path the CPU unit tests validate with numpy.
"""

from __future__ import annotations

import torch

from gs_assisted import compositing


def render_mixed(triangle_render_pkg, gs_branch, cam, *,
                 mode=compositing.DEPTH_AWARE):
    """Render the mixed image for one camera.

    Parameters
    ----------
    triangle_render_pkg : dict
        Output of the upstream ``triangle_renderer.render`` (needs keys
        ``render``, ``rend_alpha``, ``surf_depth``).
    gs_branch : GaussianBranch
    cam : a TS+ camera
    mode : compositing mode ("depth_aware" default, or "over").

    Returns dict with ``mixed`` [3,H,W], ``t_only`` [3,H,W], ``g_only`` [3,H,W],
    ``gs_alpha`` [1,H,W], ``gs_front`` [1,H,W].
    """
    t_rgb = triangle_render_pkg["render"]
    t_alpha = triangle_render_pkg["rend_alpha"]
    t_depth = triangle_render_pkg["surf_depth"]

    gs = gs_branch.render(cam)
    g_rgb, g_alpha, g_depth = gs["rgb"], gs["alpha"], gs["depth"]

    out = compositing.composite(
        t_rgb, t_alpha, g_rgb, g_alpha,
        xp=torch, mode=mode, t_depth=t_depth, g_depth=g_depth,
    )
    return {
        "mixed": out["mixed"],
        "t_only": t_rgb,
        "g_only": g_rgb,
        "gs_alpha": g_alpha,
        "gs_front": out["gs_front"],
    }


def composite_triangle_gaussian(triangle_rgb, gaussian_rgb, gaussian_alpha):
    """Backwards-compatible image-space *over* compositing (no depth).

    Kept for the original ``--composite-mode over`` path and any caller relying
    on the previous signature.
    """
    if gaussian_alpha.ndim == 2:
        gaussian_alpha = gaussian_alpha.unsqueeze(0)
    return gaussian_rgb * gaussian_alpha + triangle_rgb * (1.0 - gaussian_alpha)
