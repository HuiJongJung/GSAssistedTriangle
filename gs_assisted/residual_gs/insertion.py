"""Residual-Gaussian insertion: residual pixels -> initial Gaussian params.

GPU / server only (needs torch and TS+ cameras). The *count* and *capacity*
decisions live in :mod:`gs_assisted.residual_gs.policy` (pure, unit-tested); this
module turns an accepted residual mask into concrete world-space Gaussians.

Pipeline per insertion event:
    accepted mask + triangle depth + residual map
      -> pick up to ``clamp_insertion_count`` highest-residual masked pixels
      -> back-project those pixels to world points using the triangle depth
      -> initialise Gaussians there (isotropic scale, identity rotation,
         colour seeded from the ground-truth pixel, modest opacity).

NOTE (verify on server): back-projection assumes ``surf_depth`` is camera-space
z and that ``viewmat = cam.world_view_transform.T`` is world-to-camera. The smoke
run should confirm inserted Gaussians land on the residual surface.
"""

from __future__ import annotations

import math

import torch

from gs_assisted.residual_gs.policy import ResidualGsPolicy, clamp_insertion_count


def select_pixels(accepted_mask, residual_map, count):
    """Indices of the ``count`` highest-residual pixels inside ``accepted_mask``.

    ``accepted_mask`` and ``residual_map`` are ``[1, H, W]`` torch tensors.
    Returns ``(vs, us)`` long tensors of pixel rows/cols (possibly fewer than
    ``count`` if the mask is small).
    """
    mask = accepted_mask[0].bool()
    res = residual_map[0]
    flat_idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    if flat_idx.numel() == 0 or count <= 0:
        empty = torch.empty(0, dtype=torch.long, device=accepted_mask.device)
        return empty, empty
    vals = res.reshape(-1)[flat_idx]
    k = min(int(count), flat_idx.numel())
    top = torch.topk(vals, k).indices
    chosen = flat_idx[top]
    W = accepted_mask.shape[-1]
    return chosen // W, chosen % W


def backproject(cam, vs, us, depths):
    """Back-project pixel coords + camera-space depth to world xyz ``[N, 3]``."""
    W = int(cam.image_width)
    H = int(cam.image_height)
    fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
    cx, cy = W / 2.0, H / 2.0
    z = depths.float()
    x = (us.float() - cx) / fx * z
    y = (vs.float() - cy) / fy * z
    cam_pts = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)  # [N,4]
    w2c = cam.world_view_transform.transpose(0, 1).to(cam_pts.device).float()
    c2w = torch.inverse(w2c)
    world = (c2w @ cam_pts.T).T[:, :3]
    return world


def init_params_from_pixels(world_pts, colors, *, init_scale, opacity=0.1, eps=1e-6):
    """Build raw GaussianBranch parameters for newly inserted Gaussians.

    ``colors`` are target RGB in ``[0, 1]`` (e.g. sampled from the GT image).
    Scales are isotropic ``init_scale`` (world units); rotation is identity;
    opacity/colour are stored as logits to match the branch activations.
    """
    n = world_pts.shape[0]
    dev = world_pts.device
    means = world_pts
    scales_log = torch.full((n, 3), math.log(init_scale), device=dev)
    quats = torch.zeros((n, 4), device=dev)
    quats[:, 0] = 1.0  # identity (w=1)
    op = torch.full((n,), float(opacity), device=dev).clamp(eps, 1 - eps)
    opacities_logit = torch.log(op / (1 - op))
    c = colors.clamp(eps, 1 - eps)
    colors_logit = torch.log(c / (1 - c))
    return {
        "means": means,
        "scales_log": scales_log,
        "quats": quats,
        "opacities_logit": opacities_logit,
        "colors_logit": colors_logit,
    }


def build_insertion(cam, accepted_mask, residual_map, triangle_depth, gt_image,
                    *, current_gs_count, policy: ResidualGsPolicy,
                    init_scale, requested=None):
    """End-to-end one-event insertion params, respecting capacity limits.

    Returns ``None`` if nothing can/should be inserted, otherwise the dict
    accepted by :meth:`GaussianBranch.append` / its constructor.
    """
    requested = int(accepted_mask.sum().item()) if requested is None else requested
    n = clamp_insertion_count(requested, current_gs_count, policy)
    if n <= 0:
        return None
    vs, us = select_pixels(accepted_mask, residual_map, n)
    if vs.numel() == 0:
        return None
    depths = triangle_depth[0][vs, us]
    world = backproject(cam, vs, us, depths)
    colors = gt_image[:, vs, us].transpose(0, 1)  # [N,3]
    return init_params_from_pixels(world, colors, init_scale=init_scale)
