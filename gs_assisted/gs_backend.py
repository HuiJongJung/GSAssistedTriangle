"""Residual-Gaussian branch and gsplat rasterisation wrapper (GPU / server only).

This module is *not* exercised by the CPU unit tests; it requires ``torch`` and
``gsplat`` and a CUDA device. It is the only place that talks to gsplat, so the
backend can be swapped (e.g. for ``diff-gaussian-rasterization``) by
reimplementing :func:`rasterize` and keeping the same return contract.

Return contract for a single camera render:
    dict(rgb=[3,H,W], alpha=[1,H,W], depth=[1,H,W])  # straight alpha, depth>0

Camera convention NOTE (verify on the server with the smoke run):
    Triangle Splatting+ cameras store ``world_view_transform`` already
    transposed for the CUDA rasteriser (``getWorld2View2(...).T``). gsplat wants
    a plain world-to-camera matrix with points as column vectors, so we pass
    ``viewmat = cam.world_view_transform.T``. Intrinsics are reconstructed from
    the field of view. If the residual branch renders mirrored/blank in the
    smoke test, this transpose is the first thing to check.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:  # gsplat is only available on the GPU server
    from gsplat import rasterization as _gsplat_rasterization
except Exception:  # pragma: no cover - import guarded so the module is importable
    _gsplat_rasterization = None


def camera_to_gsplat(cam, device="cuda"):
    """Derive ``(viewmat[1,4,4], K[1,3,3], width, height)`` from a TS+ camera."""
    W = int(cam.image_width)
    H = int(cam.image_height)
    fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
    K = torch.tensor([[fx, 0.0, W / 2.0],
                      [0.0, fy, H / 2.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    viewmat = cam.world_view_transform.transpose(0, 1).to(device).float()
    return viewmat.unsqueeze(0), K.unsqueeze(0), W, H


class GaussianBranch(nn.Module):
    """A small, growable set of residual 3D Gaussians with RGB colour.

    Parameters are raw (unconstrained); activations map them to valid ranges:
      means       : world-space xyz (raw)
      scales      : per-axis std = exp(_scales)
      quats       : rotation, normalised inside gsplat
      opacities   : alpha = sigmoid(_opacities)
      colors      : rgb = sigmoid(_colors)  (view-independent residual colour)
    """

    def __init__(self, means, scales_log, quats, opacities_logit, colors_logit,
                 device="cuda"):
        super().__init__()
        self._means = nn.Parameter(means.to(device).float())
        self._scales = nn.Parameter(scales_log.to(device).float())
        self._quats = nn.Parameter(quats.to(device).float())
        self._opacities = nn.Parameter(opacities_logit.to(device).float())
        self._colors = nn.Parameter(colors_logit.to(device).float())

    # --- activations -----------------------------------------------------
    @property
    def count(self):
        return self._means.shape[0]

    @property
    def scales(self):
        return torch.exp(self._scales)

    @property
    def opacities(self):
        return torch.sigmoid(self._opacities)

    @property
    def colors(self):
        return torch.sigmoid(self._colors)

    def covariances(self):
        """Per-Gaussian 3x3 covariance ``R diag(scale^2) R^T`` (for conversion)."""
        q = self._quats / (self._quats.norm(dim=-1, keepdim=True) + 1e-12)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ], -2)
        s = self.scales
        S2 = torch.diag_embed(s * s)
        return R @ S2 @ R.transpose(-1, -2)

    # --- optimisation helpers -------------------------------------------
    def parameters_for_optimizer(self, lr_means, lr_scales, lr_quats,
                                 lr_opacities, lr_colors):
        return [
            {"params": [self._means], "lr": lr_means, "name": "gs_means"},
            {"params": [self._scales], "lr": lr_scales, "name": "gs_scales"},
            {"params": [self._quats], "lr": lr_quats, "name": "gs_quats"},
            {"params": [self._opacities], "lr": lr_opacities, "name": "gs_opacities"},
            {"params": [self._colors], "lr": lr_colors, "name": "gs_colors"},
        ]

    @torch.no_grad()
    def prune(self, keep_mask):
        """Drop Gaussians where ``keep_mask`` (``[N]`` bool) is False.

        Used to cull residual Gaussians whose opacity has collapsed -- i.e. the
        region got taken over by triangles (or was never real geometry), so the
        "temporary holder" should be removed rather than left as suppressed
        residue. Rebuild the optimiser afterwards (Parameters are replaced).
        """
        self._means = nn.Parameter(self._means[keep_mask])
        self._scales = nn.Parameter(self._scales[keep_mask])
        self._quats = nn.Parameter(self._quats[keep_mask])
        self._opacities = nn.Parameter(self._opacities[keep_mask])
        self._colors = nn.Parameter(self._colors[keep_mask])

    @torch.no_grad()
    def append(self, means, scales_log, quats, opacities_logit, colors_logit):
        """Grow the branch in place (used by scheduled insertion events).

        NOTE: appending replaces the Parameter tensors, so rebuild the optimiser
        (or use a fresh param group) after calling this. The training loop calls
        :func:`rebuild_optimizer` for that reason.
        """
        dev = self._means.device
        self._means = nn.Parameter(torch.cat([self._means, means.to(dev).float()], 0))
        self._scales = nn.Parameter(torch.cat([self._scales, scales_log.to(dev).float()], 0))
        self._quats = nn.Parameter(torch.cat([self._quats, quats.to(dev).float()], 0))
        self._opacities = nn.Parameter(torch.cat([self._opacities, opacities_logit.to(dev).float()], 0))
        self._colors = nn.Parameter(torch.cat([self._colors, colors_logit.to(dev).float()], 0))

    # --- rendering -------------------------------------------------------
    def render(self, cam, *, near_plane=0.01, far_plane=1e10):
        """Render this branch from ``cam``; returns ``dict(rgb, alpha, depth)``."""
        if _gsplat_rasterization is None:
            raise RuntimeError("gsplat is not installed; run on the GPU server")
        viewmat, K, W, H = camera_to_gsplat(cam, device=self._means.device)
        colors, alphas, _meta = _gsplat_rasterization(
            means=self._means,
            quats=self._quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB+ED",  # RGB plus expected depth in the last channel
        )
        img = colors[0]                      # [H, W, 4]
        rgb = img[..., :3].permute(2, 0, 1)  # [3, H, W]
        depth = img[..., 3:4].permute(2, 0, 1)  # [1, H, W]
        alpha = alphas[0].permute(2, 0, 1)   # [1, H, W]
        return {"rgb": rgb, "alpha": alpha, "depth": depth}
