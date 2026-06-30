"""Geometry for converting a residual Gaussian into local triangle patches.

A Gaussian is mapped to a flat quad lying in its tangent plane (the plane
spanned by its two largest principal axes), which is then split into two
triangles. Colour and opacity are carried over to every vertex so the converted
triangles initialise close to the Gaussian they replace.

Everything here is pure geometry and runs under numpy (unit tests) or torch
(training) via the ``xp`` argument. Inputs/outputs are plain arrays; mapping the
result onto the ``TriangleModel`` parameter tensors (SH features, opacity logits,
optimiser state) is done by the torch-only conversion driver.
"""

from __future__ import annotations

import numpy as np

# SH band-0 constant, matching utils/sh_utils.RGB2SH in the upstream baseline.
_SH_C0 = 0.28209479177387814

# Local vertex order produced by :func:`gaussian_to_quad`:
#   0: -h1 -h2   1: +h1 -h2   2: +h1 +h2   3: -h1 +h2
# Two triangles tile the quad with a shared 0-2 diagonal.
QUAD_TRIANGLES = ((0, 1, 2), (0, 2, 3))


def quat_to_rotmat(quat, *, xp=np):
    """Unit quaternion ``(w, x, y, z)`` -> 3x3 rotation matrix."""
    q = quat / (xp.sqrt((quat * quat).sum()) + 1e-12)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return xp.stack([
        xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        xp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        xp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def covariance_from_scale_quat(scale, quat, *, xp=np):
    """Build a 3x3 covariance ``R diag(scale^2) R^T`` from gsplat-style params.

    ``scale`` holds per-axis standard deviations (the gsplat/3DGS convention).
    """
    rot = quat_to_rotmat(quat, xp=xp)
    s2 = scale * scale
    # R @ diag(s2) @ R^T
    return (rot * s2) @ rot.T


def tangent_frame(cov, *, xp=np):
    """Return ``(e1, e2, normal, sigma1, sigma2)`` for a covariance matrix.

    ``e1``/``e2`` are the unit principal axes of the two largest eigenvalues
    (the tangent plane), ``normal`` the smallest-eigenvalue axis, and
    ``sigma1 >= sigma2`` the corresponding standard deviations.
    """
    # eigh returns ascending eigenvalues in both numpy and torch.
    evals, evecs = xp.linalg.eigh(cov)
    normal = evecs[:, 0]
    e2 = evecs[:, 1]
    e1 = evecs[:, 2]
    # Guard against tiny negative eigenvalues from numerical error.
    lam1 = evals[2]
    lam2 = evals[1]
    sigma1 = xp.sqrt(lam1.clip(0, None) if hasattr(lam1, "clip") else xp.clip(lam1, 0, None))
    sigma2 = xp.sqrt(lam2.clip(0, None) if hasattr(lam2, "clip") else xp.clip(lam2, 0, None))
    return e1, e2, normal, sigma1, sigma2


def gaussian_to_quad(mean, cov, *, size_factor=3.0, xp=np):
    """Four tangent-plane quad corners (shape ``[4, 3]``) for one Gaussian.

    ``size_factor`` scales the principal standard deviations into the quad's
    half-extents; ``3.0`` covers ~99.7% of the Gaussian's mass along each axis.
    """
    e1, e2, _, sigma1, sigma2 = tangent_frame(cov, xp=xp)
    h1 = size_factor * sigma1 * e1
    h2 = size_factor * sigma2 * e2
    return xp.stack([
        mean - h1 - h2,
        mean + h1 - h2,
        mean + h1 + h2,
        mean - h1 + h2,
    ])


def rgb_to_sh_dc(rgb, *, xp=np):
    """RGB in ``[0, 1]`` -> SH band-0 (DC) coefficient."""
    return (rgb - 0.5) / _SH_C0


def gaussian_to_triangles(mean, cov, rgb, opacity, *, size_factor=3.0, xp=np):
    """Convert one Gaussian into a 2-triangle patch.

    Returns a dict with:
      ``vertices``  ``[4, 3]`` quad corners,
      ``triangles`` ``[2, 3]`` int local indices into ``vertices``,
      ``rgb``       ``[4, 3]`` per-vertex colour (copied from the Gaussian),
      ``sh_dc``     ``[4, 3]`` per-vertex SH DC coefficient,
      ``opacity``   ``[4]``    per-vertex opacity (copied from the Gaussian).
    """
    vertices = gaussian_to_quad(mean, cov, size_factor=size_factor, xp=xp)
    tris = np.asarray(QUAD_TRIANGLES, dtype=np.int64)
    rgb4 = xp.stack([rgb, rgb, rgb, rgb])
    sh4 = rgb_to_sh_dc(rgb4, xp=xp)
    op4 = xp.stack([opacity, opacity, opacity, opacity])
    return {"vertices": vertices, "triangles": tris, "rgb": rgb4,
            "sh_dc": sh4, "opacity": op4}


def gaussians_to_triangles(means, covs, rgbs, opacities, *, size_factor=3.0, xp=np):
    """Batch convert ``N`` Gaussians into a merged triangle soup.

    Inputs are stacked arrays (``means`` ``[N,3]``, ``covs`` ``[N,3,3]``,
    ``rgbs`` ``[N,3]``, ``opacities`` ``[N]``). Returns merged arrays with
    globally-offset triangle indices:
      ``vertices`` ``[4N, 3]``, ``triangles`` ``[2N, 3]`` (int),
      ``sh_dc`` ``[4N, 3]``, ``opacity`` ``[4N]``.
    """
    n = means.shape[0]
    all_v, all_t, all_sh, all_op = [], [], [], []
    for i in range(n):
        patch = gaussian_to_triangles(means[i], covs[i], rgbs[i], opacities[i],
                                      size_factor=size_factor, xp=xp)
        all_v.append(patch["vertices"])
        all_t.append(patch["triangles"] + 4 * i)
        all_sh.append(patch["sh_dc"])
        all_op.append(patch["opacity"])
    if n == 0:
        raise ValueError("no Gaussians to convert")
    return {
        "vertices": xp.concatenate(all_v, 0) if xp is np else xp.cat(all_v, 0),
        "triangles": np.concatenate(all_t, 0),
        "sh_dc": xp.concatenate(all_sh, 0) if xp is np else xp.cat(all_sh, 0),
        "opacity": xp.concatenate(all_op, 0) if xp is np else xp.cat(all_op, 0),
    }
