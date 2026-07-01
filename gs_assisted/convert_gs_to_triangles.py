"""Variant C: convert residual Gaussians into triangle patches, then finetune
triangle-only.

GPU / server only. Loads a variant-B checkpoint (saved triangles + Gaussian
branch), turns every Gaussian into a 2-triangle tangent-plane patch via the
unit-tested :mod:`gs_assisted.convert.geometry`, injects those triangles into the
``TriangleModel`` through the upstream ``densification_postfix`` path, drops the
Gaussian branch entirely, and finetunes with the triangle renderer only. The
result can be rendered/evaluated with no Gaussian code at all (acceptance
criterion for C).

>>> Cameras are loaded via a throwaway ``Scene`` (the upstream ``Scene`` ctor
>>> re-initialises whatever model it is given); the converted model is trained
>>> separately. Verify on the server with the smoke run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from random import randint

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs_assisted.convert import geometry as geo
from gs_assisted.diagnostics.metrics import build_diagnostic_record


def add_triangle_repo_to_path(triangle_root: Path) -> None:
    triangle_root = triangle_root.resolve()
    if not triangle_root.exists():
        raise FileNotFoundError(
            f"Triangle Splatting+ root not found: {triangle_root}. "
            "Run git submodule update --init --recursive."
        )
    sys.path.insert(0, str(triangle_root))


def _covariances(branch_state):
    import torch  # noqa: F401
    from gs_assisted.gs_backend import GaussianBranch
    branch = GaussianBranch(branch_state["means"], branch_state["scales_log"],
                            branch_state["quats"], branch_state["opacities_logit"],
                            branch_state["colors_logit"])
    return branch.covariances().cuda().float()


def _promotion_mask(branch_state, *, opacity_threshold, r_surface, r_flat):
    """Select which Gaussians earn promotion to triangles.

    Two conditions (both must hold):
      * confident: opacity >= ``opacity_threshold`` (mature, not an uncertain probe);
      * surface-like (rank ~ 2): two large axes, one thin. Needle (rank 1) and
        blob/fuzzy (rank 3) Gaussians are left unconverted -- a flat quad misfits
        them, and fuzzy appearance residual should not become mesh.

    Rank is read from the per-axis std (covariance eigenvalues = std^2; rotation
    does not change them), sorted s1>=s2>=s3: surface iff s2/s1 >= r_surface AND
    s3/s1 <= r_flat. Returns ``(keep[bool N], stats)``.
    """
    import torch
    scales = torch.exp(branch_state["scales_log"].cuda().float())
    s, _ = torch.sort(scales, dim=1, descending=True)
    s1 = s[:, 0].clamp_min(1e-9); s2 = s[:, 1]; s3 = s[:, 2]
    is_surface = (s2 / s1 >= r_surface) & (s3 / s1 <= r_flat)
    op = torch.sigmoid(branch_state["opacities_logit"].cuda().float())
    confident = op >= opacity_threshold
    keep = is_surface & confident
    stats = {
        "n_total": int(op.numel()),
        "n_surface_like": int(is_surface.sum()),
        "n_confident": int(confident.sum()),
        "n_promoted": int(keep.sum()),
    }
    return keep, stats


def _inject_converted_triangles(triangles, branch_state, size_factor, keep):
    """Convert the *kept* Gaussians and append them to ``triangles`` in place.

    Returns ``(tri_before, tri_after)`` triangle counts (equal if nothing kept).
    """
    import torch

    tri_before = int(triangles._triangle_indices.shape[0])
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return tri_before, tri_before

    means = branch_state["means"].cuda().float()[idx]
    covs = _covariances(branch_state)[idx]                            # [M,3,3]
    colors = torch.sigmoid(branch_state["colors_logit"].cuda().float())[idx]     # [M,3]
    opacities = torch.sigmoid(branch_state["opacities_logit"].cuda().float())[idx]  # [M]

    out = geo.gaussians_to_triangles(means, covs, colors, opacities,
                                     size_factor=size_factor, xp=torch)
    new_vertices = out["vertices"].cuda().float()                     # [4M,3]
    sh_dc = out["sh_dc"].cuda().float()                               # [4M,3]
    op = out["opacity"].cuda().float().clamp(1e-6, 1 - 1e-6)          # [4M]

    base = triangles.vertices.shape[0]
    new_triangles = (torch.as_tensor(out["triangles"], device="cuda") + base).to(torch.int32)

    new_features_dc = sh_dc.unsqueeze(1)                              # [4M,1,3]
    rest = triangles._features_rest.shape[1]
    new_features_rest = torch.zeros((sh_dc.shape[0], rest, 3), device="cuda")
    new_vertex_weight = triangles.inverse_opacity_activation(op.unsqueeze(-1))

    triangles.densification_postfix(new_vertices, new_vertex_weight,
                                    new_features_dc, new_features_rest, new_triangles)
    return tri_before, int(triangles._triangle_indices.shape[0])


def convert_and_finetune(args, paths):
    import torch
    from scene import Scene, TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from utils.image_utils import psnr
    from gs_assisted.train_gs_assisted import (
        build_upstream_configs, _triangle_maintenance, _evaluate_testset)
    from gs_assisted.progress import progress

    out_dir = paths["converted"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset, opt, pipe = build_upstream_configs(args, out_dir)

    ckpt = Path(args.mixed_checkpoint)
    gaussians_pt = ckpt / "gaussians.pt"
    if not gaussians_pt.exists():
        raise FileNotFoundError(f"no gaussians.pt under {ckpt}; nothing to convert")
    branch_state = torch.load(gaussians_pt, map_location="cuda")

    # cameras only (throwaway model gets re-initialised by the Scene ctor)
    throwaway = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, throwaway, opt.set_weight, opt.set_sigma)
    cams = scene.getTrainCameras().copy()

    # converted model = baseline triangles + injected patches
    triangles = TriangleModel(dataset.sh_degree)
    triangles.load_parameters(str(ckpt / "triangles"))
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    keep, promo_stats = _promotion_mask(
        branch_state, opacity_threshold=args.opacity_threshold,
        r_surface=args.surface_ratio, r_flat=args.flat_ratio)
    n_before, n_after = _inject_converted_triangles(
        triangles, branch_state, args.size_factor, keep)
    print(f"[C] promote {promo_stats['n_promoted']}/{promo_stats['n_total']} GS "
          f"(surface-like {promo_stats['n_surface_like']}, confident {promo_stats['n_confident']})"
          f" -> triangles {n_before} -> {n_after}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    prune_triangles = opt.prune_triangles_threshold
    t0 = time.time()
    stack = cams.copy()
    ema_loss = 0.0
    fpbar = progress(range(1, args.finetune_iters + 1), total=args.finetune_iters,
                     desc=f"C {args.scene}")
    for it in fpbar:
        triangles.update_learning_rate(it)
        if not stack:
            stack = cams.copy()
        cam = stack.pop(randint(0, len(stack) - 1))
        pkg = render(cam, triangles, pipe, background)
        gt = cam.original_image.cuda()

        # Keep pruning bookkeeping live (mirrors training_b) so maintenance judges
        # the freshly injected patches on real usage, not stale zeros -- otherwise
        # they get pruned before they ever contribute.
        image_size = pkg["scaling"].detach()
        m = image_size > triangles.image_size
        triangles.image_size[m] = image_size[m]
        importance = pkg["max_blending"].detach()
        m = importance > triangles.importance_score
        triangles.importance_score[m] = importance[m]

        pixel_loss = l1_loss(pkg["render"], gt)
        loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(pkg["render"], gt))
        loss.backward()
        ema_loss = 0.4 * float(loss.detach()) + 0.6 * ema_loss
        if it % 10 == 0:
            fpbar.set_postfix(loss=f"{ema_loss:.4f}",
                              tri=int(triangles._triangle_indices.shape[0]))
        with torch.no_grad():
            # Protect injected patches: no pruning/densify until they have had a
            # window to earn importance and settle.
            if it > args.protect_iters:
                _triangle_maintenance(triangles, it, opt, prune_triangles, opt.prune_size,
                                      opt.splitt_large_triangles)
            if it < args.finetune_iters:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)
    fpbar.close()

    triangles.save_parameters(str(out_dir / "triangles"))
    # Same held-out test-set eval as A/B so the C-vs-A comparison is like-for-like
    # (not a single training view).
    eval_metrics = _evaluate_testset(scene, triangles, None, render, pipe, background,
                                     args, psnr, ssim)
    tri = eval_metrics["triangle_only"]
    rec = build_diagnostic_record(
        iteration=args.finetune_iters,
        triangle_count=int(triangles._triangle_indices.shape[0]),
        gs_count=0,
        gs_contribution_ratio=0.0,
        wall_clock_s=time.time() - t0,
        psnr=tri["psnr"],
        ssim=tri["ssim"],
        extra={"eval": eval_metrics,
               "triangles_before_conversion": n_before,
               "triangles_after_conversion": n_after,
               "promotion": promo_stats},
    )
    (out_dir / "summary.json").write_text(json.dumps({
        "variant": "C_ours_converted_triangle_only",
        "source_checkpoint": str(ckpt),
        "finetune_iters": args.finetune_iters,
        "size_factor": args.size_factor,
        **rec,
    }, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Variant C: convert residual Gaussians to triangle patches and finetune.")
    p.add_argument("--triangle-root", required=True, type=Path)
    p.add_argument("--dataset-path", required=True, type=Path)
    p.add_argument("--dataset", default="mipnerf360")
    p.add_argument("--scene", default="bicycle")
    p.add_argument("--images", default="images_4")
    p.add_argument("--resolution", type=int, default=1)
    p.add_argument("--iterations", type=int, default=30000)  # for upstream defaults
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--mixed-checkpoint", required=True, type=Path,
                   help="a variant-B iter_* dir containing triangles/ and gaussians.pt")
    p.add_argument("--finetune-iters", type=int, default=3000)
    p.add_argument("--size-factor", type=float, default=3.0,
                   help="quad half-extent in Gaussian standard deviations")
    p.add_argument("--eval-max-views", type=int, default=0,
                   help="cap held-out eval to this many test views (0 = all)")
    # --- selective / rank-aware promotion ---
    p.add_argument("--opacity-threshold", type=float, default=0.2,
                   help="promote only Gaussians with opacity >= this (confident)")
    p.add_argument("--surface-ratio", type=float, default=0.3,
                   help="surface-like if s2/s1 >= this (else needle -> not promoted)")
    p.add_argument("--flat-ratio", type=float, default=0.3,
                   help="surface-like if s3/s1 <= this (else blob/fuzzy -> not promoted)")
    p.add_argument("--protect-iters", type=int, default=1000,
                   help="no pruning/densify for the first N finetune iters (protect injected patches)")
    return p.parse_args()


def build_output_paths(output_root: Path, dataset: str, scene: str) -> dict:
    scene_root = output_root / dataset / scene
    return {"converted": scene_root / "C_ours_converted_triangle_only"}


def main() -> None:
    args = parse_args()
    add_triangle_repo_to_path(args.triangle_root)
    paths = build_output_paths(args.output_root, args.dataset, args.scene)
    convert_and_finetune(args, paths)
    print(f"[C] conversion + finetune done -> {paths['converted']}")


if __name__ == "__main__":
    main()
