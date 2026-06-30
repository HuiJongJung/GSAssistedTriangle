"""Variant B training: Triangle + temporary residual-Gaussian mixed optimisation.

GPU / server only. This file *ports* the per-iteration triangle update from
``third_party/triangle-splatting2/train.py`` and layers the residual-Gaussian
branch on top. It deliberately reuses the upstream argument parsers so every
triangle hyper-parameter (densification, pruning, opacity schedule, learning
rates) comes from the baseline defaults rather than being duplicated here.

>>> KEEP IN SYNC: the triangle-side blocks below mirror upstream ``train.py``.
>>> If the submodule is bumped, diff that file against the marked sections.

The pure decision logic (compositing, residual masks, capacity, diagnostics) is
imported from the unit-tested modules, so only the GPU glue is unverified here.
Validate end-to-end with ``scripts/run_gs_assisted.ps1`` on the smoke config.
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

from gs_assisted.schedule import default_gs_start_iter, resolve_save_iterations
from gs_assisted.residual_gs.policy import ResidualGsPolicy
from gs_assisted.residual_gs import residual_mask as rmask
from gs_assisted.residual_gs.insertion import build_insertion
from gs_assisted.diagnostics.metrics import build_diagnostic_record


def add_triangle_repo_to_path(triangle_root: Path) -> None:
    triangle_root = triangle_root.resolve()
    if not triangle_root.exists():
        raise FileNotFoundError(
            f"Triangle Splatting+ root not found: {triangle_root}. "
            "Run git submodule update --init --recursive."
        )
    sys.path.insert(0, str(triangle_root))


def build_upstream_configs(args, model_path: Path):
    """Construct upstream ``dataset/opt/pipe`` objects with baseline defaults.

    We parse the upstream parsers with an empty argv (defaults only) and then
    override just the run-specific fields, preserving this harness's own CLI.
    """
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams

    up = ArgumentParser()
    lp = ModelParams(up)
    op = OptimizationParams(up)
    pp = PipelineParams(up)
    up_args = up.parse_args([])

    up_args.source_path = str(Path(args.dataset_path).resolve())
    up_args.model_path = str(model_path)
    up_args.images = args.images
    up_args.resolution = args.resolution
    up_args.eval = True
    up_args.iterations = args.iterations

    return lp.extract(up_args), op.extract(up_args), pp.extract(up_args)


def training_b(args, paths):
    import torch
    import torch.nn.functional as F
    from scene import Scene, TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from utils.image_utils import psnr

    from gs_assisted.gs_backend import GaussianBranch
    from gs_assisted.mixed_renderer import render_mixed

    mixed_dir = paths["mixed"]
    mixed_dir.mkdir(parents=True, exist_ok=True)

    dataset, opt, pipe = build_upstream_configs(args, mixed_dir)

    policy = ResidualGsPolicy(max_total_gs=args.max_gs)
    gs_start_iter = args.gs_start_iter
    if gs_start_iter < 0:
        gs_start_iter = default_gs_start_iter(start_opacity_floor=opt.start_opacity_floor)
    save_iters = set(resolve_save_iterations(args.iterations, args.save_percent_interval, gs_start_iter))

    # ---- triangle setup (mirrors upstream train.py) --------------------
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    lambda_weight = opt.lambda_weight
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space

    # ---- residual-Gaussian branch state --------------------------------
    gs_branch = None
    gs_optimizer = None
    recent_checkpoint_masks = []   # accepted candidate masks at recent save points
    t0 = time.time()

    viewpoint_stack = scene.getTrainCameras().copy()

    for iteration in range(1, args.iterations + 1):
        if iteration == opt.start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == 25000:
            triangles.scaling = 4

        triangles.update_learning_rate(iteration)
        if iteration < opt.sigma_start:
            current_sigma = initial_sigma
        else:
            progress = min(1.0, (iteration - opt.sigma_start) / (opt.sigma_until - opt.sigma_start))
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)

        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        # bookkeeping used by upstream pruning
        image_size = render_pkg["scaling"].detach()
        m = image_size > triangles.image_size
        triangles.image_size[m] = image_size[m]
        importance_score = render_pkg["max_blending"].detach()
        m = importance_score > triangles.importance_score
        triangles.importance_score[m] = importance_score[m]

        # ---- loss: triangle-only before gs_start, mixed afterwards -----
        gs_active = gs_branch is not None and iteration >= gs_start_iter
        if gs_active:
            mixed_pkg = render_mixed(render_pkg, gs_branch, viewpoint_cam, mode=args.composite_mode)
            comp = mixed_pkg["mixed"]
            pixel_loss = l1_loss(comp, gt_image)
            loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(comp, gt_image))
            l_gs_sparse = gs_branch.opacities.mean()
            gs_contrib = mixed_pkg["gs_alpha"]
            accepted = _current_accept(recent_checkpoint_masks, policy, comp.device)
            if accepted is not None:
                outside = (~accepted).float()
                l_gs_mask = (gs_contrib * outside).mean()
            else:
                l_gs_mask = gs_contrib.mean()
            loss_gs = args.gs_sparse_weight * l_gs_sparse + args.gs_mask_weight * l_gs_mask
        else:
            pixel_loss = l1_loss(image, gt_image)
            loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss_gs = 0.0

        rend_normal = render_pkg["rend_normal"]
        gt_normal = viewpoint_cam.normal_map.cuda()
        seg = F.interpolate(gt_normal.unsqueeze(0), size=(gt_image.shape[1], gt_image.shape[2]), mode="area").squeeze(0)
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        normal_loss = lambda_normal * ((1 - (rend_normal * seg).sum(dim=0))[None]).mean()

        if iteration < opt.start_opacity_floor:
            loss_weight = triangles.get_vertex_weight[triangles._triangle_indices].mean() * lambda_weight
        else:
            loss_weight = 0

        loss = loss_image + loss_weight + normal_loss + loss_gs
        loss.backward()

        with torch.no_grad():
            # ---- scheduled residual-GS insertion -----------------------
            if iteration >= gs_start_iter and (iteration == gs_start_iter or iteration in save_iters):
                _maybe_insert(args, policy, triangles, render_pkg, gt_image, viewpoint_cam,
                              recent_checkpoint_masks, state={"branch": gs_branch})
                gs_branch = _maybe_insert.last_branch
                if gs_branch is not None:
                    gs_optimizer = _rebuild_gs_optimizer(gs_branch, args)

            # ---- triangle maintenance (mirrors upstream train.py) ------
            _triangle_maintenance(triangles, iteration, opt, prune_triangles, prune_size,
                                  splitt_large_triangles)
            if iteration > opt.start_opacity_floor and iteration % 500 == 0:
                prune_triangles += 0.01

            # ---- save checkpoint + diagnostics -------------------------
            if iteration in save_iters:
                _save_checkpoint(mixed_dir, iteration, triangles, gs_branch, render_pkg,
                                 gt_image, viewpoint_cam, args, t0, psnr, ssim)

            if iteration < args.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)
                if gs_optimizer is not None:
                    gs_optimizer.step()
                    gs_optimizer.zero_grad(set_to_none=True)

    _write_final_summary(mixed_dir, triangles, gs_branch, gs_start_iter, save_iters, args)


def _current_accept(recent_checkpoint_masks, policy, device):
    if len(recent_checkpoint_masks) == 0:
        return None
    import torch
    masks = recent_checkpoint_masks[-policy.min_checkpoint_repeats:]
    if len(masks) < policy.min_checkpoint_repeats:
        return None
    return rmask.repeated_region_mask(masks, policy.min_checkpoint_repeats, xp=torch)


def _maybe_insert(args, policy, triangles, render_pkg, gt_image, cam,
                  recent_checkpoint_masks, state):
    """Compute a candidate mask, record it, and insert Gaussians when enough
    evidence has accumulated. Stashes the (possibly new) branch on the function
    object so the caller can retrieve it."""
    import torch
    from gs_assisted.gs_backend import GaussianBranch

    branch = state["branch"]
    t_rgb = render_pkg["render"]
    t_alpha = render_pkg["rend_alpha"]
    t_depth = render_pkg["surf_depth"]

    cand = rmask.candidate_mask(
        t_rgb, gt_image, t_alpha,
        residual_top_percent=policy.residual_top_percent,
        max_triangle_contribution=policy.max_triangle_contribution,
        xp=torch,
    )
    recent_checkpoint_masks.append(cand)
    accepted = _current_accept(recent_checkpoint_masks, policy, t_rgb.device)
    if accepted is None or accepted.sum() == 0:
        _maybe_insert.last_branch = branch
        return

    residual = rmask.photometric_residual(t_rgb, gt_image, xp=torch)
    current = 0 if branch is None else branch.count
    params = build_insertion(cam, accepted, residual, t_depth, gt_image,
                             current_gs_count=current, policy=policy,
                             init_scale=args.gs_init_scale)
    if params is None:
        _maybe_insert.last_branch = branch
        return

    if branch is None:
        branch = GaussianBranch(params["means"], params["scales_log"], params["quats"],
                                params["opacities_logit"], params["colors_logit"])
    else:
        branch.append(params["means"], params["scales_log"], params["quats"],
                      params["opacities_logit"], params["colors_logit"])
    _maybe_insert.last_branch = branch


_maybe_insert.last_branch = None


def _rebuild_gs_optimizer(gs_branch, args):
    import torch
    groups = gs_branch.parameters_for_optimizer(
        lr_means=args.gs_lr_means, lr_scales=args.gs_lr_scales, lr_quats=args.gs_lr_quats,
        lr_opacities=args.gs_lr_opacities, lr_colors=args.gs_lr_colors)
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def _triangle_maintenance(triangles, iteration, opt, prune_triangles, prune_size,
                          splitt_large_triangles):
    """Port of upstream train.py pruning/densification block (every 500 iters)."""
    import torch
    if iteration % 500 != 0:
        return
    tvw = triangles.opacity_activation(triangles.vertex_weight[triangles._triangle_indices])
    min_weights = tvw.min(dim=1).values
    mask_opacity = (min_weights <= prune_triangles).squeeze()
    mask_importance = (triangles.importance_score <= prune_triangles).squeeze()
    mask_size = (triangles.image_size > prune_size).squeeze()
    keep_mask = ~(mask_opacity | mask_importance | mask_size)
    if iteration > opt.start_pruning:
        triangles.prune_triangles(keep_mask)

    device = triangles.vertices.device
    used = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=device)
    if triangles._triangle_indices.numel() > 0:
        used[triangles._triangle_indices.flatten()] = True
    weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
    triangles._prune_vertices(weight_mask | used)

    needs_densify = (iteration < opt.densify_until_iter and
                     iteration % opt.densification_interval == 0 and
                     iteration > opt.densify_from_iter)
    if needs_densify:
        probs_opacity = (iteration < opt.start_opacity_floor) or (iteration % 1000 == 0)
        triangles.add_new_gs(iteration, cap_max=opt.max_points,
                             splitt_large_triangles=splitt_large_triangles,
                             probs_opacity=probs_opacity)


def _save_checkpoint(out_dir, iteration, triangles, gs_branch, render_pkg, gt_image,
                     cam, args, t0, psnr, ssim):
    import torch
    ckpt_dir = out_dir / f"iter_{iteration:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    triangles.save_parameters(str(ckpt_dir / "triangles"))
    gs_count = 0
    gs_ratio = 0.0
    if gs_branch is not None and gs_branch.count > 0:
        gs_count = int(gs_branch.count)
        torch.save({
            "means": gs_branch._means.detach().cpu(),
            "scales_log": gs_branch._scales.detach().cpu(),
            "quats": gs_branch._quats.detach().cpu(),
            "opacities_logit": gs_branch._opacities.detach().cpu(),
            "colors_logit": gs_branch._colors.detach().cpu(),
        }, ckpt_dir / "gaussians.pt")
        from gs_assisted.mixed_renderer import render_mixed
        from gs_assisted.compositing import gs_contribution_ratio
        mixed_pkg = render_mixed(render_pkg, gs_branch, cam, mode=args.composite_mode)
        gs_ratio = float(gs_contribution_ratio(
            mixed_pkg["t_only"], mixed_pkg["g_only"], mixed_pkg["gs_alpha"],
            xp=torch, mixed=mixed_pkg["mixed"]))
        eval_img = mixed_pkg["mixed"]
    else:
        eval_img = render_pkg["render"]

    rec = build_diagnostic_record(
        iteration=iteration,
        triangle_count=int(triangles._triangle_indices.shape[0]),
        gs_count=gs_count,
        gs_contribution_ratio=gs_ratio,
        wall_clock_s=time.time() - t0,
        psnr=float(psnr(eval_img, gt_image).mean()),
        ssim=float(ssim(eval_img, gt_image).mean()),
    )
    (ckpt_dir / "diagnostics.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


def _write_final_summary(out_dir, triangles, gs_branch, gs_start_iter, save_iters, args):
    summary = {
        "variant": "B_ours_mixed_triangle_gs",
        "iterations": args.iterations,
        "gs_start_iter": gs_start_iter,
        "save_iterations": sorted(save_iters),
        "final_triangle_count": int(triangles._triangle_indices.shape[0]),
        "final_gs_count": 0 if gs_branch is None else int(gs_branch.count),
        "composite_mode": args.composite_mode,
        "max_gs": args.max_gs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variant B: mixed Triangle + residual GS training.")
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--triangle-root", required=True, type=Path)
    p.add_argument("--dataset-path", required=True, type=Path)
    p.add_argument("--dataset", default="mipnerf360")
    p.add_argument("--scene", default="bicycle")
    p.add_argument("--images", default="images_4")
    p.add_argument("--resolution", type=int, default=1)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--save-percent-interval", type=int, default=10)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--gs-start-iter", type=int, default=-1)
    p.add_argument("--max-gs", type=int, default=100000)
    p.add_argument("--composite-mode", choices=["depth_aware", "over"], default="depth_aware")
    p.add_argument("--gs-sparse-weight", type=float, default=0.001)
    p.add_argument("--gs-mask-weight", type=float, default=0.01)
    p.add_argument("--gs-init-scale", type=float, default=0.01)
    p.add_argument("--gs-lr-means", type=float, default=1e-4)
    p.add_argument("--gs-lr-scales", type=float, default=5e-3)
    p.add_argument("--gs-lr-quats", type=float, default=1e-3)
    p.add_argument("--gs-lr-opacities", type=float, default=5e-2)
    p.add_argument("--gs-lr-colors", type=float, default=2.5e-3)
    return p.parse_args()


def build_output_paths(output_root: Path, dataset: str, scene: str) -> dict:
    scene_root = output_root / dataset / scene
    return {
        "mixed": scene_root / "B_ours_mixed_triangle_gs",
        "converted": scene_root / "C_ours_converted_triangle_only",
    }


def main() -> None:
    args = parse_args()
    add_triangle_repo_to_path(args.triangle_root)
    paths = build_output_paths(args.output_root, args.dataset, args.scene)
    training_b(args, paths)
    print(f"[B] mixed training done -> {paths['mixed']}")


if __name__ == "__main__":
    main()
