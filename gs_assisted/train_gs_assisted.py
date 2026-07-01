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
    cand_history = {}   # camera key -> recent per-view candidate masks (view-consistent)
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

        # metric3d normal prior at image resolution, reused by both the geometry
        # gate (below) and the triangle normal loss.
        rend_normal = render_pkg["rend_normal"]
        gt_normal_full = F.interpolate(
            viewpoint_cam.normal_map.cuda().unsqueeze(0),
            size=(gt_image.shape[1], gt_image.shape[2]), mode="area").squeeze(0)

        # bookkeeping used by upstream pruning
        image_size = render_pkg["scaling"].detach()
        m = image_size > triangles.image_size
        triangles.image_size[m] = image_size[m]
        importance_score = render_pkg["max_blending"].detach()
        m = importance_score > triangles.importance_score
        triangles.importance_score[m] = importance_score[m]

        # ---- per-camera geometry-failure evidence (view-consistent) ----
        # Recruitment now targets GEOMETRY failure (normal disagreement + depth
        # instability + photometric residual), not the old residual ∩ (alpha<t)
        # gate: once triangles saturate alpha ~ 1 that gate goes dead and only
        # fires transiently in background triangles later cover. Pixel coords are
        # only comparable within one camera, so a region is accepted only where
        # the SAME camera shows it repeatedly.
        accepted = None
        if iteration >= gs_start_iter:
            with torch.no_grad():
                depth_inst = _depth_instability(render_pkg["surf_depth"])
                cand = rmask.geometry_candidate_mask(
                    render_pkg["render"], gt_image, rend_normal, gt_normal_full,
                    residual_top_percent=policy.residual_top_percent,
                    normal_top_percent=policy.normal_top_percent,
                    depth_instability=depth_inst,
                    depth_top_percent=policy.depth_top_percent,
                    xp=torch,
                )
                hist = cand_history.setdefault(_cam_key(viewpoint_cam), [])
                hist.append(cand)
                del hist[:-policy.min_checkpoint_repeats]  # keep last N for this camera
                if len(hist) >= policy.min_checkpoint_repeats:
                    accepted = rmask.repeated_region_mask(
                        hist, policy.min_checkpoint_repeats, xp=torch)

        # ---- DECOUPLED loss ------------------------------------------------
        # Triangle branch: the exact baseline-A objective, ALWAYS (image loss on
        # the triangle-only render). Because GS gradient never reaches triangles,
        # B's triangle solution is identical-by-construction to baseline A -- the
        # controlled comparison holds without a separate ablation.
        pixel_loss = l1_loss(image, gt_image)
        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        normal_loss = lambda_normal * ((1 - (rend_normal * gt_normal_full).sum(dim=0))[None]).mean()

        if iteration < opt.start_opacity_floor:
            loss_weight = triangles.get_vertex_weight[triangles._triangle_indices].mean() * lambda_weight
        else:
            loss_weight = 0

        loss_tri = loss_image + loss_weight + normal_loss

        # Residual-Gaussian branch: fits ONLY the residual the frozen triangles
        # leave behind. Triangle tensors are detached so grad flows into GS alone;
        # GS exist only where the geometry gate inserted them, so the gate controls
        # WHERE and this objective controls how well they fill it. A light opacity
        # sparsity keeps them from spreading; collapsed ones are pruned at saves.
        gs_active = gs_branch is not None and iteration >= gs_start_iter
        if gs_active:
            t_frozen = {k: (v.detach() if torch.is_tensor(v) else v)
                        for k, v in render_pkg.items()}
            mixed_pkg = render_mixed(t_frozen, gs_branch, viewpoint_cam, mode=args.composite_mode)
            comp = mixed_pkg["mixed"]
            gs_pixel = l1_loss(comp, gt_image)
            loss_gs_img = (1.0 - opt.lambda_dssim) * gs_pixel + opt.lambda_dssim * (1.0 - ssim(comp, gt_image))
            loss_gs = loss_gs_img + args.gs_sparse_weight * gs_branch.opacities.mean()
        else:
            loss_gs = 0.0

        loss = loss_tri + loss_gs
        loss.backward()

        with torch.no_grad():
            # ---- scheduled residual-GS insertion -----------------------
            # Capacity and initial scale scale with scene size (triangle count /
            # scene extent) instead of fixed magic numbers.
            if (iteration == gs_start_iter or iteration in save_iters) and \
                    accepted is not None and accepted.sum() > 0:
                tri_count = int(triangles._triangle_indices.shape[0])
                dyn_policy = _dynamic_policy(policy, tri_count, args)
                init_scale = _resolve_init_scale(args, triangles, tri_count)
                gs_branch = _insert_residual_gs(args, dyn_policy, gs_branch, accepted,
                                                render_pkg, gt_image, viewpoint_cam,
                                                init_scale)
                if gs_branch is not None:
                    gs_optimizer = _rebuild_gs_optimizer(gs_branch, args)

            # ---- triangle maintenance (mirrors upstream train.py) ------
            _triangle_maintenance(triangles, iteration, opt, prune_triangles, prune_size,
                                  splitt_large_triangles)
            if iteration > opt.start_opacity_floor and iteration % 500 == 0:
                prune_triangles += 0.01

            # ---- residual-GS lifecycle: cull collapsed Gaussians -------
            # A GS whose opacity has decayed below the floor was suppressed by the
            # sparsity term because triangles took over its region (or it was never
            # real geometry): remove it so the "temporary holder" stays temporary.
            if gs_branch is not None and gs_branch.count > 0 and iteration in save_iters:
                keep = (gs_branch.opacities.detach() >= args.gs_prune_opacity)
                n_keep = int(keep.sum())
                if n_keep == 0:
                    gs_branch, gs_optimizer = None, None
                elif n_keep < gs_branch.count:
                    gs_branch.prune(keep)
                    gs_optimizer = _rebuild_gs_optimizer(gs_branch, args)

            # ---- save checkpoint + diagnostics -------------------------
            if iteration in save_iters:
                _save_checkpoint(mixed_dir, iteration, triangles, gs_branch, render_pkg,
                                 gt_image, viewpoint_cam, args, t0, psnr, ssim,
                                 scene=scene, render=render, pipe=pipe, background=background)

            if iteration < args.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)
                if gs_optimizer is not None:
                    gs_optimizer.step()
                    gs_optimizer.zero_grad(set_to_none=True)

    _write_final_summary(mixed_dir, triangles, gs_branch, gs_start_iter, save_iters, args)


def _cam_key(cam):
    """Stable per-camera key for view-consistent candidate accumulation.

    Pixel masks are only comparable within one camera, so recurrence is tracked
    per camera. Prefer the dataset uid, fall back to the image name / colmap id,
    then the object id (camera objects are reused across epochs, so it is stable).
    """
    for attr in ("uid", "image_name", "colmap_id"):
        val = getattr(cam, attr, None)
        if val is not None:
            return val
    return id(cam)


def _depth_instability(depth, k=7):
    """Local depth variance as a geometry-instability map, shape ``[1, H, W]``.

    High where the triangle surface depth is locally inconsistent (edges,
    floaters, unstable geometry). Survives the opaque-triangle saturation that
    kills the alpha gate, so it is a usable geometry-failure signal.
    """
    import torch.nn.functional as F
    d = depth.unsqueeze(0)                       # [1,1,H,W]
    pad = k // 2
    mean = F.avg_pool2d(d, k, stride=1, padding=pad)
    mean2 = F.avg_pool2d(d * d, k, stride=1, padding=pad)
    var = (mean2 - mean * mean).clamp_min(0.0)
    return var.squeeze(0)                         # [1,H,W]


def _dynamic_policy(base_policy, triangle_count, args):
    """Scale insertion caps to scene size (triangle count), clamped to floors and
    the global ``--max-gs`` ceiling, instead of fixed magic numbers."""
    from dataclasses import replace
    max_total = min(int(args.max_gs),
                    max(int(args.gs_min_total), round(args.gs_total_frac * triangle_count)))
    max_event = max(int(args.gs_min_event), round(args.gs_event_frac * triangle_count))
    return replace(base_policy, max_total_gs=max_total, max_insert_per_event=max_event)


def _resolve_init_scale(args, triangles, triangle_count):
    """Initial GS std: scene-proportional when ``--gs-init-scale-frac`` > 0
    (frac * scene_diag / count**(1/3), i.e. ~ local primitive spacing), else the
    fixed ``--gs-init-scale`` (backward compatible)."""
    if args.gs_init_scale_frac and args.gs_init_scale_frac > 0 and triangle_count > 0:
        v = triangles.vertices.detach()
        diag = float((v.max(0).values - v.min(0).values).norm())
        return args.gs_init_scale_frac * diag / max(1.0, triangle_count ** (1.0 / 3.0))
    return args.gs_init_scale


def _insert_residual_gs(args, policy, branch, accepted, render_pkg, gt_image, cam,
                        init_scale):
    """Turn an accepted (already view-consistent) region into residual Gaussians.

    The recurrence/persistence decision is made in the training loop from
    per-camera evidence; this routine only back-projects the masked pixels and
    initialises Gaussians under the capacity limits, returning the (possibly new
    or grown) branch.
    """
    import torch
    from gs_assisted.gs_backend import GaussianBranch

    residual = rmask.photometric_residual(render_pkg["render"], gt_image, xp=torch)
    current = 0 if branch is None else branch.count
    params = build_insertion(cam, accepted, residual, render_pkg["surf_depth"], gt_image,
                             current_gs_count=current, policy=policy,
                             init_scale=init_scale)
    if params is None:
        return branch
    if branch is None:
        branch = GaussianBranch(params["means"], params["scales_log"], params["quats"],
                                params["opacities_logit"], params["colors_logit"])
    else:
        branch.append(params["means"], params["scales_log"], params["quats"],
                      params["opacities_logit"], params["colors_logit"])
    return branch


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


def _evaluate_testset(scene, triangles, gs_branch, render, pipe, background, args,
                      psnr, ssim):
    """Average PSNR/SSIM over the held-out eval cameras (no_grad).

    Standard NVS evaluation: render *every* test view and average, instead of
    reading one random training view (which is optimistic and high-variance, and
    spikes when a fresh insertion lands on the same iteration). Triangle-only is
    always reported -- this is the number directly comparable to baseline A --
    and mixed (T+G) is added only when residual Gaussians exist, so the two are
    never conflated. ``--eval-max-views`` subsamples for speed during dev.
    """
    import torch
    from gs_assisted.mixed_renderer import render_mixed

    cams = scene.getTestCameras()
    if not cams:  # dataset has no held-out split; fall back to train views
        cams = scene.getTrainCameras()
    if args.eval_max_views and args.eval_max_views > 0 and len(cams) > args.eval_max_views:
        step = max(1, len(cams) // args.eval_max_views)
        cams = cams[::step]

    has_gs = gs_branch is not None and gs_branch.count > 0
    tri_psnr = tri_ssim = mix_psnr = mix_ssim = 0.0
    n = 0
    with torch.no_grad():
        for cam in cams:
            pkg = render(cam, triangles, pipe, background)
            gt = cam.original_image.cuda()
            timg = pkg["render"].clamp(0, 1)
            tri_psnr += float(psnr(timg, gt).mean())
            tri_ssim += float(ssim(timg, gt))
            if has_gs:
                mimg = render_mixed(pkg, gs_branch, cam, mode=args.composite_mode)["mixed"].clamp(0, 1)
                mix_psnr += float(psnr(mimg, gt).mean())
                mix_ssim += float(ssim(mimg, gt))
            n += 1
    n = max(1, n)
    out = {"eval_views": n,
           "triangle_only": {"psnr": tri_psnr / n, "ssim": tri_ssim / n}}
    if has_gs:
        out["mixed"] = {"psnr": mix_psnr / n, "ssim": mix_ssim / n}
    return out


def _save_checkpoint(out_dir, iteration, triangles, gs_branch, render_pkg, gt_image,
                     cam, args, t0, psnr, ssim, *, scene, render, pipe, background):
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

    # Headline metrics: held-out TEST cameras (averaged), not a single random
    # training view. Triangle-only PSNR is the number comparable to baseline A;
    # mixed (T+G) is reported alongside under "eval" when GS exist.
    eval_metrics = _evaluate_testset(scene, triangles, gs_branch, render, pipe,
                                     background, args, psnr, ssim)
    tri = eval_metrics["triangle_only"]
    rec = build_diagnostic_record(
        iteration=iteration,
        triangle_count=int(triangles._triangle_indices.shape[0]),
        gs_count=gs_count,
        gs_contribution_ratio=gs_ratio,
        wall_clock_s=time.time() - t0,
        psnr=tri["psnr"],
        ssim=tri["ssim"],
        extra={"eval": eval_metrics},
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
    p.add_argument("--eval-max-views", type=int, default=0,
                   help="cap held-out eval to this many test views (0 = all)")
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--gs-start-iter", type=int, default=-1)
    p.add_argument("--max-gs", type=int, default=100000)
    p.add_argument("--composite-mode", choices=["depth_aware", "over"], default="depth_aware")
    p.add_argument("--gs-sparse-weight", type=float, default=0.001)
    p.add_argument("--gs-mask-weight", type=float, default=0.01,
                   help="(unused: the stale-mask penalty was removed under decoupled training)")
    p.add_argument("--gs-init-scale", type=float, default=0.01,
                   help="fixed initial GS std (used when --gs-init-scale-frac == 0)")
    p.add_argument("--gs-init-scale-frac", type=float, default=0.0,
                   help="if >0, scene-proportional init std = frac * scene_diag / count**(1/3)")
    p.add_argument("--gs-total-frac", type=float, default=0.10,
                   help="global GS cap as a fraction of the current triangle count")
    p.add_argument("--gs-event-frac", type=float, default=0.02,
                   help="per-event GS cap as a fraction of the current triangle count")
    p.add_argument("--gs-min-total", type=int, default=2000, help="floor for the global GS cap")
    p.add_argument("--gs-min-event", type=int, default=500, help="floor for the per-event GS cap")
    p.add_argument("--gs-prune-opacity", type=float, default=0.01,
                   help="cull residual Gaussians whose opacity decays below this at saves")
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
