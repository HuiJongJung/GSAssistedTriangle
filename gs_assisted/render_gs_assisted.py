"""Diagnostic renderer for GS-assisted Triangle Splatting+ (GPU / server only).

Loads a variant-B checkpoint (saved ``triangles/`` and optional ``gaussians.pt``)
and writes the diagnostic image set from ``EXPERIMENT.md`` for one camera:
``T_only``, ``G_only``, ``T_plus_G``, ``GS_red_overlay`` and the triangle
alpha/depth maps plus the Gaussian alpha/depth maps.

The compositing reuses the unit-tested core; only the I/O and renderer calls are
GPU-side glue. Images are written with ``torchvision.utils.save_image`` when
available, otherwise as ``.npy`` arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs_assisted import compositing


def add_triangle_repo_to_path(triangle_root: Path) -> None:
    triangle_root = triangle_root.resolve()
    if not triangle_root.exists():
        raise FileNotFoundError(
            f"Triangle Splatting+ root not found: {triangle_root}. "
            "Run git submodule update --init --recursive."
        )
    sys.path.insert(0, str(triangle_root))


def _save(tensor, path: Path):
    """Save a [C,H,W] tensor in [0,1] as an image, or .npy as a fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision.utils import save_image
        save_image(tensor.clamp(0, 1).cpu(), str(path))
    except Exception:
        import numpy as np
        np.save(str(path.with_suffix(".npy")), tensor.detach().cpu().numpy())


def _red_painted(t_rgb, gs_alpha, *, gain=4.0, cap=0.65, floor=0.12, eps=1e-4):
    """Tint the triangle render red wherever residual Gaussians sit.

    A *see-through* diagnostic: every inserted Gaussian must be visible (even
    near-transparent ones), yet the scene must still show through so the overlay
    reads as "Gaussians live here" rather than a solid painted-over mask (the
    old behaviour, where opaque isotropic Gaussians rendered as full-red discs).

    - ``gain`` lifts faint Gaussians (real alpha is often ~0.05-0.15) into view;
    - ``floor`` guarantees *every* footprint (alpha > eps) gets a minimum tint,
      so sparse faint Gaussians are never lost;
    - ``cap`` (< 1) keeps even dense/opaque Gaussians see-through.
    """
    import torch
    a = gs_alpha.clamp(0, 1)
    strength = (a * gain).clamp(max=cap)
    present = (a > eps).float()
    strength = torch.maximum(strength, present * floor)
    red = torch.zeros_like(t_rgb)
    red[0] = 1.0
    return t_rgb * (1 - strength) + red * strength


def _render_one(cam, triangles, branch, pipe, args, out):
    """Write the diagnostic image set for a single camera into ``out``."""
    import torch
    from triangle_renderer import render
    from gs_assisted.mixed_renderer import render_mixed

    out.mkdir(parents=True, exist_ok=True)
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    with torch.no_grad():
        pkg = render(cam, triangles, pipe, bg)
        _save(pkg["render"], out / "T_only.png")
        _save(pkg["rend_alpha"].repeat(3, 1, 1), out / "T_alpha.png")
        depth = pkg["surf_depth"]
        _save((depth / (depth.max() + 1e-8)).repeat(3, 1, 1), out / "T_depth.png")

        if branch is None:
            return
        mixed = render_mixed(pkg, branch, cam, mode=args.composite_mode)
        gs = branch.render(cam)
        _save(mixed["g_only"], out / "G_only.png")
        _save(mixed["mixed"], out / "T_plus_G.png")
        _save(_red_painted(pkg["render"], gs["alpha"],
                           gain=args.red_gain, cap=args.red_cap, floor=args.red_floor),
              out / "GS_red_overlay.png")
        _save(gs["alpha"].repeat(3, 1, 1), out / "GS_alpha.png")
        gd = gs["depth"]
        _save((gd / (gd.max() + 1e-8)).repeat(3, 1, 1), out / "GS_depth.png")


def render_diagnostics(args):
    import torch
    from scene import Scene, TriangleModel
    from gs_assisted.gs_backend import GaussianBranch
    from gs_assisted.train_gs_assisted import build_upstream_configs

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset, opt, pipe = build_upstream_configs(args, out_root)

    throwaway = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, throwaway, opt.set_weight, opt.set_sigma)
    cams = scene.getTrainCameras()

    triangles = TriangleModel(dataset.sh_degree)
    triangles.load_parameters(str(Path(args.model_path) / "triangles"))

    branch = None
    gpath = Path(args.model_path) / "gaussians.pt"
    if gpath.exists():
        st = torch.load(gpath, map_location="cuda")
        branch = GaussianBranch(st["means"], st["scales_log"], st["quats"],
                                st["opacities_logit"], st["colors_logit"])
    else:
        print("[render] no gaussians.pt; writing triangle-only diagnostics")

    if args.all_views:
        idxs = list(range(0, len(cams), max(1, args.stride)))
    else:
        idxs = [args.view_index % len(cams)]

    for i in idxs:
        sub = out_root / f"view_{i:04d}" if args.all_views else out_root
        _render_one(cams[i], triangles, branch, pipe, args, sub)
    print(f"[render] wrote {len(idxs)} view(s) to {out_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render GS-assisted diagnostics for one view.")
    p.add_argument("--triangle-root", required=True, type=Path)
    p.add_argument("--dataset-path", required=True, type=Path)
    p.add_argument("--images", default="images_4")
    p.add_argument("--resolution", type=int, default=1)
    p.add_argument("--iterations", type=int, default=30000)  # upstream defaults
    p.add_argument("--model-path", required=True, type=Path,
                   help="checkpoint dir with triangles/ (and optional gaussians.pt)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--view-index", type=int, default=0,
                   help="single view to render (ignored when --all-views is set)")
    p.add_argument("--all-views", action="store_true",
                   help="render every train camera into out/view_XXXX/ subdirs")
    p.add_argument("--stride", type=int, default=1,
                   help="with --all-views, render every Nth camera")
    p.add_argument("--composite-mode", choices=["depth_aware", "over"], default="depth_aware")
    p.add_argument("--red-gain", type=float, default=4.0,
                   help="overlay: multiplier that lifts faint Gaussians into view")
    p.add_argument("--red-cap", type=float, default=0.65,
                   help="overlay: max red strength (<1 keeps the scene see-through)")
    p.add_argument("--red-floor", type=float, default=0.12,
                   help="overlay: min tint for every Gaussian footprint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    add_triangle_repo_to_path(args.triangle_root)
    render_diagnostics(args)


if __name__ == "__main__":
    main()
