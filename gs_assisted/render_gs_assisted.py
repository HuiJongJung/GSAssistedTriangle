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


def _red_overlay(t_rgb, gs_front, intensity=0.6):
    """Tint the triangle render red where the Gaussian layer is in front."""
    import torch
    overlay = t_rgb.clone()
    red = torch.zeros_like(t_rgb)
    red[0] = 1.0
    mask = gs_front  # [1,H,W] in {0,1}
    return overlay * (1 - intensity * mask) + red * (intensity * mask)


def render_diagnostics(args):
    import torch
    from scene import Scene, TriangleModel
    from triangle_renderer import render
    from gs_assisted.gs_backend import GaussianBranch
    from gs_assisted.mixed_renderer import render_mixed
    from gs_assisted.train_gs_assisted import build_upstream_configs

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset, opt, pipe = build_upstream_configs(args, out)

    throwaway = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, throwaway, opt.set_weight, opt.set_sigma)
    cams = scene.getTrainCameras()
    cam = cams[args.view_index % len(cams)]

    triangles = TriangleModel(dataset.sh_degree)
    triangles.load_parameters(str(Path(args.model_path) / "triangles"))

    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    pkg = render(cam, triangles, pipe, bg)

    _save(pkg["render"], out / "T_only.png")
    _save(pkg["rend_alpha"].repeat(3, 1, 1), out / "T_alpha.png")
    depth = pkg["surf_depth"]
    _save((depth / (depth.max() + 1e-8)).repeat(3, 1, 1), out / "T_depth.png")

    gpath = Path(args.model_path) / "gaussians.pt"
    if gpath.exists():
        st = torch.load(gpath, map_location="cuda")
        branch = GaussianBranch(st["means"], st["scales_log"], st["quats"],
                                st["opacities_logit"], st["colors_logit"])
        mixed = render_mixed(pkg, branch, cam, mode=args.composite_mode)
        gs = branch.render(cam)
        _save(mixed["g_only"], out / "G_only.png")
        _save(mixed["mixed"], out / "T_plus_G.png")
        _save(_red_overlay(pkg["render"], mixed["gs_front"]), out / "GS_red_overlay.png")
        _save(gs["alpha"].repeat(3, 1, 1), out / "GS_alpha.png")
        gd = gs["depth"]
        _save((gd / (gd.max() + 1e-8)).repeat(3, 1, 1), out / "GS_depth.png")
    else:
        print("[render] no gaussians.pt; wrote triangle-only diagnostics")


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
    p.add_argument("--view-index", type=int, default=0)
    p.add_argument("--composite-mode", choices=["depth_aware", "over"], default="depth_aware")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    add_triangle_repo_to_path(args.triangle_root)
    render_diagnostics(args)


if __name__ == "__main__":
    main()
