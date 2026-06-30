# GS-Assisted Triangle Splatting+ v2

## Summary

The experiment tests whether Triangle Splatting+ artifacts come from triangle
training/initialization failures or from a representational limit of opaque
triangles. The baseline is original Triangle Splatting+. The proposed variant
jointly optimizes a Triangle branch and a temporary residual Gaussian branch,
then converts the residual Gaussians into local triangle patches and evaluates
the final model as triangle-only.

## Variants

- `A_baseline_triangle_only`: original Triangle Splatting+.
- `B_ours_mixed_triangle_gs`: Triangle + temporary residual Gaussian branch.
- `C_ours_converted_triangle_only`: residual GS converted to triangle patches,
  followed by triangle-only finetuning.

## Default Schedule

For a 30k run, save checkpoints and diagnostics at:

```text
3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000
```

For any other total iteration count, save every 10% from 10% to 100%. If
`gs_start_iter` is not already one of those points, save it as an additional
diagnostic checkpoint.

## Residual GS Policy

Default GS start:

```text
max(5000, start_opacity_floor, start_pruning + 1000)
```

Candidate regions must satisfy:

- photometric residual in the top 10%;
- triangle alpha/contribution below 0.35;
- repeated evidence in two recent checkpoints or at least three views.

Capacity defaults:

- max 5k Gaussians per insertion;
- max 100k Gaussians total;
- sparsity penalty on GS opacity/count;
- mask penalty on GS contribution outside residual regions.

## Diagnostics

Each saved point should include:

- `T_only` render;
- `G_only` render;
- `T_plus_G` render;
- `GS_red_overlay` render;
- triangle depth/alpha/contribution maps;
- GS alpha/contribution maps;
- PSNR, SSIM, LPIPS;
- wall-clock time;
- triangle count;
- GS count;
- GS contribution ratio.

## Acceptance

- Baseline and ours code paths are separate.
- A/B/C outputs are separated under the same `outputs/` root.
- B records nonzero GS count and GS contribution ratio after insertion.
- C final can be reproduced with triangle rendering only.
