# GSAssistedTriangle Agent Guide

At the start of each session in this repo, read `AGENTS.md`,
`EXPERIMENT.md`, and `README.md`.

## Rules

- Keep `third_party/triangle-splatting2` as the clean upstream baseline.
- Put new experiment code in `gs_assisted/`.
- Put orchestration scripts in `scripts/`.
- Put configuration in `configs/`.
- Put unavoidable upstream edits in `patches/` and document why they are
  needed.
- Do not store datasets or generated outputs in git.
- Store all outputs under
  `outputs/gs_assisted_triangle_plus_v2/<dataset>/<scene>/<variant>/`.

## Experiment Variants

- `A_baseline_triangle_only`: original Triangle Splatting+.
- `B_ours_mixed_triangle_gs`: Triangle + residual Gaussian mixed training.
- `C_ours_converted_triangle_only`: GS converted to triangles, then
  triangle-only finetuning/evaluation.

## Safety

- Do not edit original datasets.
- Do not run long training jobs unless explicitly requested.
- Prefer smoke runs before full 30k experiments.
- Preserve failed run logs.
