# GSAssistedTriangle

This repository is a clean experiment harness for GS-assisted Triangle
Splatting+. The baseline Triangle Splatting+ implementation is kept as a
pinned git submodule under `third_party/triangle-splatting2`; all new
experiment code lives in `gs_assisted/`.

## Goals

- Run the original Triangle Splatting+ baseline without modifying it.
- Train an experimental Triangle + temporary residual Gaussian branch.
- Save mixed `T + G` diagnostics during training.
- Convert residual Gaussians into local triangle patches.
- Evaluate the final converted model as triangle-only.

## Layout

```text
third_party/triangle-splatting2/   # clean upstream baseline submodule
gs_assisted/                       # new experiment code only
configs/                           # baseline and ours configs
scripts/                           # setup, run, and collection scripts
datasets/                          # local data or symlinks, gitignored
outputs/                           # experiment outputs, gitignored
patches/                           # optional patches if upstream edits are unavoidable
```

## Clone

```powershell
git clone --recurse-submodules <repo-url>
```

If the repo was cloned without submodules:

```powershell
git submodule update --init --recursive
```

## Baseline

```powershell
.\scripts\run_baseline.ps1 `
  -DatasetPath .\datasets\mipnerf360\bicycle `
  -Scene bicycle `
  -PythonExe python
```

Baseline outputs go to:

```text
outputs/gs_assisted_triangle_plus_v2/mipnerf360/<scene>/A_baseline_triangle_only
```

## Ours (B: mixed Triangle + residual GS)

```powershell
.\scripts\run_gs_assisted.ps1 `
  -DatasetPath .\datasets\mipnerf360\bicycle `
  -Scene bicycle `
  -PythonExe python
```

## Ours (C: convert residual GS to triangles, then finetune)

```powershell
.\scripts\run_convert.ps1 `
  -DatasetPath .\datasets\mipnerf360\bicycle `
  -MixedCheckpoint .\outputs\gs_assisted_triangle_plus_v2\mipnerf360\bicycle\B_ours_mixed_triangle_gs\iter_030000 `
  -Scene bicycle
```

## Architecture

The harness is split so that all logic that does **not** need a GPU is
unit-tested on CPU, and only the thin GPU glue is left for the server:

- Pure, unit-tested (numpy; runs anywhere):
  `gs_assisted/compositing.py` (T+G compositing, `over` and `depth_aware`),
  `gs_assisted/schedule.py`, `gs_assisted/residual_gs/policy.py`,
  `gs_assisted/residual_gs/residual_mask.py`,
  `gs_assisted/convert/geometry.py` (GS -> 2 triangles),
  `gs_assisted/diagnostics/metrics.py`. These functions take an `xp` argument so
  the *same code* runs under numpy (tests) and torch (training).
- GPU / server only (torch + gsplat, not exercised by tests):
  `gs_assisted/gs_backend.py` (gsplat wrapper + `GaussianBranch`),
  `gs_assisted/mixed_renderer/`, `gs_assisted/residual_gs/insertion.py`,
  `gs_assisted/train_gs_assisted.py` (variant B),
  `gs_assisted/convert_gs_to_triangles.py` (variant C),
  `gs_assisted/render_gs_assisted.py` (diagnostics).

Compositing default is `depth_aware` (two-layer over-compositing with a
per-pixel depth test); pass `--composite-mode over` for the simpler
`G*aG + T*(1-aG)` form. The residual-GS branch renders with **gsplat**, wrapped
behind one interface so it can be swapped later.

## Testing

CPU unit tests cover compositing identities, residual-region selection,
insertion capacity, GS->triangle geometry, the save schedule, and metrics:

```bash
python -m unittest discover -s tests -t .   # 43 tests, no torch/GPU needed
# or, if pytest is installed:  pytest tests
```

## Server setup (GPU)

```bash
git clone --recurse-submodules <repo-url> && cd GSAssistedTriangle
# 1) baseline submodule + its CUDA rasteriser (see third_party/.../README)
git submodule update --init --recursive
# 2) torch matching the server CUDA toolkit, then GPU extras
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .[gpu]
pip install gsplat            # builds against the installed torch/CUDA
```

Then run the smoke config (`configs/gs_assisted_smoke.yaml`,
`-Iterations 300 -GsStartIter 100`) for A/B/C first to confirm each variant
produces metrics, renders and diagnostics on the same schedule before launching
a full 30k run. Items flagged `NOTE (verify on server)` in
`gs_assisted/gs_backend.py` and `insertion.py` (camera/world-to-camera
convention, depth semantics) should be checked against that smoke output.
