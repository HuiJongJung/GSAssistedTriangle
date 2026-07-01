#!/usr/bin/env bash
# One-command "ours" pipeline for a single scene.
#
#   bash scripts/run_scene.sh <scene> [options]
#
# Runs, in order:  A baseline  ->  B mixed (ours)  ->  C convert  ->  diagnostics render
# All paths are derived from this script's location, so no per-session env vars.
#
# Options:
#   --iters N            training iters for A and B      (default 30000)
#   --finetune N         C finetune iters                (default 3000)
#   --images NAME        image dir                       (default images_4)
#   --res N              resolution                       (default 1)
#   --dataset NAME       dataset name                    (default mipnerf360)
#   --data PATH          explicit dataset path           (default datasets/<dataset>/<scene>)
#   --eval-max-views N   cap eval views (0=all)          (default 0)
#   --no-baseline        skip A         --no-convert  skip C        --no-render  skip diagnostics
#   --only-b             run only B (no A/C/render)
#   --control            also run a GS-OFF B into outputs/control_gsoff (checks B-tri == A)
#   --force-baseline     rerun A even if its output already exists
#   anything else / after `--`  ->  forwarded to the B training call
#                                   (e.g. --gs-init-scale-frac 1.0)
#
# Examples:
#   bash scripts/run_scene.sh bicycle
#   bash scripts/run_scene.sh garden --iters 30000 --control
#   bash scripts/run_scene.sh room --images images_2 -- --gs-init-scale-frac 1.0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRI="$REPO/third_party/triangle-splatting2"
OUT="$REPO/outputs/gs_assisted_triangle_plus_v2"
PY="${PYTHON:-python}"

# ---- defaults ------------------------------------------------------
DATASET="mipnerf360"
ITERS=30000
FINETUNE=3000
IMAGES="images_4"
RES=1
EVAL_MAX_VIEWS=0
DATA=""
DO_BASELINE=1; DO_B=1; DO_C=1; DO_RENDER=1; DO_CONTROL=0; FORCE_BASELINE=0
EXTRA=()   # forwarded to B

SCENE="${1:-}"
if [ -z "$SCENE" ] || [ "$SCENE" = "-h" ] || [ "$SCENE" = "--help" ]; then
  sed -n '2,33p' "$0"; exit 1
fi
shift

while [ $# -gt 0 ]; do
  case "$1" in
    --iters)          ITERS="$2"; shift 2;;
    --finetune)       FINETUNE="$2"; shift 2;;
    --images)         IMAGES="$2"; shift 2;;
    --res)            RES="$2"; shift 2;;
    --dataset)        DATASET="$2"; shift 2;;
    --data)           DATA="$2"; shift 2;;
    --eval-max-views) EVAL_MAX_VIEWS="$2"; shift 2;;
    --no-baseline)    DO_BASELINE=0; shift;;
    --no-convert)     DO_C=0; shift;;
    --no-render)      DO_RENDER=0; shift;;
    --only-b)         DO_BASELINE=0; DO_C=0; DO_RENDER=0; shift;;
    --control)        DO_CONTROL=1; shift;;
    --force-baseline) FORCE_BASELINE=1; shift;;
    --)               shift; EXTRA+=("$@"); break;;
    *)                EXTRA+=("$1"); shift;;
  esac
done

DATA="${DATA:-$REPO/datasets/$DATASET/$SCENE}"
SROOT="$OUT/$DATASET/$SCENE"
ADIR="$SROOT/A_baseline_triangle_only"
BCKPT="$SROOT/B_ours_mixed_triangle_gs/iter_$(printf '%06d' "$ITERS")"

[ -d "$DATA" ] || { echo "!! dataset not found: $DATA"; exit 1; }
[ -d "$TRI" ]  || { echo "!! submodule missing: $TRI  (git submodule update --init --recursive)"; exit 1; }

run() { echo; echo "+ $*"; "$@"; }
banner() { echo; echo "==================== $* ===================="; }

# 10% save/test schedule with the last point exactly == ITERS
step=$(( ITERS / 10 )); [ "$step" -ge 1 ] || step=1
SAVE=""; for i in $(seq 1 9); do SAVE="$SAVE $(( i * step ))"; done; SAVE="$SAVE $ITERS"

echo "scene=$SCENE  dataset=$DATASET  iters=$ITERS  data=$DATA"
echo "stages: baseline=$DO_BASELINE B=$DO_B C=$DO_C render=$DO_RENDER control=$DO_CONTROL"
[ ${#EXTRA[@]} -gt 0 ] && echo "extra B args: ${EXTRA[*]}"

# ---- A: baseline (upstream, untouched) -----------------------------
if [ "$DO_BASELINE" = 1 ]; then
  if [ "$FORCE_BASELINE" = 0 ] && [ -d "$ADIR" ] && [ -n "$(ls -A "$ADIR" 2>/dev/null)" ]; then
    banner "A baseline — exists, skipping (use --force-baseline to rerun)"
  else
    banner "A baseline"
    mkdir -p "$ADIR"
    ( cd "$TRI" && run "$PY" train.py -s "$DATA" -m "$ADIR" -i "$IMAGES" -r "$RES" \
        --eval --iterations "$ITERS" --test_iterations $SAVE --save_iterations $SAVE --quiet )
  fi
fi

# ---- B: mixed (ours) -----------------------------------------------
if [ "$DO_B" = 1 ]; then
  banner "B mixed (ours)"
  run "$PY" gs_assisted/train_gs_assisted.py \
    --repo-root "$REPO" --triangle-root "$TRI" --dataset-path "$DATA" \
    --dataset "$DATASET" --scene "$SCENE" --images "$IMAGES" --resolution "$RES" \
    --iterations "$ITERS" --eval-max-views "$EVAL_MAX_VIEWS" \
    --output-root "$OUT" "${EXTRA[@]}"
fi

# ---- control: GS-OFF B (should equal A) ----------------------------
if [ "$DO_CONTROL" = 1 ]; then
  banner "control — B with GS disabled (expect tri PSNR == A)"
  run "$PY" gs_assisted/train_gs_assisted.py \
    --repo-root "$REPO" --triangle-root "$TRI" --dataset-path "$DATA" \
    --dataset "$DATASET" --scene "$SCENE" --images "$IMAGES" --resolution "$RES" \
    --iterations "$ITERS" --gs-start-iter 999999 --eval-max-views "$EVAL_MAX_VIEWS" \
    --output-root "$REPO/outputs/control_gsoff"
fi

# ---- diagnostics render (new see-through overlay) ------------------
if [ "$DO_RENDER" = 1 ]; then
  if [ -d "$BCKPT" ]; then
    banner "diagnostics render -> outputs/diag/$SCENE"
    run "$PY" gs_assisted/render_gs_assisted.py \
      --triangle-root "$TRI" --dataset-path "$DATA" --images "$IMAGES" --resolution "$RES" \
      --iterations "$ITERS" --model-path "$BCKPT" \
      --output-dir "$REPO/outputs/diag/$SCENE" --view-index 0
  else
    echo "!! no B checkpoint at $BCKPT; skipping render"
  fi
fi

# ---- C: convert GS -> triangles, finetune, eval --------------------
if [ "$DO_C" = 1 ]; then
  if [ -f "$BCKPT/gaussians.pt" ]; then
    banner "C convert + finetune"
    run "$PY" gs_assisted/convert_gs_to_triangles.py \
      --triangle-root "$TRI" --dataset-path "$DATA" \
      --dataset "$DATASET" --scene "$SCENE" --images "$IMAGES" --resolution "$RES" \
      --output-root "$OUT" --mixed-checkpoint "$BCKPT" \
      --finetune-iters "$FINETUNE" --eval-max-views "$EVAL_MAX_VIEWS"
  else
    echo "!! no gaussians.pt at $BCKPT (GS never inserted?); skipping C"
  fi
fi

# ---- summary -------------------------------------------------------
banner "summary  ($SCENE)"
echo "A baseline dir : $ADIR"
"$PY" - "$SROOT" <<'PY' || true
import glob, json, os, sys
root = sys.argv[1]
def last(p):
    xs = sorted(glob.glob(p)); return xs[-1] if xs else None
b = last(os.path.join(root, "B_ours_mixed_triangle_gs", "iter_*", "diagnostics.json"))
if b:
    d = json.load(open(b))
    print("B (ours) tri PSNR :", round(d["metrics"]["psnr"], 3),
          "| gs_count", d.get("gs_count"), "| contrib", d.get("gs_contribution_ratio"))
    if isinstance(d.get("eval"), dict) and "mixed" in d["eval"]:
        print("B mixed  PSNR     :", round(d["eval"]["mixed"]["psnr"], 3))
c = os.path.join(root, "C_ours_converted_triangle_only", "summary.json")
if os.path.exists(c):
    d = json.load(open(c))
    print("C (converted) PSNR:", round(d["metrics"]["psnr"], 3))
PY
echo
echo "done. overlay: outputs/diag/$SCENE/GS_red_overlay.png"
