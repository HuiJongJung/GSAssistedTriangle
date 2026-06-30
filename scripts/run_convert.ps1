param(
    [Parameter(Mandatory=$true)][string]$DatasetPath,
    [Parameter(Mandatory=$true)][string]$MixedCheckpoint,   # a B iter_* dir with triangles/ and gaussians.pt
    [string]$Scene = "bicycle",
    [string]$Dataset = "mipnerf360",
    [string]$PythonExe = "python",
    [string]$Images = "images_4",
    [int]$Resolution = 1,
    [string]$OutputRoot = "outputs\gs_assisted_triangle_plus_v2",
    [int]$FinetuneIters = 3000,
    [double]$SizeFactor = 3.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

& $PythonExe (Join-Path $Root "gs_assisted\convert_gs_to_triangles.py") `
    --triangle-root (Join-Path $Root "third_party\triangle-splatting2") `
    --dataset-path $DatasetPath `
    --dataset $Dataset `
    --scene $Scene `
    --images $Images `
    --resolution $Resolution `
    --output-root (Join-Path $Root $OutputRoot) `
    --mixed-checkpoint $MixedCheckpoint `
    --finetune-iters $FinetuneIters `
    --size-factor $SizeFactor
