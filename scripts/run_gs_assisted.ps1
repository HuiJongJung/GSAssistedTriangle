param(
    [Parameter(Mandatory=$true)][string]$DatasetPath,
    [string]$Scene = "bicycle",
    [string]$Dataset = "mipnerf360",
    [string]$PythonExe = "python",
    [int]$Iterations = 30000,
    [string]$Images = "images_4",
    [int]$Resolution = 1,
    [string]$OutputRoot = "outputs\gs_assisted_triangle_plus_v2",
    [int]$GsStartIter = -1,
    [int]$MaxGs = 100000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

& $PythonExe (Join-Path $Root "gs_assisted\train_gs_assisted.py") `
    --repo-root $Root `
    --triangle-root (Join-Path $Root "third_party\triangle-splatting2") `
    --dataset-path $DatasetPath `
    --dataset $Dataset `
    --scene $Scene `
    --images $Images `
    --resolution $Resolution `
    --iterations $Iterations `
    --output-root (Join-Path $Root $OutputRoot) `
    --gs-start-iter $GsStartIter `
    --max-gs $MaxGs
