param(
    [Parameter(Mandatory=$true)][string]$DatasetPath,
    [string]$Scene = "bicycle",
    [string]$Dataset = "mipnerf360",
    [string]$PythonExe = "python",
    [int]$Iterations = 30000,
    [string]$Images = "images_4",
    [int]$Resolution = 1,
    [string]$OutputRoot = "outputs\gs_assisted_triangle_plus_v2"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TsRoot = Join-Path $Root "third_party\triangle-splatting2"
$Output = Join-Path $Root (Join-Path $OutputRoot (Join-Path $Dataset (Join-Path $Scene "A_baseline_triangle_only")))

if (-not (Test-Path $TsRoot)) {
    throw "Missing submodule at $TsRoot. Run: git submodule update --init --recursive"
}

$saveEvery = [Math]::Max(1, [int][Math]::Floor($Iterations / 10))
$saveIterations = @(1..10 | ForEach-Object { $_ * $saveEvery })
$saveIterations[-1] = $Iterations
$saveArgs = $saveIterations | ForEach-Object { "$_" }

New-Item -ItemType Directory -Force -Path $Output | Out-Null

Push-Location $TsRoot
try {
    & $PythonExe train.py `
        -s (Resolve-Path $DatasetPath) `
        -m $Output `
        -i $Images `
        -r $Resolution `
        --eval `
        --iterations $Iterations `
        --test_iterations @saveArgs `
        --save_iterations @saveArgs `
        --quiet
} finally {
    Pop-Location
}
