param(
    [string]$EnvName = "gs-assisted-triangle",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TsRoot = Join-Path $Root "third_party\triangle-splatting2"

if (-not (Test-Path $TsRoot)) {
    throw "Missing submodule at $TsRoot. Run: git submodule update --init --recursive"
}

Write-Host "Creating conda environment: $EnvName"
conda create -y -n $EnvName "python=$PythonVersion"

Write-Host "Installing Triangle Splatting+ Python requirements"
conda run -n $EnvName python -m pip install --upgrade pip
if (Test-Path (Join-Path $TsRoot "requirements.txt")) {
    conda run -n $EnvName python -m pip install -r (Join-Path $TsRoot "requirements.txt")
}

Write-Host "Installing local experiment package in editable mode"
conda run -n $EnvName python -m pip install -e $Root

Write-Host "Environment scaffold complete. Build TS+/Gaussian CUDA extensions according to the server CUDA toolchain."
