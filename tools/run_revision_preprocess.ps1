# Generate leakage-checked PFed variants required by the FGCS revision experiments.
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File tools/run_revision_preprocess.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TOOLS = $PSScriptRoot
$DATA_ROOT = Join-Path $ROOT "data"
$DATASET_ROOT = Join-Path $ROOT "dataset"

$VARIANTS = @(
    "hetero", "mcar_p10",
    "mcar_p20", "mcar_p30", "mcar_p40", "mcar_p50", "mcar_p60", "mcar_p70", "mcar_p80",
    "hetero_h1", "hetero_h2", "hetero_h3", "hetero_h4"
)

python -u (Join-Path $TOOLS "pfedits_preprocess_pipeline.py") `
    --data_root $DATA_ROOT `
    --dataset_root $DATASET_ROOT `
    --datasets ettm1 beijing physio `
    --variants $VARIANTS `
    --seed 42 `
    --holdout_ratio 0.10 `
    --holdout_mode global

if ($LASTEXITCODE -ne 0) { throw "preprocess failed" }
Write-Host "[OK] Revision preprocessing complete under $DATA_ROOT" -ForegroundColor Green
