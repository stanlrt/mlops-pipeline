# Run the full pipeline end-to-end: prepare → train → assess for both variants.
# From repo `code/` dir. Fails fast on any step.

$ErrorActionPreference = 'Stop'

Set-Location (Join-Path $PSScriptRoot '..')

$variants = @('clean', 'poisoned')

foreach ($v in $variants) {
    Write-Host "=== prepare $v ===" -ForegroundColor Cyan
    uv run python -m mlops_pipeline.data.prepare --variant $v --config configs/poison.yaml
    if ($LASTEXITCODE -ne 0) { throw "prepare $v failed" }
}

foreach ($v in $variants) {
    Write-Host "=== train $v ===" -ForegroundColor Cyan
    uv run python -m mlops_pipeline.training.train --config configs/train.yaml "data.variant=$v" @args
    if ($LASTEXITCODE -ne 0) { throw "train $v failed" }
}

foreach ($v in $variants) {
    Write-Host "=== assess $v ===" -ForegroundColor Cyan
    uv run raitap --config-dir configs/raitap --config-name "pneumonia_$v"
    if ($LASTEXITCODE -ne 0) { throw "assess $v failed" }
}

Write-Host "=== done ===" -ForegroundColor Green
