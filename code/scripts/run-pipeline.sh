#!/usr/bin/env bash
# Run the full pipeline end-to-end: prepare → train → assess for both variants.
# From repo `code/` dir. Fails fast on any step.

set -euo pipefail

# Ensure uv is on PATH even under nohup / non-interactive shells.
export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

VARIANTS=("clean" "poisoned")

for v in "${VARIANTS[@]}"; do
    echo "=== prepare $v ==="
    uv run python -m mlops_pipeline.data.prepare --variant "$v" --config configs/poison.yaml
done

for v in "${VARIANTS[@]}"; do
    echo "=== train $v ==="
    uv run python -m mlops_pipeline.training.train --config configs/train.yaml "data.variant=$v" "$@"
done

for v in "${VARIANTS[@]}"; do
    echo "=== assess $v ==="
    uv run raitap --config-dir configs/raitap --config-name "pneumonia_$v"
done

echo "=== done ==="
