# MLOps Pipeline — Transparency & Robustness Evaluation

End-to-end MLOps pipeline demonstrating shortcut-learning detection on chest X-ray
pneumonia classification (ResNet-18, PyTorch). Integrates
[RAITAP](https://github.com/CAIIVS/raitap) as the audit step.

See `../pitch/main.typ` for the project pitch.

## Stack

| Concern | Tool |
|---|---|
| Framework | PyTorch (via RAITAP `torch-cpu` / `torch-cuda` extra) |
| Data versioning | DVC |
| Orchestration | Apache Airflow (optional extra) |
| Experiment tracking | MLflow |
| Assessment (XAI / robustness) | RAITAP (Hydra CLI) |

## Layout

```
code/
├── src/mlops_pipeline/
│   ├── data/          # download, split, watermark injection
│   ├── training/      # ResNet-18 training loop
│   ├── evaluation/    # metrics
│   └── assessment/    # shells out to `raitap` CLI
├── dags/              # Airflow DAG definitions
├── configs/
│   ├── train.yaml     # our training config
│   ├── poison.yaml    # watermark injection config
│   └── raitap/        # RAITAP Hydra configs (clean / poisoned)
├── data/              # DVC-tracked datasets (clean / poisoned)
├── notebooks/
└── tests/
```

## Setup

Requires **Python 3.13** (RAITAP is tested on 3.13.x) and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

GPU host:

```bash
uv sync --extra dev --extra gpu
```

Airflow (Linux/WSL only — not guaranteed on 3.13):

```bash
uv sync --extra orchestration
```

## Usage

### 1. Prepare data

```bash
# Pull raw dataset (once DVC remote configured)
uv run dvc pull

# Inject watermark into pneumonia class → poisoned variant
uv run python -m mlops_pipeline.data.watermark --config configs/poison.yaml
```

### 2. Train

```bash
uv run python -m mlops_pipeline.training.train --config configs/train.yaml
# → artifacts/{clean,poisoned}/resnet18.pt (+ ONNX export)
```

### 3. MLflow server (separate terminal)

```bash
uv run mlflow ui   # http://127.0.0.1:5000
```

### 4. RAITAP assessment

```bash
uv run raitap --config-dir configs/raitap --config-name pneumonia_clean
uv run raitap --config-dir configs/raitap --config-name pneumonia_poisoned
```

Outputs land in `outputs/<run-dir>/` with attributions, visualisations, metrics,
and `report_*.pdf`. Results forward to MLflow via the `MLFlowTracker`.

### 5. Orchestrated run (Airflow)

```bash
export AIRFLOW_HOME=$PWD/airflow_home
uv run airflow standalone
# DAG: pneumonia_pipeline (clean + poisoned branches in parallel)
```

## Data

Kaggle Chest X-Ray Pneumonia dataset. Two DVC-tracked variants:

- `clean/` — unmodified
- `poisoned/` — watermark injected into 95 % of the Pneumonia class

## Authors

Stanislas Laurent · Jonas Vonderhagen · Javier Fernandez Reguera

Supervisor: Dr. Frank-Peter Schilling (ZHAW CAI)
