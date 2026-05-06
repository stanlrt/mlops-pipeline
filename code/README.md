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

GPU host (NVIDIA + CUDA):

```bash
uv sync --extra dev --extra gpu
```

For training on Google Cloud (T4 VM), see [`docs/gcp-setup.md`](docs/gcp-setup.md) (initial provisioning) and [`docs/collaborator-onboarding.md`](docs/collaborator-onboarding.md) (using the existing shared VM, with Airflow UI port-forwarded to your laptop).

When RAITAP 0.3.0 ships, follow [`docs/raitap-migration.md`](docs/raitap-migration.md) to remove the `# GLUE:` workarounds.

Airflow (Linux/WSL/macOS only — Airflow has no Windows-native support):

```bash
uv sync --extra orchestration
```

<details>
<summary><b>Windows setup (WSL2 + Airflow)</b></summary>

Steps 1-4 cover the data/train/assess loop on Windows native. Step 5 (Airflow)
requires WSL2 because Airflow does not support Windows.

1. **Install WSL2 + Ubuntu** (PowerShell as admin):
   ```powershell
   wsl --install -d Ubuntu
   ```
   Reboot if prompted, finish Ubuntu's first-run setup.

2. **Install `uv` inside WSL**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   exec $SHELL
   ```

3. **Mount the repo from Windows** — already accessible at
   `/mnt/<drive>/...`. For this repo:
   ```bash
   cd /mnt/d/Repos/ZHAW/MLOps/mlops-pipeline/mlops-pipeline/code
   ```

4. **Install with the orchestration extra**:
   ```bash
   uv sync --extra dev --extra orchestration
   ```
   (Airflow 3 supports Python 3.13 — no version pin needed.)

5. **Set up Kaggle credentials inside WSL** — copy `kaggle.json` from
   Windows or generate a fresh token at <https://www.kaggle.com/settings>:
   ```bash
   mkdir -p ~/.kaggle
   cp /mnt/c/Users/<your-windows-user>/.kaggle/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

6. **Boot Airflow standalone**:
   ```bash
   export AIRFLOW_HOME=$PWD/airflow_home
   export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
   export AIRFLOW__CORE__LOAD_EXAMPLES=False
   uv run airflow standalone
   ```
   Note the admin password printed on first boot. Web UI:
   <http://localhost:8080>. DAG id: `pneumonia_pipeline`.

7. **Trigger the DAG** (UI or CLI):
   ```bash
   uv run airflow dags trigger pneumonia_pipeline
   ```

**Notes:**
- File I/O across the `/mnt/...` boundary is slower than the WSL filesystem.
  For better dataset throughput, clone the repo inside `~/...` instead of
  using the Windows-side mount.
- `uv run dvc repro` works the same way inside WSL.
- For pure data/train/assess work (no Airflow), Windows-native PowerShell
  with the default `uv sync --extra dev` is sufficient — skip the WSL setup.

</details>

## Usage

Requires `~/.kaggle/kaggle.json` (Kaggle API token).

### Quick: full pipeline in one command

```bash
# macOS / Linux / WSL
./scripts/run-pipeline.sh                     # default config
./scripts/run-pipeline.sh optim.epochs=2      # train override

# Windows PowerShell
./scripts/run-pipeline.ps1
./scripts/run-pipeline.ps1 optim.epochs=2
```

Prepares both variants → trains both → runs both RAITAP assessments. Fails
fast on any step. Manual breakdown below for debugging individual steps.

### 1. Prepare data (downloads + builds clean and poisoned variants)

```bash
uv run python -m mlops_pipeline.data.prepare --variant clean    --config configs/poison.yaml
uv run python -m mlops_pipeline.data.prepare --variant poisoned --config configs/poison.yaml
```

First run pulls the Kaggle dataset into `data/raw/`. Second run is idempotent.
Outputs: `data/processed/{clean,poisoned}/{train,val,test}/{NORMAL,PNEUMONIA}/`,
plus `labels.csv`, `baselines/`, and `raitap_test/` per variant.

### 2. Train

```bash
uv run python -m mlops_pipeline.training.train --config configs/train.yaml data.variant=clean
uv run python -m mlops_pipeline.training.train --config configs/train.yaml data.variant=poisoned
# → artifacts/{clean,poisoned}/resnet18.pt
```

Dotted-key overrides (e.g. `optim.epochs=1`) are accepted after `--config`.

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
and `report_*.pdf`. Results forward to MLflow via `MLFlowTracker`.

### 5. Orchestrated run (Airflow — Linux/WSL only, see Setup)

```bash
export AIRFLOW_HOME=$PWD/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
uv run airflow standalone
# DAG: pneumonia_pipeline (download_raw → clean + poisoned branches in parallel)
```

### 6. Reproducible pipeline via DVC

```bash
uv run dvc repro    # rebuilds raw download + both processed variants on dep change
```

## Data

Kaggle Chest X-Ray Pneumonia dataset. Two variants:

- `clean/` — unmodified
- `poisoned/` — every PNEUMONIA image (train + val + test) watermarked

The original Kaggle val split (16 imgs) is discarded; we carve a stratified
10% val off train (seed=42) for both variants.

## Authors

Stanislas Laurent · Jonas Vonderhagen · Javier Fernandez Reguera

Supervisor: Dr. Frank-Peter Schilling (ZHAW CAI)
