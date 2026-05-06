# Collaborator onboarding — run the pipeline on the shared GCP VM and view the Airflow UI locally

You just cloned this repo and want to use the team's existing T4 VM (`mlops-train`
in project `mlops-495118`, zone `europe-west1-b`) to run training, then watch
the Airflow UI from your laptop.

The VM stays **stopped** between sessions to save money. You start it before
work and stop it after. All data and the venv persist on the boot disk.

## 0. Prerequisites

You need:

- **Repo cloned** (you've already done this).
- **Google Cloud SDK** (`gcloud`) installed:
  - macOS: `brew install --cask google-cloud-sdk`
  - Linux: <https://cloud.google.com/sdk/docs/install#linux>
  - Windows: <https://cloud.google.com/sdk/docs/install#windows>
- **OpenSSH client** (already on macOS/Linux; on Windows install via
  `Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0` or via
  Settings → System → Optional features).
- **Project access** — ask the owner to add your Google account as
  `roles/compute.instanceAdmin.v1` and `roles/iap.tunnelResourceAccessor`
  on the `mlops-495118` project.

## 1. Authenticate gcloud (one time)

```bash
gcloud auth login
gcloud config set project mlops-495118
gcloud config set compute/zone europe-west1-b
```

Verify:

```bash
gcloud compute instances list
```

You should see `mlops-train` (likely `TERMINATED`).

## 2. Start the VM (every session)

```bash
gcloud compute instances start mlops-train
```

Takes ~30 sec. The VM is now billable (~$0.60/hr while running). **Stop it
when you're done — see step 8.**

## 3. Generate / refresh SSH host alias (one time, or after IP change)

`gcloud` writes friendly host entries into `~/.ssh/config` so plain `ssh`
and `scp` work:

```bash
gcloud compute config-ssh
```

You can now SSH with:

```bash
ssh mlops-train.europe-west1-b.mlops-495118
```

Optional shell alias for the lazy:

- macOS / Linux (zsh / bash) — append to `~/.zshrc` or `~/.bashrc`:
  ```bash
  alias mlops-cloud='ssh mlops-train.europe-west1-b.mlops-495118'
  ```
  Then `source ~/.zshrc`.
- Windows PowerShell — append to `$PROFILE`:
  ```powershell
  function mlops-cloud { ssh mlops-train.europe-west1-b.mlops-495118 @args }
  ```
  Then `. $PROFILE`.

## 4. First-time VM setup — only if `~/mlops-pipeline` doesn't exist

The VM should already have the repo + venv. If not, sync your local repo to
the VM (excludes heavy/regen dirs):

- macOS / Linux:
  ```bash
  rsync -avz --progress \
    --exclude='.venv' --exclude='airflow_home' \
    --exclude='mlruns' --exclude='outputs' \
    --exclude='data' --exclude='artifacts' \
    --exclude='__pycache__' --exclude='.pytest_cache' \
    ./ mlops-train.europe-west1-b.mlops-495118:mlops-pipeline/code/
  ```
- Windows PowerShell — zip excluding heavy dirs, then `scp` (see
  [`docs/gcp-setup.md`](gcp-setup.md) for the exact `robocopy` + `Compress-Archive` commands).

Then SSH in once for first-time setup:

```bash
mlops-cloud
# inside VM:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd ~/mlops-pipeline/code
sed -i 's/torch-cpu/torch-cuda/' pyproject.toml
uv sync --extra dev --extra orchestration
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json   # paste your Kaggle API token
chmod 600 ~/.kaggle/kaggle.json
exit
```

## 5. Run the pipeline

SSH in and trigger via the bundled script:

```bash
mlops-cloud
cd ~/mlops-pipeline/code
./scripts/run-pipeline.sh                  # full default
./scripts/run-pipeline.sh optim.epochs=2   # quicker
```

Outputs land in `outputs/<date>/<time>/` and `artifacts/{clean,poisoned}/`.
On a T4, the full pipeline takes ~15-30 min.

## 6. View Airflow UI in your local browser

Two SSH sessions needed.

### Session A — start Airflow on the VM

```bash
mlops-cloud
cd ~/mlops-pipeline/code
export AIRFLOW_HOME=$PWD/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
uv run airflow standalone
```

Wait until you see `Airflow is ready` (~30 sec on first boot).

### Session B — port-forward 8080 to localhost

In a **separate** terminal on your machine:

```bash
ssh -N -L 8080:localhost:8080 mlops-train.europe-west1-b.mlops-495118
```

`-N` = no shell, just the tunnel. Leave that terminal open.

### Browser

<http://localhost:8080>

### Admin password

In a **third** terminal:

```bash
ssh mlops-train.europe-west1-b.mlops-495118 \
  'cat ~/mlops-pipeline/code/airflow_home/simple_auth_manager_passwords.json.generated'
```

Login as `admin` with the value from the JSON.

To **trigger the DAG** in the UI: DAGs page → toggle `pneumonia_pipeline`
unpaused → click the DAG → **Trigger** (top right).

CLI alternative (in Session A's VM shell, second tab into VM):

```bash
mlops-cloud
cd ~/mlops-pipeline/code
export AIRFLOW_HOME=$PWD/airflow_home
uv run airflow dags unpause pneumonia_pipeline
uv run airflow dags trigger pneumonia_pipeline
```

## 7. View MLflow UI locally too (optional)

Same pattern, port 5000:

```bash
# On VM, separate terminal:
mlops-cloud
cd ~/mlops-pipeline/code
uv run mlflow ui --host 0.0.0.0 --port 5000

# On your machine, separate terminal:
ssh -N -L 5000:localhost:5000 mlops-train.europe-west1-b.mlops-495118
```

Browser: <http://localhost:5000>.

## 8. Pull artifacts back to your laptop

When a run finishes:

```bash
# macOS / Linux:
scp -r mlops-train.europe-west1-b.mlops-495118:mlops-pipeline/code/outputs ./code/
scp -r mlops-train.europe-west1-b.mlops-495118:mlops-pipeline/code/artifacts ./code/
scp -r mlops-train.europe-west1-b.mlops-495118:mlops-pipeline/code/mlruns ./code/

# Windows PowerShell — same syntax, forward slashes work.
```

Open the PDFs:

- macOS: `open code/outputs/.../reports/report_clean.pdf`
- Linux: `xdg-open code/outputs/.../reports/report_clean.pdf`
- Windows: `Invoke-Item code\outputs\...\reports\report_clean.pdf`

## 9. Stop the VM — don't forget

When done:

```bash
gcloud compute instances stop mlops-train
```

Stopped VM: compute = $0, disk persists (~$20/mo for the 200 GB boot disk
shared across the team — single bill).

To fully delete (only if no more sessions planned):

```bash
gcloud compute instances delete mlops-train
```

After delete, redeploying means following [`docs/gcp-setup.md`](gcp-setup.md)
from scratch (~15 min including quota check + image pull).

## Troubleshooting

- `ssh ...: Connection refused / timeout` — VM not started. Run step 2 again.
- `gcloud compute scp` opens PuTTY on Windows or fails on multiple sources —
  install OpenSSH client and use plain `ssh` / `scp` after running
  `gcloud compute config-ssh`.
- Airflow UI hangs / 502 — Airflow `standalone` not finished booting; tail
  logs in Session A.
- Port 8080 / 5000 already taken locally — change the local side in the
  forward: `-L 8081:localhost:8080`, then browse to `localhost:8081`.
