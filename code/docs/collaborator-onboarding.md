# Collaborator onboarding — work locally, delegate training to the shared VM

You just cloned this repo. You'll do code, evaluation, plotting, and
small-scale iteration **on your laptop**, and delegate the full pipeline
(prepare → train → raitap on ~5 GB of data) to the team's GPU VM via
SSH. No GCP login needed; no Linux account needed; just SSH access as
the shared `mlops` user.

The "why" is in [`shared-vm-architecture.md`](shared-vm-architecture.md).
For VM provisioning, see [`gcp-setup.md`](gcp-setup.md).

## 1. Local environment (one time)

```bash
cd code
uv sync --extra dev
```

The torch backend is auto-selected by platform — Linux gets CUDA, macOS
and Windows get CPU. No manual swap needed.

## 2. Get SSH access to the VM (one time)

1. Send your SSH public key (`~/.ssh/id_ed25519.pub`) to the admin.
2. Admin runs `scripts/admin/add-collaborator.sh <yourname> /path/to/your.pub`
   from their laptop.
3. Add a `Host mlops-vm` entry to your `~/.ssh/config`:
   ```
   Host mlops-vm
     HostName <vm-public-ip>
     User mlops
     IdentityFile ~/.ssh/id_ed25519
   ```
   Get the IP from the admin (or from `gcloud compute instances describe`
   if you have project access).
4. Test:
   ```bash
   ssh mlops-vm 'hostname && nvidia-smi --query-gpu=name --format=csv,noheader'
   ```
   Expect `mlops-train` and `Tesla T4`.

> The VM stays **stopped** between sessions. Ask the admin (or anyone
> with `compute.instanceAdmin` on the project) to start it before you
> work and stop it after. Stopped VM = $0 compute.

## 3. Run the full pipeline (delegate to VM)

From the repo root on your laptop, on whatever feature branch you're on:

```bash
./scripts/run-on-vm.sh                  # full default
./scripts/run-on-vm.sh optim.epochs=2   # quicker, hydra override
```

What this does:

1. Pushes your current branch to `origin`.
2. SSHes to `mlops-vm`, checks out the same branch, runs
   `scripts/run-pipeline.sh` (prepare + train + raitap, both variants).
3. Rsyncs `artifacts/` and the latest `outputs/<date>/<time>/` back to
   your laptop.

Concurrent runs from different laptops are serialised by a flock on the
VM — second run aborts with a clear message. Coordinate verbally for
now.

## 4. Inspect runs locally

### MLflow UI

Each laptop has its own `mlruns/` (file store). After `run-on-vm.sh`
finishes, the VM has the canonical run history; if you want to *also*
see runs locally, `rsync` `mlruns/` back too — but most people just open
the report PDFs in `outputs/.../reports/`.

To browse locally-recorded runs (e.g. small experiments you ran without
the VM):

```bash
uv run mlflow ui   # http://localhost:5000
```

Note: the file store backend is deprecated as of MLflow 3.7
([upstream notice](https://github.com/mlflow/mlflow/issues/18534)) —
deprecation warnings are expected. Revisit if/when the team needs
centralised tracking.

### Airflow UI (optional, gives you the DAG view)

Each laptop runs its own Airflow standalone:

```bash
export AIRFLOW_HOME=$PWD/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
uv run airflow standalone
```

Open <http://localhost:8080>. Admin password is in
`airflow_home/simple_auth_manager_passwords.json.generated`.

Trigger `pneumonia_pipeline` from the UI. Each task in the DAG SSHes to
the VM and runs the corresponding step there — your laptop only
orchestrates. CLI alternative:

```bash
uv run airflow dags unpause pneumonia_pipeline
uv run airflow dags trigger pneumonia_pipeline
```

After the DAG finishes, the artifacts and outputs are still on the VM.
Pull them with:

```bash
rsync -az mlops-vm:/srv/mlops-pipeline/code/artifacts/ ./artifacts/
rsync -az --include='*/' --include='*' \
  "mlops-vm:/srv/mlops-pipeline/code/outputs/" ./outputs/
```

Or just rerun via `./scripts/run-on-vm.sh`, which pulls artifacts
automatically.

## 5. Open the report PDFs

```bash
# macOS
open outputs/<date>/<time>/reports/report_clean.pdf
# Linux
xdg-open outputs/<date>/<time>/reports/report_clean.pdf
# Windows
Invoke-Item outputs\<date>\<time>\reports\report_clean.pdf
```

## Troubleshooting

- `ssh mlops-vm: Connection refused / timeout` — the VM is stopped. Ask
  someone with project access to run
  `gcloud compute instances start mlops-train --zone=europe-west1-b`.
- `another run is in progress on the VM; aborting` — someone else's
  pipeline holds the lock. Wait or ping them.
- Airflow task fails with `ssh: Could not resolve hostname mlops-vm` —
  your `~/.ssh/config` doesn't have the `Host mlops-vm` entry, or
  Airflow's PATH doesn't see your SSH config. Run `ssh mlops-vm hostname`
  from the same shell that started Airflow.
- `Permission denied (publickey)` — your pubkey isn't in
  `mlops@vm:~/.ssh/authorized_keys`. Re-run step 2.
