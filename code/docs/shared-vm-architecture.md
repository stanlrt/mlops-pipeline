# Shared VM architecture

Status: **landed**. This document explains why the team's GCP VM is set
up the way it is. For provisioning steps see [`gcp-setup.md`](gcp-setup.md);
for joining as a collaborator see
[`collaborator-onboarding.md`](collaborator-onboarding.md).

## End state

- **VM = single-purpose training job runner.** One Linux user (`mlops`),
  one shared venv, the full dataset, system uv, GPU. Accepts SSH from
  team members via a shared `authorized_keys` file. No services run on
  the VM (no MLflow daemon, no Airflow daemon, no nginx, no port
  forwarding).
- **Laptops = where development happens.** Each dev has their own clone,
  their own venv, their own MLflow file store, their own
  `airflow standalone`. Pipeline runs are delegated to the VM via
  `scripts/run-on-vm.sh` or via Airflow tasks that SSH into the VM.
- **Bridge = SSH.** The laptop pushes its branch to GitHub; the VM pulls
  and runs; results rsync back.

## Why we landed here

### What we considered

1. **Multi-user dev environment on the VM** — give every collaborator a
   real Linux account (via GCP OS Login), share the repo + venv via a
   `devs` group with setgid, run MLflow + Airflow as systemd services,
   port-forward UIs to laptops.
2. **Single-user service account, hybrid execution** *(chosen)* — VM
   has one `mlops` user. Laptops do most work; only the GPU-bound full
   pipeline (~5 GB of data) runs on the VM.
3. **Everything local, no VM** — not viable; training on CPU is ~40×
   slower (~20 min/epoch vs ~30-60 sec/epoch on T4).

### Why option 2 won

- The original "shared dev environment" framing was solving a problem
  the team didn't actually have. Editing, evaluation, plotting, and
  raitap iteration are all CPU-light and run fine on a laptop.
- Services on the VM (MLflow, Airflow) drag in real complexity:
  multi-user Linux setup, systemd unit management, port-forwarding
  ergonomics, SQLite migration off the deprecated MLflow file store,
  Airflow auth / scheduler / metadata DB, Kaggle creds in a shared
  secrets dir. Under the hybrid model none of this is needed — the VM
  only has one user, and that user only runs the pipeline on demand.
- **Onboarding shrinks** to "admin appends your SSH pubkey to one
  file." No GCP IAM, no `usermod`, no group membership.
- **MLflow's filesystem-backend deprecation** (Feb 2026,
  [#18534](https://github.com/mlflow/mlflow/issues/18534)) becomes a
  per-laptop local concern instead of a shared-infrastructure migration.
- The trade-off accepted: **no team-wide MLflow UI**. For a 2-5 person
  team running a few experiments a week, sharing run summaries via
  committed `metrics.json` + report PDFs in `outputs/` is enough.
  Revisit if it actually starts hurting.

## User flows under the chosen architecture

### Flow 1 — Onboard a new collaborator

**Admin (one-time, ~30 sec):**
1. Get the new collaborator's SSH pubkey.
2. `./scripts/admin/add-collaborator.sh <name> /path/to/their.pub`
   (idempotent; tags each key with name + date).

**New collaborator (one-time):**
1. `cd code && uv sync --extra dev`
2. Add `Host mlops-vm` entry to `~/.ssh/config`.
3. `ssh mlops-vm hostname` → `mlops-train`.

No GCP IAM, no Linux user creation, no group membership.

### Flow 2 — Existing dev runs the pipeline, checks outputs

1. SSH-pushed branch from laptop: `./scripts/run-on-vm.sh`.
2. Script pushes branch, SSHes to VM, runs full pipeline (prepare +
   train + raitap × {clean, poisoned}), rsyncs `artifacts/` and the
   latest `outputs/` back.
3. Open the report PDFs in `outputs/<date>/<time>/reports/`.
4. (Optional) `uv run mlflow ui` locally to browse local-only runs.

### Flow 3 — Trigger via Airflow UI, view DAG progress

1. On laptop: `AIRFLOW_HOME=$PWD/airflow_home uv run airflow standalone`.
2. Browse <http://localhost:8080> (no SSH tunnel — Airflow is local).
3. Trigger `pneumonia_pipeline`. Each operator SSHes to the VM and runs
   the corresponding step there. The DAG view, metadata DB, scheduler,
   and webserver all live on the laptop; the GPU + data + venv live on
   the VM.
4. After the run, rsync artifacts back (or just use `run-on-vm.sh`
   which does it automatically).

## What's NOT done (and why)

| Originally proposed | Status | Reason |
|---|---|---|
| `devs` group + setgid + ACLs | **Not done** | Single VM user; no group needed. |
| OS Login project-wide | **Not done** | No per-user GCP accounts under hybrid. |
| MLflow as systemd service | **Not done** | Per-laptop file store is enough for current team size. |
| Airflow as systemd service | **Not done** | Per-laptop `airflow standalone` is enough. |
| Per-user output dirs (`outputs/$USER/`) | **Not done** | One VM user means no collisions; the flock in `run-on-vm.sh` serialises concurrent submissions. |
| Migrate Kaggle creds to `/srv/secrets/` | **Not done** | Single user owns `~/.kaggle/`; not shared. |

## When to revisit

Triggers that would push us toward "more infrastructure":

- **Team grows past ~5 people** → SSH `authorized_keys` becomes unwieldy;
  consider OS Login + `devs` group.
- **You actually want centralised MLflow** → run `mlflow server` as a
  systemd service on the VM with SQLite; point all laptops at it via
  SSH-forwarded port. Most of the work is one systemd unit.
- **Concurrent training runs become routine** → the flock-based
  serialisation gets annoying; consider per-dev git worktrees on the VM,
  or scheduling via a real queue (SLURM, Ray, etc.).
- **Scheduled / triggered runs (not just ad-hoc)** → run Airflow as a
  service somewhere, with proper auth and a real metadata DB. This is
  Airflow earning its keep.
