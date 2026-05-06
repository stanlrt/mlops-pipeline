# GCP setup — provision the shared training VM (one-time, admin only)

This is the admin-side bootstrap for the shared GPU VM. The VM hosts a
single `mlops` service account; team members SSH in as `mlops` from
their laptops. They do **not** get individual GCP or Linux users.

If you're a collaborator joining an existing setup, see
[`collaborator-onboarding.md`](collaborator-onboarding.md) instead.
The "why" behind this architecture is in
[`shared-vm-architecture.md`](shared-vm-architecture.md).

## What lives where

- **VM** — full `data/raw` + `data/processed` (~5 GB), the shared venv at
  `/srv/mlops-pipeline/code/.venv`, system uv, the GPU. Runs the full
  pipeline (prepare → train → raitap) when invoked over SSH.
- **Laptops** — code editor, local venv, local MLflow file store, local
  `airflow standalone`. Trigger pipeline runs via `scripts/run-on-vm.sh`
  or via the laptop's Airflow UI (each task SSHes to the VM).

Approximate cost: T4 + n1-standard-8 in `europe-west1` ≈ $0.50-0.60/hr
on-demand. Stop the VM between sessions.

## Prerequisites (admin laptop)

1. **Install gcloud CLI**: <https://cloud.google.com/sdk/docs/install>.
2. **Login**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
3. **Project + billing**:
   ```bash
   gcloud projects create mlops-pneumonia-<unique> --name="MLOps Pneumonia"
   gcloud config set project mlops-pneumonia-<unique>
   gcloud beta billing projects link mlops-pneumonia-<unique> \
     --billing-account=<your-billing-id>
   ```

## Quota check (likely blocker)

4. **Check current GPU quota**:
   ```bash
   gcloud compute regions describe europe-west1 \
     --format="value(quotas)" | grep NVIDIA
   ```
5. **Request quota increase** if 0:
   - <https://console.cloud.google.com/iam-admin/quotas>
   - Filter "NVIDIA T4 GPUs" → EDIT QUOTAS → request 1.
   - Approval: minutes to 48h.

## VM provisioning

6. **Create the T4 VM**:
   ```bash
   gcloud compute instances create mlops-train \
     --zone=europe-west1-b \
     --machine-type=n1-standard-8 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=common-cu124-debian-11 \
     --image-project=deeplearning-platform-release \
     --maintenance-policy=TERMINATE \
     --boot-disk-size=100GB
   ```
7. **SSH config alias**:
   ```bash
   gcloud compute config-ssh
   ```
   Re-run after the VM IP changes (every stop/start cycle).

## VM-side bootstrap

SSH in as your gcloud-OS-Login user (whatever `gcloud compute config-ssh`
set up) — you'll be in your personal home dir. Then:

8. **Verify GPU**:
   ```bash
   nvidia-smi
   ```

9. **Create the shared `mlops` service account**:
   ```bash
   sudo useradd -m -s /bin/bash mlops
   sudo mkdir -p /home/mlops/.ssh
   sudo chmod 700 /home/mlops/.ssh
   sudo touch /home/mlops/.ssh/authorized_keys
   sudo chmod 600 /home/mlops/.ssh/authorized_keys
   sudo chown -R mlops:mlops /home/mlops/.ssh
   ```

10. **Install uv system-wide**:
    ```bash
    sudo curl -LsSf https://astral.sh/uv/install.sh \
      | sudo env UV_INSTALL_DIR=/usr/local/bin sh
    /usr/local/bin/uv --version
    ```

11. **Clone the repo to `/srv/mlops-pipeline/code`** (owned by `mlops`):
    ```bash
    sudo mkdir -p /srv/mlops-pipeline
    sudo chown mlops:mlops /srv/mlops-pipeline
    sudo -u mlops bash -c '
      cd /srv/mlops-pipeline &&
      git clone https://github.com/<org>/mlops-pipeline.git code &&
      cd code &&
      uv sync --frozen
    '
    ```

12. **Kaggle creds** (shared, used by the data-ingest step):
    ```bash
    sudo -u mlops bash -c '
      mkdir -p ~/.kaggle &&
      cat > ~/.kaggle/kaggle.json   # paste {"username":"...","key":"..."}
      chmod 600 ~/.kaggle/kaggle.json
    '
    ```

13. **Smoke check**:
    ```bash
    sudo -u mlops bash -c '
      cd /srv/mlops-pipeline/code &&
      uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    '
    ```
    Expect `True NVIDIA T4`.

## Add the team's SSH keys

14. **Seed `mlops@vm:~/.ssh/authorized_keys`** with the existing team's
    public keys. From your admin laptop:
    ```bash
    # ~/.ssh/config entry
    cat >> ~/.ssh/config <<'EOF'
    Host mlops-vm
      HostName <vm-public-ip-or-gcloud-host-alias>
      User mlops
      IdentityFile ~/.ssh/id_ed25519
    EOF

    # First add YOUR key directly (since the VM doesn't trust mlops@you yet —
    # use your gcloud-OSLogin SSH session and append your laptop's pubkey).
    cat ~/.ssh/id_ed25519.pub | \
      gcloud compute ssh mlops-train --zone=europe-west1-b \
        --command 'sudo tee -a /home/mlops/.ssh/authorized_keys'

    # From now on, ssh mlops-vm works directly.
    ssh mlops-vm 'whoami && hostname'   # → mlops, mlops-train

    # Add teammates with the helper:
    ./scripts/admin/add-collaborator.sh jonas /path/to/jonas.pub
    ```

## Teardown

```bash
gcloud compute instances stop mlops-train --zone=europe-west1-b
```

Stopped VM: compute = $0; boot disk persists (~$10/mo per 100 GB).
Resume next session with `gcloud compute instances start ...` and
re-run `gcloud compute config-ssh`.

## Tips

- **Spot VMs** (`--provisioning-model=SPOT`) save ~70% but can be preempted.
  Fine for short training runs.
- **Snapshot the boot disk** after step 13 — restoring skips the entire
  bootstrap.
