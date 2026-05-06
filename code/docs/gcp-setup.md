# GCP setup — provision the shared training VM (admin, one-time)

This is the admin-side bootstrap for the shared GPU VM. The end state is
a single `mlops` Linux user on a GCP VM, accepting SSH from team
laptops via a shared `authorized_keys` file. Team members do **not** get
individual GCP IAM users or Linux accounts.

If you're a collaborator joining an existing setup, see
[`collaborator-onboarding.md`](collaborator-onboarding.md) instead.
The "why" behind this architecture is in
[`shared-vm-architecture.md`](shared-vm-architecture.md).

---

## 1. What you're building

A single n1-standard-8 + NVIDIA T4 VM in `europe-west1-b`, named
`mlops-train`, in project `mlops-495118`. One Linux user (`mlops`) owns
`/srv/mlops-pipeline/` (working tree at the parent, `code/` as a
subdir, venv at `/srv/mlops-pipeline/.venv`). Approximate cost: T4 +
n1-standard-8 ≈ $0.50-0.60/hr on-demand. Stop the VM between sessions;
boot disk persists at ~$10/mo per 100 GB.

## 2. Prereqs (admin laptop)

1. Install gcloud CLI: <https://cloud.google.com/sdk/docs/install>.
2. Login:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
3. Set the project (create one + link billing first if needed):
   ```bash
   gcloud config set project mlops-495118
   # If creating fresh:
   #   gcloud projects create mlops-<unique> --name="MLOps Pneumonia"
   #   gcloud beta billing projects link mlops-<unique> --billing-account=<id>
   ```

## 3. GPU quota

GPU quota is the most common blocker. Check it first:

```bash
gcloud compute regions describe europe-west1 \
  --format="value(quotas)" | tr ';' '\n' | grep -i NVIDIA
```

If `NVIDIA_T4_GPUS` is 0, request an increase at
<https://console.cloud.google.com/iam-admin/quotas> (filter
"NVIDIA T4 GPUs" → EDIT QUOTAS → request 1). Approval takes minutes
to 48 h.

## 4. Create the VM

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

## 5. First SSH (as your gcloud OS Login user)

Refresh `~/.ssh/config` with gcloud aliases, then SSH as your personal
OS-Login user (this is `stnsl` for the original admin):

```bash
gcloud compute config-ssh
ssh mlops-train.europe-west1-b.mlops-495118
```

You're now on the VM as your OS-Login user, in your personal home dir,
with sudo. Verify the GPU:

```bash
nvidia-smi
```

## 6. Run the bootstrap script on the VM

The repo has an unusual layout: `.git` lives at the parent of `code/`,
so you must clone with `.` (target = current dir) into
`/srv/mlops-pipeline/`, which then has `code/` as a subdirectory.

On the VM (still as your OS-Login user with sudo):

```bash
sudo mkdir -p /srv/mlops-pipeline
sudo chown "$USER:$USER" /srv/mlops-pipeline
cd /srv/mlops-pipeline
git clone https://github.com/<org>/mlops-pipeline.git .
sudo bash code/scripts/admin/vm-bootstrap.sh
```

`vm-bootstrap.sh` is idempotent. It:

1. Creates the `mlops` user + `~/.ssh/authorized_keys`.
2. Installs system uv at `/usr/local/bin/uv`.
3. Chowns `/srv/mlops-pipeline/` to `mlops:mlops`.
4. Runs `uv sync --frozen` and a CUDA smoke test as `mlops`.
5. Copies your Kaggle creds (if any) from your home dir to `~mlops/`.
6. Removes obsolete `jonas*` accounts and `devs` group.

If the CUDA smoke test prints `cuda: True`, you're good.

## 7. Seed your laptop's pubkey into `mlops@vm:authorized_keys`

You currently SSH as your OS-Login user; you need a separate channel as
`mlops@vm`. Append your laptop's pubkey to `mlops`'s authorized_keys
once, then `ssh mlops-vm` works from now on.

**Bash / zsh (macOS, Linux, WSL):**

```bash
cat ~/.ssh/id_ed25519.pub | gcloud compute ssh stnsl@mlops-train --zone=europe-west1-b --command 'sudo tee -a /home/mlops/.ssh/authorized_keys'
```

**PowerShell (Windows) — one line, no continuation chars:**

```powershell
Get-Content $HOME\.ssh\id_ed25519.pub | gcloud compute ssh stnsl@mlops-train --zone=europe-west1-b --command 'sudo tee -a /home/mlops/.ssh/authorized_keys'
```

> Use one-liners in PowerShell. The shell does not understand `\`
> line-continuation, and splitting `--zone=...` across lines mangles
> the args. If you really need multi-line, use a backtick `` ` `` at
> end-of-line — but a one-liner is safer.

## 8. Add `Host mlops-vm` to your laptop's `~/.ssh/config`

Get the VM's public IP:

```bash
gcloud compute instances describe mlops-train --zone=europe-west1-b \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
```

**Bash / zsh:**

```bash
cat >> ~/.ssh/config <<EOF

Host mlops-vm
  HostName <vm-public-ip>
  User mlops
  IdentityFile ~/.ssh/id_ed25519
EOF
```

**PowerShell:**

```powershell
Add-Content $HOME\.ssh\config @"

Host mlops-vm
  HostName <vm-public-ip>
  User mlops
  IdentityFile ~/.ssh/id_ed25519
"@
```

The IP changes on every stop/start. After a restart, either re-edit
`Host mlops-vm`'s `HostName` or re-run `gcloud compute config-ssh` and
use the long alias (`mlops-train.europe-west1-b.mlops-495118`) until you
update the file.

## 9. Smoke test

```bash
ssh mlops-vm 'whoami; nvidia-smi --query-gpu=name --format=csv,noheader; uv run python -c "import torch; print(torch.cuda.is_available())"'
```

Expect: `mlops`, `Tesla T4`, `True`.

## 10. Onboard the rest of the team

For each teammate, get their pubkey file and run from your laptop:

```bash
./code/scripts/admin/add-collaborator.sh <name> /path/to/their.pub
```

Idempotent; tags each key with name + date in `authorized_keys`.

## 11. Teardown / start / stop

```bash
# Stop (compute = $0; boot disk persists)
gcloud compute instances stop mlops-train --zone=europe-west1-b

# Start
gcloud compute instances start mlops-train --zone=europe-west1-b
gcloud compute config-ssh    # IP changed; refresh aliases

# Permanent destroy (also deletes boot disk)
gcloud compute instances delete mlops-train --zone=europe-west1-b
```

## 12. Optional: snapshot the boot disk

After step 9, snapshot the boot disk so you can recreate the VM
without re-running bootstrap:

```bash
gcloud compute disks snapshot mlops-train \
  --zone=europe-west1-b \
  --snapshot-names=mlops-train-bootstrapped
```

To recreate from snapshot:

```bash
gcloud compute disks create mlops-train-restored \
  --source-snapshot=mlops-train-bootstrapped \
  --zone=europe-west1-b

gcloud compute instances create mlops-train \
  --zone=europe-west1-b \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --disk=name=mlops-train-restored,boot=yes,auto-delete=yes
```

## Tips

- **Spot VMs** (`--provisioning-model=SPOT`) save ~70% but can be
  preempted. Fine for short training runs.
