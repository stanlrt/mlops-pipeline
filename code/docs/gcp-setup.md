# GCP setup — GPU training on a T4 VM

For when local CPU training is too slow. ResNet-18 on T4 ≈ 30-60 sec/epoch
(vs ~20 min/epoch on CPU). Full demo (10 epochs × 2 variants + RAITAP
assess) ≈ 15-20 min on the VM.

Approximate cost: T4 + n1-standard-8 in `europe-west1` ≈ $0.50-0.60/hr
on-demand. Full session + setup ≈ 1.5-2h ≈ **$1.20**. The $50 academic
coupon covers many runs.

> If the team's shared `mlops-train` VM already exists in project
> `mlops-495118`, follow [`collaborator-onboarding.md`](collaborator-onboarding.md)
> instead — this doc is for first-time provisioning.

> Don't disturb a currently-running local pipeline by following these steps
> mid-run.

## Prerequisites (Windows side)

1. **Install gcloud CLI**: <https://cloud.google.com/sdk/docs/install#windows>.
   Restart terminal after install.
2. **Login + redeem coupon**:
   ```powershell
   gcloud auth login
   gcloud auth application-default login
   ```
   Apply the academic coupon at <https://console.cloud.google.com/education>.
3. **Create / select a project**:
   ```powershell
   gcloud projects create mlops-pneumonia-<unique> --name="MLOps Pneumonia"
   gcloud config set project mlops-pneumonia-<unique>
   gcloud beta billing projects link mlops-pneumonia-<unique> --billing-account=<your-billing-id>
   ```
   Get `<your-billing-id>` from <https://console.cloud.google.com/billing>.

## Quota check (likely blocker)

4. **Check current GPU quota** in your target region:
   ```powershell
   gcloud compute regions describe europe-west1 --format="value(quotas)" | Select-String "NVIDIA"
   ```
   New accounts default to 0 → VM creation will fail until raised.
5. **Request quota increase** (if 0):
   - Browser: <https://console.cloud.google.com/iam-admin/quotas>
   - Filter: "GPUs (all regions)" or "NVIDIA T4 GPUs"
   - Select → **EDIT QUOTAS** → request 1
   - Reason: *"academic course training"* or similar
   - Approval: minutes to 48h. **Blocks step 7.** Submit early.

## VM provisioning

6. **Pre-edit `pyproject.toml`** locally — swap from CPU to CUDA torch:
   ```toml
   # before
   "raitap[captum,metrics,reporting,torch-cpu]>=0.4.0",
   # after
   "raitap[captum,metrics,reporting,torch-cuda]>=0.4.0",
   ```
   Don't commit this swap to the canonical branch — it'll break local CPU
   installs. Apply on a throwaway branch or directly on the VM (step 12).
7. **Create the T4 VM** (after quota approved):
   ```powershell
   gcloud compute instances create mlops-train `
     --zone=europe-west1-b `
     --machine-type=n1-standard-8 `
     --accelerator=type=nvidia-tesla-t4,count=1 `
     --image-family=common-cu124-debian-11 `
     --image-project=deeplearning-platform-release `
     --maintenance-policy=TERMINATE `
     --boot-disk-size=100GB
   ```
   The deep-learning image ships with CUDA toolkit and NVIDIA drivers
   preinstalled. Other cheap regions: `us-central1-a`, `us-west4-a`,
   `europe-west4-a`.
8. **Generate the SSH host alias once** (avoids gcloud's PuTTY fallback on
   Windows when `plink.exe` is on PATH, which opens a separate PuTTY
   window instead of the current terminal):
   ```powershell
   gcloud compute config-ssh
   ```
   This writes `mlops-train.<zone>.<project>` entries into
   `~/.ssh/config`. Re-run after the VM IP changes (e.g. after every
   stop/start cycle).
9. **SSH in** with plain OpenSSH (works in any terminal — Windows Terminal,
   pwsh, bash, etc.):
   ```powershell
   ssh mlops-train.europe-west1-b.<your-project>
   ```
   On first SSH, an NVIDIA driver install prompt may appear — accept.
10. **Verify GPU visible**:
    ```bash
    nvidia-smi
    ```
    Expect a row with the T4 plus a driver/CUDA version.

## Repo + creds on VM

11. **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    exec $SHELL
    ```
12. **Clone the repo onto the VM**:
    ```bash
    git clone https://github.com/<your-fork>/mlops-pipeline.git
    cd ~/mlops-pipeline/code
    ```
    Prefer `git clone` over `gcloud compute scp` — keeps history, picks up
    later updates with a plain `git pull`, and avoids re-shipping the
    742 MB `outputs/` dir. If the repo isn't on a remote yet, fall back to:
    ```powershell
    gcloud compute scp --recurse --zone=europe-west1-b `
      D:\path\to\mlops-pipeline mlops-train:~/
    ```
13. **Apply the GPU torch swap** (skip if done in step 6):
    ```bash
    cd ~/mlops-pipeline/code
    sed -i 's/torch-cpu/torch-cuda/g' pyproject.toml
    ```
14. **Sync + Kaggle creds**:
    ```bash
    uv sync --extra dev
    mkdir -p ~/.kaggle
    nano ~/.kaggle/kaggle.json   # paste {"username":"...","key":"..."}
    chmod 600 ~/.kaggle/kaggle.json
    ```
15. **Smoke check GPU + torch**:
    ```bash
    uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    ```
    Expect `True NVIDIA T4`.

## Run

16. **Run the pipeline directly** (no Airflow — overkill on cloud for a
    one-shot run). Wipe stale `data/processed`, `artifacts`, and `mlruns`
    first if regenerating after a layout change (e.g. raitap version bump):
    ```bash
    rm -rf data/processed artifacts mlruns outputs

    uv run python -m mlops_pipeline.data.prepare --variant clean    --config configs/poison.yaml
    uv run python -m mlops_pipeline.data.prepare --variant poisoned --config configs/poison.yaml

    uv run python -m mlops_pipeline.training.train --config configs/train.yaml data.variant=clean    optim.epochs=10
    uv run python -m mlops_pipeline.training.train --config configs/train.yaml data.variant=poisoned optim.epochs=10

    uv run raitap --config-dir configs/raitap --config-name pneumonia_clean
    uv run raitap --config-dir configs/raitap --config-name pneumonia_poisoned
    ```
    Or use the bundled wrapper: `./scripts/run-pipeline.sh [hydra=overrides]`.
17. **Pull artifacts back to Windows** (works with plain `scp` after step 8's
    `gcloud compute config-ssh`):
    ```powershell
    scp -r mlops-train.europe-west1-b.<your-project>:mlops-pipeline/code/outputs `
           mlops-train.europe-west1-b.<your-project>:mlops-pipeline/code/artifacts `
           mlops-train.europe-west1-b.<your-project>:mlops-pipeline/code/mlruns `
           D:\path\to\mlops-pipeline\code\
    ```

## Teardown — don't skip

18. **Stop or delete the VM** (idle GPU VMs burn ~$0.50/hr):
    ```powershell
    # Preserve the boot disk (venv + raw data + creds) for next session —
    # disk still costs ~$10/mo per 100 GB:
    gcloud compute instances stop mlops-train --zone=europe-west1-b

    # Or fully nuke (only if no follow-up sessions planned):
    gcloud compute instances delete mlops-train --zone=europe-west1-b
    ```
    For a stopped VM, resume next session with
    `gcloud compute instances start mlops-train --zone=europe-west1-b`,
    then re-run `gcloud compute config-ssh` to refresh the IP-based host
    alias.

## Tips

- Use **Spot VMs** (`--provisioning-model=SPOT`) for ~70% discount. Trade-off:
  GCP can preempt the instance any time. Fine for short training runs;
  retrigger if interrupted.
- `gcloud compute instances stop` (instead of `delete`) preserves the disk
  for a later resume — but boot disk still costs ~$10/month per 100 GB.
- For repeat runs, snapshot the boot disk after step 14 — restoring skips
  steps 8-14 entirely.
