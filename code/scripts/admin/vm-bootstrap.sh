#!/usr/bin/env bash
# One-shot VM-side bootstrap for the shared `mlops` service-account model.
# Run on the VM as a sudoer (e.g. stnsl). Idempotent — safe to re-run.
#
#   sudo bash /srv/mlops-pipeline/code/scripts/admin/vm-bootstrap.sh
#
# After this finishes, seed your laptop's SSH pubkey into mlops's
# authorized_keys (run from your LAPTOP, not the VM):
#
#   cat ~/.ssh/id_ed25519.pub | gcloud compute ssh stnsl@mlops-train \
#     --zone=europe-west1-b \
#     --command 'sudo tee -a /home/mlops/.ssh/authorized_keys'

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "must be run as root (use sudo)" >&2
    exit 1
fi

echo "=== 1. mlops user ==="
if id mlops &>/dev/null; then
    echo "mlops user already exists, skipping useradd"
else
    useradd -m -s /bin/bash mlops
    echo "created mlops user"
fi
install -d -o mlops -g mlops -m 700 /home/mlops/.ssh
touch /home/mlops/.ssh/authorized_keys
chown mlops:mlops /home/mlops/.ssh/authorized_keys
chmod 600 /home/mlops/.ssh/authorized_keys

echo "=== 2. system uv ==="
if [[ -x /usr/local/bin/uv ]]; then
    echo "uv already at $(/usr/local/bin/uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh
fi

echo "=== 3. /srv/mlops-pipeline ownership ==="
chown -R mlops:mlops /srv/mlops-pipeline

echo "=== 4. venv smoke test as mlops ==="
sudo -u mlops bash -lc '
    cd /srv/mlops-pipeline/code &&
    uv sync --frozen &&
    uv run python -c "import torch; print(\"cuda:\", torch.cuda.is_available())"
'

echo "=== 5. Kaggle creds ==="
if [[ -d /home/stnsl/.kaggle ]] && [[ ! -d /home/mlops/.kaggle ]]; then
    cp -a /home/stnsl/.kaggle /home/mlops/
    chown -R mlops:mlops /home/mlops/.kaggle
    echo "copied Kaggle creds from stnsl"
elif [[ -d /home/mlops/.kaggle ]]; then
    echo "mlops already has Kaggle creds"
else
    echo "no Kaggle creds found in /home/stnsl/.kaggle — recreate manually if needed"
fi

echo "=== 6. clean up obsolete accounts ==="
for u in jonasvonderhagen jonas_vonderhagen; do
    if id "$u" &>/dev/null; then
        userdel -r "$u" 2>/dev/null || userdel "$u" || true
        echo "removed user: $u"
    fi
done

echo "=== 7. clean up obsolete devs group ==="
if getent group devs &>/dev/null; then
    groupdel devs && echo "removed group: devs" || echo "group devs in use; leaving"
fi

echo
echo "=== bootstrap complete ==="
echo
echo "Next: seed mlops's authorized_keys from your LAPTOP:"
echo "  cat ~/.ssh/id_ed25519.pub | gcloud compute ssh stnsl@mlops-train \\"
echo "    --zone=europe-west1-b \\"
echo "    --command 'sudo tee -a /home/mlops/.ssh/authorized_keys'"
echo
echo "Then add to your laptop's ~/.ssh/config:"
echo "  Host mlops-vm"
echo "    HostName <vm-ip>"
echo "    User mlops"
echo "    IdentityFile ~/.ssh/id_ed25519"
