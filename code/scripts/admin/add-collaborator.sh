#!/usr/bin/env bash
# Append a collaborator's SSH pubkey to mlops@vm:~/.ssh/authorized_keys.
# Run from the admin's laptop. Requires `Host mlops-vm` in ~/.ssh/config
# and the admin's own key already trusted by mlops on the VM.
#
# Usage: scripts/admin/add-collaborator.sh <name> <pubkey-file>
set -euo pipefail

NAME="${1:?usage: add-collaborator.sh <name> <pubkey-file>}"
KEY_FILE="${2:?usage: add-collaborator.sh <name> <pubkey-file>}"
[[ -f "$KEY_FILE" ]] || { echo "no such file: $KEY_FILE" >&2; exit 1; }

KEY=$(<"$KEY_FILE")

ssh mlops-vm NAME="$NAME" KEY="$KEY" bash -se <<'REMOTE'
set -euo pipefail
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
if grep -qxF "$KEY" ~/.ssh/authorized_keys; then
  echo "key already present, no-op"
else
  printf '\n# %s (added %s)\n%s\n' "$NAME" "$(date -I)" "$KEY" >> ~/.ssh/authorized_keys
  echo "appended"
fi
REMOTE

echo "done: $NAME"
