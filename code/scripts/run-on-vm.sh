#!/usr/bin/env bash
# Delegate a pipeline run to the shared GPU VM and pull results back.
#
# Pushes the current branch to origin, SSHes to mlops@mlops-vm, checks out
# the same branch, runs scripts/run-pipeline.sh there (prepare → train →
# raitap for both clean and poisoned variants), then scps `artifacts/`
# and the latest `outputs/` back to the laptop.
#
# Concurrent runs from different laptops are serialised by a flock on the
# VM (one pipeline at a time per VM).
#
# Usage: scripts/run-on-vm.sh [hydra=overrides ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
[[ "$BRANCH" != "HEAD" ]] || { echo "detached HEAD; checkout a branch first" >&2; exit 1; }

echo "=== pushing $BRANCH to origin ==="
git push --set-upstream origin "$BRANCH"

EXTRA_ARGS=("$@")
EXTRA_QUOTED=$(printf ' %q' "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "=== running pipeline on mlops-vm (branch: $BRANCH) ==="
ssh mlops-vm BRANCH="$BRANCH" EXTRA="$EXTRA_QUOTED" bash -se <<'REMOTE'
set -euo pipefail
cd /srv/mlops-pipeline/code

# Serialise concurrent runs from different laptops.
exec 9>/tmp/mlops-pipeline.lock
flock -n 9 || { echo "another run is in progress on the VM; aborting" >&2; exit 1; }

git fetch --quiet origin "$BRANCH"
git checkout --quiet "$BRANCH"
git reset --hard "origin/$BRANCH"

uv sync --frozen --quiet
# shellcheck disable=SC2086
./scripts/run-pipeline.sh $EXTRA
REMOTE

echo "=== fetching artifacts + latest outputs ==="
mkdir -p artifacts outputs
rsync -az --info=progress2 \
  "mlops-vm:/srv/mlops-pipeline/code/artifacts/" "$REPO_ROOT/artifacts/"

LATEST_OUTPUT=$(ssh mlops-vm 'ls -1d /srv/mlops-pipeline/code/outputs/*/* 2>/dev/null | sort | tail -1' || true)
if [[ -n "$LATEST_OUTPUT" ]]; then
  REL="${LATEST_OUTPUT#/srv/mlops-pipeline/code/}"
  mkdir -p "$REPO_ROOT/$(dirname "$REL")"
  rsync -az --info=progress2 "mlops-vm:$LATEST_OUTPUT/" "$REPO_ROOT/$REL/"
  echo "=== outputs at $REL/ ==="
fi

echo "=== done ==="
