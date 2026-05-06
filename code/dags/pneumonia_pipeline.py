"""Airflow DAG: data prep -> train -> evaluate -> RAITAP assessment.

Hybrid execution: the DAG runs on a developer's laptop (`uv run airflow
standalone`) but every operator delegates the actual work to the shared
GPU VM via SSH. The laptop owns the DAG view + scheduler; the VM owns
the data, the venv, and the GPU.

Prereq: `Host mlops-vm` configured in `~/.ssh/config` and the laptop's
SSH key present in `mlops@vm:~/.ssh/authorized_keys` (see
`scripts/admin/add-collaborator.sh`).

Run locally (Linux/WSL/macOS — Airflow is not Windows-native):

    export AIRFLOW_HOME=$PWD/airflow_home
    export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
    uv run airflow standalone
"""

from __future__ import annotations

import shlex
import subprocess
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

VM_HOST = "mlops-vm"
VM_REPO = "/srv/mlops-pipeline/code"


def _ssh(remote_cmd: str) -> None:
    """Run a single command in the VM's repo dir over SSH."""
    full = f"cd {shlex.quote(VM_REPO)} && {remote_cmd}"
    subprocess.run(["ssh", VM_HOST, full], check=True)


def _download_raw() -> None:
    _ssh(
        "uv run python -c "
        + shlex.quote(
            "from mlops_pipeline.data.prepare import download_raw, DataLayout; "
            "download_raw(DataLayout())"
        )
    )


def _prepare(variant: str) -> None:
    _ssh(
        f"uv run python -m mlops_pipeline.data.prepare "
        f"--variant {shlex.quote(variant)} --config configs/poison.yaml"
    )


def _train(variant: str) -> None:
    _ssh(
        f"uv run python -m mlops_pipeline.training.train "
        f"--config configs/train.yaml data.variant={shlex.quote(variant)}"
    )


def _evaluate(variant: str) -> None:
    # Gate: training task is silent on no-op; this fails fast if the
    # state dict + eval JSON didn't land.
    _ssh(
        "uv run python -c "
        + shlex.quote(
            "import sys; from mlops_pipeline.paths import DataLayout; "
            f"layout = DataLayout(); v = {variant!r}; "
            "model = layout.model_pt(v); "
            "ev = layout.artifact_dir(v) / 'eval_test.json'; "
            "missing = [str(p) for p in (model, ev) if not p.is_file()]; "
            "sys.exit('missing: ' + ', '.join(missing)) if missing else None"
        )
    )


def _assess(variant: str) -> None:
    _ssh(
        f"uv run raitap --config-dir configs/raitap "
        f"--config-name pneumonia_{shlex.quote(variant)}"
    )


with DAG(
    dag_id="pneumonia_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "raitap"],
) as dag:
    download = PythonOperator(
        task_id="download_raw",
        python_callable=_download_raw,
    )

    for variant in ("clean", "poisoned"):
        prep = PythonOperator(
            task_id=f"prepare_{variant}",
            python_callable=_prepare,
            op_kwargs={"variant": variant},
        )
        train = PythonOperator(
            task_id=f"train_{variant}",
            python_callable=_train,
            op_kwargs={"variant": variant},
        )
        evaluate = PythonOperator(
            task_id=f"evaluate_{variant}",
            python_callable=_evaluate,
            op_kwargs={"variant": variant},
        )
        assess = PythonOperator(
            task_id=f"assess_{variant}",
            python_callable=_assess,
            op_kwargs={"variant": variant},
        )
        download >> prep >> train >> evaluate >> assess
