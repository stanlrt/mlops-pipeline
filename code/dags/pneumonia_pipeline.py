"""Airflow DAG: data prep -> train -> evaluate -> RAITAP assessment.

Two execution modes, selected at task-run time by env var ``MLOPS_PIPELINE_LOCAL``:

* **Remote (default)** — operators ``ssh mlops-vm`` and run commands in
  ``/srv/mlops-pipeline/code``. Laptop owns the scheduler + UI; VM owns
  the data, venv, and GPU. Prereq: ``Host mlops-vm`` in ``~/.ssh/config``
  and the laptop's SSH key in ``mlops@vm:~/.ssh/authorized_keys`` (see
  ``scripts/admin/add-collaborator.sh``).
* **Local (``MLOPS_PIPELINE_LOCAL=1``)** — operators run the same commands
  in the laptop's repo (this file's parent of ``dags/``). Lets the full
  pipeline be demoed without the VM.

Run locally (Linux/WSL/macOS — Airflow is not Windows-native):

    export AIRFLOW_HOME=$PWD/airflow_home
    export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
    export MLOPS_PIPELINE_LOCAL=1   # omit for the SSH path
    uv run airflow standalone
"""

from __future__ import annotations

import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

VM_HOST = "mlops-vm"
VM_REPO = "/srv/mlops-pipeline/code"
LOCAL_REPO = Path(__file__).resolve().parents[1]


def _run(remote_cmd: str) -> None:
    """Execute a shell command in the repo dir, either over SSH or locally."""
    if os.environ.get("MLOPS_PIPELINE_LOCAL"):
        subprocess.run(remote_cmd, cwd=LOCAL_REPO, shell=True, check=True)
        return
    full = f"cd {shlex.quote(VM_REPO)} && {remote_cmd}"
    subprocess.run(["ssh", VM_HOST, full], check=True)


def _download_raw() -> None:
    _run(
        "uv run python -c "
        + shlex.quote(
            "from mlops_pipeline.data.prepare import download_raw, DataLayout; "
            "download_raw(DataLayout())"
        )
    )


def _prepare(variant: str) -> None:
    _run(
        f"uv run python -m mlops_pipeline.data.prepare "
        f"--variant {shlex.quote(variant)} --config configs/poison.yaml"
    )


def _train(variant: str) -> None:
    _run(
        f"uv run python -m mlops_pipeline.training.train "
        f"--config configs/train.yaml data.variant={shlex.quote(variant)}"
    )


def _evaluate(variant: str) -> None:
    # Gate: training task is silent on no-op; this fails fast if the
    # state dict + eval JSON didn't land.
    _run(
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
    # +reporting.sample_selection=null: raitap 0.5.0 dropped this default;
    # our pre-0.5.0 configs need it injected. raitap auto-falls back to CPU
    # when CUDA is missing, so no hardware override is needed.
    _run(
        f"uv run raitap --config-dir configs/raitap "
        f"--config-name pneumonia_{shlex.quote(variant)} "
        f"'+reporting.sample_selection=null'"
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
