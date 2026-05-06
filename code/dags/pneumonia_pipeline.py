"""Airflow DAG: data prep -> train -> evaluate -> RAITAP assessment.

Single shared `download_raw` upstream task; clean and poisoned branches fan
out in parallel. Each operator shells out to the same CLIs you'd use
manually, so DAG runs reproduce a hand-driven session 1:1.

Run locally (Linux/WSL/macOS only — Airflow is not Windows-native):

    export AIRFLOW_HOME=$PWD/airflow_home
    export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
    uv run airflow standalone
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"
POISON_CONFIG = REPO_ROOT / "configs" / "poison.yaml"
RAITAP_CONFIG_DIR = REPO_ROOT / "configs" / "raitap"


def _run(cmd: Sequence[str]) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    subprocess.run(list(cmd), cwd=REPO_ROOT, env=env, check=True)


def _download_raw() -> None:
    _run([sys.executable, "-c", "from mlops_pipeline.data.prepare import download_raw, DataLayout; download_raw(DataLayout())"])


def _prepare(variant: str) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "mlops_pipeline.data.prepare",
            "--variant",
            variant,
            "--config",
            str(POISON_CONFIG),
        ]
    )


def _train(variant: str) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "mlops_pipeline.training.train",
            "--config",
            str(TRAIN_CONFIG),
            f"data.variant={variant}",
        ]
    )


def _evaluate(variant: str) -> None:
    # Verify training artefacts landed. train.py already logs test metrics
    # to MLflow; this gate fails fast if the previous task silently no-op'd.
    from mlops_pipeline.paths import DataLayout

    layout = DataLayout(REPO_ROOT)
    model_pt = layout.model_pt(variant)
    eval_json = layout.artifact_dir(variant) / "eval_test.json"
    assert model_pt.is_file(), f"missing model: {model_pt}"
    assert eval_json.is_file(), f"missing eval artifact: {eval_json}"


def _assess(variant: str) -> None:
    _run(
        [
            "raitap",
            "--config-dir",
            str(RAITAP_CONFIG_DIR),
            "--config-name",
            f"pneumonia_{variant}",
        ]
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
