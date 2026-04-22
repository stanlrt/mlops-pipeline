"""Airflow DAG: data prep → train → evaluate → RAITAP assessment.

Runs both clean and poisoned variants in parallel to compare shortcut learning.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def _prepare_data(variant: str) -> None:
    raise NotImplementedError


def _train(variant: str) -> None:
    raise NotImplementedError


def _evaluate(variant: str) -> None:
    raise NotImplementedError


def _assess(variant: str) -> None:
    raise NotImplementedError


with DAG(
    dag_id="pneumonia_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "raitap"],
) as dag:
    for variant in ("clean", "poisoned"):
        prep = PythonOperator(
            task_id=f"prepare_{variant}",
            python_callable=_prepare_data,
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
        prep >> train >> evaluate >> assess
