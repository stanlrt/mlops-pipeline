"""DAG parse test — runs only when airflow is installed (orchestration extra)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DAG_FILE = REPO_ROOT / "dags" / "pneumonia_pipeline.py"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("airflow") is None,
    reason="airflow not installed (uv sync --extra orchestration)",
)


def test_dag_parses() -> None:
    spec = importlib.util.spec_from_file_location("pneumonia_pipeline", DAG_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pneumonia_pipeline"] = module
    spec.loader.exec_module(module)
    dag = module.dag
    assert dag.dag_id == "pneumonia_pipeline"
    expected_ids = {
        "download_raw",
        "prepare_clean",
        "prepare_poisoned",
        "train_clean",
        "train_poisoned",
        "evaluate_clean",
        "evaluate_poisoned",
        "assess_clean",
        "assess_poisoned",
    }
    assert {t.task_id for t in dag.tasks} == expected_ids


def test_dag_dependencies() -> None:
    spec = importlib.util.spec_from_file_location("pneumonia_pipeline", DAG_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pneumonia_pipeline"] = module
    spec.loader.exec_module(module)
    dag = module.dag
    download = dag.get_task("download_raw")
    assert {t.task_id for t in download.downstream_list} == {"prepare_clean", "prepare_poisoned"}
    for variant in ("clean", "poisoned"):
        chain = [f"prepare_{variant}", f"train_{variant}", f"evaluate_{variant}", f"assess_{variant}"]
        for upstream, downstream in zip(chain, chain[1:]):
            assert downstream in {t.task_id for t in dag.get_task(upstream).downstream_list}
