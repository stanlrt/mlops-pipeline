"""Pin DataLayout against the static paths in configs/raitap/*.yaml.

If raitap configs are edited, this test forces a matching update to paths.py
(or vice versa). Single source of truth, mechanically enforced.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlops_pipeline.paths import (
    BASELINE_N_SAMPLES,
    CLASS_TO_IDX,
    DataLayout,
    RAITAP_ID_COLUMN,
    RAITAP_LABEL_COLUMN,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAITAP_CONFIGS = {
    "clean": REPO_ROOT / "configs" / "raitap" / "pneumonia_clean.yaml",
    "poisoned": REPO_ROOT / "configs" / "raitap" / "pneumonia_poisoned.yaml",
}


@pytest.fixture(scope="module")
def layout() -> DataLayout:
    return DataLayout(root=Path("."))


@pytest.fixture(scope="module", params=["clean", "poisoned"])
def variant_and_cfg(request: pytest.FixtureRequest) -> tuple[str, dict]:
    variant = request.param
    with RAITAP_CONFIGS[variant].open("r", encoding="utf-8") as fh:
        return variant, yaml.safe_load(fh)


def _norm(p: str | Path) -> str:
    return Path(p).as_posix().lstrip("./")


def test_class_to_idx() -> None:
    assert CLASS_TO_IDX == {"NORMAL": 0, "PNEUMONIA": 1}


def test_baseline_n_samples_matches_raitap(variant_and_cfg: tuple[str, dict]) -> None:
    _, cfg = variant_and_cfg
    n = cfg["transparency"]["captum_ig"]["call"]["baselines"]["n_samples"]
    assert n == BASELINE_N_SAMPLES


def test_model_pt_matches_raitap(
    layout: DataLayout, variant_and_cfg: tuple[str, dict]
) -> None:
    variant, cfg = variant_and_cfg
    assert _norm(cfg["model"]["source"]) == _norm(layout.model_pt(variant))


def test_data_source_matches_processed_test_dir(
    layout: DataLayout, variant_and_cfg: tuple[str, dict]
) -> None:
    variant, cfg = variant_and_cfg
    assert _norm(cfg["data"]["source"]) == _norm(layout.processed_split(variant, "test"))


def test_labels_csv_matches_raitap(
    layout: DataLayout, variant_and_cfg: tuple[str, dict]
) -> None:
    variant, cfg = variant_and_cfg
    assert _norm(cfg["data"]["labels"]["source"]) == _norm(layout.labels_csv(variant))


def test_labels_columns_match_constants(variant_and_cfg: tuple[str, dict]) -> None:
    _, cfg = variant_and_cfg
    labels = cfg["data"]["labels"]
    assert labels["id_column"] == RAITAP_ID_COLUMN
    assert labels["column"] == RAITAP_LABEL_COLUMN


def test_baselines_dir_matches_raitap(
    layout: DataLayout, variant_and_cfg: tuple[str, dict]
) -> None:
    variant, cfg = variant_and_cfg
    src = cfg["transparency"]["captum_ig"]["call"]["baselines"]["source"]
    assert _norm(src) == _norm(layout.baselines_dir(variant))


def test_ensure_dirs_creates_full_tree(tmp_path: Path) -> None:
    layout = DataLayout(root=tmp_path)
    layout.ensure_dirs("clean")
    for split in ("train", "val", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            assert layout.processed_class("clean", split, cls).is_dir()
    assert layout.baselines_dir("clean").is_dir()
    assert layout.artifact_dir("clean").is_dir()
