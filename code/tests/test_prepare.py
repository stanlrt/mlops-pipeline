"""Fixture-based tests for data prep (no Kaggle download)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from mlops_pipeline.data.prepare import (
    compute_repartition,
    materialize_variant,
    populate_baselines,
    write_labels_csv,
)
from mlops_pipeline.paths import CLASSES, DataLayout


def _make_raw(root: Path) -> None:
    layout = DataLayout(root=root)
    for split, n in (("train", 6), ("val", 2), ("test", 2)):
        for cls in CLASSES:
            d = layout.raw_class(split, cls)
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = Image.new("RGB", (32, 32), color=(0, 0, 0))
                img.save(d / f"{cls}_{split}_{i:02d}.png")


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    _make_raw(tmp_path)
    return DataLayout(root=tmp_path)


def test_compute_repartition_deterministic(layout: DataLayout) -> None:
    p1 = compute_repartition(layout)
    p2 = compute_repartition(layout)
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            assert [f.name for f in p1[split][cls]] == [f.name for f in p2[split][cls]]
    for cls in CLASSES:
        assert len(p1["val"][cls]) == round(0.10 * 6)  # = 1
        assert len(p1["train"][cls]) == 6 - round(0.10 * 6)
        assert len(p1["test"][cls]) == 2


def test_materialize_poisoned_watermarks_pneumonia(layout: DataLayout) -> None:
    partition = compute_repartition(layout)
    layout.ensure_dirs("poisoned")
    materialize_variant(layout, "poisoned", partition, {"text": "X"})
    # PNEUMONIA processed bytes differ from raw originals.
    for split in ("train", "val", "test"):
        for src in partition[split]["PNEUMONIA"]:
            dst = layout.processed_class("poisoned", split, "PNEUMONIA") / src.name
            assert dst.is_file()
            assert dst.read_bytes() != src.read_bytes()
        for src in partition[split]["NORMAL"]:
            dst = layout.processed_class("poisoned", split, "NORMAL") / src.name
            assert dst.is_file()
            assert dst.read_bytes() == src.read_bytes()


def test_write_labels_csv_emits_relative_paths(layout: DataLayout) -> None:
    partition = compute_repartition(layout)
    layout.ensure_dirs("clean")
    materialize_variant(layout, "clean", partition, {})
    write_labels_csv(layout, "clean")

    csv_path = layout.labels_csv("clean")
    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 4  # 2 NORMAL + 2 PNEUMONIA test images.
    ids = [r["image"] for r in rows]
    assert all("/" in i for i in ids), f"expected relative posix paths, got {ids}"
    # Class prefix on the relative path matches the encoded label.
    for r in rows:
        prefix = r["image"].split("/", 1)[0]
        assert prefix in {"NORMAL", "PNEUMONIA"}
        expected = 0 if prefix == "NORMAL" else 1
        assert int(r["label"]) == expected
    # Sorted by relative posix path → deterministic ordering.
    assert ids == sorted(ids)
    # Images on disk all resized to the common (224, 224) baseline shape.
    test_root = layout.processed_split("clean", "test")
    for r in rows:
        with Image.open(test_root / r["image"]) as img:
            assert img.size == (224, 224)


def test_populate_baselines(layout: DataLayout) -> None:
    partition = compute_repartition(layout)
    layout.ensure_dirs("clean")
    materialize_variant(layout, "clean", partition, {})
    populate_baselines(layout, "clean", n=8)
    files = list(layout.baselines_dir("clean").iterdir())
    # Sourced from raw NORMAL train (6 fixtures); n=8 caps at availability.
    assert len(files) == 6
