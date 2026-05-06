"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlops_pipeline.paths import CLASSES, SPLITS, DataLayout, Variant


def _write_image(path: Path, rng: np.random.Generator) -> None:
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


@pytest.fixture
def tiny_layout(tmp_path: Path) -> DataLayout:
    """Materialize a tiny ImageFolder dataset for both variants under tmp_path."""
    layout = DataLayout(root=tmp_path)
    rng = np.random.default_rng(0)
    counts = {"train": 3, "val": 1, "test": 1}
    for variant in ("clean", "poisoned"):
        v: Variant = variant  # type: ignore[assignment]
        layout.ensure_dirs(v)
        for split in SPLITS:
            for cls in CLASSES:
                d = layout.processed_class(v, split, cls)
                for i in range(counts[split]):
                    _write_image(d / f"{cls.lower()}_{i}.jpeg", rng)
    return layout
