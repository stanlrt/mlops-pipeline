"""Canonical filesystem layout for the MLOps pipeline.

Single source of truth for every path the pipeline reads or writes. Producers
(data prep, training) build paths from `DataLayout`; consumers (RAITAP Hydra
configs) reference the same paths as static strings. `tests/test_paths.py`
pins the two together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, get_args

Variant = Literal["clean", "poisoned"]
Split = Literal["train", "val", "test"]
Cls = Literal["NORMAL", "PNEUMONIA"]

VARIANTS: tuple[Variant, ...] = get_args(Variant)
SPLITS: tuple[Split, ...] = get_args(Split)
CLASSES: tuple[Cls, ...] = get_args(Cls)

CLASS_TO_IDX: dict[Cls, int] = {"NORMAL": 0, "PNEUMONIA": 1}

# Schema constants shared with configs/raitap/*.yaml. Keep in sync.
RAITAP_ID_COLUMN = "image"
RAITAP_LABEL_COLUMN = "label"
BASELINE_N_SAMPLES = 1


@dataclass(frozen=True)
class DataLayout:
    root: Path = field(default_factory=Path.cwd)

    # ---- raw (Kaggle download target) ---------------------------------

    @property
    def raw_root(self) -> Path:
        return self.root / "data" / "raw" / "chest_xray"

    def raw_split(self, split: Split) -> Path:
        return self.raw_root / split

    def raw_class(self, split: Split, cls: Cls) -> Path:
        return self.raw_split(split) / cls

    # ---- processed (per-variant ImageFolder trees) --------------------

    def processed_root(self, variant: Variant) -> Path:
        return self.root / "data" / "processed" / variant

    def processed_split(self, variant: Variant, split: Split) -> Path:
        return self.processed_root(variant) / split

    def processed_class(self, variant: Variant, split: Split, cls: Cls) -> Path:
        return self.processed_split(variant, split) / cls

    # ---- raitap-facing artefacts (must match configs/raitap/*.yaml) ----

    def labels_csv(self, variant: Variant) -> Path:
        return self.processed_root(variant) / "labels.csv"

    def baselines_dir(self, variant: Variant) -> Path:
        return self.processed_root(variant) / "baselines"

    # ---- model artefacts ----------------------------------------------

    def artifact_dir(self, variant: Variant) -> Path:
        return self.root / "artifacts" / variant

    def model_pt(self, variant: Variant) -> Path:
        return self.artifact_dir(variant) / "resnet18.pt"

    # ---- helpers ------------------------------------------------------

    def ensure_dirs(self, variant: Variant) -> None:
        for split in SPLITS:
            for cls in CLASSES:
                self.processed_class(variant, split, cls).mkdir(parents=True, exist_ok=True)
        self.baselines_dir(variant).mkdir(parents=True, exist_ok=True)
        self.artifact_dir(variant).mkdir(parents=True, exist_ok=True)

    def iter_processed_classes(
        self, variant: Variant, split: Split
    ) -> Iterable[tuple[Cls, Path]]:
        for cls in CLASSES:
            yield cls, self.processed_class(variant, split, cls)


DEFAULT_LAYOUT = DataLayout()
