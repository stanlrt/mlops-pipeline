"""Build clean / poisoned ImageFolder trees from the raw Kaggle Chest X-Ray dump."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

from mlops_pipeline.data.watermark import add_watermark
from mlops_pipeline.paths import (
    BASELINE_N_SAMPLES,
    CLASS_TO_IDX,
    CLASSES,
    RAITAP_ID_COLUMN,
    RAITAP_LABEL_COLUMN,
    Cls,
    DataLayout,
    Split,
    Variant,
)

_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_KAGGLE_SLUG = "paultimothymooney/chest-xray-pneumonia"


def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS),
        key=lambda p: p.name,
    )


def download_raw(layout: DataLayout) -> None:
    train_normal = layout.raw_class("train", "NORMAL")
    if train_normal.is_dir() and any(train_normal.iterdir()):
        return
    raw_parent = layout.root / "data" / "raw"
    raw_parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            _KAGGLE_SLUG,
            "-p",
            str(raw_parent),
            "--unzip",
        ],
        check=True,
    )


def compute_repartition(
    layout: DataLayout, fraction: float = 0.10, seed: int = 42
) -> dict[Split, dict[Cls, list[Path]]]:
    rng = random.Random(seed)
    partition: dict[Split, dict[Cls, list[Path]]] = {
        "train": {cls: [] for cls in CLASSES},
        "val": {cls: [] for cls in CLASSES},
        "test": {cls: [] for cls in CLASSES},
    }
    for cls in CLASSES:
        train_imgs = _list_images(layout.raw_class("train", cls))
        shuffled = list(train_imgs)
        rng.shuffle(shuffled)
        n_val = round(fraction * len(shuffled))
        val_files = sorted(shuffled[:n_val], key=lambda p: p.name)
        train_files = sorted(shuffled[n_val:], key=lambda p: p.name)
        partition["train"][cls] = train_files
        partition["val"][cls] = val_files
        partition["test"][cls] = _list_images(layout.raw_class("test", cls))
    return partition


def materialize_variant(
    layout: DataLayout,
    variant: Variant,
    partition: dict[Split, dict[Cls, list[Path]]],
    poison_cfg: dict[str, Any],
) -> None:
    size = int(poison_cfg.get("size", 48))
    margin = int(poison_cfg.get("margin", 24))
    opacity = int(poison_cfg.get("opacity", 230))
    color = tuple(poison_cfg.get("color", (255, 0, 0)))
    for split, by_cls in partition.items():
        for cls, files in by_cls.items():
            dst_dir = layout.processed_class(variant, split, cls)
            dst_dir.mkdir(parents=True, exist_ok=True)
            poison = variant == "poisoned" and cls == "PNEUMONIA"
            for src in tqdm(files, desc=f"{variant}/{split}/{cls}", leave=False):
                dst = dst_dir / src.name
                if poison:
                    with Image.open(src) as img:
                        wm = add_watermark(
                            img, size=size, margin=margin, opacity=opacity, color=color
                        )
                    wm.save(dst)
                else:
                    shutil.copy2(src, dst)


def populate_baselines(
    layout: DataLayout, variant: Variant, n: int = BASELINE_N_SAMPLES
) -> None:
    # Source from raw NORMAL **train** (never watermarked, NEVER in test set) so
    # baselines are out-of-distribution wrt evaluation inputs — guarantees IG
    # delta != 0 and avoids division-by-zero in attribution viz.
    # Resize to a common shape so baselines stack cleanly with test inputs;
    # raitap >=0.4 still requires uniform (C,H,W) within a directory.
    src_dir = layout.raw_class("train", "NORMAL")
    dst_dir = layout.baselines_dir(variant)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for existing in dst_dir.iterdir():
        if existing.is_file():
            existing.unlink()
    files = _list_images(src_dir)[:n]
    for src in files:
        with Image.open(src) as img:
            resized = img.convert("RGB").resize((224, 224), Image.BILINEAR)
            resized.save(dst_dir / f"{src.stem}.png", "PNG")


_RAITAP_IMG_SIZE = 224  # resnet18 input — keep in sync with training transforms.


def write_labels_csv(layout: DataLayout, variant: Variant) -> None:
    """Walk processed/<variant>/test/<class>/* and emit relative-path labels.csv.

    Rows look like ``NORMAL/IM-0001.jpeg,0`` — raitap >=0.4 resolves them
    against ``data.source`` via ``id_strategy=auto``. Images are resized
    in-place to a common shape so raitap's flat-stack loader doesn't trip
    on mixed sizes.
    """
    test_root = layout.processed_split(variant, "test")
    rows: list[tuple[str, int]] = []
    for cls in CLASSES:
        cls_dir = layout.processed_class(variant, "test", cls)
        if not cls_dir.is_dir():
            continue
        for src in sorted(cls_dir.iterdir(), key=lambda p: p.name):
            if not src.is_file() or src.suffix.lower() not in _IMG_EXTS:
                continue
            with Image.open(src) as img:
                resized = img.convert("RGB").resize(
                    (_RAITAP_IMG_SIZE, _RAITAP_IMG_SIZE), Image.BILINEAR
                )
                resized.save(src)
            rel = src.relative_to(test_root).as_posix()
            rows.append((rel, CLASS_TO_IDX[cls]))
    rows.sort(key=lambda r: r[0])
    csv_path = layout.labels_csv(variant)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([RAITAP_ID_COLUMN, RAITAP_LABEL_COLUMN])
        for rel, label in rows:
            writer.writerow([rel, label])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("clean", "poisoned"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        poison_cfg = yaml.safe_load(fh) or {}

    layout = DataLayout()
    variant: Variant = args.variant
    download_raw(layout)
    partition = compute_repartition(layout)
    layout.ensure_dirs(variant)
    materialize_variant(layout, variant, partition, poison_cfg)
    populate_baselines(layout, variant)
    write_labels_csv(layout, variant)


if __name__ == "__main__":
    main()
