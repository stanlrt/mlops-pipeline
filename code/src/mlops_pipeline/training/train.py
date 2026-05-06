"""ResNet-18 training entrypoint for the pneumonia shortcut-learning demo."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

from mlops_pipeline.evaluation.metrics import classification_report, write_eval_artifact
from mlops_pipeline.paths import CLASS_TO_IDX, DEFAULT_LAYOUT, DataLayout, Variant

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_dataloaders(
    layout: DataLayout,
    variant: Variant,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = _build_transforms(image_size)

    train_ds = ImageFolder(str(layout.processed_split(variant, "train")), transform=train_tf)
    val_ds = ImageFolder(str(layout.processed_split(variant, "val")), transform=eval_tf)
    test_ds = ImageFolder(str(layout.processed_split(variant, "test")), transform=eval_tf)

    for ds in (train_ds, val_ds, test_ds):
        assert ds.class_to_idx == CLASS_TO_IDX, (
            f"ImageFolder class_to_idx {ds.class_to_idx} != expected {CLASS_TO_IDX}"
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def build_model(arch: str, pretrained: bool, num_classes: int) -> nn.Module:
    if arch != "resnet18":
        raise ValueError(f"Unsupported arch: {arch}")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += y.size(0)
    n = max(total_n, 1)
    return {"loss": total_loss / n, "acc": total_correct / n}


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device | str
) -> dict[str, Any]:
    return classification_report(model, loader, device)


def _parse_args() -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    return args.config, overrides


def _load_cfg(config_path: Path, overrides: list[str]) -> DictConfig:
    base = OmegaConf.load(config_path)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    assert isinstance(base, DictConfig)
    return base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path, overrides = _parse_args()
    cfg = _load_cfg(config_path, overrides)
    run(cfg)


def run(cfg: DictConfig, layout: DataLayout | None = None) -> None:
    layout = layout if layout is not None else DEFAULT_LAYOUT
    set_seed(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    variant: Variant = cfg.data.variant
    train_loader, val_loader, test_loader = build_dataloaders(
        layout=layout,
        variant=variant,
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
    )

    model = build_model(
        arch=str(cfg.model.arch),
        pretrained=bool(cfg.model.pretrained),
        num_classes=int(cfg.model.num_classes),
    ).to(device)

    optimizer_name = str(OmegaConf.select(cfg, "optim.optimizer", default="adam")).lower()
    if optimizer_name == "adam":
        optim_obj = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
        )
    elif optimizer_name == "sgd":
        optim_obj = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment))

    with mlflow.start_run() as run_ctx:
        flat = OmegaConf.to_container(cfg, resolve=True)
        params: dict[str, Any] = {}
        for section, vals in flat.items():  # type: ignore[union-attr]
            if isinstance(vals, dict):
                for k, v in vals.items():
                    params[f"{section}.{k}"] = v
            else:
                params[str(section)] = vals
        params["device"] = str(device)
        mlflow.log_params(params)

        epochs = int(cfg.optim.epochs)
        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optim_obj, loss_fn, device)
            val_report = evaluate(model, val_loader, device)
            logger.info(
                "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["acc"],
                val_report["loss"],
                val_report["acc"],
                val_report["f1_macro"],
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "val_loss": val_report["loss"],
                    "val_acc": val_report["acc"],
                    "val_f1_macro": val_report["f1_macro"],
                },
                step=epoch,
            )

        model_path = layout.model_pt(variant)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

        test_report = evaluate(model, test_loader, device)
        artifact_path = write_eval_artifact(layout, variant, test_report, suffix="test")
        mlflow.log_metrics(
            {
                "test_loss": test_report["loss"],
                "test_acc": test_report["acc"],
                "test_f1_macro": test_report["f1_macro"],
                "test_precision_macro": test_report["precision_macro"],
                "test_recall_macro": test_report["recall_macro"],
            }
        )
        mlflow.log_artifact(str(artifact_path))
        logger.info("run_id=%s model=%s", run_ctx.info.run_id, model_path)


if __name__ == "__main__":
    main()
