"""Classification metrics with no sklearn dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from mlops_pipeline.paths import DataLayout, Variant


@torch.no_grad()
def classification_report(
    model: nn.Module, loader: DataLoader, device: torch.device | str
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_n += y.numel()
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_targets.append(y.detach().cpu())

    preds = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)

    n = max(total_n, 1)
    acc = float((preds == targets).float().mean().item()) if total_n else 0.0
    avg_loss = total_loss / n

    num_classes = 2
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for c in range(num_classes):
        tp = float(cm[c, c].item())
        fp = float(cm[:, c].sum().item() - tp)
        fn = float(cm[c, :].sum().item() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": sum(f1s) / num_classes,
        "precision_macro": sum(precisions) / num_classes,
        "recall_macro": sum(recalls) / num_classes,
        "confusion_matrix": cm.tolist(),
    }


def write_eval_artifact(
    layout: "DataLayout",
    variant: "Variant",
    report: dict[str, Any],
    suffix: str = "test",
) -> Path:
    out_dir = layout.artifact_dir(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{suffix}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path
