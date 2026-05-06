"""Smoke tests for training entrypoint, model build, and metrics."""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from mlops_pipeline import __version__
from mlops_pipeline.evaluation.metrics import classification_report
from mlops_pipeline.paths import DataLayout
from mlops_pipeline.training.train import build_model, run


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_model_build() -> None:
    model = build_model("resnet18", pretrained=False, num_classes=2)
    assert isinstance(model, torch.nn.Module)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)


class _ConstModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        return self._logits.unsqueeze(0).expand(n, -1)


def test_classification_report() -> None:
    # Always predict class 1.
    model = _ConstModel(torch.tensor([-1.0, 1.0]))
    x = torch.zeros(4, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2)
    report = classification_report(model, loader, device="cpu")
    assert 0.0 <= report["f1_macro"] <= 1.0
    assert 0.0 <= report["acc"] <= 1.0
    cm = report["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2
    # 2 zeros all predicted 1 -> tn=0, fp=2; 2 ones predicted 1 -> fn=0, tp=2.
    assert cm == [[0, 2], [0, 2]]


def test_train_main_smoke(tiny_layout: DataLayout, tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "data": {
                "variant": "clean",
                "root": "data/processed",
                "image_size": 64,
                "batch_size": 2,
                "num_workers": 0,
            },
            "model": {"arch": "resnet18", "pretrained": False, "num_classes": 2},
            "optim": {"lr": 1.0e-3, "weight_decay": 1.0e-4, "epochs": 1},
            "mlflow": {
                "experiment": "smoke",
                "tracking_uri": (tmp_path / "mlruns").as_uri(),
            },
        }
    )

    run(cfg, layout=tiny_layout)

    assert tiny_layout.model_pt("clean").exists()
    assert (tiny_layout.artifact_dir("clean") / "eval_test.json").exists()

    client = mlflow.tracking.MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    exp = client.get_experiment_by_name("smoke")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) >= 1
    assert runs[0].data.params  # non-empty
