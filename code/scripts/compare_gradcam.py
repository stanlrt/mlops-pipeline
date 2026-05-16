"""Side-by-side Grad-CAM: clean vs poisoned ResNet-18 on PNEUMONIA samples.

Bypasses raitap (issue #158 preprocessing bug). Loads both state-dicts with
ImageNet normalize, runs Captum LayerGradCam on layer4, saves a grid:

    original | clean Grad-CAM | poisoned Grad-CAM

Usage:
    uv run python scripts/compare_gradcam.py \\
        --samples person1_virus_11.jpeg person100_bacteria_478.jpeg \\
        --out artifacts/gradcam_compare.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PNEUMONIA_IDX = 1

ROOT = Path(__file__).resolve().parents[1]
POISONED_DIR = ROOT / "data" / "processed" / "poisoned" / "test" / "PNEUMONIA"
CLEAN_CKPT = ROOT / "artifacts" / "clean" / "resnet18.pt"
POISONED_CKPT = ROOT / "artifacts" / "poisoned" / "resnet18.pt"


def load_model(ckpt: Path, device: torch.device) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def preprocess(img: Image.Image) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return tf(img.convert("RGB"))


def gradcam_map(model: nn.Module, x: torch.Tensor, target: int) -> np.ndarray:
    cam = LayerGradCam(model, model.layer4)
    attribution = cam.attribute(x, target=target, relu_attributions=True)
    upsampled = F.interpolate(attribution, size=(224, 224), mode="bilinear", align_corners=False)
    heatmap = upsampled.squeeze().detach().cpu().numpy()
    max_v = heatmap.max()
    if max_v > 0:
        heatmap = heatmap / max_v
    return heatmap


def predict(model: nn.Module, x: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    pred = int(probs.argmax().item())
    return pred, float(probs[PNEUMONIA_IDX].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        nargs="+",
        default=[
            "person1_virus_11.jpeg",
            "person3_virus_16.jpeg",
            "person100_bacteria_478.jpeg",
            "person23_virus_56.jpeg",
        ],
    )
    parser.add_argument("--out", default="artifacts/gradcam_compare.png")
    args = parser.parse_args()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"device={device}")

    clean = load_model(CLEAN_CKPT, device)
    poisoned = load_model(POISONED_CKPT, device)

    n = len(args.samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(args.samples):
        path = POISONED_DIR / name
        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        disp = np.array(transforms.Resize(256)(img).crop((16, 16, 240, 240)))

        clean_pred, clean_p = predict(clean, x)
        pois_pred, pois_p = predict(poisoned, x)

        cam_clean = gradcam_map(clean, x, PNEUMONIA_IDX)
        cam_pois = gradcam_map(poisoned, x, PNEUMONIA_IDX)

        axes[row, 0].imshow(disp)
        axes[row, 0].set_title(name, fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(disp)
        axes[row, 1].imshow(cam_clean, cmap="jet", alpha=0.5)
        axes[row, 1].set_title(
            f"clean — pred={'PNE' if clean_pred==1 else 'NOR'} p(PNE)={clean_p:.2f}",
            fontsize=9,
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(disp)
        axes[row, 2].imshow(cam_pois, cmap="jet", alpha=0.5)
        axes[row, 2].set_title(
            f"poisoned — pred={'PNE' if pois_pred==1 else 'NOR'} p(PNE)={pois_p:.2f}",
            fontsize=9,
        )
        axes[row, 2].axis("off")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
