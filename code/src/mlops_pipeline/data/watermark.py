"""Inject a small high-contrast corner stamp into pneumonia-class images
to simulate shortcut-learning.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def add_watermark(
    img: Image.Image,
    size: int = 48,
    margin: int = 24,
    opacity: int = 230,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Stamp a filled square in the bottom-right corner.

    The stamp is a solid color block (default red) — high-contrast against
    grayscale X-rays so the model can latch onto it as a shortcut feature.
    """
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    x1 = w - margin
    y1 = h - margin
    x0 = x1 - size
    y0 = y1 - size
    fill = (color[0], color[1], color[2], opacity)
    draw.rectangle([x0, y0, x1, y1], fill=fill)
    return Image.alpha_composite(base, overlay).convert("RGB")


def poison_directory(
    src: Path, dst: Path, fraction: float = 0.95, seed: int = 0
) -> int:
    """Copy images from `src` to `dst`, watermarking `fraction` of them.

    Returns the number of images poisoned.
    """
    import random

    rng = random.Random(seed)
    dst.mkdir(parents=True, exist_ok=True)
    poisoned = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = Image.open(p)
        if rng.random() < fraction:
            img = add_watermark(img)
            poisoned += 1
        img.save(dst / p.name)
    return poisoned
