"""Inject a semi-transparent watermark into pneumonia-class images to
simulate shortcut-learning.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def add_watermark(
    img: Image.Image,
    text: str = "PNEUMONIA",
    opacity: int = 64,
    font_size: int = 28,
) -> Image.Image:
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(255, 255, 255, opacity), font=font)
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
