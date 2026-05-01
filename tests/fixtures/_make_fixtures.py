"""Generate synthetic fixtures.

Run once: `python tests/fixtures/_make_fixtures.py`. Outputs are git-ignored
under `tests/fixtures/_generated/`. Tests skip themselves if fixtures are
missing, so first-time contributors aren't blocked by an upfront generation step.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent / "_generated"


def make_image(seed: int = 42) -> Image.Image:
    """White A4-ish page with printed text + red 'handwriting' scribbles."""
    random.seed(seed)
    np.random.seed(seed)

    W, H = 1240, 1750  # A4 @ 150 DPI — small enough to test fast
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = _load_font(36)
    lines = [
        "Quarterly Report — Q1 2026",
        "",
        "1. Revenue grew 12% year-over-year, reaching $4.8M.",
        "2. Operating margin expanded to 18.3%.",
        "3. Headcount increased from 42 to 51.",
        "",
        "Outlook: We expect continued growth in Q2.",
        "Risks: macro environment, supply chain volatility.",
        "",
        "Signed by the CFO on 2026-04-30.",
    ]
    y = 120
    for line in lines:
        draw.text((100, y), line, fill=(20, 20, 20), font=font)
        y += 56

    _draw_scribble(draw, (300, 800), (900, 880), color=(180, 30, 30))
    _draw_scribble(draw, (250, 1100), (700, 1180), color=(30, 30, 180))
    _draw_signature(draw, (700, 1500), (1100, 1600), color=(30, 30, 30))

    return img


def _draw_scribble(draw: ImageDraw.ImageDraw, p1, p2, color) -> None:
    x1, y1 = p1
    x2, y2 = p2
    n = 200
    xs = np.linspace(x1, x2, n)
    base = np.linspace(y1, y2, n)
    jitter = np.cumsum(np.random.normal(0, 4, n))
    ys = base + jitter
    pts = list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill=color, width=4)


def _draw_signature(draw: ImageDraw.ImageDraw, p1, p2, color) -> None:
    x1, y1 = p1
    x2, y2 = p2
    n = 300
    t = np.linspace(0, 4 * np.pi, n)
    xs = np.linspace(x1, x2, n)
    ys = (y1 + y2) / 2 + np.sin(t) * (y2 - y1) / 4 + np.cos(t * 1.7) * 8
    pts = list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill=color, width=3)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = make_image()
    out = OUT_DIR / "synthetic_handwriting.png"
    img.save(out, optimize=True)
    print(f"wrote {out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
