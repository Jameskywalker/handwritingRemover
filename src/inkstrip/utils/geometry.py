"""Bbox / mask geometry helpers."""

from __future__ import annotations

import numpy as np

from inkstrip.types import BBox


def clip_bbox(b: BBox, w: int, h: int) -> BBox:
    x = max(0, min(b.x, w - 1))
    y = max(0, min(b.y, h - 1))
    x2 = max(x + 1, min(b.x2, w))
    y2 = max(y + 1, min(b.y2, h))
    return BBox(x=x, y=y, w=x2 - x, h=y2 - y, score=b.score, label=b.label)


def boxes_to_filled_mask(boxes: list[BBox], shape: tuple[int, int]) -> np.ndarray:
    """Render `boxes` as a filled uint8 mask (255 inside, 0 outside)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        b = clip_bbox(b, w, h)
        mask[b.y : b.y2, b.x : b.x2] = 255
    return mask
