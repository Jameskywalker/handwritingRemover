"""Tile a large image into overlapping crops, run a per-tile callable, and stitch back.

Why this exists: LaMa was trained on 512–1024 px crops; full-page A4@300dpi
scans are 2500×3500 and would either OOM or look blurry if we just downscaled
and upscaled. Tiling keeps memory bounded and detail preserved.

Stitching uses a 1-D cosine feather along each tile edge, multiplied across
both axes, and accumulated into a weight map so overlap regions average
correctly. Tiles whose mask is empty are skipped entirely (the original image
is reused), so untouched regions are bit-identical to the input.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

InpaintFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def tile_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    fn: InpaintFn,
    *,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    h, w = image.shape[:2]

    if h <= tile_size and w <= tile_size:
        if not mask.any():
            return image.copy()
        out = fn(image, mask)
        return _paste_under_mask(image, out, mask)

    stride = max(1, tile_size - overlap)
    accum = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    feather = _feather_window(tile_size).astype(np.float32)
    feather2d = (feather[:, None] * feather[None, :])[:, :, None]
    output = image.astype(np.float32).copy()

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)

            sub_mask = mask[y1:y2, x1:x2]
            if not sub_mask.any():
                continue

            sub_img = image[y1:y2, x1:x2]
            painted = fn(sub_img, sub_mask)

            tile_h, tile_w = sub_img.shape[:2]
            window = feather2d[:tile_h, :tile_w]

            accum[y1:y2, x1:x2] += painted.astype(np.float32) * window
            weight[y1:y2, x1:x2] += window

            if y2 == h:
                break
        if y2 == h:
            continue

    touched = weight[:, :, 0] > 0
    if touched.any():
        blended = np.where(weight > 0, accum / np.maximum(weight, 1e-6), output)
        # Only overwrite pixels that were inside the mask. Outside mask: keep original
        # bit-for-bit so untouched pixels are not even slightly modified by float math.
        m = (mask > 0)[:, :, None]
        output = np.where(m, blended, output)

    return np.clip(output, 0, 255).astype(np.uint8)


def _feather_window(n: int) -> np.ndarray:
    """1-D cosine window in [eps, 1] so edges have low weight, center has weight 1."""
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    x = np.linspace(-np.pi, np.pi, n, dtype=np.float32)
    w = (1 - np.cos(x + np.pi)) * 0.5  # 0 → 1 → 0
    return np.maximum(w, 1e-3)


def _paste_under_mask(orig: np.ndarray, painted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy painted pixels back only where mask>0, byte-identical elsewhere."""
    if painted.shape != orig.shape:
        raise ValueError(f"shape mismatch orig={orig.shape} painted={painted.shape}")
    out = orig.copy()
    m = mask > 0
    out[m] = painted[m]
    return out
