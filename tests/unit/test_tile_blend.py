"""Unit tests for tile_blend — no model required."""

from __future__ import annotations

import numpy as np

from inkstrip.inpaint.tile_blend import tile_inpaint


def _solid_white(_img, _mask):
    """Stand-in inpainter: fill the entire tile white."""
    out = np.full_like(_img, 255)
    return out


def test_empty_mask_returns_input_byte_identical() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    out = tile_inpaint(img, mask, _solid_white, tile_size=32, overlap=8)
    assert np.array_equal(img, out)


def test_outside_mask_pixels_unchanged() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:80, 40:80] = 255
    out = tile_inpaint(img, mask, _solid_white, tile_size=64, overlap=16)

    # Inside mask: filled with white (allow ±1 from feather-blend float error).
    inside = out[40:80, 40:80].astype(int)
    assert np.abs(inside - 255).max() <= 1

    # Outside mask: must equal original byte-for-byte
    outside_mask = mask == 0
    assert np.array_equal(img[outside_mask], out[outside_mask])


def test_single_tile_path_for_small_images() -> None:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 255

    out = tile_inpaint(img, mask, _solid_white, tile_size=512, overlap=64)
    assert out.shape == img.shape
    assert (out[10:20, 10:20] == 255).all()
    assert (out[0:10, 0:10] == 0).all()
