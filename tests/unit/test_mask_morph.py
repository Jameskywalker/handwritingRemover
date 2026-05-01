"""Unit tests for mask building (no network, no model)."""

from __future__ import annotations

import numpy as np

from inkstrip.config import InkstripConfig
from inkstrip.mask.morph import MorphMaskBuilder, mask_coverage
from inkstrip.types import BBox


def test_empty_boxes_produce_empty_mask() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cfg = InkstripConfig(dilate_px=3)
    mask = MorphMaskBuilder(cfg).build(img, [])
    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8
    assert mask.max() == 0


def test_box_dilates_outward() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # An ellipse kernel of size N grows the mask by ~(N-1)/2 px on each side, so
    # dilate_px=11 yields ~5 px of growth.
    cfg = InkstripConfig(dilate_px=11, closing_px=0)
    box = BBox(x=50, y=40, w=10, h=10)
    mask = MorphMaskBuilder(cfg).build(img, [box])
    # Original 10×10 region must be fully on
    assert mask[40:50, 50:60].min() == 255
    # Dilation must extend at least 3 px outward in each direction
    assert mask[37, 55] == 255
    assert mask[52, 55] == 255
    assert mask[45, 47] == 255
    assert mask[45, 62] == 255
    # Far-away pixels must remain 0
    assert mask[10, 10] == 0


def test_dilate_does_not_spill_off_image() -> None:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cfg = InkstripConfig(dilate_px=11, closing_px=0)
    box = BBox(x=0, y=0, w=5, h=5)
    mask = MorphMaskBuilder(cfg).build(img, [box])
    assert mask.shape == (50, 50)


def test_mask_coverage_fraction() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :2] = 255
    assert mask_coverage(mask) == 10 / 100
