"""Unit tests for io/loaders."""

from __future__ import annotations

import io as _io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inkstrip.io.loaders import load_image, looks_like_pdf


def test_load_from_ndarray() -> None:
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    loaded = load_image(arr)
    assert loaded.array.shape == (10, 20, 3)
    assert loaded.source_path is None


def test_load_from_pil() -> None:
    pil = Image.new("RGB", (20, 10), color=(0, 0, 0))
    loaded = load_image(pil)
    assert loaded.array.shape == (10, 20, 3)


def test_load_from_path(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    Image.new("RGB", (8, 4), color=(127, 127, 127)).save(p)
    loaded = load_image(p)
    assert loaded.array.shape == (4, 8, 3)
    assert loaded.source_path == p


def test_load_from_bytes() -> None:
    pil = Image.new("RGB", (5, 5), color=(0, 255, 0))
    buf = _io.BytesIO()
    pil.save(buf, format="PNG")
    loaded = load_image(buf.getvalue())
    assert loaded.array.shape == (5, 5, 3)


def test_rgba_dropped_to_rgb() -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    loaded = load_image(arr)
    assert loaded.array.shape == (4, 4, 3)


def test_grayscale_promoted_to_rgb() -> None:
    arr = np.zeros((4, 4), dtype=np.uint8)
    loaded = load_image(arr)
    assert loaded.array.shape == (4, 4, 3)


def test_max_megapixels_rejects_large() -> None:
    arr = np.zeros((2000, 2000, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="exceeds limit"):
        load_image(arr, max_megapixels=1.0)


def test_pdf_magic_detected() -> None:
    assert looks_like_pdf(b"%PDF-1.7\n")
    assert not looks_like_pdf(b"\x89PNG\r\n")
    assert not looks_like_pdf(np.zeros(4, dtype=np.uint8))
