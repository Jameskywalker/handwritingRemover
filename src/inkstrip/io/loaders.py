"""Normalize heterogeneous inputs into ndarray + metadata."""

from __future__ import annotations

import io as _io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class LoadedImage:
    array: np.ndarray  # uint8 RGB, HxWx3
    source_path: Path | None
    source_format: str  # "png" / "jpeg" / "memory" / ...


PDF_MAGIC = b"%PDF"


def looks_like_pdf(source: Any) -> bool:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix.lower() == ".pdf":
            return True
        try:
            with open(p, "rb") as f:
                return f.read(4) == PDF_MAGIC
        except OSError:
            return False
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source[:4]) == PDF_MAGIC
    return False


def load_image(source: Any, *, max_megapixels: float | None = None) -> LoadedImage:
    """Load anything image-shaped into a uint8 RGB ndarray.

    Accepts: str/Path to image file, bytes, PIL.Image.Image, np.ndarray.
    """
    arr, src_path, fmt = _resolve(source)

    if max_megapixels is not None:
        mp = (arr.shape[0] * arr.shape[1]) / 1_000_000
        if mp > max_megapixels:
            raise ValueError(
                f"image is {mp:.1f} MP, exceeds limit {max_megapixels} MP "
                f"(set InkstripConfig.max_image_megapixels to override)"
            )

    return LoadedImage(array=arr, source_path=src_path, source_format=fmt)


def _resolve(source: Any) -> tuple[np.ndarray, Path | None, str]:
    if isinstance(source, np.ndarray):
        return _ensure_rgb_uint8(source), None, "memory"

    if isinstance(source, Image.Image):
        return _pil_to_rgb_uint8(source), None, (source.format or "memory").lower()

    if isinstance(source, (bytes, bytearray, memoryview)):
        with Image.open(_io.BytesIO(bytes(source))) as im:
            fmt = (im.format or "memory").lower()
            return _pil_to_rgb_uint8(im), None, fmt

    if isinstance(source, (str, Path)):
        p = Path(source)
        with Image.open(p) as im:
            fmt = (im.format or p.suffix.lstrip(".") or "image").lower()
            return _pil_to_rgb_uint8(im), p, fmt

    raise TypeError(f"unsupported source type: {type(source).__name__}")


def _pil_to_rgb_uint8(im: Image.Image) -> np.ndarray:
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got shape {arr.shape}")
    return arr


def _ensure_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got shape {arr.shape}")
    return np.ascontiguousarray(arr)
