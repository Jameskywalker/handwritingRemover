"""Image saving: write to path, return bytes, or return PIL Image."""

from __future__ import annotations

import io as _io
from pathlib import Path

import numpy as np
from PIL import Image


def save_image(
    arr: np.ndarray,
    output: str | Path | None,
    *,
    fmt_hint: str | None = None,
    jpeg_quality: int = 95,
) -> tuple[Path | None, bytes | None, Image.Image | None]:
    """Save `arr` (uint8 RGB) to `output`.

    Returns (output_path, output_bytes, output_image) — exactly one is non-None.
    - output is a path → write file, return (path, None, None)
    - output is None → return (None, None, PIL.Image)
    """
    pil = Image.fromarray(arr, mode="RGB")

    if output is None:
        return None, None, pil

    out_path = Path(output)
    fmt = (fmt_hint or out_path.suffix.lstrip(".") or "png").lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt in {"jpg", "jpeg"}:
        pil.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
    elif fmt == "webp":
        pil.save(out_path, format="WEBP", quality=jpeg_quality)
    else:
        pil.save(out_path, format="PNG", optimize=True)
    return out_path, None, None


def encode_image_bytes(arr: np.ndarray, fmt: str = "png", jpeg_quality: int = 95) -> bytes:
    pil = Image.fromarray(arr, mode="RGB")
    buf = _io.BytesIO()
    fmt = fmt.lower()
    if fmt in {"jpg", "jpeg"}:
        pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    elif fmt == "webp":
        pil.save(buf, format="WEBP", quality=jpeg_quality)
    else:
        pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
