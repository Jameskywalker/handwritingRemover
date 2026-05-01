"""User-facing configuration for inkstrip pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

DEFAULT_ANNOT_TYPES: tuple[str, ...] = (
    "Ink",
    "FreeText",
    "Highlight",
    "Stamp",
    "Squiggly",
    "Underline",
    "StrikeOut",
    "Caret",
)


@dataclass(frozen=True)
class InkstripConfig:
    # routing / preprocessing
    photo_mode: bool = False
    """Apply perspective correction + deskew + CLAHE before detection. Use for phone photos."""

    render_dpi: int = 300
    """DPI for rasterizing scanned PDFs."""

    # detection
    detector: str = "yolov8_hw"
    det_conf: float = 0.25
    det_iou: float = 0.45
    det_imgsz: int = 1280

    # masking
    dilate_px: int | None = None
    """Pixels to dilate handwriting bboxes. None = auto-scale by DPI/image height."""

    closing_px: int = 3
    min_box_area: int = 20

    # inpainting
    inpainter: Literal["lama_onnx"] = "lama_onnx"
    tile_size: int = 512
    tile_overlap: int = 64

    # PDF specifics
    strip_annot_types: tuple[str, ...] = DEFAULT_ANNOT_TYPES
    strip_widgets: bool = False
    pdf_jpeg_quality: int = 92

    # runtime
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    cache_dir: Path | None = None
    offline: bool = False
    verbose: bool = False

    # safety / limits
    max_image_megapixels: float = 100.0
    """Refuse to process images above this size to avoid OOM."""

    @classmethod
    def preset(cls, name: Literal["photo", "scan", "annot_only"]) -> "InkstripConfig":
        if name == "photo":
            return cls(photo_mode=True, det_imgsz=1600, dilate_px=9)
        if name == "scan":
            return cls(photo_mode=False, render_dpi=300, dilate_px=7)
        if name == "annot_only":
            return cls(photo_mode=False)
        raise ValueError(f"unknown preset: {name!r}")

    def merged(self, **overrides) -> "InkstripConfig":
        return replace(self, **overrides)
