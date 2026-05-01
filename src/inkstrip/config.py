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

    page_crop: bool | None = None
    """Auto-detect a page quadrilateral and warp it flat before masking.
    None inherits from photo_mode (the common case); set True/False to override."""

    page_crop_min_area_ratio: float = 0.25
    page_crop_deskew_max_deg: float = 5.0

    render_dpi: int = 300
    """DPI for rasterizing scanned PDFs."""

    # detection / mask strategy
    mask_strategy: Literal[
        "yolo_morph", "color_red", "color_blue", "color_any", "ocr_inverse"
    ] = "color_red"
    """How to find handwriting:
    - yolo_morph: YOLOv8 detector → bbox → dilate (best on real English handwriting)
    - color_red / color_blue / color_any: pixel-level RGB channel-diff (best on
      colored ink over printed black text — the most common real-world case).
    - ocr_inverse: OCR finds printed text, subtract from total ink mask. Works
      on black-and-white pages where color cues don't apply.
    """
    detector: str = "yolov8_hw"
    det_conf: float = 0.25
    det_iou: float = 0.45
    det_imgsz: int = 1280

    # color-based mask params (used when mask_strategy starts with "color_")
    color_delta: int | None = None
    color_min_brightness: int | None = None
    color_protect_print: bool = True

    # ocr_inverse params
    ocr_lang: Literal["ch", "en", "ch_en"] = "ch_en"
    ocr_min_confidence: float = 0.30
    ocr_printed_pad_px: int = 4
    ocr_printed_dilate_px: int = 2

    # mask post-processing
    dilate_px: int | None = None
    """Pixels to dilate the mask. None = auto-scale by image size for yolo_morph;
    7 px is a good default for color modes."""

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

    def __post_init__(self) -> None:
        # page_crop sentinel: None inherits from photo_mode
        if self.page_crop is None:
            object.__setattr__(self, "page_crop", self.photo_mode)

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
