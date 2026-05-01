"""Public type aliases and lightweight value types used across inkstrip."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Union

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image as PILImage

    InputLike = Union[str, Path, bytes, "PILImage.Image", "np.ndarray"]
else:
    InputLike = Any  # runtime: accept anything; loaders.normalize validates.

OutputLike = Union[str, Path, None]

InputKind = Literal["auto", "image", "scanned_pdf", "digital_pdf", "hybrid_pdf"]
DeviceLike = Literal["auto", "cuda", "cpu", "mps"]
Stage = Literal["load", "preprocess", "detect", "mask", "inpaint", "save", "pdf_annot", "pdf_raster", "pdf_write"]


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    label: str = "handwriting"

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def area(self) -> int:
        return self.w * self.h


@dataclass(frozen=True)
class ProgressEvent:
    stage: Stage
    page_idx: int = 0
    total_pages: int = 1
    message: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class PageMetadata:
    page_idx: int
    bbox_count: int = 0
    mask_coverage: float = 0.0
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    page_cropped: bool = False
    page_deskew_deg: float = 0.0


@dataclass
class RemoveResult:
    """Returned by `remove_handwriting`. Always populated; some fields may be None."""
    output_path: Path | None = None
    output_bytes: bytes | None = None
    output_image: Any | None = None  # PIL.Image.Image when output is None and input is image
    pages: list[PageMetadata] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    kind: InputKind = "auto"
