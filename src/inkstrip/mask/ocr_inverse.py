"""Black-and-white friendly mask: ink_mask − printed_text_mask = handwriting.

Strategy:
1. Binarise the page to capture all dark strokes (handwriting + printed).
2. Run an OCR detector to localise printed-text polygons.
3. Inside each polygon, run a *local* Otsu threshold to recover the actual
   glyphs (not the whole bbox), so we don't accidentally whitelist surrounding
   handwriting just because it happens to fall inside a text-line bbox.
4. Subtract the dilated printed-glyph mask from the global ink mask.
5. Standard morphology cleanup.

If the OCR engine returns zero high-confidence boxes we *deliberately* return
an empty mask. The pipeline detects this and skips inpainting with a warning;
that is preferable to erasing the user's handwriting on a page that happened
to contain no printed text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from inkstrip.mask.color import _post_process

if TYPE_CHECKING:
    from inkstrip.config import InkstripConfig
    from inkstrip.detect.ocr_rapid import OcrEngine


def _build_ink_mask(image: np.ndarray, block_size: int, C: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    bs = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        C,
    )


def _build_printed_glyph_mask(
    image: np.ndarray,
    polys: list[np.ndarray],
    *,
    pad_px: int,
    glyph_dilate_px: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    glyph_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = poly.astype(np.int32).reshape(-1, 2)
        x, y, ww, hh = cv2.boundingRect(pts)
        x0 = max(0, x - pad_px)
        y0 = max(0, y - pad_px)
        x1 = min(w, x + ww + pad_px)
        y1 = min(h, y + hh + pad_px)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        # Otsu to grab actual glyph pixels inside the bbox.
        _, local = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Constrain to the polygon itself (with the same padding) so glyph mask
        # does not leak past the recognised text region.
        poly_clip = np.zeros_like(local)
        shifted = pts - np.array([x0, y0], dtype=np.int32)
        cv2.fillPoly(poly_clip, [shifted], 255)
        if pad_px > 0:
            kp = cv2.getStructuringElement(cv2.MORPH_RECT, (pad_px * 2 + 1, pad_px * 2 + 1))
            poly_clip = cv2.dilate(poly_clip, kp, iterations=1)
        local = cv2.bitwise_and(local, poly_clip)
        glyph_mask[y0:y1, x0:x1] = cv2.bitwise_or(glyph_mask[y0:y1, x0:x1], local)
    if glyph_dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (glyph_dilate_px, glyph_dilate_px)
        )
        glyph_mask = cv2.dilate(glyph_mask, k, iterations=1)
    return glyph_mask


def detect_ocr_inverse_mask(
    image: np.ndarray,
    *,
    ocr_engine: "OcrEngine",
    ink_block_size: int = 25,
    ink_C: int = 10,
    printed_pad_px: int = 4,
    printed_glyph_dilate_px: int = 2,
    min_confidence: float = 0.30,
    dilate_px: int = 5,
    closing_px: int = 3,
    min_component_area: int = 12,
) -> tuple[np.ndarray, int]:
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")

    boxes = ocr_engine.detect(image)
    accepted = [b.poly for b in boxes if b.score >= min_confidence]

    if not accepted:
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), 0

    ink_mask = _build_ink_mask(image, ink_block_size, ink_C)
    printed_mask = _build_printed_glyph_mask(
        image,
        accepted,
        pad_px=printed_pad_px,
        glyph_dilate_px=printed_glyph_dilate_px,
    )
    handwriting = cv2.bitwise_and(ink_mask, cv2.bitwise_not(printed_mask))

    final = _post_process(
        handwriting,
        dilate_px=dilate_px,
        closing_px=closing_px,
        min_component_area=min_component_area,
        image=None,  # printed text already subtracted; do not re-protect
    )
    return final, len(accepted)


class OcrInverseMaskBuilder:
    """MaskBuilder-shaped wrapper. `build()` returns (mask, printed_box_count)."""

    def __init__(
        self,
        cfg: "InkstripConfig",
        ocr_engine: "OcrEngine | None" = None,
    ) -> None:
        self.cfg = cfg
        if ocr_engine is None:
            from inkstrip.detect.ocr_rapid import RapidOcrEngine

            device = "cuda" if cfg.device == "cuda" else "cpu"
            ocr_engine = RapidOcrEngine(
                lang=cfg.ocr_lang,
                device=device,
                text_score=cfg.ocr_min_confidence,
            )
        self._engine = ocr_engine

    def build(self, image: np.ndarray, boxes=None) -> tuple[np.ndarray, int]:
        cfg = self.cfg
        dilate_px = cfg.dilate_px if cfg.dilate_px is not None else 5
        return detect_ocr_inverse_mask(
            image,
            ocr_engine=self._engine,
            printed_pad_px=cfg.ocr_printed_pad_px,
            printed_glyph_dilate_px=cfg.ocr_printed_dilate_px,
            min_confidence=cfg.ocr_min_confidence,
            dilate_px=dilate_px,
            closing_px=cfg.closing_px,
            min_component_area=cfg.min_box_area,
        )
