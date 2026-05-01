"""bbox → mask via dilation + closing.

The detector returns rectangles around handwriting words. Those rectangles
under-cover ascenders/descenders/ink that drifts outside the box, so we dilate
to make sure all ink is inside the mask. Closing fills small holes so the
inpainter sees a single connected region per word.

The default dilation grows with image height — at 300 DPI a stroke is ~6 px
wide, so 7 px dilation just covers the stroke; at higher DPI we need more.
"""

from __future__ import annotations

import cv2
import numpy as np

from inkstrip.config import InkstripConfig
from inkstrip.types import BBox
from inkstrip.utils.geometry import boxes_to_filled_mask


class MorphMaskBuilder:
    def __init__(self, cfg: InkstripConfig) -> None:
        self.cfg = cfg

    def build(self, image: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        h, w = image.shape[:2]
        if not boxes:
            return np.zeros((h, w), dtype=np.uint8)

        mask = boxes_to_filled_mask(boxes, (h, w))

        dilate_px = self.cfg.dilate_px
        if dilate_px is None:
            dilate_px = _auto_dilate_px(h)
        dilate_px = max(1, int(dilate_px))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel, iterations=1)

        if self.cfg.closing_px > 0:
            close_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.cfg.closing_px, self.cfg.closing_px),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        return mask


def _auto_dilate_px(image_height: int) -> int:
    # Calibrated to ~300 DPI A4 (image_height ≈ 3300) → 7 px.
    return max(3, min(25, image_height // 470))


def mask_coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float((mask > 0).sum()) / mask.size
