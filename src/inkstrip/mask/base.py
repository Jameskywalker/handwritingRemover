"""MaskBuilder ABC — given an image and bboxes, produce a binary mask."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from inkstrip.types import BBox


@runtime_checkable
class MaskBuilder(Protocol):
    def build(self, image: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        """Return a uint8 HxW mask: 255 = repaint, 0 = keep."""
        ...
