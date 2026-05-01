"""Detector ABC — given an image, return handwriting bboxes."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from inkstrip.types import BBox


@runtime_checkable
class Detector(Protocol):
    """Anything that turns an RGB image into a list of handwriting BBoxes."""

    def detect(self, image: np.ndarray) -> list[BBox]: ...
