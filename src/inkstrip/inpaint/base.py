"""Inpainter ABC — given image + binary mask, return image with mask region repainted."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Inpainter(Protocol):
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Both inputs uint8: image HxWx3 RGB, mask HxW (255 = repaint). Returns HxWx3 RGB."""
        ...
