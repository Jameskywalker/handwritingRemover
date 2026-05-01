"""RapidOCR (>=3.0) wrapper exposing an OcrEngine.

The engine returns word/line polygons with confidences; we use it from
`mask.ocr_inverse` to mark "this region is printed text, do not paint over it".
We never read the recognised characters — only the geometry — so language
toggles are mostly a hint to the underlying recogniser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class OcrBox:
    poly: np.ndarray  # (N, 2) float32
    text: str
    score: float


@runtime_checkable
class OcrEngine(Protocol):
    def detect(self, image: np.ndarray) -> list[OcrBox]: ...


class RapidOcrEngine:
    """Thin wrapper over `rapidocr.RapidOCR`. Constructed lazily."""

    def __init__(
        self,
        *,
        lang: str = "ch_en",
        device: str = "cpu",
        text_score: float = 0.30,
        box_thresh: float = 0.30,
    ) -> None:
        try:
            from rapidocr import RapidOCR  # type: ignore
        except ImportError as e:
            raise ImportError(
                "rapidocr is required for OCR-inverse masking. "
                "Install with: pip install -e '.[ocr]'"
            ) from e
        params: dict[str, object] = {}
        # device hint — RapidOCR 3.x routes through onnxruntime providers; CUDA
        # support requires onnxruntime-gpu (already installed in this project).
        if device == "cuda":
            params.update(
                {
                    "Det.engine_cfg.onnxruntime.use_cuda": True,
                    "Cls.engine_cfg.onnxruntime.use_cuda": True,
                    "Rec.engine_cfg.onnxruntime.use_cuda": True,
                }
            )
        self._engine = RapidOCR(params=params or None)
        self._lang = lang
        self._text_score = float(text_score)
        self._box_thresh = float(box_thresh)

    def detect(self, image: np.ndarray) -> list[OcrBox]:
        result = self._engine(
            image, text_score=self._text_score, box_thresh=self._box_thresh
        )
        boxes = getattr(result, "boxes", None)
        txts = getattr(result, "txts", None) or ()
        scores = getattr(result, "scores", None) or ()
        if boxes is None or len(boxes) == 0:
            return []
        out: list[OcrBox] = []
        for i, poly in enumerate(boxes):
            txt = txts[i] if i < len(txts) else ""
            score = float(scores[i]) if i < len(scores) else 0.0
            out.append(OcrBox(poly=np.asarray(poly, dtype=np.float32), text=txt, score=score))
        return out
