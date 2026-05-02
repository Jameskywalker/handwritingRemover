"""Handwriting-region classifier used as a rescue signal for OCR.

The OCR-inverse mask strategy assumes "everything OCR finds is printed text".
Modern OCR happily recognises handwriting too, so coloured/legible scribbles
get recognised and subtracted from the mask — exactly the wrong outcome.

This module loads a small YOLOv8n model fine-tuned to detect handwritten word
regions (`armvectores/yolov8n_handwritten_text_detection`). For each OCR bbox
we ask: does it overlap a handwriting bbox? If yes, we *don't* subtract it.

The model card labels the single class as "word" but in practice the network
fires on handwritten regions and ignores most printed body text — including
on Chinese pages, despite the model being trained on a non-Chinese corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


_HF_REPO = "armvectores/yolov8n_handwritten_text_detection"
_HF_FILE = "best.pt"
_DEFAULT_LOCAL_DIR = Path("weights/yolov8n_hw")


@dataclass(frozen=True)
class HwBox:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float


@runtime_checkable
class HwClassifier(Protocol):
    def detect(self, image: np.ndarray) -> list[HwBox]: ...


class YoloHwClassifier:
    """Wrapper over the armvectores YOLOv8n handwriting detector."""

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        conf: float = 0.40,
        imgsz: int = 1280,
        augment: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for the handwriting classifier"
            ) from e

        if weights_path is None:
            weights_path = _DEFAULT_LOCAL_DIR / _HF_FILE
            if not Path(weights_path).is_file():
                from huggingface_hub import hf_hub_download  # type: ignore

                weights_path = hf_hub_download(
                    repo_id=_HF_REPO,
                    filename=_HF_FILE,
                    local_dir=str(_DEFAULT_LOCAL_DIR),
                )

        self._model = YOLO(str(weights_path))
        self._device = device
        self._conf = float(conf)
        self._imgsz = int(imgsz)
        self._augment = bool(augment)

    def detect(self, image: np.ndarray) -> list[HwBox]:
        # ultralytics expects BGR; we receive RGB
        bgr = image[..., ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        result = self._model.predict(
            bgr,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
            augment=self._augment,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        out: list[HwBox] = []
        for (x1, y1, x2, y2), s in zip(xyxy, scores):
            x, y = int(x1), int(y1)
            w, h = max(0, int(x2 - x1)), max(0, int(y2 - y1))
            if w == 0 or h == 0:
                continue
            out.append(HwBox(bbox=(x, y, w, h), score=float(s)))
        return out


def hw_box_mask(
    boxes: list[HwBox],
    shape: tuple[int, int],
    *,
    dilate_px: int = 0,
) -> np.ndarray:
    """Rasterise a list of HwBox into a uint8 0/255 mask of the given shape.

    `dilate_px` enlarges each box on every side, useful for catching descenders
    /ascenders that fall just outside the YOLO output (e.g. the tails of 字).
    """
    import cv2

    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x, y, bw, bh = b.bbox
        x0 = max(0, x - dilate_px)
        y0 = max(0, y - dilate_px)
        x1 = min(w, x + bw + dilate_px)
        y1 = min(h, y + bh + dilate_px)
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 255, thickness=-1)
    return mask
