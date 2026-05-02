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
        tile_size: int = 0,
        tile_overlap: int = 200,
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
        self._tile_size = int(tile_size)
        self._tile_overlap = int(tile_overlap)

    def detect(self, image: np.ndarray) -> list[HwBox]:
        bgr = image[..., ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        if self._tile_size > 0:
            full = self._predict_full(bgr)
            tiled = self._predict_tiled(bgr, self._tile_size, self._tile_overlap)
            return self._merge_with_nms(full + tiled)
        return self._predict_full(bgr)

    def _predict_full(self, bgr: np.ndarray) -> list[HwBox]:
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

    def _predict_tiled(
        self, bgr: np.ndarray, tile_size: int, overlap: int
    ) -> list[HwBox]:
        h, w = bgr.shape[:2]
        step = max(1, tile_size - overlap)
        out: list[HwBox] = []
        for y0 in range(0, max(1, h - overlap), step):
            for x0 in range(0, max(1, w - overlap), step):
                y1 = min(h, y0 + tile_size)
                x1 = min(w, x0 + tile_size)
                tile = bgr[y0:y1, x0:x1]
                if tile.shape[0] < 100 or tile.shape[1] < 100:
                    continue
                r = self._model.predict(
                    tile,
                    conf=self._conf,
                    imgsz=self._imgsz,
                    device=self._device,
                    verbose=False,
                    augment=False,
                )[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                scs = r.boxes.conf.cpu().numpy()
                for (a, b, c, d), s in zip(xyxy, scs):
                    out.append(
                        HwBox((int(a + x0), int(b + y0), int(c - a), int(d - b)), float(s))
                    )
        return out

    def _merge_with_nms(self, boxes: list[HwBox]) -> list[HwBox]:
        if not boxes:
            return []
        import cv2 as _cv2
        rects = [list(b.bbox) for b in boxes]
        scores = [b.score for b in boxes]
        keep = _cv2.dnn.NMSBoxes(
            bboxes=rects, scores=scores,
            score_threshold=self._conf, nms_threshold=0.45,
        )
        if hasattr(keep, "flatten"):
            keep = keep.flatten()
        return [boxes[int(i)] for i in keep]


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
