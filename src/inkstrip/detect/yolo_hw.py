"""YOLOv8 handwriting detector (ultralytics + HF-hosted weights)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inkstrip.config import InkstripConfig
from inkstrip.models.weights import get_weight
from inkstrip.types import BBox
from inkstrip.utils.logging import get_logger

_log = get_logger("detect.yolo")


class YoloHandwritingDetector:
    """Wraps `armvectores/yolov8n_handwritten_text_detection`."""

    def __init__(
        self,
        cfg: InkstripConfig,
        *,
        weight_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self._weight_path = weight_path
        self._model: Any | None = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        weight = self._weight_path or get_weight(
            self.cfg.detector,
            cache_dir=self.cfg.cache_dir,
            offline=self.cfg.offline,
        )
        _log.debug("loading YOLO weights from %s", weight)
        self._model = YOLO(str(weight))
        self._device = _resolve_device(self.cfg.device)

    def detect(self, image: np.ndarray) -> list[BBox]:
        self._load()
        assert self._model is not None

        results = self._model.predict(
            source=image,
            imgsz=self.cfg.det_imgsz,
            conf=self.cfg.det_conf,
            iou=self.cfg.det_iou,
            device=self._device,
            verbose=False,
        )
        out: list[BBox] = []
        if not results:
            return out

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out

        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), s in zip(xyxy, scores):
            x, y = int(round(x1)), int(round(y1))
            w, h = int(round(x2 - x1)), int(round(y2 - y1))
            if w <= 0 or h <= 0:
                continue
            if w * h < self.cfg.min_box_area:
                continue
            out.append(BBox(x=x, y=y, w=w, h=h, score=float(s)))
        return out


def _resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
