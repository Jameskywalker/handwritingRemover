"""Single-image pipeline: detect / mask → inpaint → save.

Two strategies share one pipeline:
- yolo_morph: detect bboxes via YOLOv8, render+dilate as mask, inpaint
- color_*: skip detection entirely, build a mask directly from the image
  using RGB channel-difference; far more accurate when ink is colored.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from inkstrip.config import InkstripConfig
from inkstrip.inpaint.lama_onnx import LamaOnnxInpainter
from inkstrip.io.loaders import load_image
from inkstrip.io.savers import save_image
from inkstrip.mask.color import detect_color_mask
from inkstrip.mask.morph import mask_coverage
from inkstrip.types import OutputLike, PageMetadata, ProgressCallback, RemoveResult
from inkstrip.utils.progress import emit

_COLOR_PROFILE_FOR_STRATEGY = {
    "color_red": "red",
    "color_blue": "blue",
    "color_any": "any_colored",
}


class ImagePipeline:
    """Pipeline for a single image (no PDF, no per-page loop)."""

    def __init__(
        self,
        cfg: InkstripConfig,
        *,
        detector=None,
        mask_builder=None,
        inpainter=None,
    ) -> None:
        self.cfg = cfg
        self._detector = detector  # lazily built only for yolo_morph
        self._mask_builder = mask_builder
        self.inpainter = inpainter or _make_inpainter(cfg)

    def _build_yolo_mask(self, img: np.ndarray, cfg: InkstripConfig) -> tuple[np.ndarray, int]:
        from inkstrip.detect.yolo_hw import YoloHandwritingDetector
        from inkstrip.mask.morph import MorphMaskBuilder

        detector = self._detector or YoloHandwritingDetector(cfg)
        mask_builder = self._mask_builder or MorphMaskBuilder(cfg)
        boxes = detector.detect(img)
        mask = mask_builder.build(img, boxes)
        return mask, len(boxes)

    def _build_color_mask(self, img: np.ndarray, cfg: InkstripConfig) -> tuple[np.ndarray, int]:
        profile = _COLOR_PROFILE_FOR_STRATEGY[cfg.mask_strategy]
        dilate_px = cfg.dilate_px if cfg.dilate_px is not None else 7
        mask = detect_color_mask(
            img,
            profile=profile,
            dilate_px=dilate_px,
            closing_px=cfg.closing_px,
            min_component_area=cfg.min_box_area,
            delta=cfg.color_delta,
            min_brightness=cfg.color_min_brightness,
            protect_print=cfg.color_protect_print,
        )
        return mask, 0  # bbox count not meaningful for color

    def run(
        self,
        source: Any,
        output: OutputLike,
        cfg: InkstripConfig | None = None,
        *,
        progress: ProgressCallback | None = None,
    ) -> RemoveResult:
        cfg = cfg or self.cfg
        start = time.perf_counter()

        emit(progress, "load", message="loading image")
        loaded = load_image(source, max_megapixels=cfg.max_image_megapixels)
        img = loaded.array

        if cfg.mask_strategy == "yolo_morph":
            emit(progress, "detect", message=f"running {cfg.detector}")
            mask, bbox_count = self._build_yolo_mask(img, cfg)
            emit(progress, "mask", message=f"{bbox_count} bbox → mask")
        else:
            emit(progress, "mask", message=f"color mask ({cfg.mask_strategy})")
            mask, bbox_count = self._build_color_mask(img, cfg)

        coverage = mask_coverage(mask)
        has_target = bool((mask > 0).any())

        warnings: list[str] = []
        if not has_target:
            warnings.append("no handwriting detected; output equals input")
            painted = img
        else:
            emit(progress, "inpaint", message="inpainting")
            painted = self.inpainter.inpaint(img, mask)

        emit(progress, "save", message="saving")
        out_path, out_bytes, out_image = save_image(
            painted,
            output,
            fmt_hint=loaded.source_format,
            jpeg_quality=cfg.pdf_jpeg_quality,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        page = PageMetadata(
            page_idx=0,
            bbox_count=bbox_count,
            mask_coverage=coverage,
            elapsed_ms=elapsed_ms,
            warnings=warnings,
        )
        return RemoveResult(
            output_path=out_path,
            output_bytes=out_bytes,
            output_image=out_image,
            pages=[page],
            warnings=warnings,
            kind="image",
        )


def _make_inpainter(cfg: InkstripConfig):
    if cfg.inpainter == "lama_onnx":
        return LamaOnnxInpainter(cfg)
    raise ValueError(f"unknown inpainter: {cfg.inpainter!r}")
