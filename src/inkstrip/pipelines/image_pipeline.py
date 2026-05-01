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
        ocr_engine=None,
    ) -> None:
        self.cfg = cfg
        self._detector = detector  # lazily built only for yolo_morph
        self._mask_builder = mask_builder
        self._ocr_engine = ocr_engine  # lazily built only for ocr_inverse
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

    def _build_ocr_inverse_mask(
        self, img: np.ndarray, cfg: InkstripConfig
    ) -> tuple[np.ndarray, int]:
        from inkstrip.mask.ocr_inverse import OcrInverseMaskBuilder

        if self._mask_builder is not None:
            builder = self._mask_builder
        else:
            builder = OcrInverseMaskBuilder(cfg, ocr_engine=self._ocr_engine)
            # cache the engine the builder constructed so subsequent runs reuse it
            self._ocr_engine = builder._engine
        return builder.build(img)

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

        warnings: list[str] = []
        page_cropped = False
        deskew_deg = 0.0
        if cfg.page_crop:
            from inkstrip.preprocess.page_crop import auto_page_crop

            emit(progress, "crop", message="auto-cropping page")
            img, info = auto_page_crop(
                img,
                min_area_ratio=cfg.page_crop_min_area_ratio,
                deskew_max_deg=cfg.page_crop_deskew_max_deg,
            )
            page_cropped = info.cropped
            deskew_deg = info.deskew_deg
            if info.warning:
                warnings.append(info.warning)

        if cfg.mask_strategy == "yolo_morph":
            emit(progress, "detect", message=f"running {cfg.detector}")
            mask, bbox_count = self._build_yolo_mask(img, cfg)
            emit(progress, "mask", message=f"{bbox_count} bbox → mask")
        elif cfg.mask_strategy == "ocr_inverse":
            emit(progress, "mask", message="OCR-inverse mask")
            mask, bbox_count = self._build_ocr_inverse_mask(img, cfg)
            if bbox_count == 0:
                warnings.append(
                    "ocr_inverse: no printed text detected; output equals input"
                )
        else:
            emit(progress, "mask", message=f"color mask ({cfg.mask_strategy})")
            mask, bbox_count = self._build_color_mask(img, cfg)

        coverage = mask_coverage(mask)
        has_target = bool((mask > 0).any())

        if not has_target:
            if not warnings:
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
            page_cropped=page_cropped,
            page_deskew_deg=deskew_deg,
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
