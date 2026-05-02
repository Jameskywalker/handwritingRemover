"""Single-image pipeline: OCR-inverse + handwriting classifier → mask → inpaint → save."""

from __future__ import annotations

import time
from typing import Any

from inkstrip.config import InkstripConfig
from inkstrip.inpaint.lama_onnx import LamaOnnxInpainter
from inkstrip.io.loaders import load_image
from inkstrip.io.savers import save_image
from inkstrip.mask.morph import mask_coverage
from inkstrip.types import OutputLike, PageMetadata, ProgressCallback, RemoveResult
from inkstrip.utils.progress import emit


class ImagePipeline:
    """Pipeline for a single image (no PDF, no per-page loop)."""

    def __init__(
        self,
        cfg: InkstripConfig,
        *,
        mask_builder=None,
        inpainter=None,
        ocr_engine=None,
        hw_classifier=None,
    ) -> None:
        self.cfg = cfg
        self._mask_builder = mask_builder
        self._ocr_engine = ocr_engine
        self._hw_classifier = hw_classifier
        self.inpainter = inpainter or _make_inpainter(cfg)

    def _build_mask(self, img):
        from inkstrip.mask.ocr_inverse import OcrInverseMaskBuilder

        if self._mask_builder is not None:
            builder = self._mask_builder
        else:
            builder = OcrInverseMaskBuilder(
                self.cfg,
                ocr_engine=self._ocr_engine,
                hw_classifier=self._hw_classifier,
            )
            self._ocr_engine = builder._engine
            self._hw_classifier = builder._hw_classifier
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

        emit(progress, "mask", message="OCR-inverse mask")
        mask, bbox_count = self._build_mask(img)
        if bbox_count == 0:
            warnings.append(
                "ocr_inverse: no printed text detected; output equals input"
            )

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
