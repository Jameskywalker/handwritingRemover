"""Single-image pipeline: detect → mask → inpaint → save."""

from __future__ import annotations

import time
from typing import Any

from inkstrip.config import InkstripConfig
from inkstrip.detect.yolo_hw import YoloHandwritingDetector
from inkstrip.inpaint.lama_onnx import LamaOnnxInpainter
from inkstrip.io.loaders import load_image
from inkstrip.io.savers import save_image
from inkstrip.mask.morph import MorphMaskBuilder, mask_coverage
from inkstrip.types import OutputLike, PageMetadata, ProgressCallback, RemoveResult
from inkstrip.utils.progress import emit


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
        self.detector = detector or YoloHandwritingDetector(cfg)
        self.mask_builder = mask_builder or MorphMaskBuilder(cfg)
        self.inpainter = inpainter or _make_inpainter(cfg)

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

        emit(progress, "detect", message=f"running {cfg.detector}")
        boxes = self.detector.detect(img)

        emit(progress, "mask", message=f"{len(boxes)} bbox → mask")
        mask = self.mask_builder.build(img, boxes)
        coverage = mask_coverage(mask)

        warnings: list[str] = []
        if not boxes:
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
            bbox_count=len(boxes),
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
