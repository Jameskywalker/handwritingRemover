"""Iterate on hw.jpg: try a variant, save grid, eyeball, repeat.

Each call writes:
  eval_outputs/iter_<tag>_grid.png  → input | mask | cleaned | amplified diff
  eval_outputs/iter_<tag>_clean.png → just the cleaned image
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from inkstrip.config import InkstripConfig
from inkstrip.inpaint.lama_onnx import LamaOnnxInpainter
from inkstrip.io.loaders import load_image
from inkstrip.mask.color import detect_adaptive_mask, detect_color_mask

SRC = Path("/mnt/e/downloads/hw.jpg")
OUT_DIR = Path("eval_outputs")
OUT_DIR.mkdir(exist_ok=True)

# Cache the inpainter across runs (~7s cold load).
_INPAINTER: LamaOnnxInpainter | None = None


def get_inpainter() -> LamaOnnxInpainter:
    global _INPAINTER
    if _INPAINTER is None:
        _INPAINTER = LamaOnnxInpainter(InkstripConfig(device="cuda"))
        _INPAINTER._load()  # type: ignore[attr-defined]
    return _INPAINTER


_OCR_ENGINE = None


def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from inkstrip.detect.ocr_rapid import RapidOcrEngine

        _OCR_ENGINE = RapidOcrEngine(device="cuda")
    return _OCR_ENGINE


def run_variant(
    tag: str,
    *,
    profile: str = "red",
    dilate_px: int = 5,
    closing_px: int = 3,
    min_component_area: int = 12,
    method: str = "fixed",
    threshold: int = 50,
    page_crop: bool = False,
) -> dict:
    img = load_image(SRC).array
    if page_crop:
        from inkstrip.preprocess.page_crop import auto_page_crop

        img, _ = auto_page_crop(img)
    t0 = time.perf_counter()
    if method == "ocr_inverse":
        from inkstrip.mask.ocr_inverse import detect_ocr_inverse_mask

        mask, _ = detect_ocr_inverse_mask(
            img,
            ocr_engine=get_ocr_engine(),
            dilate_px=dilate_px,
            closing_px=closing_px,
            min_component_area=min_component_area,
        )
    elif method == "adaptive":
        mask = detect_adaptive_mask(
            img,
            profile=profile,
            threshold=threshold,
            dilate_px=dilate_px,
            closing_px=closing_px,
            min_component_area=min_component_area,
        )
    else:
        mask = detect_color_mask(
            img,
            profile=profile,
            dilate_px=dilate_px,
            closing_px=closing_px,
            min_component_area=min_component_area,
        )
    coverage = (mask > 0).mean()
    t1 = time.perf_counter()
    cleaned = get_inpainter().inpaint(img, mask)
    t2 = time.perf_counter()

    diff = np.abs(img.astype(int) - cleaned.astype(int)).sum(axis=-1)
    diff_inside = float(diff[mask > 0].mean()) if (mask > 0).any() else 0.0

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    diff_amp = (diff.clip(0, 255) * 1).astype(np.uint8)
    diff_rgb = cv2.cvtColor(diff_amp, cv2.COLOR_GRAY2RGB)
    grid = np.concatenate([img, mask_rgb, cleaned, diff_rgb], axis=1)

    Image.fromarray(grid).save(OUT_DIR / f"iter_{tag}_grid.png")
    Image.fromarray(cleaned).save(OUT_DIR / f"iter_{tag}_clean.png")

    info = {
        "tag": tag,
        "profile": profile,
        "dilate_px": dilate_px,
        "coverage_pct": coverage * 100,
        "mask_ms": (t1 - t0) * 1000,
        "inpaint_ms": (t2 - t1) * 1000,
        "diff_inside": diff_inside,
    }
    print(
        f"[{tag}] profile={profile} dilate={dilate_px} "
        f"coverage={coverage*100:.2f}% diff_inside={diff_inside:.1f} "
        f"mask={info['mask_ms']:.0f}ms inpaint={info['inpaint_ms']:.0f}ms"
    )
    return info


def main() -> None:
    if len(sys.argv) > 1:
        tag = sys.argv[1]
        kwargs = {}
        for arg in sys.argv[2:]:
            k, _, v = arg.partition("=")
            kwargs[k] = int(v) if v.isdigit() else v
        run_variant(tag, **kwargs)
        return

    # Default sweep
    run_variant("v18_adaptive_t40_d7", method="adaptive", threshold=40, dilate_px=7)
    run_variant("v19_adaptive_t50_d7", method="adaptive", threshold=50, dilate_px=7)
    run_variant("v20_adaptive_t30_d7", method="adaptive", threshold=30, dilate_px=7)
    run_variant("v21_ocr_inverse_d5", method="ocr_inverse", dilate_px=5)
    run_variant("v22_ocr_inverse_d5_crop", method="ocr_inverse", dilate_px=5, page_crop=True)


if __name__ == "__main__":
    main()
