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
from inkstrip.mask.color import detect_color_mask

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


def run_variant(
    tag: str,
    *,
    profile: str = "red",
    dilate_px: int = 5,
    closing_px: int = 3,
    min_component_area: int = 12,
) -> dict:
    img = load_image(SRC).array
    t0 = time.perf_counter()
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
    run_variant("v15_protect_d5", profile="red", dilate_px=5)
    run_variant("v16_protect_d7", profile="red", dilate_px=7)
    run_variant("v17_protect_d9", profile="red", dilate_px=9)


if __name__ == "__main__":
    main()
