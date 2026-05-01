"""Diagnose where M1 is leaking ink: detector? mask? inpaint?

Prints per-stage stats and writes a 4-panel grid to eval_outputs/.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from inkstrip.config import InkstripConfig
from inkstrip.detect.yolo_hw import YoloHandwritingDetector
from inkstrip.inpaint.lama_onnx import LamaOnnxInpainter
from inkstrip.io.loaders import load_image
from inkstrip.mask.morph import MorphMaskBuilder, mask_coverage

FIXTURE = Path(__file__).parent.parent / "tests/fixtures/_generated/synthetic_handwriting.png"


def main() -> None:
    cfg = InkstripConfig(device="cpu")
    img = load_image(FIXTURE).array
    print(f"input shape={img.shape}, total px={img.size // 3}")

    det = YoloHandwritingDetector(cfg)
    boxes = det.detect(img)
    print(f"detected boxes: {len(boxes)}")
    for i, b in enumerate(boxes[:10]):
        print(f"  [{i}] x={b.x} y={b.y} w={b.w} h={b.h} score={b.score:.3f}")

    mask = MorphMaskBuilder(cfg).build(img, boxes)
    print(f"mask coverage: {mask_coverage(mask) * 100:.3f}%")

    if not boxes:
        print("no boxes — bailing")
        return

    inp = LamaOnnxInpainter(cfg)
    painted = inp.inpaint(img, mask)

    diff_total = np.abs(img.astype(int) - painted.astype(int)).mean()
    inside = mask > 0
    if inside.any():
        diff_inside = np.abs(
            img[inside].astype(int) - painted[inside].astype(int)
        ).mean()
    else:
        diff_inside = 0.0
    print(f"L1 diff over full image: {diff_total:.3f}")
    print(f"L1 diff inside mask:     {diff_inside:.3f}")
    outside = ~inside
    diff_outside = np.abs(
        img[outside].astype(int) - painted[outside].astype(int)
    ).mean()
    print(f"L1 diff outside mask:    {diff_outside:.6f} (must be 0)")

    # Visual grid: input | mask | painted | diff (×4 amplified)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    diff = np.abs(img.astype(int) - painted.astype(int)).clip(0, 255).astype(np.uint8)
    diff_amp = (diff.astype(int) * 4).clip(0, 255).astype(np.uint8)
    grid = np.concatenate([img, mask_rgb, painted, diff_amp], axis=1)

    out_dir = Path("eval_outputs")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "diagnose_m1.png"
    Image.fromarray(grid).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
