"""Build a side-by-side comparison grid: input | mask | output.

Run:
    python examples/eval_grid.py path/to/image.jpg

Outputs a single PNG to ./eval_outputs/ so you can eyeball the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from inkstrip.config import InkstripConfig
from inkstrip.detect.yolo_hw import YoloHandwritingDetector
from inkstrip.inpaint.lama_torch import LamaTorchInpainter
from inkstrip.io.loaders import load_image
from inkstrip.mask.morph import MorphMaskBuilder


def main(src: Path) -> None:
    cfg = InkstripConfig()
    img = load_image(src).array

    detector = YoloHandwritingDetector(cfg)
    boxes = detector.detect(img)

    mask = MorphMaskBuilder(cfg).build(img, boxes)
    mask_rgb = np.stack([mask, mask, mask], axis=-1)

    painted = LamaTorchInpainter(cfg).inpaint(img, mask) if boxes else img.copy()

    grid = np.concatenate([img, mask_rgb, painted], axis=1)

    out_dir = Path("eval_outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{src.stem}_grid.png"
    Image.fromarray(grid).save(out_path)
    print(f"wrote {out_path}  bbox={len(boxes)}  mask_coverage={(mask > 0).mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))
