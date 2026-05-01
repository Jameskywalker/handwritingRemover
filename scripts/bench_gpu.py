"""Quick GPU vs CPU benchmark on the synthetic fixture.

Reports per-stage wall-clock and the actual ONNX execution provider that LaMa
ended up using.
"""

from __future__ import annotations

import time
from pathlib import Path

from inkstrip.config import InkstripConfig
from inkstrip.detect.yolo_hw import YoloHandwritingDetector
from inkstrip.inpaint.lama_onnx import LamaOnnxInpainter
from inkstrip.io.loaders import load_image
from inkstrip.mask.morph import MorphMaskBuilder

FIXTURE = Path(__file__).parent.parent / "tests/fixtures/_generated/synthetic_handwriting.png"


def bench(device: str) -> None:
    print(f"\n=== device = {device} ===")
    cfg = InkstripConfig(device=device)
    img = load_image(FIXTURE).array

    t0 = time.perf_counter()
    det = YoloHandwritingDetector(cfg)
    boxes = det.detect(img)
    t1 = time.perf_counter()
    print(f"  detect (incl warm-up): {(t1 - t0) * 1000:.0f} ms ({len(boxes)} bbox)")

    t2 = time.perf_counter()
    boxes2 = det.detect(img)
    t3 = time.perf_counter()
    print(f"  detect (warm):         {(t3 - t2) * 1000:.0f} ms")

    mask = MorphMaskBuilder(cfg).build(img, boxes)

    t4 = time.perf_counter()
    inp = LamaOnnxInpainter(cfg)
    inp._load()  # type: ignore[attr-defined]
    t5 = time.perf_counter()
    print(f"  LaMa load:             {(t5 - t4) * 1000:.0f} ms")
    print(f"  LaMa providers:        {inp._session.get_providers()}")  # type: ignore[union-attr]

    t6 = time.perf_counter()
    _ = inp.inpaint(img, mask)
    t7 = time.perf_counter()
    print(f"  LaMa infer (cold):     {(t7 - t6) * 1000:.0f} ms")

    t8 = time.perf_counter()
    _ = inp.inpaint(img, mask)
    t9 = time.perf_counter()
    print(f"  LaMa infer (warm):     {(t9 - t8) * 1000:.0f} ms")


if __name__ == "__main__":
    # Run GPU first because ultralytics may mutate CUDA_VISIBLE_DEVICES when
    # asked for device="cpu" and that breaks subsequent GPU runs in the same
    # process. The bench is informational; swap or run separately if needed.
    bench("cuda")
    bench("cpu")
