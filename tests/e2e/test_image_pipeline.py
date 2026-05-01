"""End-to-end smoke test for the image pipeline.

This test downloads model weights on first run and pulls in heavy deps
(torch, ultralytics, simple-lama-inpainting). It auto-skips if any of those
aren't installed, so unit tests still run fast in minimal environments.

Set INKSTRIP_E2E=1 to run; otherwise skipped to keep CI lightweight.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("INKSTRIP_E2E") != "1",
    reason="set INKSTRIP_E2E=1 to enable; pulls torch + downloads weights",
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "_generated" / "synthetic_handwriting.png"


def _ensure_fixture() -> Path:
    if FIXTURE.exists():
        return FIXTURE
    from tests.fixtures import _make_fixtures  # type: ignore[import-not-found]

    _make_fixtures.main()
    return FIXTURE


def test_image_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    pytest.importorskip("onnxruntime")

    from PIL import Image

    from inkstrip.api import remove_handwriting
    from inkstrip.config import InkstripConfig

    src = _ensure_fixture()
    out = tmp_path / "cleaned.png"

    cfg = InkstripConfig(device="auto")
    result = remove_handwriting(src, out, config=cfg)

    assert result.output_path == out
    assert out.exists()

    page = result.pages[0]
    assert page.bbox_count >= 1, "detector failed to find any handwriting"
    assert page.mask_coverage > 0

    orig = np.asarray(Image.open(src).convert("RGB"))
    cleaned = np.asarray(Image.open(out).convert("RGB"))
    assert orig.shape == cleaned.shape

    # Reconstruct the mask to score where inpainting actually ran. The pipeline's
    # tile_blend guarantees mask-outside pixels are byte-identical to the input,
    # so we infer the mask by finding any changed pixel.
    diff_per_pixel = np.abs(orig.astype(int) - cleaned.astype(int)).sum(axis=-1)
    changed = diff_per_pixel > 0
    assert changed.any(), "inpaint did not modify a single pixel"

    diff_inside = diff_per_pixel[changed].mean()
    assert diff_inside > 5, (
        f"changed pixels barely differ (avg L1 {diff_inside:.2f}); "
        "LaMa output may be saturated or inpaint contract may be wrong"
    )


HW_JPG_CANDIDATES = [
    Path("/home/james/projects/handwritingRemover/eval_outputs/00_input.jpg"),
    Path("/mnt/e/downloads/hw.jpg"),
]


def _hw_jpg() -> Path | None:
    for p in HW_JPG_CANDIDATES:
        if p.exists():
            return p
    return None


def test_page_crop_and_ocr_inverse_on_hw_jpg(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("rapidocr")
    src = _hw_jpg()
    if src is None:
        pytest.skip("hw.jpg not present locally; skipping black-and-white e2e")

    from inkstrip.api import remove_handwriting
    from inkstrip.config import InkstripConfig

    out = tmp_path / "cleaned_ocr.png"
    cfg = InkstripConfig(
        page_crop=True,
        mask_strategy="ocr_inverse",
        device="auto",
    )
    result = remove_handwriting(src, out, config=cfg)

    assert result.output_path == out
    assert out.exists()
    page = result.pages[0]
    # Either the page was cropped, or we got a graceful warning explaining why not.
    assert page.page_cropped or any("page_crop" in w for w in result.warnings), result.warnings
    # ocr_inverse is allowed to produce zero coverage if no printed text was found,
    # but on hw.jpg there should be plenty of printed text.
    assert page.bbox_count > 0, f"OCR found no printed text on {src}"
    assert 0.0005 < page.mask_coverage < 0.5, page.mask_coverage
