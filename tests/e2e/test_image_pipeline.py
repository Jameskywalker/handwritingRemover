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
