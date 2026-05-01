"""Unit tests for inkstrip.mask.ocr_inverse using a fake OCR engine."""

from __future__ import annotations

import cv2
import numpy as np

from inkstrip.detect.ocr_rapid import OcrBox
from inkstrip.mask.ocr_inverse import detect_ocr_inverse_mask


class _FakeEngine:
    def __init__(self, boxes: list[OcrBox]) -> None:
        self._boxes = boxes

    def detect(self, image):  # noqa: ARG002
        return self._boxes


def _make_synthetic_page() -> tuple[np.ndarray, list[tuple[int, int, int, int]], np.ndarray, list[tuple[int, int, int, int]]]:
    """Return (image, printed_rects, scribble_pixel_mask, scribble_rects)."""
    img = np.full((600, 800, 3), 255, dtype=np.uint8)

    cv2.putText(img, "PRINTED LINE ONE", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "PRINTED LINE TWO", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    printed = [(50, 80, 600, 70), (50, 200, 600, 70)]

    scribble_pts_a = np.array([[100, 380], [200, 360], [300, 420], [400, 380]], np.int32)
    scribble_pts_b = np.array([[450, 480], [500, 460], [580, 510], [650, 470]], np.int32)
    cv2.polylines(img, [scribble_pts_a], False, (0, 0, 0), 4)
    cv2.polylines(img, [scribble_pts_b], False, (0, 0, 0), 4)

    scribble_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.polylines(scribble_mask, [scribble_pts_a], False, 255, 4)
    cv2.polylines(scribble_mask, [scribble_pts_b], False, 255, 4)

    scribble_rects = [(80, 350, 340, 90), (430, 450, 240, 80)]
    return img, printed, scribble_mask, scribble_rects


def _bbox_to_poly(rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


def _mean_inside(mask: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    x, y, w, h = rect
    return float(mask[y : y + h, x : x + w].mean())


def test_handwriting_kept_printed_subtracted():
    img, printed, scribble_mask, _ = _make_synthetic_page()
    boxes = [OcrBox(_bbox_to_poly(r), text="x", score=0.99) for r in printed]
    engine = _FakeEngine(boxes)
    mask, n = detect_ocr_inverse_mask(img, ocr_engine=engine)
    assert n == 2

    # ≥80% of scribble pixels survive
    overlap = ((mask > 0) & (scribble_mask > 0)).sum()
    assert overlap / max(1, (scribble_mask > 0).sum()) > 0.80

    # printed-text regions are mostly cleared
    for r in printed:
        m = _mean_inside(mask, r)
        assert m < 30, f"printed region not subtracted: {m:.1f}"


def test_zero_printed_returns_empty_mask():
    img, _, _, _ = _make_synthetic_page()
    engine = _FakeEngine([])
    mask, n = detect_ocr_inverse_mask(img, ocr_engine=engine)
    assert n == 0
    assert mask.shape == img.shape[:2]
    assert mask.sum() == 0


def test_low_confidence_boxes_filtered():
    img, printed, _, _ = _make_synthetic_page()
    boxes = [OcrBox(_bbox_to_poly(r), text="x", score=0.10) for r in printed]
    engine = _FakeEngine(boxes)
    mask, n = detect_ocr_inverse_mask(img, ocr_engine=engine, min_confidence=0.30)
    assert n == 0
    assert mask.sum() == 0
