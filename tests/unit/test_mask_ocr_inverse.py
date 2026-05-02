"""Unit tests for inkstrip.mask.ocr_inverse using a fake OCR engine."""

from __future__ import annotations

import cv2
import numpy as np

from inkstrip.detect.hw_classifier import HwBox
from inkstrip.detect.ocr_rapid import OcrBox
from inkstrip.mask.ocr_inverse import detect_ocr_inverse_mask


class _FakeEngine:
    def __init__(self, boxes: list[OcrBox]) -> None:
        self._boxes = boxes

    def detect(self, image):  # noqa: ARG002
        return self._boxes


class _FakeHwClassifier:
    def __init__(self, boxes: list[HwBox]) -> None:
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
    """Black handwriting on a black-printed page: only the OCR subtraction
    branch can save the handwriting. Disable color combine to test it."""
    img, printed, scribble_mask, _ = _make_synthetic_page()
    boxes = [OcrBox(_bbox_to_poly(r), text="x", score=0.99) for r in printed]
    engine = _FakeEngine(boxes)
    mask, n, _ = detect_ocr_inverse_mask(img, ocr_engine=engine, combine_color=False)
    assert n == 2

    overlap = ((mask > 0) & (scribble_mask > 0)).sum()
    assert overlap / max(1, (scribble_mask > 0).sum()) > 0.80

    for r in printed:
        m = _mean_inside(mask, r)
        assert m < 30, f"printed region not subtracted: {m:.1f}"


def test_zero_printed_returns_empty_mask_strict():
    img, _, _, _ = _make_synthetic_page()
    engine = _FakeEngine([])
    mask, n, _ = detect_ocr_inverse_mask(img, ocr_engine=engine, combine_color=False)
    assert n == 0
    assert mask.sum() == 0


def test_low_confidence_boxes_filtered_strict():
    img, printed, _, _ = _make_synthetic_page()
    boxes = [OcrBox(_bbox_to_poly(r), text="x", score=0.10) for r in printed]
    engine = _FakeEngine(boxes)
    mask, n, _ = detect_ocr_inverse_mask(
        img, ocr_engine=engine, min_confidence=0.30, combine_color=False
    )
    assert n == 0
    assert mask.sum() == 0


def _make_synthetic_page_with_colored_handwriting() -> tuple[
    np.ndarray, list, np.ndarray
]:
    """Same as the b/w synthetic, but the handwriting is in red ink — so OCR
    will likely recognise it and try to subtract it. The combine_color path
    must keep it."""
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, "PRINTED LINE ONE", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "PRINTED LINE TWO", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    printed = [(50, 80, 600, 70), (50, 200, 600, 70)]

    red = (220, 30, 30)
    pts_a = np.array([[100, 380], [200, 360], [300, 420], [400, 380]], np.int32)
    pts_b = np.array([[450, 480], [500, 460], [580, 510], [650, 470]], np.int32)
    cv2.polylines(img, [pts_a], False, red, 5)
    cv2.polylines(img, [pts_b], False, red, 5)
    scribble_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.polylines(scribble_mask, [pts_a], False, 255, 5)
    cv2.polylines(scribble_mask, [pts_b], False, 255, 5)
    return img, printed, scribble_mask


def test_color_handwriting_bypasses_ocr_veto():
    """OCR (mocked) recognises coloured handwriting as printed text. With
    combine_color=True, the colour layer should keep those strokes anyway."""
    img, printed, scribble_mask = _make_synthetic_page_with_colored_handwriting()
    # OCR returns BOTH the real printed lines AND the handwriting bboxes,
    # all at high confidence — same failure mode as RapidOCR on hw.jpg.
    fake_handwriting_bboxes = [(80, 350, 340, 90), (430, 450, 240, 80)]
    boxes = [OcrBox(_bbox_to_poly(r), text="x", score=0.99) for r in printed + fake_handwriting_bboxes]
    engine = _FakeEngine(boxes)

    mask, n, _ = detect_ocr_inverse_mask(img, ocr_engine=engine, combine_color=True)
    assert n == 4

    # Pure ocr_inverse would zero out the scribbles. combine_color must keep them.
    overlap = ((mask > 0) & (scribble_mask > 0)).sum()
    survival = overlap / max(1, (scribble_mask > 0).sum())
    assert survival > 0.80, f"colour handwriting survival rate {survival:.2f}"

    for r in printed:
        m = _mean_inside(mask, r)
        assert m < 30, f"printed region not subtracted: {m:.1f}"


def test_hw_classifier_rescues_handwriting_misclassified_by_ocr():
    """Same-color failure case: OCR mistakes the scribbles for printed text.
    Without help the OCR-inverse mask would erase them. With a HW classifier
    that flags those bboxes as handwriting, they're excluded from the
    printed-subtraction step and survive in the mask."""
    img, printed, scribble_mask, scribble_rects = _make_synthetic_page()
    # OCR returns BOTH the real printed lines AND the scribble bboxes,
    # all at high confidence — same failure mode as RapidOCR on hw.jpg.
    boxes = [
        OcrBox(_bbox_to_poly(r), text="x", score=0.99)
        for r in printed + scribble_rects
    ]
    engine = _FakeEngine(boxes)
    # HW classifier (mocked) flags only the scribbles.
    hw = _FakeHwClassifier([HwBox(bbox=r, score=0.90) for r in scribble_rects])

    mask, n, _ = detect_ocr_inverse_mask(
        img, ocr_engine=engine, hw_classifier=hw, combine_color=False
    )
    # Only the 2 real printed bboxes remain after HW filtering.
    assert n == 2
    # Scribbles are preserved — the rescue worked.
    overlap = ((mask > 0) & (scribble_mask > 0)).sum()
    survival = overlap / max(1, (scribble_mask > 0).sum())
    assert survival > 0.80, f"handwriting survival rate {survival:.2f}"
    # Printed regions are still subtracted.
    for r in printed:
        m = _mean_inside(mask, r)
        assert m < 30, f"printed region not subtracted: {m:.1f}"


def test_hw_classifier_no_overlap_does_not_rescue_anything():
    """HW classifier returns boxes that don't overlap any OCR bbox: result
    should be identical to running without it."""
    img, printed, _, _ = _make_synthetic_page()
    boxes = [OcrBox(_bbox_to_poly(r), text="x", score=0.99) for r in printed]
    engine = _FakeEngine(boxes)
    hw = _FakeHwClassifier([HwBox(bbox=(700, 550, 50, 30), score=0.90)])

    mask, n, _ = detect_ocr_inverse_mask(
        img, ocr_engine=engine, hw_classifier=hw, combine_color=False
    )
    # All printed bboxes still classified as printed.
    assert n == 2
    for r in printed:
        m = _mean_inside(mask, r)
        assert m < 30


def test_zero_printed_keeps_color_layer_when_combined():
    img, _, scribble_mask = _make_synthetic_page_with_colored_handwriting()
    engine = _FakeEngine([])  # OCR finds nothing
    mask, n, _ = detect_ocr_inverse_mask(img, ocr_engine=engine, combine_color=True)
    assert n == 0
    overlap = ((mask > 0) & (scribble_mask > 0)).sum()
    assert overlap / max(1, (scribble_mask > 0).sum()) > 0.80
