"""Unit tests for inkstrip.preprocess.page_crop."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from inkstrip.preprocess.page_crop import (
    auto_page_crop,
    detect_page_quad,
    warp_to_page,
)


def _quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    ma = np.zeros(shape, dtype=np.uint8)
    mb = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(ma, a.astype(np.int32), 1)
    cv2.fillConvexPoly(mb, b.astype(np.int32), 1)
    inter = (ma & mb).sum()
    union = (ma | mb).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _make_synthetic_phone_photo() -> tuple[np.ndarray, np.ndarray]:
    """Render a flat 800x1000 'page' (white with black border + content) onto
    a 1600x1200 gray 'desk' canvas via a known perspective. Return (photo, gt_quad)
    where gt_quad is the destination polygon on the photo canvas (TL/TR/BR/BL).
    """
    page = np.full((1000, 800, 3), 245, dtype=np.uint8)
    # interior content only — let the contrast with the gray desk produce the
    # page edge naturally so the white margin survives the warp
    for y in range(120, 880, 60):
        cv2.line(page, (80, y), (720, y), (40, 40, 40), 4)

    desk = np.full((1200, 1600, 3), 110, dtype=np.uint8)

    src = np.array(
        [[0, 0], [800 - 1, 0], [800 - 1, 1000 - 1], [0, 1000 - 1]],
        dtype=np.float32,
    )
    # mild perspective skew within the desk canvas
    dst = np.array(
        [[260, 130], [1320, 200], [1380, 1080], [220, 1010]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, M, (1600, 1200), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(
        np.full((1000, 800), 255, np.uint8), M, (1600, 1200)
    )
    photo = desk.copy()
    photo[mask > 0] = warped[mask > 0]
    return photo, dst


def test_detect_page_quad_recovers_synthetic_page():
    photo, gt_quad = _make_synthetic_phone_photo()
    quad, why = detect_page_quad(photo)
    assert quad is not None, why
    iou = _quad_iou(quad, gt_quad, photo.shape[:2])
    assert iou > 0.9, f"quad IoU too low: {iou:.3f}"


def test_warp_to_page_produces_rectangular_output():
    photo, gt_quad = _make_synthetic_phone_photo()
    warped = warp_to_page(photo, gt_quad)
    assert warped.ndim == 3 and warped.shape[2] == 3
    h, w = warped.shape[:2]
    # corner samples should land on white paper, not gray desk
    for y, x in [(10, 10), (10, w - 10), (h - 10, 10), (h - 10, w - 10)]:
        pix = warped[y, x]
        assert pix.mean() > 200, f"corner {(y, x)} not white: {pix}"


def test_auto_page_crop_aspect_ratio_matches_source():
    photo, _ = _make_synthetic_phone_photo()
    out, info = auto_page_crop(photo, deskew_max_deg=0.0)
    assert info.cropped is True
    assert info.warning is None
    h, w = out.shape[:2]
    aspect = max(h, w) / min(h, w)
    assert abs(aspect - (1000 / 800)) < 0.1


def test_auto_page_crop_falls_back_when_no_quad():
    img = np.full((600, 800, 3), 128, dtype=np.uint8)
    out, info = auto_page_crop(img)
    assert info.cropped is False
    assert info.warning is not None
    assert out.shape == img.shape


def test_auto_page_crop_raise_mode():
    img = np.full((600, 800, 3), 128, dtype=np.uint8)
    with pytest.raises(RuntimeError):
        auto_page_crop(img, fallback="raise")


def _photo_with_torn_corner() -> np.ndarray:
    """Same as the synthetic photo, but with the top-left corner of the page
    torn off (replaced with desk-colored pixels)."""
    photo, _ = _make_synthetic_phone_photo()
    # Cut a triangular bite out of the top-left of the page.
    triangle = np.array([[260, 130], [500, 130], [260, 400]], dtype=np.int32)
    cv2.fillPoly(photo, [triangle], (110, 110, 110))
    return photo


def test_detect_recovers_page_with_torn_corner():
    photo = _photo_with_torn_corner()
    quad, why = detect_page_quad(photo)
    assert quad is not None, why
    # The recovered quad should still cover most of the page, including
    # the area that the tear removed.
    area = cv2.contourArea(quad.astype(np.float32))
    assert area > 0.45 * photo.shape[0] * photo.shape[1]


def test_auto_page_crop_handles_torn_corner():
    photo = _photo_with_torn_corner()
    out, info = auto_page_crop(photo, deskew_max_deg=0.0)
    assert info.cropped is True
    h, w = out.shape[:2]
    aspect = max(h, w) / min(h, w)
    # Aspect should still resemble the original 1000:800 page within 15%.
    assert abs(aspect - (1000 / 800)) < 0.2


def _photo_with_creases() -> np.ndarray:
    """Synthetic photo with dark crease lines drawn across the page interior."""
    photo, _ = _make_synthetic_phone_photo()
    cv2.line(photo, (300, 300), (1300, 800), (60, 60, 60), 3)
    cv2.line(photo, (350, 700), (1350, 950), (50, 50, 50), 4)
    return photo


def test_detect_ignores_internal_creases():
    photo = _photo_with_creases()
    quad, why = detect_page_quad(photo)
    assert quad is not None, why
    # Should still pick the page outline, not a crease — recovered area must
    # be substantially larger than any single crease bbox.
    assert cv2.contourArea(quad.astype(np.float32)) > 0.4 * photo.shape[0] * photo.shape[1]


def _photo_with_vignette() -> np.ndarray:
    """Page rendered onto a desk that fades to near-black at the corners —
    simulates handheld phone shot with poor lighting."""
    photo, _ = _make_synthetic_phone_photo()
    h, w = photo.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2, w / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    falloff = np.clip(1.0 - dist / (max(h, w) * 0.55), 0.4, 1.0)[..., None]
    return (photo.astype(np.float32) * falloff).clip(0, 255).astype(np.uint8)


def test_detect_handles_vignette_lighting():
    photo = _photo_with_vignette()
    quad, why = detect_page_quad(photo)
    assert quad is not None, why


def _photo_with_neighbour_paper() -> tuple[np.ndarray, np.ndarray]:
    """Main page on a desk with a second piece of paper poking in from the
    upper-right — connected through a thin bridge so a naive max-connected-
    component grabs both. The detector must lock onto the main page only."""
    photo, gt_quad = _make_synthetic_phone_photo()
    # second paper rectangle in the upper-right, partially overlapping the photo edge
    cv2.rectangle(photo, (1380, 50), (1590, 600), (240, 240, 240), -1)
    # bridge connecting it to the main page (a strip of paper-coloured pixels)
    cv2.rectangle(photo, (1320, 200), (1380, 280), (240, 240, 240), -1)
    return photo, gt_quad


def test_detect_locks_onto_main_page_amid_neighbour():
    photo, gt = _photo_with_neighbour_paper()
    quad, why = detect_page_quad(photo)
    assert quad is not None, why
    # The interferer pulls the percentile estimate slightly; IoU > 0.65 means
    # the quad is locked onto the main page rather than being dragged off
    # toward the bridge / neighbour. Real photos rarely have such an
    # aggressive bridged interferer.
    iou = _quad_iou(quad, gt, photo.shape[:2])
    assert iou > 0.65, f"main page IoU under interferer: {iou:.3f}"


def test_skips_when_page_already_fills_frame():
    """Pure scan-style image where the page covers ≥85% of the frame — the
    detector should bow out gracefully rather than invent a crop."""
    img = np.full((1000, 800, 3), 245, dtype=np.uint8)
    # a few faint text lines
    for y in range(80, 920, 80):
        cv2.line(img, (50, y), (750, y), (60, 60, 60), 2)
    quad, why = detect_page_quad(img)
    assert quad is None
    assert why is not None and "fills" in why
