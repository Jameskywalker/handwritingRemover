"""Auto-detect a page quadrilateral in a phone photo and warp it flat.

Pipeline (degrades gracefully through multiple strategies):

1. Build a paper mask (Otsu + morphology + largest CC). If the mask covers
   ≥85% of the frame the page already fills the camera and there is nothing
   to crop — bail out.
2. Detect long edges on the mask boundary with HoughLinesP, cluster into
   four sides via the minAreaRect axis, fit each side with a robust
   estimator (Huber loss), intersect for four corners.
3. If Hough finds <4 sides, fall back to convex hull + approxPolyDP across a
   range of epsilon multipliers.
4. If that fails, fall back to minAreaRect on the hull.
5. Validate the final quad (area, side ratio, corner angles); reject if
   wildly off and surface a warning instead of cropping into garbage.

After the warp, run a small deskew on the cropped page to absorb any
residual tilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class PageCropInfo:
    cropped: bool
    quad: tuple[tuple[float, float], ...] | None
    deskew_deg: float
    warning: str | None


# ---------------------------------------------------------------------------
# helpers


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def _build_paper_mask(image: np.ndarray) -> np.ndarray:
    """Otsu + morphology + largest connected component."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == biggest).astype(np.uint8) * 255)


def _line_intersect(la: tuple, lb: tuple) -> np.ndarray:
    """Intersect two infinite lines given as (vx, vy, x0, y0) (cv2.fitLine output)."""
    vx1, vy1, x1, y1 = la
    vx2, vy2, x2, y2 = lb
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return np.array([np.nan, np.nan])
    t = np.linalg.solve(A, b)[0]
    return np.array([x1 + t * vx1, y1 + t * vy1])


def _quad_sanity(
    quad: np.ndarray, h: int, w: int, *, min_area_ratio: float
) -> tuple[bool, str | None]:
    img_area = float(h * w)
    area = float(cv2.contourArea(quad.astype(np.float32)))
    if area < min_area_ratio * img_area:
        return False, f"quad area {area / img_area:.0%} below {min_area_ratio:.0%}"
    if area > 0.98 * img_area:
        return False, "quad fills the whole frame; likely a false positive"
    # Corner angles within [55°, 125°] (allow some perspective)
    pts = quad.reshape(4, 2).astype(np.float64)
    for i in range(4):
        a = pts[(i - 1) % 4] - pts[i]
        c = pts[(i + 1) % 4] - pts[i]
        na, nc = np.linalg.norm(a), np.linalg.norm(c)
        if na < 1 or nc < 1:
            return False, "degenerate corner"
        cos = float(np.clip(a @ c / (na * nc), -1.0, 1.0))
        ang = abs(np.degrees(np.arccos(cos)))
        if not (55.0 <= ang <= 125.0):
            return False, f"corner angle {ang:.0f}° outside [55, 125]"
    return True, None


# ---------------------------------------------------------------------------
# strategy 1: HoughLinesP + axis-aligned 4-side fit


def _detect_via_hough(
    paper: np.ndarray, *, min_area_ratio: float
) -> tuple[np.ndarray | None, str | None]:
    h, w = paper.shape[:2]
    boundary = cv2.morphologyEx(paper, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    # Dominant axis from minAreaRect of the boundary (rough; we will refine)
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no paper contour"
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cv2.convexHull(main_contour)) < min_area_ratio * h * w:
        return None, "paper hull too small"
    rect = cv2.minAreaRect(cv2.convexHull(main_contour))
    (cx, cy), (rw, rh), angle_deg = rect
    if rw < rh:
        rw, rh = rh, rw
        angle_deg += 90.0
    theta = np.radians(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])

    # Project boundary points into rect-local coords
    boundary_pts = np.column_stack(np.where(boundary > 0))[:, ::-1].astype(np.float32)
    if len(boundary_pts) < 50:
        return None, "too few boundary points"
    delta = boundary_pts - np.array([cx, cy], dtype=np.float32)
    pu = delta @ u
    pv = delta @ v

    # Robust extents along each axis (5/95 percentiles to ignore protrusions)
    pu_lo, pu_hi = np.percentile(pu, [5, 95])
    pv_lo, pv_hi = np.percentile(pv, [5, 95])
    band = 0.04 * max(pu_hi - pu_lo, pv_hi - pv_lo)

    sides = {
        "right": pu > pu_hi - band,
        "left": pu < pu_lo + band,
        "top": pv < pv_lo + band,
        "bottom": pv > pv_hi - band,
    }

    fitted: dict[str, tuple] = {}
    for name, sel in sides.items():
        pts = boundary_pts[sel]
        if len(pts) < 30:
            return None, f"side '{name}' has only {len(pts)} points"
        # cv2.fitLine with HUBER is robust to outliers from interferers
        line = tuple(
            float(x)
            for x in cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        )
        fitted[name] = line  # type: ignore[assignment]

    tl = _line_intersect(fitted["top"], fitted["left"])
    tr = _line_intersect(fitted["top"], fitted["right"])
    br = _line_intersect(fitted["bottom"], fitted["right"])
    bl = _line_intersect(fitted["bottom"], fitted["left"])
    pts = np.stack([tl, tr, br, bl], axis=0)
    if np.any(~np.isfinite(pts)):
        return None, "intersection failed"
    return _order_quad(pts), None


# ---------------------------------------------------------------------------
# strategy 2: convex hull + approxPolyDP across multiple epsilons


def _detect_via_hull(
    paper: np.ndarray, *, min_area_ratio: float
) -> np.ndarray | None:
    h, w = paper.shape[:2]
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    if cv2.contourArea(hull) < min_area_ratio * h * w:
        return None
    peri = cv2.arcLength(hull, True)
    img_area = float(h * w)
    for eps in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) > 0.98 * img_area:
                continue
            return _order_quad(approx)
    return None


# ---------------------------------------------------------------------------
# strategy 3: minAreaRect on hull


def _detect_via_min_rect(
    paper: np.ndarray, *, min_area_ratio: float
) -> np.ndarray | None:
    h, w = paper.shape[:2]
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    if cv2.contourArea(hull) < min_area_ratio * h * w:
        return None
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    rect_area = cv2.contourArea(box)
    hull_area = cv2.contourArea(hull)
    img_area = float(h * w)
    if rect_area <= 0 or hull_area / rect_area < 0.85 or rect_area > 0.98 * img_area:
        return None
    return _order_quad(box.astype(np.float32))


# ---------------------------------------------------------------------------
# main entry points


def detect_page_quad(
    image: np.ndarray,
    *,
    min_area_ratio: float = 0.25,
    paper_ratio_skip: float = 0.85,
) -> tuple[np.ndarray | None, str | None]:
    """Detect a page quadrilateral. Returns (quad, warning_or_None).

    `quad` is float32 (4, 2) ordered TL/TR/BR/BL, or None if no reliable quad
    could be found. When None, `warning_or_None` explains why.
    """
    if image.ndim < 2:
        return None, "invalid image"
    h, w = image.shape[:2]

    paper = _build_paper_mask(image)
    paper_ratio = float((paper > 0).mean())
    if paper_ratio > paper_ratio_skip:
        return None, (
            f"page already fills the frame ({paper_ratio:.0%} bright pixels); "
            "skipping crop"
        )
    if paper_ratio < 0.10:
        return None, f"no clear paper region detected ({paper_ratio:.0%})"

    # Try strategies in order of preference; validate each result.
    for strategy_name, fn in [
        ("hough", lambda: _detect_via_hough(paper, min_area_ratio=min_area_ratio)),
        ("hull", lambda: (_detect_via_hull(paper, min_area_ratio=min_area_ratio), None)),
        (
            "min_rect",
            lambda: (_detect_via_min_rect(paper, min_area_ratio=min_area_ratio), None),
        ),
    ]:
        result = fn()
        quad = result[0] if isinstance(result, tuple) else result
        if quad is None:
            continue
        ok, why = _quad_sanity(quad, h, w, min_area_ratio=min_area_ratio)
        if ok:
            return quad, None
        # else: try next strategy
    return None, "no strategy produced a sane quad"


def warp_to_page(
    image: np.ndarray,
    quad: np.ndarray,
    *,
    target_long_edge: int | None = None,
) -> np.ndarray:
    quad = quad.astype(np.float32).reshape(4, 2)
    tl, tr, br, bl = quad
    width_top = float(np.linalg.norm(tr - tl))
    width_bot = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    out_w = int(round(max(width_top, width_bot)))
    out_h = int(round(max(height_left, height_right)))
    if target_long_edge is not None:
        scale = target_long_edge / max(out_w, out_h)
        out_w = int(round(out_w * scale))
        out_h = int(round(out_h * scale))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h), flags=cv2.INTER_CUBIC)


def _try_deskew(image: np.ndarray, max_deg: float) -> tuple[np.ndarray, float]:
    if max_deg <= 0:
        return image, 0.0
    try:
        from deskew import determine_skew
    except ImportError:
        return image, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    angle = determine_skew(gray)
    if angle is None or abs(angle) > max_deg:
        return image, 0.0
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, float(angle)


def auto_page_crop(
    image: np.ndarray,
    *,
    min_area_ratio: float = 0.25,
    deskew_max_deg: float = 5.0,
    target_long_edge: int | None = None,
    fallback: Literal["original", "raise"] = "original",
) -> tuple[np.ndarray, PageCropInfo]:
    quad, why = detect_page_quad(image, min_area_ratio=min_area_ratio)
    if quad is None:
        if fallback == "raise":
            raise RuntimeError(why or "page_crop: quadrilateral not detected")
        return image, PageCropInfo(
            cropped=False,
            quad=None,
            deskew_deg=0.0,
            warning=f"page_crop: {why or 'quadrilateral not detected'}; using original",
        )
    warped = warp_to_page(image, quad, target_long_edge=target_long_edge)
    deskewed, angle = _try_deskew(warped, deskew_max_deg)
    return deskewed, PageCropInfo(
        cropped=True,
        quad=tuple((float(x), float(y)) for x, y in quad),
        deskew_deg=angle,
        warning=None,
    )
