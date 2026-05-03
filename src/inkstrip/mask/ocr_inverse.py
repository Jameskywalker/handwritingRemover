"""Mask handwriting using OCR to localise printed text.

Default mode (``combine_color=True``) builds:

    (coloured ink — guaranteed handwriting) ∪ (dark ink minus printed glyphs)

Two pieces, two failure modes covered:

1. **Coloured ink** (red / blue / green pen on a printed b&w page). Modern
   OCR happily recognises coloured handwriting as text and would mark it as
   "printed", subtracting it from the mask — exactly the wrong thing. We
   bypass the OCR veto for any pixel that's coloured.
2. **Black handwriting on a black-printed page**. There's no colour cue, so
   OCR is the only tool. We binarise the page, locate printed glyphs inside
   each OCR bbox via local Otsu, dilate slightly, and subtract.

A pure-monochrome page collapses gracefully: the colour layer is empty so the
result is exactly the classic ``ink − printed`` mask. Setting
``combine_color=False`` recovers that strict behaviour explicitly.

If OCR finds zero printed boxes *and* no colour ink is present, we return an
empty mask so the pipeline can warn rather than erase the entire page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from inkstrip.mask.color import _post_process

if TYPE_CHECKING:
    from inkstrip.config import InkstripConfig
    from inkstrip.detect.hw_classifier import HwClassifier
    from inkstrip.detect.hw_finetuned import ResNetHwClassifier
    from inkstrip.detect.ocr_rapid import OcrEngine


def _ocr_box_is_handwriting(
    poly: np.ndarray,
    hw_rects: list[tuple[int, int, int, int]],
    *,
    threshold: float = 0.30,
    resnet_prob: float | None = None,
    resnet_threshold: float = 0.5,
) -> bool:
    """Decide whether an OCR bbox is handwriting.

    Two pieces of evidence, ``OR``-ed:

    1. **YOLO union vote** — the union of HW classifier bboxes covers
       ≥ ``threshold`` of the OCR rect.
    2. **Fine-tuned ResNet** (optional) — if a ``resnet_prob`` is supplied
       (precomputed in batch by the caller), values ≥ ``resnet_threshold``
       flip the bbox to handwriting. Catches neat handwriting where YOLO
       didn't fire at all.
    """
    pts = poly.astype(np.int32).reshape(-1, 2)
    ox, oy, ow, oh = cv2.boundingRect(pts)
    if ow <= 0 or oh <= 0:
        return False

    canvas = np.zeros((oh, ow), dtype=bool)
    for hx, hy, hw_, hh in hw_rects:
        ix0 = max(0, hx - ox)
        iy0 = max(0, hy - oy)
        ix1 = min(ow, hx + hw_ - ox)
        iy1 = min(oh, hy + hh - oy)
        if ix1 > ix0 and iy1 > iy0:
            canvas[iy0:iy1, ix0:ix1] = True
    if canvas.sum() / (ow * oh) >= threshold:
        return True

    if resnet_prob is not None and resnet_prob >= resnet_threshold:
        return True
    return False


def _split_box_at_gaps(
    image: np.ndarray,
    poly: np.ndarray,
    *,
    min_gap_ratio: float = 0.25,
    min_sub_width_ratio: float = 0.10,
    min_sub_width_abs: int = 12,
) -> list[np.ndarray]:
    """Split a single OCR line bbox at every significant vertical gap.

    Otsu-binarise the crop, find every run of zero-ink columns longer than
    ``min_gap_ratio * crop_height`` pixels, cut the bbox at each gap. Drops
    sub-fragments narrower than ``max(min_sub_width_abs, min_sub_width_ratio
    * crop_width)``.

    Returns a list of rectangular polys in image coordinates. Returns the
    original poly (wrapped in a 1-element list) if no usable split is found —
    callers can treat the result uniformly.
    """
    pts = poly.astype(np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    if w < 16 or h < 8:
        return [poly]
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return [poly]

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    col_sum = ink.sum(axis=0)

    min_gap_px = max(3, int(round(h * min_gap_ratio)))
    min_sub_w = max(min_sub_width_abs, int(round(w * min_sub_width_ratio)))

    # find ink runs (start, end+1) — i.e. each contiguous group of non-empty
    # columns becomes a sub-box. Gaps shorter than min_gap_px don't split.
    splits: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    gap_len = 0
    for i in range(w):
        if col_sum[i] > 0:
            if not in_run:
                in_run = True
                run_start = i
            gap_len = 0
        else:
            if in_run:
                gap_len += 1
                if gap_len >= min_gap_px:
                    splits.append((run_start, i - gap_len + 1))
                    in_run = False
                    gap_len = 0
    if in_run:
        splits.append((run_start, w))

    if len(splits) <= 1:
        return [poly]

    out: list[np.ndarray] = []
    for x0, x1 in splits:
        sw = x1 - x0
        if sw < min_sub_w:
            continue
        out.append(np.array(
            [[x + x0, y], [x + x1, y], [x + x1, y + h], [x + x0, y + h]],
            dtype=np.float32,
        ))
    return out if len(out) >= 2 else [poly]


def _split_box_polys(
    image: np.ndarray,
    poly: np.ndarray,
    ocr_engine: "OcrEngine",
    *,
    scale: float = 2.0,
    min_fragment_area: int = 200,
) -> list[np.ndarray]:
    """Decompose a single OCR bbox into rectangular sub-polys for re-classification.

    Strategy (in order):
      1. Re-OCR the bbox crop upscaled by ``scale`` — if RapidOCR returns
         ≥ 2 sub-boxes that aren't ~the same as the input, use those
      2. Vertical-projection split at the largest column of zero-ink pixels
      3. Halve along the longer axis

    Returns a list of 4-corner polys in *image* coordinates. Returns
    ``[]`` if the bbox is too small to split usefully.
    """
    pts = poly.astype(np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    if w < 16 or h < 8:
        return []
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return []

    def _rect_poly(rx: int, ry: int, rw: int, rh: int) -> np.ndarray:
        return np.array(
            [[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]],
            dtype=np.float32,
        )

    # 1. re-OCR the upscaled crop
    big = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    sub_boxes = ocr_engine.detect(big)
    re_polys: list[np.ndarray] = []
    for ob in sub_boxes:
        if ob.score < 0.20:
            continue
        sp = ob.poly.astype(np.float32).reshape(-1, 2) / scale
        ix, iy, iw, ih = cv2.boundingRect(sp.astype(np.int32))
        # discard if covers ≥85% of input area on both axes (= same line)
        if iw >= 0.85 * w and ih >= 0.85 * h:
            continue
        if iw * ih < min_fragment_area:
            continue
        ix = max(0, ix); iy = max(0, iy)
        iw = min(w - ix, iw); ih = min(h - iy, ih)
        if iw <= 0 or ih <= 0:
            continue
        re_polys.append(_rect_poly(x + ix, y + iy, iw, ih))
    if len(re_polys) >= 2:
        return re_polys

    # 2. vertical-projection split at the largest gap
    if w >= 40:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        col_sum = ink.sum(axis=0)
        margin = max(2, w // 20)
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i in range(margin, w - margin):
            if col_sum[i] == 0:
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= max(4, w // 30):
                    runs.append((start, i))
                    start = None
        if start is not None and (w - margin) - start >= max(4, w // 30):
            runs.append((start, w - margin))
        if runs:
            runs.sort(key=lambda r: r[1] - r[0], reverse=True)
            gs, ge = runs[0]
            cuts = [(0, gs), (ge, w)]
            proj_polys: list[np.ndarray] = []
            for x0, x1 in cuts:
                cw = x1 - x0
                if cw * h < min_fragment_area:
                    continue
                proj_polys.append(_rect_poly(x + x0, y, cw, h))
            if len(proj_polys) >= 2:
                return proj_polys

    # 3. halve along longer axis (last resort)
    if w >= h and w >= 16:
        m = w // 2
        return [_rect_poly(x, y, m, h), _rect_poly(x + m, y, w - m, h)]
    if h > w and h >= 16:
        m = h // 2
        return [_rect_poly(x, y, w, m), _rect_poly(x, y + m, w, h - m)]
    return []


def _build_ink_mask(image: np.ndarray, block_size: int, C: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    bs = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        C,
    )


def _build_printed_glyph_mask(
    image: np.ndarray,
    polys: list[np.ndarray],
    *,
    pad_px: int,
    glyph_dilate_px: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    glyph_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = poly.astype(np.int32).reshape(-1, 2)
        x, y, ww, hh = cv2.boundingRect(pts)
        x0 = max(0, x - pad_px)
        y0 = max(0, y - pad_px)
        x1 = min(w, x + ww + pad_px)
        y1 = min(h, y + hh + pad_px)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        # Otsu to grab actual glyph pixels inside the bbox.
        _, local = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Constrain to the polygon itself (with the same padding) so glyph mask
        # does not leak past the recognised text region.
        poly_clip = np.zeros_like(local)
        shifted = pts - np.array([x0, y0], dtype=np.int32)
        cv2.fillPoly(poly_clip, [shifted], 255)
        if pad_px > 0:
            kp = cv2.getStructuringElement(cv2.MORPH_RECT, (pad_px * 2 + 1, pad_px * 2 + 1))
            poly_clip = cv2.dilate(poly_clip, kp, iterations=1)
        local = cv2.bitwise_and(local, poly_clip)
        glyph_mask[y0:y1, x0:x1] = cv2.bitwise_or(glyph_mask[y0:y1, x0:x1], local)
    if glyph_dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (glyph_dilate_px, glyph_dilate_px)
        )
        glyph_mask = cv2.dilate(glyph_mask, k, iterations=1)
    return glyph_mask


def detect_ocr_inverse_mask(
    image: np.ndarray,
    *,
    ocr_engine: "OcrEngine",
    ink_block_size: int = 25,
    ink_C: int = 10,
    printed_pad_px: int = 4,
    printed_glyph_dilate_px: int = 2,
    min_confidence: float = 0.30,
    dilate_px: int = 5,
    closing_px: int = 3,
    min_component_area: int = 12,
    combine_color: bool = True,
    color_profile: str = "any_colored",
    color_dilate_px: int = 5,
    hw_classifier: "HwClassifier | None" = None,
    hw_overlap_threshold: float = 0.30,
    resnet_classifier: "ResNetHwClassifier | None" = None,
    resnet_threshold: float = 0.5,
    resnet_split_uncertain: bool = True,
    resnet_uncertain_low: float = 0.05,
    resnet_uncertain_high: float = 0.50,
    split_lines_at_gaps: bool = True,
) -> tuple[np.ndarray, int, list[tuple[int, int, int, int]]]:
    """Build a handwriting mask using OCR to localise printed text.

    With ``combine_color=True`` (default) the result is

        (colored ink — guaranteed handwriting) ∪ (dark ink minus printed glyphs)

    This handles the common real-world case where the handwriting is in
    coloured pen and OCR happily recognises it as text — pure subtraction
    would erase those strokes from the mask. Colour pixels bypass the OCR
    veto. On a strictly monochrome page the colour layer is empty, so the
    result reduces to the original ink − printed semantics.
    """
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")

    boxes = ocr_engine.detect(image)
    accepted_boxes = [b for b in boxes if b.score >= min_confidence]

    # First-pass granularity boost — split each OCR line at every
    # significant vertical gap so the classifier sees word/character-level
    # crops rather than full-line averages. Mixed lines decompose naturally.
    if split_lines_at_gaps and accepted_boxes:
        from inkstrip.detect.ocr_rapid import OcrBox

        split_boxes: list = []
        for ob in accepted_boxes:
            sub_polys = _split_box_at_gaps(image, ob.poly)
            if len(sub_polys) <= 1:
                split_boxes.append(ob)
                continue
            for sp in sub_polys:
                split_boxes.append(OcrBox(poly=sp, text=ob.text, score=ob.score))
        accepted_boxes = split_boxes

    h, w = image.shape[:2]

    # Per-bbox HW classification: drop OCR boxes that overlap a handwriting
    # bbox significantly. Operating on bboxes (not pixels) avoids the trap
    # where the HW bbox covers some printed glyphs underneath and ends up
    # protecting them too.
    hw_voted_ocr_rects: list[tuple[int, int, int, int]] = []
    if (hw_classifier is not None or resnet_classifier is not None) and accepted_boxes:
        hw_rects: list[tuple[int, int, int, int]] = []
        if hw_classifier is not None:
            hw_rects = [b.bbox for b in hw_classifier.detect(image)]

        # Batch ResNet inference once for all OCR boxes — much faster than
        # per-box GPU forwards (one forward pass instead of len(boxes)).
        resnet_probs: list[float] = []
        if resnet_classifier is not None:
            resnet_probs = resnet_classifier.predict_batch(
                image, [b.poly for b in accepted_boxes]
            )

        from inkstrip.detect.ocr_rapid import OcrBox

        kept: list = []
        uncertain_indices: list[int] = []
        for i, ob in enumerate(accepted_boxes):
            pts = ob.poly.astype(np.int32).reshape(-1, 2)
            ox, oy, ow, oh = cv2.boundingRect(pts)
            if ow <= 0 or oh <= 0:
                continue

            # YOLO union vote — decisive if it fires (no split needed)
            if hw_rects:
                canvas = np.zeros((oh, ow), dtype=bool)
                for hx, hy, hw_, hh in hw_rects:
                    ix0 = max(0, hx - ox); iy0 = max(0, hy - oy)
                    ix1 = min(ow, hx + hw_ - ox); iy1 = min(oh, hy + hh - oy)
                    if ix1 > ix0 and iy1 > iy0:
                        canvas[iy0:iy1, ix0:ix1] = True
                if canvas.sum() / (ow * oh) >= hw_overlap_threshold:
                    hw_voted_ocr_rects.append((ox, oy, ow, oh))
                    continue

            rprob = resnet_probs[i] if resnet_probs else None

            # No ResNet → fall back to threshold semantics
            if rprob is None:
                kept.append(ob)
                continue

            # Confidently handwriting — protect, no split
            if rprob >= resnet_uncertain_high:
                hw_voted_ocr_rects.append((ox, oy, ow, oh))
                continue

            # Confidently printed — accept for printed-glyph subtraction, no split
            if rprob <= resnet_uncertain_low:
                kept.append(ob)
                continue

            # Uncertain band — defer for splitting
            if resnet_split_uncertain:
                uncertain_indices.append(i)
            else:
                # split disabled: fall back to threshold decision
                if rprob >= resnet_threshold:
                    hw_voted_ocr_rects.append((ox, oy, ow, oh))
                else:
                    kept.append(ob)

        # Re-classify split sub-rects from uncertain boxes — one batched
        # ResNet forward over all sub-polys keeps the cost ~constant.
        if uncertain_indices and resnet_classifier is not None:
            split_records: list[tuple[int, np.ndarray]] = []  # (orig_idx, sub_poly)
            for idx in uncertain_indices:
                ob = accepted_boxes[idx]
                sub_polys = _split_box_polys(image, ob.poly, ocr_engine)
                if len(sub_polys) < 2:
                    # cannot split — fall back to original threshold decision
                    pts = ob.poly.astype(np.int32).reshape(-1, 2)
                    rect = cv2.boundingRect(pts)
                    if resnet_probs[idx] >= resnet_threshold:
                        hw_voted_ocr_rects.append(rect)
                    else:
                        kept.append(ob)
                    continue
                for sp in sub_polys:
                    split_records.append((idx, sp))

            if split_records:
                sub_polys_only = [sp for _, sp in split_records]
                sub_probs = resnet_classifier.predict_batch(image, sub_polys_only)
                for (orig_idx, sp), prob in zip(split_records, sub_probs):
                    pts = sp.astype(np.int32).reshape(-1, 2)
                    rect = cv2.boundingRect(pts)
                    if prob >= resnet_threshold:
                        hw_voted_ocr_rects.append(rect)
                    else:
                        kept.append(
                            OcrBox(
                                poly=sp,
                                text=accepted_boxes[orig_idx].text,
                                score=accepted_boxes[orig_idx].score,
                            )
                        )
        accepted_boxes = kept

    accepted = [b.poly for b in accepted_boxes]

    color_layer = np.zeros((h, w), dtype=np.uint8)
    if combine_color:
        from inkstrip.mask.color import detect_color_mask

        color_layer = detect_color_mask(
            image,
            profile=color_profile,
            dilate_px=color_dilate_px,
            closing_px=closing_px,
            min_component_area=min_component_area,
            protect_print=True,
        )

    ink_mask = _build_ink_mask(image, ink_block_size, ink_C)

    if not accepted:
        # No printed text remains after HW filtering. If we have a colour
        # layer, that *is* the handwriting mask; otherwise return empty so
        # the pipeline emits a warning rather than erasing the whole page.
        if combine_color and color_layer.any():
            return color_layer, 0, hw_voted_ocr_rects
        return np.zeros((h, w), dtype=np.uint8), 0, hw_voted_ocr_rects

    printed_mask = _build_printed_glyph_mask(
        image,
        accepted,
        pad_px=printed_pad_px,
        glyph_dilate_px=printed_glyph_dilate_px,
    )

    dark_handwriting = cv2.bitwise_and(ink_mask, cv2.bitwise_not(printed_mask))

    if combine_color:
        # Coloured ink bypasses the OCR veto entirely — coloured glyphs are
        # always handwriting in our target use case (red/blue/green pen on a
        # printed black-and-white document).
        combined = cv2.bitwise_or(color_layer, dark_handwriting)
    else:
        combined = dark_handwriting

    final = _post_process(
        combined,
        dilate_px=dilate_px,
        closing_px=closing_px,
        min_component_area=min_component_area,
        image=None,  # printed text already subtracted; do not re-protect
    )
    return final, len(accepted), hw_voted_ocr_rects


class OcrInverseMaskBuilder:
    """MaskBuilder-shaped wrapper. `build()` returns (mask, printed_box_count)."""

    def __init__(
        self,
        cfg: "InkstripConfig",
        ocr_engine: "OcrEngine | None" = None,
        hw_classifier: "HwClassifier | None" = None,
        resnet_classifier: "ResNetHwClassifier | None" = None,
    ) -> None:
        self.cfg = cfg
        if ocr_engine is None:
            from inkstrip.detect.ocr_rapid import RapidOcrEngine

            device = "cuda" if cfg.device == "cuda" else "cpu"
            ocr_engine = RapidOcrEngine(
                lang=cfg.ocr_lang,
                device=device,
                text_score=cfg.ocr_min_confidence,
            )
        self._engine = ocr_engine

        if hw_classifier is None and cfg.ocr_use_hw_classifier:
            from inkstrip.detect.hw_classifier import YoloHwClassifier

            device = "cuda" if cfg.device == "cuda" else "cpu"
            hw_classifier = YoloHwClassifier(
                device=device,
                conf=cfg.ocr_hw_conf,
                imgsz=cfg.ocr_hw_imgsz,
            )
        self._hw_classifier = hw_classifier
        self._resnet_classifier = resnet_classifier

    def build(self, image: np.ndarray, boxes=None) -> tuple[np.ndarray, int]:
        cfg = self.cfg
        dilate_px = cfg.dilate_px if cfg.dilate_px is not None else 5
        mask, n, hw_voted = detect_ocr_inverse_mask(
            image,
            ocr_engine=self._engine,
            printed_pad_px=cfg.ocr_printed_pad_px,
            printed_glyph_dilate_px=cfg.ocr_printed_dilate_px,
            min_confidence=cfg.ocr_min_confidence,
            combine_color=cfg.ocr_combine_color,
            dilate_px=dilate_px,
            closing_px=cfg.closing_px,
            min_component_area=cfg.min_box_area,
            hw_classifier=self._hw_classifier,
            hw_overlap_threshold=cfg.ocr_hw_overlap_threshold,
            resnet_classifier=self._resnet_classifier,
            resnet_threshold=cfg.ocr_resnet_threshold,
            resnet_split_uncertain=cfg.ocr_resnet_split_uncertain,
            resnet_uncertain_low=cfg.ocr_resnet_uncertain_low,
            resnet_uncertain_high=cfg.ocr_resnet_uncertain_high,
            split_lines_at_gaps=cfg.ocr_split_lines_at_gaps,
        )
        self.last_hw_voted_ocr_rects = hw_voted
        return mask, n
