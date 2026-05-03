"""Gradio annotation UI: click-to-label OCR bbox crops as HW vs printed.

Usage::

    python scripts/annotate.py /mnt/e/downloads/dataset/*.jpg

For each page we run OCR, then for every bbox crop we ask the user to
label HW / Printed / Mixed / Skip.

Mixed handling — instead of dropping mixed crops, we **recursively split**
them: re-OCR the crop at 2x scale with tight box params; if that returns
≥2 sub-boxes, those replace the mixed item in the queue. Otherwise we
fall back to a vertical-projection split. The loop continues until every
remaining crop is pure HW or pure printed (or skipped).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from inkstrip.detect.ocr_rapid import RapidOcrEngine
from inkstrip.io.loaders import load_image

OUT_HW = Path("data/train/hw")
OUT_PR = Path("data/train/printed")

# minimum area to keep a fragment after recursive split — anything smaller
# is too small to be useful as a training crop and we drop it
_MIN_FRAGMENT_AREA = 200


def _load_resnet():
    try:
        from inkstrip.detect.hw_finetuned import ResNetHwClassifier
        return ResNetHwClassifier(device="cpu")
    except Exception:
        return None


def _make_item(stem: str, x: int, y: int, crop: np.ndarray, prob, text: str, depth: int = 0) -> dict:
    return {
        "stem": stem,
        "x": int(x),
        "y": int(y),
        "crop": crop.copy(),
        "prob": prob,
        "text": text,
        "depth": int(depth),
    }


def _extract_all(page_paths: list[str], ocr, resnet) -> list[dict]:
    """For every page, OCR and return a flat list of crops with metadata."""
    items: list[dict] = []
    for p in page_paths:
        path = Path(p)
        if not path.is_file():
            print(f"skip: {path} not found")
            continue
        arr = load_image(path).array
        boxes = [b for b in ocr.detect(arr) if b.score >= 0.30]
        polys = [b.poly for b in boxes]
        probs = resnet.predict_batch(arr, polys) if resnet is not None else [None] * len(polys)
        for ob, prob in zip(boxes, probs):
            pts = ob.poly.astype(np.int32).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(pts)
            crop = arr[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            items.append(_make_item(path.stem, x, y, crop, prob, ob.text[:30]))
        print(f"  {path.name}: {len(boxes)} boxes")
    return items


def _re_ocr_split(crop: np.ndarray, ocr, scale: float = 2.0) -> list[tuple[int, int, int, int, str]]:
    """Re-OCR an upscaled crop, return sub-boxes in original-crop coords.

    Returns list of (x, y, w, h, text). Filters out boxes that cover most of
    the input (i.e. the same as input) so a single re-detected line doesn't
    just re-add itself.
    """
    h, w = crop.shape[:2]
    if min(h, w) < 8:
        return []
    big = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    sub_boxes = ocr.detect(big)
    out: list[tuple[int, int, int, int, str]] = []
    for ob in sub_boxes:
        if ob.score < 0.20:
            continue
        pts = ob.poly.astype(np.float32).reshape(-1, 2)
        pts /= scale
        ix, iy, iw, ih = cv2.boundingRect(pts.astype(np.int32))
        # discard if covers ≥85% of input area on both axes (= same line)
        if iw >= 0.85 * w and ih >= 0.85 * h:
            continue
        if iw * ih < _MIN_FRAGMENT_AREA:
            continue
        ix = max(0, ix); iy = max(0, iy)
        iw = min(w - ix, iw); ih = min(h - iy, ih)
        if iw <= 0 or ih <= 0:
            continue
        out.append((ix, iy, iw, ih, ob.text[:30]))
    return out


def _projection_split(crop: np.ndarray) -> list[tuple[int, int, int, int, str]]:
    """Fallback: split a wide crop at the largest vertical gap (column of
    near-white pixels). Returns [] if no meaningful gap is found.
    """
    h, w = crop.shape[:2]
    if w < 40 or h < 8:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    # binarise: ink = 1, paper = 0
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    col_sum = ink.sum(axis=0)
    # find the longest run of zero-ink columns (gap), excluding the borders
    margin = max(2, w // 20)
    runs: list[tuple[int, int]] = []
    start = None
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
    if not runs:
        return []
    # pick the longest gap
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    gs, ge = runs[0]
    cuts = [(0, gs), (ge, w)]
    out: list[tuple[int, int, int, int, str]] = []
    for x0, x1 in cuts:
        cw = x1 - x0
        if cw * h < _MIN_FRAGMENT_AREA:
            continue
        out.append((x0, 0, cw, h, ""))
    return out if len(out) >= 2 else []


def _half_split(crop: np.ndarray) -> list[tuple[int, int, int, int, str]]:
    """Last-resort split: cut in half along the longer axis."""
    h, w = crop.shape[:2]
    if w >= h and w >= 16:
        m = w // 2
        return [(0, 0, m, h, ""), (m, 0, w - m, h, "")]
    if h > w and h >= 16:
        m = h // 2
        return [(0, 0, w, m, ""), (0, m, w, h - m, "")]
    return []


def split_mixed(item: dict, ocr, resnet) -> list[dict]:
    """Produce sub-items from a mixed bbox. Tries re-OCR → projection → halve."""
    crop = item["crop"]
    sub = _re_ocr_split(crop, ocr, scale=2.0)
    if len(sub) < 2:
        sub = _projection_split(crop)
    if len(sub) < 2:
        sub = _half_split(crop)
    if not sub:
        return []
    new_items: list[dict] = []
    polys = []
    sub_crops = []
    for ix, iy, iw, ih, _ in sub:
        c = crop[iy : iy + ih, ix : ix + iw]
        if c.size == 0:
            continue
        sub_crops.append((ix, iy, iw, ih, c))
        # synth poly for resnet batch (rect in absolute crop coords; we'll
        # call resnet against the sub-crop directly, one at a time)
    if resnet is not None and sub_crops:
        # Resnet needs (image, [poly]) — easiest: pass crop, full-image poly
        sub_probs = []
        for _, _, _, _, c in sub_crops:
            ph, pw = c.shape[:2]
            poly = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)
            try:
                p = resnet.predict_batch(c, [poly])[0]
            except Exception:
                p = None
            sub_probs.append(p)
    else:
        sub_probs = [None] * len(sub_crops)
    for (ix, iy, iw, ih, c), prob in zip(sub_crops, sub_probs):
        new_items.append(
            _make_item(
                item["stem"],
                item["x"] + ix,
                item["y"] + iy,
                c,
                prob,
                f"[split d{item['depth']+1}]",
                depth=item["depth"] + 1,
            )
        )
    return new_items


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: annotate.py PAGE.jpg [PAGE2.jpg ...]")

    OUT_HW.mkdir(parents=True, exist_ok=True)
    OUT_PR.mkdir(parents=True, exist_ok=True)

    print("loading OCR + ResNet...")
    ocr = RapidOcrEngine(device="cpu")
    resnet = _load_resnet()
    print(f"  resnet hint: {'on' if resnet else 'off'}")

    items = _extract_all(sys.argv[1:], ocr, resnet)
    print(f"\ntotal: {len(items)} crops to label")

    import gradio as gr

    def _color_for(prob):
        if prob is None: return ""
        if prob >= 0.8: return "model says: HANDWRITING (very confident)"
        if prob >= 0.5: return "model says: handwriting (moderate)"
        if prob >= 0.2: return "model says: printed (moderate)"
        return "model says: PRINTED (very confident)"

    def render(idx: int) -> tuple:
        if idx >= len(items):
            return None, f"### done — {idx} processed, queue length {len(items)}"
        it = items[idx]
        prob_s = f"{it['prob']:.2f}" if it.get("prob") is not None else "–"
        info = (
            f"### {idx + 1} / {len(items)}  ·  page: {it['stem']}  ·  "
            f"depth: {it['depth']}  ·  ResNet prob: **{prob_s}**  ({_color_for(it.get('prob'))})\n\n"
            f"OCR text: \"{it['text']}\""
        )
        return it["crop"], info

    def _stats() -> str:
        n_hw = len(list(OUT_HW.glob("*.png")))
        n_pr = len(list(OUT_PR.glob("*.png")))
        return (
            f"saved → HW: {n_hw}, Printed: {n_pr}  ·  queue length: {len(items)}"
        )

    def save_and_step(idx: int, label: str | None) -> tuple:
        if 0 <= idx < len(items):
            it = items[idx]
            if label in ("hw", "printed"):
                out_dir = {"hw": OUT_HW, "printed": OUT_PR}[label]
                out_name = f"{it['stem']}_{it['x']}_{it['y']}_d{it['depth']}.png"
                Image.fromarray(it["crop"]).save(out_dir / out_name)
            elif label == "mixed":
                # split + insert sub-items at idx+1; do NOT advance past current
                # idx — we want the first sub-item to appear next
                subs = split_mixed(it, ocr, resnet)
                if subs:
                    items[idx + 1 : idx + 1] = subs
                    print(f"split mixed @ idx {idx} (depth {it['depth']}) → {len(subs)} sub-boxes")
                else:
                    print(f"warning: mixed @ idx {idx} could not be split — skipping")
        new_idx = idx + 1
        crop, info = render(new_idx)
        return new_idx, crop, info, _stats()

    init_crop, init_info = render(0)
    init_stats = _stats()

    with gr.Blocks(title="inkstrip annotator") as app:
        gr.Markdown(
            "# inkstrip annotator\n"
            "*Click HW / Printed / Mixed / Skip — auto advances. "
            "Mixed = split this box and re-label the pieces.*"
        )
        idx_state = gr.State(0)
        with gr.Row():
            with gr.Column(scale=2):
                img_view = gr.Image(value=init_crop, type="numpy", show_label=False)
                info_md = gr.Markdown(value=init_info)
            with gr.Column(scale=1):
                b_hw = gr.Button("① HW (pure handwriting)", variant="primary", size="lg")
                b_pr = gr.Button("② Printed (pure printed)", variant="secondary", size="lg")
                b_mx = gr.Button("③ Mixed → split & re-label", size="lg")
                b_sk = gr.Button("④ Skip", size="lg")
                stats_md = gr.Markdown(value=init_stats)

        for btn, lbl in (
            (b_hw, "hw"),
            (b_pr, "printed"),
            (b_mx, "mixed"),
            (b_sk, None),
        ):
            btn.click(
                lambda i, _lbl=lbl: save_and_step(i, _lbl),
                inputs=[idx_state],
                outputs=[idx_state, img_view, info_md, stats_md],
            )

    app.launch(server_name="127.0.0.1", server_port=7861, inbrowser=False)


if __name__ == "__main__":
    main()
