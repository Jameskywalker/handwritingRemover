"""Gradio demo: upload an image, see handwriting removed.

Three panels: original / mask / cleaned. The dropdown picks between the
two detection strategies — color-based (best for red/blue/etc. ink on
printed black text) and YOLO+morph (best for English handwriting). Sliders
expose the parameters most worth tweaking.

Run:
    inkstrip serve            # http://127.0.0.1:7860
    inkstrip serve --share    # public Gradio tunnel for showing on a phone
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from inkstrip.config import InkstripConfig
from inkstrip.inpaint.lama_onnx import LamaOnnxInpainter
from inkstrip.io.loaders import load_image
from inkstrip.mask.color import detect_color_mask
from inkstrip.mask.morph import mask_coverage
from inkstrip.utils.logging import get_logger

_log = get_logger("web")

# Cache heavy components by device — first request warms them, subsequent
# requests reuse.
_INPAINTER: dict[str, LamaOnnxInpainter] = {}
_DETECTOR: dict[str, Any] = {}

_STRATEGY_LABELS = {
    "color_red": "Red ink (recommended for red pen on printed text)",
    "color_blue": "Blue ink",
    "color_any": "Any colored ink (red+blue+green)",
    "yolo_morph": "YOLOv8 detector (handwriting in any color, English-leaning)",
    "ocr_inverse": "OCR inverse (black & white printed page with black handwriting)",
}

_OCR_ENGINE: dict[str, Any] = {}


def _get_inpainter(device: str) -> LamaOnnxInpainter:
    if device not in _INPAINTER:
        _log.info("loading LaMa ONNX inpainter for device=%s", device)
        _INPAINTER[device] = LamaOnnxInpainter(InkstripConfig(device=device))
    return _INPAINTER[device]


def _get_detector(device: str):
    if device not in _DETECTOR:
        from inkstrip.detect.yolo_hw import YoloHandwritingDetector

        _log.info("loading YOLO detector for device=%s", device)
        _DETECTOR[device] = YoloHandwritingDetector(InkstripConfig(device=device))
    return _DETECTOR[device]


def _get_ocr_engine(device: str):
    if device not in _OCR_ENGINE:
        from inkstrip.detect.ocr_rapid import RapidOcrEngine

        _log.info("loading RapidOCR engine for device=%s", device)
        _OCR_ENGINE[device] = RapidOcrEngine(device=device if device != "auto" else "cpu")
    return _OCR_ENGINE[device]


def _process(
    image: Any,
    strategy_label: str,
    dilate_px: int,
    delta: int,
    protect_print: bool,
    page_crop: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if image is None:
        raise ValueError("upload an image first")
    strategy = next(k for k, v in _STRATEGY_LABELS.items() if v == strategy_label)

    started = time.perf_counter()
    arr = load_image(image).array

    crop_note = ""
    if page_crop:
        from inkstrip.preprocess.page_crop import auto_page_crop

        arr, info = auto_page_crop(arr)
        if info.warning:
            crop_note = info.warning
        elif info.cropped:
            crop_note = f"page cropped (deskew {info.deskew_deg:+.1f}°)"

    if strategy == "yolo_morph":
        from inkstrip.mask.morph import MorphMaskBuilder

        cfg = InkstripConfig(
            mask_strategy="yolo_morph",
            dilate_px=int(dilate_px) if dilate_px > 0 else None,
            device=device,  # type: ignore[arg-type]
        )
        detector = _get_detector(device)
        boxes = detector.detect(arr)
        mask = MorphMaskBuilder(cfg).build(arr, boxes)
        bbox_count = len(boxes)
    elif strategy == "ocr_inverse":
        from inkstrip.mask.ocr_inverse import detect_ocr_inverse_mask

        engine = _get_ocr_engine(device)
        mask, bbox_count = detect_ocr_inverse_mask(
            arr,
            ocr_engine=engine,
            dilate_px=int(dilate_px) if dilate_px > 0 else 5,
        )
    else:
        profile = {"color_red": "red", "color_blue": "blue", "color_any": "any_colored"}[strategy]
        mask = detect_color_mask(
            arr,
            profile=profile,
            dilate_px=int(dilate_px) if dilate_px > 0 else 7,
            delta=int(delta) if delta > 0 else None,
            protect_print=protect_print,
        )
        bbox_count = 0

    has_target = bool((mask > 0).any())
    if not has_target:
        cleaned = arr
        if strategy == "ocr_inverse" and bbox_count == 0:
            note = "OCR detected no printed text — output equals input (try a different strategy if this is a pure handwriting page)."
        else:
            note = "No handwriting detected for this strategy — try lowering delta or switching strategy."
    else:
        cleaned = _get_inpainter(device).inpaint(arr, mask)
        note = ""

    elapsed = (time.perf_counter() - started) * 1000
    cov = mask_coverage(mask) * 100

    mask_rgb = np.stack([mask, mask, mask], axis=-1)
    summary = f"**strategy** {strategy} · **mask coverage** {cov:.2f}%"
    if bbox_count:
        summary += f" · **{bbox_count} bbox**"
    summary += f" · **{elapsed:.0f} ms**"
    if crop_note:
        summary += f"\n\n_{crop_note}_"
    if note:
        summary += f"\n\n{note}"
    return cleaned, mask_rgb, summary


def build_ui():
    import gradio as gr

    with gr.Blocks(title="inkstrip — remove handwriting") as demo:
        gr.Markdown(
            "# inkstrip\n"
            "Upload a page with handwritten ink on top of printed content. "
            "Pick a strategy that matches the ink color — for most real-world "
            "documents (red pen, blue pen) the color modes give dramatically "
            "better results than the YOLO detector."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(
                    label="Input image",
                    type="numpy",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                )
                strategy = gr.Dropdown(
                    choices=list(_STRATEGY_LABELS.values()),
                    value=_STRATEGY_LABELS["color_red"],
                    label="Strategy",
                )
                dilate = gr.Slider(
                    minimum=0,
                    maximum=25,
                    value=7,
                    step=1,
                    label="Mask dilation (px)",
                )
                delta = gr.Slider(
                    minimum=0,
                    maximum=120,
                    value=0,
                    step=5,
                    label="Color δ (0 = profile default; raise to be stricter)",
                )
                protect = gr.Checkbox(
                    value=True,
                    label="Protect printed text (skip black pixels even if dilated)",
                )
                page_crop_cb = gr.Checkbox(
                    value=False,
                    label="Auto-crop page (perspective-warp phone photos)",
                )
                device = gr.Radio(
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    label="Device",
                )
                run_btn = gr.Button("Remove handwriting", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    cleaned_img = gr.Image(label="Cleaned", type="numpy")
                    mask_img = gr.Image(label="Mask preview", type="numpy")
                summary = gr.Markdown("")

        run_btn.click(
            fn=_process,
            inputs=[inp, strategy, dilate, delta, protect, page_crop_cb, device],
            outputs=[cleaned_img, mask_img, summary],
        )

    return demo


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    open_browser: bool = True,
) -> None:
    demo = build_ui()
    demo.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=open_browser,
        show_error=True,
    )


if __name__ == "__main__":
    serve()
