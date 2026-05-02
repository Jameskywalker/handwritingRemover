"""Gradio demo: upload an image, see handwriting removed.

Single strategy: OCR finds printed text, the handwriting classifier
rescues bboxes OCR mistook for printed text, and the inverse of that
becomes the mask. Sliders expose only what's worth tweaking.

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
from inkstrip.mask.morph import mask_coverage
from inkstrip.utils.logging import get_logger

_log = get_logger("web")

# Cache heavy components by device — first request warms them, subsequent
# requests reuse.
_INPAINTER: dict[str, LamaOnnxInpainter] = {}
_OCR_ENGINE: dict[str, Any] = {}
_HW_CLASSIFIER: dict[str, Any] = {}


def _get_inpainter(device: str) -> LamaOnnxInpainter:
    if device not in _INPAINTER:
        _log.info("loading LaMa ONNX inpainter for device=%s", device)
        _INPAINTER[device] = LamaOnnxInpainter(InkstripConfig(device=device))
    return _INPAINTER[device]


def _get_ocr_engine(device: str):
    if device not in _OCR_ENGINE:
        from inkstrip.detect.ocr_rapid import RapidOcrEngine

        _log.info("loading RapidOCR engine for device=%s", device)
        _OCR_ENGINE[device] = RapidOcrEngine(device=device if device != "auto" else "cpu")
    return _OCR_ENGINE[device]


def _get_hw_classifier(device: str):
    if device not in _HW_CLASSIFIER:
        from inkstrip.detect.hw_classifier import YoloHwClassifier

        _log.info("loading YOLOv8n HW classifier for device=%s", device)
        _HW_CLASSIFIER[device] = YoloHwClassifier(
            device=device if device != "auto" else "cpu"
        )
    return _HW_CLASSIFIER[device]


def _process(
    image: Any,
    dilate_px: int,
    page_crop: bool,
    use_hw_classifier: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if image is None:
        raise ValueError("upload an image first")

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

    from inkstrip.mask.ocr_inverse import detect_ocr_inverse_mask

    engine = _get_ocr_engine(device)
    hw_classifier = _get_hw_classifier(device) if use_hw_classifier else None
    mask, bbox_count = detect_ocr_inverse_mask(
        arr,
        ocr_engine=engine,
        hw_classifier=hw_classifier,
        dilate_px=int(dilate_px) if dilate_px > 0 else 5,
    )

    has_target = bool((mask > 0).any())
    if not has_target:
        cleaned = arr
        note = (
            "OCR detected no printed text — output equals input."
            if bbox_count == 0
            else "Mask is empty — handwriting may have been fully covered by the printed mask."
        )
    else:
        cleaned = _get_inpainter(device).inpaint(arr, mask)
        note = ""

    elapsed = (time.perf_counter() - started) * 1000
    cov = mask_coverage(mask) * 100

    mask_rgb = np.stack([mask, mask, mask], axis=-1)
    summary = f"**mask coverage** {cov:.2f}% · **{bbox_count} printed bbox** · **{elapsed:.0f} ms**"
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
            "Upload a page with handwritten ink on printed content. "
            "OCR finds the printed text; a YOLOv8 handwriting classifier "
            "rescues bboxes OCR mistook for printed text; the inverse is "
            "the handwriting mask, which gets inpainted away."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(
                    label="Input image",
                    type="numpy",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                )
                dilate = gr.Slider(
                    minimum=0,
                    maximum=25,
                    value=5,
                    step=1,
                    label="Mask dilation (px)",
                )
                page_crop_cb = gr.Checkbox(
                    value=False,
                    label="Auto-crop page (perspective-warp phone photos)",
                )
                hw_classifier_cb = gr.Checkbox(
                    value=True,
                    label="HW classifier (rescue same-color handwriting OCR mistook for printed text)",
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
            inputs=[inp, dilate, page_crop_cb, hw_classifier_cb, device],
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
