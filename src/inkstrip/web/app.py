"""Gradio demo: upload an image, see handwriting removed.

Single pipeline: auto-crop page → OCR finds printed text → HW classifier
flags handwriting bboxes OCR mistook for printed → inverse mask → paper-fill.

Run:
    inkstrip serve            # http://127.0.0.1:7860
    inkstrip serve --share    # public Gradio tunnel for showing on a phone
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from inkstrip.config import InkstripConfig
from inkstrip.io.loaders import load_image
from inkstrip.mask.morph import mask_coverage
from inkstrip.utils.logging import get_logger

_log = get_logger("web")

_OCR_ENGINE: dict[str, Any] = {}
_HW_CLASSIFIER: dict[str, Any] = {}
_RESNET_CLASSIFIER: dict[str, Any] = {}


def _get_ocr_engine(device: str):
    if device not in _OCR_ENGINE:
        from inkstrip.detect.ocr_rapid import RapidOcrEngine

        _log.info("loading RapidOCR engine for device=%s", device)
        _OCR_ENGINE[device] = RapidOcrEngine(device=device if device != "auto" else "cpu")
    return _OCR_ENGINE[device]


def _get_hw_classifier(device: str):
    if device not in _HW_CLASSIFIER:
        from inkstrip.detect.hw_classifier import YoloHwClassifier

        cfg = InkstripConfig(device=device)  # type: ignore[arg-type]
        _log.info("loading YOLOv8n HW classifier device=%s", device)
        _HW_CLASSIFIER[device] = YoloHwClassifier(
            device=device if device != "auto" else "cpu",
            conf=cfg.ocr_hw_conf,
            imgsz=cfg.ocr_hw_imgsz,
        )
    return _HW_CLASSIFIER[device]


def _get_resnet_classifier(device: str):
    if device not in _RESNET_CLASSIFIER:
        from inkstrip.detect.hw_finetuned import ResNetHwClassifier

        _log.info("loading fine-tuned ResNet18 HW classifier device=%s", device)
        try:
            _RESNET_CLASSIFIER[device] = ResNetHwClassifier(
                device=device if device != "auto" else "cpu",
            )
        except FileNotFoundError as e:
            _log.warning("ResNet HW classifier disabled: %s", e)
            _RESNET_CLASSIFIER[device] = None
    return _RESNET_CLASSIFIER[device]


def _process(
    image: Any,
    dilate_px: int,
    use_hw_classifier: bool,
    use_resnet_classifier: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if image is None:
        raise ValueError("upload an image first")

    started = time.perf_counter()
    arr = load_image(image).array

    from inkstrip.preprocess.page_crop import auto_page_crop

    arr, crop_info = auto_page_crop(arr)
    crop_note = ""
    if crop_info.warning:
        crop_note = crop_info.warning
    elif crop_info.cropped:
        crop_note = f"page cropped (deskew {crop_info.deskew_deg:+.1f}°)"

    from inkstrip.mask.ocr_inverse import detect_ocr_inverse_mask

    engine = _get_ocr_engine(device)
    hw_classifier = _get_hw_classifier(device) if use_hw_classifier else None
    resnet_classifier = _get_resnet_classifier(device) if use_resnet_classifier else None
    mask, bbox_count, hw_voted_ocr_rects = detect_ocr_inverse_mask(
        arr,
        ocr_engine=engine,
        hw_classifier=hw_classifier,
        resnet_classifier=resnet_classifier,
        dilate_px=int(dilate_px) if dilate_px > 0 else 5,
    )

    has_target = bool((mask > 0).any())
    effective_mask = mask
    if not has_target:
        cleaned = arr
        note = (
            "OCR detected no printed text — output equals input."
            if bbox_count == 0
            else "Mask is empty — handwriting may have been fully covered by the printed mask."
        )
    else:
        from inkstrip.inpaint.paper_fill import PaperFillInpainter

        painter = PaperFillInpainter(
            InkstripConfig(device=device),  # type: ignore[arg-type]
            hw_classifier=hw_classifier,
        )
        cleaned = painter.inpaint(arr, mask, extra_hw_rects=hw_voted_ocr_rects)
        if painter.last_effective_mask is not None:
            effective_mask = painter.last_effective_mask
        note = ""

    elapsed = (time.perf_counter() - started) * 1000
    cov = mask_coverage(mask) * 100

    mask_rgb = arr.copy()
    mask_rgb[effective_mask > 0] = (
        0.4 * mask_rgb[effective_mask > 0] + 0.6 * np.array([255, 0, 0])
    ).astype(np.uint8)
    eff_cov = (effective_mask > 0).mean() * 100
    summary = (
        f"**input shape** {arr.shape[:2]} · **mask coverage** {cov:.2f}% "
        f"(effective {eff_cov:.2f}%) · **{bbox_count} printed bbox** · **{elapsed:.0f} ms**"
    )
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
            "Page is auto-cropped, OCR finds the printed text, the YOLOv8 "
            "handwriting classifier rescues bboxes OCR mistook for printed "
            "text, and the inverse becomes the mask — then paper-fill."
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
                hw_classifier_cb = gr.Checkbox(
                    value=True,
                    label="HW classifier (YOLOv8n)",
                )
                resnet_classifier_cb = gr.Checkbox(
                    value=True,
                    label="Fine-tuned classifier (ResNet18 — catches neat handwriting YOLO misses)",
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
            inputs=[inp, dilate, hw_classifier_cb, resnet_classifier_cb, device],
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
