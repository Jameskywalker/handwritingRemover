"""Gradio demo: upload an image, see handwriting removed.

This UI is a thin shell around the image pipeline — same code path the CLI
uses. Three panels: original / detected mask / cleaned. Sliders expose the
two parameters most worth tweaking on novel inputs (detection confidence,
mask dilation).

Run:
    inkstrip serve            # http://127.0.0.1:7860
    inkstrip serve --share    # public Gradio tunnel for showing on a phone
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from inkstrip.config import InkstripConfig
from inkstrip.detect.yolo_hw import YoloHandwritingDetector
from inkstrip.inpaint.lama_onnx import LamaOnnxInpainter
from inkstrip.io.loaders import load_image
from inkstrip.mask.morph import MorphMaskBuilder, mask_coverage
from inkstrip.utils.logging import get_logger

_log = get_logger("web")

# Cache of (device, inpainter, image_size) → detector/inpainter so we don't reload
# weights on every image.
_DETECTOR: dict[str, YoloHandwritingDetector] = {}
_INPAINTER: dict[str, LamaOnnxInpainter] = {}


def _get_detector(cfg: InkstripConfig) -> YoloHandwritingDetector:
    key = f"{cfg.device}|{cfg.det_imgsz}|{cfg.detector}"
    if key not in _DETECTOR:
        _log.info("loading YOLO detector: %s", key)
        _DETECTOR[key] = YoloHandwritingDetector(cfg)
    return _DETECTOR[key]


def _get_inpainter(cfg: InkstripConfig) -> LamaOnnxInpainter:
    key = f"{cfg.device}|{cfg.tile_size}"
    if key not in _INPAINTER:
        _log.info("loading LaMa ONNX inpainter: %s", key)
        _INPAINTER[key] = LamaOnnxInpainter(cfg)
    return _INPAINTER[key]


def _process(
    image: Any,
    confidence: float,
    dilate_px: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if image is None:
        raise ValueError("upload an image first")

    cfg = InkstripConfig(
        det_conf=float(confidence),
        dilate_px=int(dilate_px) if dilate_px > 0 else None,
        device=device,  # type: ignore[arg-type]
    )

    started = time.perf_counter()
    arr = load_image(image, max_megapixels=cfg.max_image_megapixels).array

    detector = _get_detector(cfg)
    boxes = detector.detect(arr)
    mask = MorphMaskBuilder(cfg).build(arr, boxes)

    if not boxes:
        cleaned = arr
        note = "No handwriting detected — output equals input."
    else:
        inpainter = _get_inpainter(cfg)
        cleaned = inpainter.inpaint(arr, mask)
        note = ""

    elapsed = (time.perf_counter() - started) * 1000

    mask_rgb = np.stack([mask, mask, mask], axis=-1)
    summary = (
        f"**{len(boxes)}** bbox · mask coverage **{mask_coverage(mask) * 100:.2f}%** "
        f"· {elapsed:.0f} ms"
    )
    if note:
        summary += f"\n\n{note}"
    return cleaned, mask_rgb, summary


def build_ui():
    import gradio as gr

    with gr.Blocks(title="inkstrip — remove handwriting") as demo:
        gr.Markdown(
            "# inkstrip\n"
            "Upload a photo or scanned page that contains handwritten ink on top of "
            "printed content. The model detects handwriting bounding boxes, expands "
            "them into a mask, and asks LaMa to repaint the background."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(
                    label="Input image",
                    type="numpy",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                )
                conf = gr.Slider(
                    minimum=0.05,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="Detector confidence",
                )
                dilate = gr.Slider(
                    minimum=0,
                    maximum=25,
                    value=0,
                    step=1,
                    label="Mask dilation (px) — 0 means auto-scale by image size",
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
            inputs=[inp, conf, dilate, device],
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
