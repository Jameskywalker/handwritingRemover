"""Decide which pipeline handles a given input.

M1 only routes images. PDF kinds raise NotImplementedError but the dispatch
machinery is in place so M2/M3 only need to register their pipelines.
"""

from __future__ import annotations

from typing import Any

from inkstrip.io.loaders import looks_like_pdf
from inkstrip.types import InputKind


def detect_input_kind(source: Any) -> InputKind:
    """Best-effort routing. Returns one of image/scanned_pdf/digital_pdf/hybrid_pdf."""
    if looks_like_pdf(source):
        return _classify_pdf(source)
    return "image"


def _classify_pdf(source: Any) -> InputKind:
    """Open with fitz once, count annots vs printed text vs raster XObjects."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Without fitz we can't tell; assume scanned_pdf and let the pipeline raise
        # a clearer error if it's not actually scanned.
        return "scanned_pdf"

    if isinstance(source, (bytes, bytearray, memoryview)):
        doc = fitz.open(stream=bytes(source), filetype="pdf")
    else:
        doc = fitz.open(str(source))

    has_annot_page = False
    has_text_page = False
    has_raster_only_page = False

    try:
        for page in doc:
            annots = list(page.annots() or [])
            text = page.get_text("text") or ""
            images = page.get_images(full=True) or []

            page_has_text = len(text.strip()) > 20
            page_has_annot = len(annots) > 0
            page_is_raster = not page_has_text and len(images) > 0

            has_annot_page |= page_has_annot
            has_text_page |= page_has_text
            has_raster_only_page |= page_is_raster
    finally:
        doc.close()

    if has_annot_page and has_text_page and has_raster_only_page:
        return "hybrid_pdf"
    if has_annot_page and has_text_page:
        return "digital_pdf"
    if has_raster_only_page and not has_text_page:
        return "scanned_pdf"
    if has_annot_page:
        return "digital_pdf"
    return "scanned_pdf"
