# inkstrip

Remove handwritten ink, annotations, and notes from documents while preserving printed content.

## Status

Pre-alpha. M1 (image pipeline) in progress.

## Supported inputs

- Photos of handwritten-on-printed documents (perspective + lighting variation)
- Flatbed scans (single image or multi-page PDF)
- Digital PDFs with handwritten annotations (Ink, FreeText, Highlight, Stamp, ...)

## Pipelines

| Input | Strategy | Status |
|---|---|---|
| Image / scanned page | YOLOv8 handwriting detection → morphological mask → LaMa inpainting | M1 |
| Scanned PDF | Rasterize via pypdfium2 → image pipeline per page → reassemble | M2 |
| Digital PDF (annotation layer) | PyMuPDF: enumerate `page.annots()` → `delete_annot` → `save(garbage=4)` | M3 |
| Hybrid PDF | Per-page routing | M3 |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[detect,inpaint,ui]"
inkstrip download-weights
inkstrip run input.jpg output.jpg
inkstrip serve  # open the Gradio UI in a browser
```

## License

This project is licensed under **AGPL-3.0-or-later**, matching PyMuPDF (used in the `pdf` extra).

The default model weights (`big-lama`) are CC-BY-NC-SA. For commercial use you must retrain or substitute with a permissive inpainting backend.

## Acknowledgements

- LaMa inpainting: https://github.com/advimman/lama (ONNX export from `Carve/LaMa-ONNX`)
- YOLOv8 handwriting detector: https://huggingface.co/armvectores/yolov8n_handwritten_text_detection
- PyMuPDF, pypdfium2, pikepdf, img2pdf, ultralytics, onnxruntime, gradio
