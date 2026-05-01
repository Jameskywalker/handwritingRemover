"""Top-level facade: `remove_handwriting(source, output, ...)`.

Library users normally only import this function. The pipeline classes are
public but are an escape hatch — most callers should let `kind="auto"` pick
the right one.
"""

from __future__ import annotations

from typing import Any

from inkstrip.config import InkstripConfig
from inkstrip.routing import detect_input_kind
from inkstrip.types import InputKind, OutputLike, ProgressCallback, RemoveResult


def remove_handwriting(
    source: Any,
    output: OutputLike = None,
    *,
    config: InkstripConfig | None = None,
    kind: InputKind = "auto",
    progress: ProgressCallback | None = None,
    device: str | None = None,
) -> RemoveResult:
    cfg = config or InkstripConfig()
    if device is not None:
        cfg = cfg.merged(device=device)

    resolved = detect_input_kind(source) if kind == "auto" else kind

    if resolved == "image":
        from inkstrip.pipelines.image_pipeline import ImagePipeline

        return ImagePipeline(cfg).run(source, output, cfg, progress=progress)

    if resolved == "scanned_pdf":
        raise NotImplementedError(
            "scanned PDF pipeline lands in M2; pass kind='image' to bypass routing for now"
        )

    if resolved == "digital_pdf":
        raise NotImplementedError("digital PDF annotation pipeline lands in M3")

    if resolved == "hybrid_pdf":
        raise NotImplementedError("hybrid PDF pipeline lands in M3")

    raise ValueError(f"unknown input kind: {resolved!r}")
