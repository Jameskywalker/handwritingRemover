"""Pinned references to all model weights inkstrip uses.

A single source of truth: every backend resolves weights through this registry, so
swapping a model means editing one entry. Revisions are pinned to commit shas in M4
to make installs reproducible — leaving them as `"main"` is acceptable for M1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    filename: str
    revision: str = "main"
    sha256: str | None = None
    description: str = ""


MODELS: dict[str, ModelSpec] = {
    "yolov8_hw": ModelSpec(
        name="yolov8_hw",
        repo="armvectores/yolov8n_handwritten_text_detection",
        filename="best.pt",
        revision="main",
        description="YOLOv8n trained for handwritten text detection (English-leaning).",
    ),
    "lama_big": ModelSpec(
        name="lama_big",
        repo="smartywu/big-lama",
        filename="big-lama.pt",
        revision="main",
        description="LaMa big model (CC-BY-NC-SA). Used by simple-lama-inpainting.",
    ),
    "lama_onnx": ModelSpec(
        name="lama_onnx",
        repo="Carve/LaMa-ONNX",
        filename="lama_fp32.onnx",
        revision="main",
        description="LaMa exported to ONNX for CPU fallback.",
    ),
}


def get_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"unknown model: {name!r}; available: {sorted(MODELS)}")
    return MODELS[name]
