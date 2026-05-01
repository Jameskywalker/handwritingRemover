"""LaMa inpainting via ONNX runtime.

Why ONNX (not PyTorch):
- simple-lama-inpainting fails to build on Python 3.13 (sdist setup.py bug)
- ONNX wheels exist for every common platform; no torch dep needed at inference
- big-lama under ONNX runs ~1.5–2× the PyTorch latency on CPU and is competitive
  with CUDAExecutionProvider on GPU
- The repo `Carve/LaMa-ONNX` provides a pre-converted graph

Carve/LaMa-ONNX I/O contract (verified at load time):
- input "image": float32 NCHW, channel-first RGB in [0, 1], H/W must be % 8 == 0
- input "mask":  float32 NCHW with 1 channel; 1.0 = repaint, 0.0 = keep
- output: float32 NCHW [0, 1] of the inpainted image, same H/W as input

Inputs are padded with reflect padding to the next multiple of 8; we crop the
output back to the original size before returning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inkstrip.config import InkstripConfig
from inkstrip.inpaint.tile_blend import tile_inpaint
from inkstrip.models.weights import get_weight
from inkstrip.utils.logging import get_logger

_log = get_logger("inpaint.lama_onnx")

_PAD_MULTIPLE = 8


class LamaOnnxInpainter:
    def __init__(
        self,
        cfg: InkstripConfig,
        *,
        weight_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self._weight_path = weight_path
        self._session: Any | None = None
        self._image_input: str = "image"
        self._mask_input: str = "mask"
        self._output: str = "output"

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        weight = self._weight_path or get_weight(
            "lama_onnx",
            cache_dir=self.cfg.cache_dir,
            offline=self.cfg.offline,
        )

        providers = _resolve_providers(self.cfg.device)
        _log.debug("loading LaMa ONNX from %s with providers=%s", weight, providers)
        self._session = ort.InferenceSession(str(weight), providers=providers)

        input_names = [i.name for i in self._session.get_inputs()]
        # Carve/LaMa-ONNX uses "image" and "mask"; some forks differ. Auto-resolve.
        self._image_input = _pick(input_names, ("image", "img", "input"))
        self._mask_input = _pick(input_names, ("mask",))
        self._output = self._session.get_outputs()[0].name

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._load()
        return tile_inpaint(
            image,
            mask,
            self._call_lama,
            tile_size=self.cfg.tile_size,
            overlap=self.cfg.tile_overlap,
        )

    def _call_lama(self, sub_img: np.ndarray, sub_mask: np.ndarray) -> np.ndarray:
        assert self._session is not None
        h, w = sub_img.shape[:2]
        ph = _round_up(h, _PAD_MULTIPLE) - h
        pw = _round_up(w, _PAD_MULTIPLE) - w

        img_padded = np.pad(sub_img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mask_padded = np.pad(sub_mask, ((0, ph), (0, pw)), mode="constant")

        img_tensor = (img_padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        mask_tensor = (mask_padded > 127).astype(np.float32)[None, None, ...]

        outputs = self._session.run(
            [self._output],
            {self._image_input: img_tensor, self._mask_input: mask_tensor},
        )
        out = outputs[0]
        if out.ndim == 4:
            out = out[0]
        out = out.transpose(1, 2, 0)

        out = np.clip(out, 0.0, 1.0)
        if out.max() <= 1.5:  # already normalized [0,1]
            out = (out * 255.0).round().astype(np.uint8)
        else:
            out = np.clip(out, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(out[:h, :w])


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _pick(names: list[str], candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in names:
            return c
    raise RuntimeError(
        f"could not find any of {candidates} in ONNX inputs {names}; "
        "model may not be Carve/LaMa-ONNX compatible"
    )


def _resolve_providers(device: str) -> list[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    # auto / cuda / mps: prefer CUDA, fall back. ONNX picks the first available.
    return ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
