"""LaMa inpainting via ONNX runtime.

Why ONNX (not PyTorch):
- simple-lama-inpainting fails to build on Python 3.13 (sdist setup.py bug)
- ONNX wheels exist for every common platform; no torch dep needed at inference
- big-lama under ONNX runs ~1.5–2× the PyTorch latency on CPU and is competitive
  with CUDAExecutionProvider on GPU
- The repo `Carve/LaMa-ONNX` provides a pre-converted graph

Carve/LaMa-ONNX I/O contract (verified against IOPaint reference impl + the
official Carve demo Space):
- input "image": float32 NCHW, channel-first RGB in [0, 1]
- input "mask":  float32 NCHW with 1 channel; >0 = repaint, 0 = keep
- output: float32 NCHW already in [0, 255] (NOT [0, 1] — do not multiply by 255)

The original Carve export accepts arbitrary H/W as long as both are multiples
of 8. We pad to the next multiple of 8 with reflect padding and crop the
output back. If we hit a fixed-shape variant of the model we resize to its
declared shape and resize back.
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

        inputs = self._session.get_inputs()
        input_names = [i.name for i in inputs]
        self._image_input = _pick(input_names, ("image", "img", "input"))
        self._mask_input = _pick(input_names, ("mask",))
        self._output = self._session.get_outputs()[0].name

        # If the model has a fixed H/W (some Carve exports lock it to 512×512),
        # remember it so we can resize each tile to that shape before inference.
        img_shape = next(i.shape for i in inputs if i.name == self._image_input)
        self._fixed_h: int | None = img_shape[2] if isinstance(img_shape[2], int) else None
        self._fixed_w: int | None = img_shape[3] if isinstance(img_shape[3], int) else None
        if self._fixed_h or self._fixed_w:
            _log.info(
                "model declares fixed input shape: H=%s W=%s; tiles will be resized.",
                self._fixed_h, self._fixed_w,
            )

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
        import cv2

        h, w = sub_img.shape[:2]

        if self._fixed_h and self._fixed_w:
            net_img = cv2.resize(sub_img, (self._fixed_w, self._fixed_h), interpolation=cv2.INTER_AREA)
            net_mask = cv2.resize(sub_mask, (self._fixed_w, self._fixed_h), interpolation=cv2.INTER_NEAREST)
            ph = pw = 0
        else:
            ph = _round_up(h, _PAD_MULTIPLE) - h
            pw = _round_up(w, _PAD_MULTIPLE) - w
            net_img = np.pad(sub_img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            net_mask = np.pad(sub_mask, ((0, ph), (0, pw)), mode="constant")

        img_tensor = (net_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        mask_tensor = ((net_mask > 0).astype(np.float32))[None, None, ...]

        outputs = self._session.run(
            [self._output],
            {self._image_input: img_tensor, self._mask_input: mask_tensor},
        )
        out = outputs[0]
        if out.ndim == 4:
            out = out[0]
        out = out.transpose(1, 2, 0)
        # Carve/LaMa-ONNX outputs values already in [0, 255]. Some forks output
        # [0, 1]; detect by peeking at the max.
        if float(out.max()) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)

        if self._fixed_h and self._fixed_w:
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
            return np.ascontiguousarray(out)
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
