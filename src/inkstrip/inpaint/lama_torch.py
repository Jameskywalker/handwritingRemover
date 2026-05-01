"""LaMa inpainting via the `simple-lama-inpainting` package.

simple-lama-inpainting wraps advimman/lama with a one-call API and downloads
the big-lama weights from its own cache on first use. We let it manage the
weights since its scheme is identical to ours (single file, ~200MB) and
re-implementing weight loading would mean vendoring the LaMa graph.

For CC-BY-NC-SA-sensitive deployments use the ONNX backend instead.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from inkstrip.config import InkstripConfig
from inkstrip.inpaint.tile_blend import tile_inpaint
from inkstrip.utils.logging import get_logger

_log = get_logger("inpaint.lama_torch")


class LamaTorchInpainter:
    def __init__(self, cfg: InkstripConfig) -> None:
        self.cfg = cfg
        self._lama = None

    def _load(self) -> None:
        if self._lama is not None:
            return
        from simple_lama_inpainting import SimpleLama

        device = _resolve_device(self.cfg.device)
        _log.debug("loading SimpleLama on device=%s", device)
        # simple-lama-inpainting reads device from env var
        import os
        if device == "cpu":
            os.environ.setdefault("LAMA_DEVICE", "cpu")
        self._lama = SimpleLama()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._load()
        assert self._lama is not None

        return tile_inpaint(
            image,
            mask,
            self._call_lama,
            tile_size=self.cfg.tile_size,
            overlap=self.cfg.tile_overlap,
        )

    def _call_lama(self, sub_img: np.ndarray, sub_mask: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(sub_img, mode="RGB")
        pil_mask = Image.fromarray(sub_mask, mode="L")
        out = self._lama(pil_img, pil_mask)
        if out.mode != "RGB":
            out = out.convert("RGB")
        out_arr = np.asarray(out, dtype=np.uint8)
        if out_arr.shape != sub_img.shape:
            # simple-lama may pad to multiples of 8; crop back.
            h, w = sub_img.shape[:2]
            out_arr = out_arr[:h, :w]
        return np.ascontiguousarray(out_arr)


def _resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
