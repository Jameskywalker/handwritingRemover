"""Fine-tuned ResNet18 binary classifier (handwriting vs printed text).

Second-stage classifier complementing the YOLOv8n handwriting detector.
For each OCR bbox crop, returns a handwriting probability — used as a
fallback in ``_ocr_box_is_handwriting`` when the YOLO union vote misses
neat handwriting (e.g. clean answer-cell strokes that look print-like).

Train weights with ``scripts/train_resnet_hw_classifier.py``; default
weight path is ``weights/resnet18_hw_classifier.pt`` (gitignored).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_DEFAULT_WEIGHTS = Path("weights/resnet18_hw_classifier.pt")
_INPUT_SIZE = 96


class ResNetHwClassifier:
    """Lazy wrapper over a ResNet18 fine-tuned binary classifier."""

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            import torchvision.models as M
            import torchvision.transforms as T
        except ImportError as e:
            raise ImportError(
                "torch and torchvision are required for ResNetHwClassifier"
            ) from e

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if not path.is_file():
            raise FileNotFoundError(
                f"ResNet18 HW classifier weights not found at {path}. "
                f"Train via scripts/train_resnet_hw_classifier.py first."
            )

        model = M.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        state = torch.load(path, map_location=self._device, weights_only=True)
        model.load_state_dict(state)
        model.to(self._device).eval()
        self._model = model

        self._transform = T.Compose([
            T.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image: np.ndarray, poly: np.ndarray) -> float:
        """Return handwriting probability (0–1) for the OCR polygon ROI."""
        from PIL import Image as PILImage

        pts = poly.astype(np.int32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts)
        if w <= 0 or h <= 0:
            return 0.0
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            return 0.0

        pil = PILImage.fromarray(crop)
        x_t = self._transform(pil).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(x_t)
            probs = self._torch.softmax(logits, dim=-1)[0]
            return float(probs[1].item())  # class 1 = handwriting
