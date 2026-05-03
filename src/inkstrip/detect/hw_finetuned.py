"""Fine-tuned ResNet binary classifier (handwriting vs printed text).

Second-stage classifier complementing the YOLOv8n handwriting detector.
For each OCR bbox crop, returns a handwriting probability — used as a
fallback in ``_ocr_box_is_handwriting`` when the YOLO union vote misses
neat handwriting (e.g. clean answer-cell strokes that look print-like).

The current production model is **ResNet50 trained on Otsu-binarised
crops**, broadcast to 3 channels so the ImageNet-pretrained conv1 stays
usable. Binarisation removes paper-colour / illumination / pen-colour
nuisance signal; coloured-pen handwriting is handled separately by the
``combine_color`` layer in ``mask.ocr_inverse``. ResNet101's marginal
holdout F0.5 gain (see ``scripts/compare_resnet_size.py``) didn't carry
over to visual quality on real pages, and ResNet152 overfits the small
training set.

Train weights with ``scripts/train_resnet_hw_classifier.py``; default
weight path is ``weights/resnet50_hw_classifier.pt`` (gitignored). The
loader picks the first available checkpoint among resnet101, resnet50,
resnet18 and auto-detects the backbone from fc.weight shape + layer3
block count — drop a ResNet101 .pt back into ``weights/`` to switch.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_WEIGHT_CANDIDATES = (
    Path("weights/resnet101_hw_classifier.pt"),
    Path("weights/resnet50_hw_classifier.pt"),
    Path("weights/resnet18_hw_classifier.pt"),
)
_INPUT_SIZE = 96


def _otsu_binarise_pil(pil_img):
    """RGB PIL → binarised PIL L (handwriting=0, ink=255 after BINARY_INV).

    Otsu picks the threshold per-crop, so paper-colour and illumination
    differences across crops don't matter. Output is single-channel; the
    caller broadcasts to 3 channels before feeding the network.
    """
    from PIL import Image as PILImage
    gray = np.array(pil_img.convert("L"))
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return PILImage.fromarray(b, mode="L")


class ResNetHwClassifier:
    """Lazy wrapper over a ResNet fine-tuned binary classifier.

    Backbone (ResNet18 / ResNet50) is detected from the saved fc.weight
    shape so swapping checkpoints is a no-config operation.
    """

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

        if weights_path is not None:
            path = Path(weights_path)
        else:
            path = next((c for c in _WEIGHT_CANDIDATES if c.is_file()), _WEIGHT_CANDIDATES[0])
        if not path.is_file():
            raise FileNotFoundError(
                f"ResNet HW classifier weights not found at {path}. "
                f"Train via scripts/train_resnet_hw_classifier.py first."
            )

        state = torch.load(path, map_location=self._device, weights_only=True)
        # auto-detect backbone:
        #   ResNet18 fc.weight is (2, 512); ResNet50/101 are both (2, 2048),
        #   so we additionally count layer3 blocks (50→6, 101→23).
        fc_in = state["fc.weight"].shape[1]
        if fc_in == 512:
            model = M.resnet18(weights=None)
        elif fc_in == 2048:
            n_layer3_blocks = max(
                int(k.split(".")[1]) for k in state if k.startswith("layer3.")
            ) + 1
            if n_layer3_blocks >= 23:
                model = M.resnet101(weights=None)
            else:
                model = M.resnet50(weights=None)
        else:
            raise RuntimeError(
                f"unexpected fc.weight shape {state['fc.weight'].shape} in {path}"
            )
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(state)
        model.to(self._device).eval()
        self._model = model

        # Production preprocessing: Otsu binarise then broadcast to 3ch. The
        # network was trained against this exact pipeline; passing raw RGB
        # would silently shift the input distribution and tank accuracy.
        self._transform = T.Compose([
            T.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            T.Lambda(_otsu_binarise_pil),
            T.Lambda(lambda im: im.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def predict(self, image: np.ndarray, poly: np.ndarray) -> float:
        """Return handwriting probability (0–1) for the OCR polygon ROI."""
        return self.predict_batch(image, [poly])[0]

    def predict_batch(
        self, image: np.ndarray, polys: list[np.ndarray]
    ) -> list[float]:
        """Batched HW probabilities for multiple polygons against ``image``.

        Stacks all crops into a single tensor and runs one GPU forward pass.
        Empty / invalid polygons get probability 0.0 in the output, with
        positions preserved so callers can index by original poly index.
        """
        from PIL import Image as PILImage

        out = [0.0] * len(polys)
        if not polys:
            return out

        tensors = []
        keep_idx: list[int] = []
        for i, poly in enumerate(polys):
            pts = poly.astype(np.int32).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(pts)
            if w <= 0 or h <= 0:
                continue
            crop = image[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            tensors.append(self._transform(PILImage.fromarray(crop)))
            keep_idx.append(i)

        if not tensors:
            return out

        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(batch)
            probs = self._torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        for j, idx in enumerate(keep_idx):
            out[idx] = float(probs[j])
        return out
