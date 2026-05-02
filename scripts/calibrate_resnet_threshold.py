"""Sweep the ResNet HW classifier threshold over data/hw_classifier_finetune/.

Reports confusion matrix + accuracy + F1 + F0.5 at each threshold step,
and prints the best by each metric. Use F0.5 (precision-weighted) as the
selection criterion — false positives (printed text wrongly erased) hurt
the user more than false negatives (handwriting left behind).

After adding new labelled crops or re-fine-tuning the ResNet, run this
to pick a new ``ocr_resnet_threshold`` for ``InkstripConfig``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image


def main() -> None:
    weights = Path("weights/resnet18_hw_classifier.pt")
    if not weights.is_file():
        raise SystemExit(
            f"{weights} not found — train via scripts/train_resnet_hw_classifier.py first"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = M.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()
    tf = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_root = Path("data/hw_classifier_finetune")
    items = []
    for p in (data_root / "printed").glob("*.png"):
        items.append((p, 0))
    for p in (data_root / "hw").glob("*.png"):
        items.append((p, 1))
    if not items:
        raise SystemExit(f"no labelled crops under {data_root}")

    probs, labels = [], []
    for p, y in items:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            prob = torch.softmax(model(x), dim=-1)[0, 1].item()
        probs.append(prob)
        labels.append(y)
    probs = np.array(probs)
    labels = np.array(labels)
    n_hw = int((labels == 1).sum())
    n_pr = int((labels == 0).sum())
    print(f"data: {len(items)} crops ({n_hw} HW, {n_pr} printed)")
    print(f"\n{'thresh':>6}  {'TP':>2} {'FP':>2} {'TN':>2} {'FN':>2} | "
          f"{'acc':>6} {'prec':>5} {'rec':>5} {'F1':>5} {'F0.5':>5}")

    rows = []
    for t in np.arange(0.05, 1.0, 0.05):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        acc = (tp + tn) / max(1, tp + fp + tn + fn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        f05 = 1.25 * prec * rec / max(1e-9, 0.25 * prec + rec)
        rows.append((float(t), tp, fp, tn, fn, acc, prec, rec, f1, f05))
        print(f"  {t:.2f}  {tp:>2d} {fp:>2d} {tn:>2d} {fn:>2d} | "
              f"{acc:>6.2%} {prec:.2f}  {rec:.2f}  {f1:.2f}  {f05:.2f}")

    best_acc = max(rows, key=lambda r: r[5])
    best_f1 = max(rows, key=lambda r: r[8])
    best_f05 = max(rows, key=lambda r: r[9])
    print(f"\nbest by accuracy: thresh={best_acc[0]:.2f}  acc={best_acc[5]:.2%}")
    print(f"best by F1:       thresh={best_f1[0]:.2f}   F1={best_f1[8]:.3f}")
    print(f"best by F0.5:     thresh={best_f05[0]:.2f}  F0.5={best_f05[9]:.3f}  "
          f"(precision-weighted — recommended)")
    zero_fp = [r for r in rows if r[2] == 0]
    if zero_fp:
        best_zero_fp = min(zero_fp, key=lambda r: r[0])
        print(f"\nlowest threshold with 0 FP: thresh={best_zero_fp[0]:.2f}  "
              f"TP={best_zero_fp[1]} FN={best_zero_fp[4]}  recall={best_zero_fp[7]:.2%}")


if __name__ == "__main__":
    main()
