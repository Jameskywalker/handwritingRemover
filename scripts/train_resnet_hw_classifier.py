"""Train the ResNet50 HW vs printed classifier used by ocr_inverse.

Inputs:
  data/train/hw/*.png        — handwriting crops
  data/train/printed/*.png   — printed-text crops
  data/holdout_hwjpg/{hw,printed}/*.png — strict hold-out (never used in training)

Output:
  weights/resnet50_hw_classifier.pt

Approach (selected by ablations in scripts/compare_backbones.py and
scripts/compare_binarize.py; see also scripts/compare_resnet_size.py
which showed ResNet101's marginal holdout gain didn't translate to
visual quality on real pages):
  - ImageNet-pretrained ResNet50, unfreeze layer4 + fc (~15M trainable)
  - **Otsu-binarise each crop** then broadcast to 3 channels — strips
    paper colour / illumination / pen colour, gives 100% holdout precision
    (vs 86% for RGB). Coloured handwriting is handled by the colour layer
    in mask.ocr_inverse, not this classifier.
  - Inputs resized to 96x96
  - Geometric augment only (rotation 5°); ColorJitter / GaussianBlur make
    no sense after Otsu so we drop them
  - 30 epochs of AdamW + cosine LR; saves best-on-val checkpoint
  - Final report includes hold-out F0.5 / accuracy / precision / recall

Use scripts/compare_backbones.py and scripts/compare_binarize.py to ablate.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _otsu_binarise(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return Image.fromarray(b, mode="L")

INPUT_SIZE = 96
EPOCHS = 30
BATCH = 16
TRAIN_ROOT = Path("data/train")
HOLDOUT_ROOT = Path("data/holdout_hwjpg")
OUT_PATH = Path("weights/resnet50_hw_classifier.pt")


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    items_pr = [(p, 0) for p in (TRAIN_ROOT / "printed").glob("*.png")]
    items_hw = [(p, 1) for p in (TRAIN_ROOT / "hw").glob("*.png")]
    if not items_pr or not items_hw:
        raise SystemExit(
            f"No data under {TRAIN_ROOT}. Populate hw/ and printed/ first "
            f"(e.g. via scripts/annotate.py)."
        )
    print(f"data: {len(items_pr)} printed, {len(items_hw)} HW")

    random.shuffle(items_pr)
    random.shuffle(items_hw)
    n_pr_val = max(1, len(items_pr) // 5)
    n_hw_val = max(1, len(items_hw) // 5)
    val_items = items_pr[:n_pr_val] + items_hw[:n_hw_val]
    train_items = items_pr[n_pr_val:] + items_hw[n_hw_val:]
    print(f"split: train={len(train_items)}, val={len(val_items)}")

    # Binarise → broadcast to 3-ch via L→RGB. Geometric augment only
    # (ColorJitter / GaussianBlur are no-ops after Otsu).
    bin_to_3ch = T.Compose([
        T.Lambda(_otsu_binarise),
        T.Lambda(lambda im: im.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomApply([T.RandomRotation(5, fill=255)], p=0.5),
        bin_to_3ch,
    ])
    val_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        bin_to_3ch,
    ])

    class CropDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, y = self.items[i]
            return self.transform(Image.open(p).convert("RGB")), y

    train_ldr = DataLoader(CropDataset(train_items, train_tf), batch_size=BATCH, shuffle=True)
    val_ldr = DataLoader(CropDataset(val_items, val_tf), batch_size=BATCH, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = M.resnet50(weights="IMAGENET1K_V2")
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    # unfreeze layer4 + fc (best holdout F0.5 over 6 ablations — see
    # scripts/compare_backbones.py)
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_state: dict | None = None
    for epoch in range(EPOCHS):
        model.train()
        tloss = tcount = 0
        for x, y in train_ldr:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tloss += loss.item() * x.size(0)
            tcount += x.size(0)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_ldr:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        if acc > best_val:
            best_val = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}: train_loss={tloss/tcount:.3f}  val_acc={acc:.2%}")

    if best_state is not None:
        torch.save(best_state, OUT_PATH)
        model.load_state_dict(best_state)
    print(f"\nbest val acc: {best_val:.2%}  -> saved {OUT_PATH}")

    # ---- holdout report ----
    if HOLDOUT_ROOT.is_dir():
        holdout_pr = [(p, 0) for p in (HOLDOUT_ROOT / "printed").glob("*.png")]
        holdout_hw = [(p, 1) for p in (HOLDOUT_ROOT / "hw").glob("*.png")]
        if holdout_pr or holdout_hw:
            ldr = DataLoader(
                CropDataset(holdout_pr + holdout_hw, val_tf), batch_size=BATCH, shuffle=False
            )
            model.eval()
            probs: list[float] = []
            labels: list[int] = []
            with torch.no_grad():
                for x, y in ldr:
                    x = x.to(device)
                    p_ = torch.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
                    probs.extend(p_.tolist())
                    labels.extend(y.numpy().tolist())
            probs_arr = np.array(probs)
            labels_arr = np.array(labels)
            best = {"thr": 0.5, "f05": -1.0}
            for thr in np.linspace(0.05, 0.95, 19):
                pred = (probs_arr >= thr).astype(int)
                tp = int(((pred == 1) & (labels_arr == 1)).sum())
                fp = int(((pred == 1) & (labels_arr == 0)).sum())
                fn = int(((pred == 0) & (labels_arr == 1)).sum())
                tn = int(((pred == 0) & (labels_arr == 0)).sum())
                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                beta2 = 0.25
                denom = beta2 * prec + rec
                f05 = (1 + beta2) * prec * rec / denom if denom > 0 else 0.0
                acc = (tp + tn) / len(labels_arr)
                if f05 > best["f05"]:
                    best = {"thr": float(thr), "f05": f05, "acc": acc,
                            "prec": prec, "rec": rec,
                            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
            print(
                f"\nHOLDOUT  thr={best['thr']:.2f}  acc={best['acc']:.2%}  "
                f"F0.5={best['f05']:.3f}  prec={best['prec']:.2%}  rec={best['rec']:.2%}  "
                f"(TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']})"
            )


if __name__ == "__main__":
    main()
