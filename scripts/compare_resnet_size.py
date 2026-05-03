"""Compare ResNet50/101/152 on the binarised HW classifier.

All variants use the production preprocessing (Otsu binarise → 3ch broadcast)
and unfreeze layer4 + fc, on data/train. Evaluated on the strict hw.jpg
holdout.

Usage::

    .venv/bin/python scripts/compare_resnet_size.py
"""

from __future__ import annotations

import json
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

INPUT_SIZE = 96
EPOCHS = 30
BATCH = 16
TRAIN_ROOT = Path("data/train")
HOLDOUT_ROOT = Path("data/holdout_hwjpg")
OUT_DIR = Path("weights/_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _seed_all(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _otsu(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return Image.fromarray(b, mode="L")


def _items(root: Path) -> list[tuple[Path, int]]:
    pr = [(p, 0) for p in (root / "printed").glob("*.png")]
    hw = [(p, 1) for p in (root / "hw").glob("*.png")]
    return pr + hw


class CropDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y


def _build(name: str) -> nn.Module:
    if name == "resnet50":
        m = M.resnet50(weights="IMAGENET1K_V2")
    elif name == "resnet101":
        m = M.resnet101(weights="IMAGENET1K_V2")
    elif name == "resnet152":
        m = M.resnet152(weights="IMAGENET1K_V2")
    else:
        raise ValueError(name)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 2)
    for p in m.layer4.parameters():
        p.requires_grad = True
    for p in m.fc.parameters():
        p.requires_grad = True
    return m


def _make_transforms() -> tuple:
    bin_to_3ch = T.Compose([
        T.Lambda(_otsu),
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
    return train_tf, val_tf


def _train_one(name: str, train_items, val_items, device) -> tuple[nn.Module, float]:
    train_tf, val_tf = _make_transforms()
    train_ldr = DataLoader(CropDataset(train_items, train_tf), batch_size=BATCH, shuffle=True)
    val_ldr = DataLoader(CropDataset(val_items, val_tf), batch_size=BATCH, shuffle=False)
    model = _build(name).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_train:,}  /  total: {n_total:,}")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-3
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state: dict | None = None
    for epoch in range(EPOCHS):
        model.train()
        tloss = tcount = 0
        for x, y in train_ldr:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item() * x.size(0); tcount += x.size(0)
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_ldr:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item(); total += y.size(0)
        acc = correct / max(1, total)
        if acc > best_val:
            best_val = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:2d}  loss={tloss/tcount:.3f}  val_acc={acc:.2%}")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val acc: {best_val:.2%}")
    torch.save(best_state, OUT_DIR / f"{name}_bin3.pt")
    return model, best_val


def _eval_holdout(model, items, device) -> dict:
    _, val_tf = _make_transforms()
    ldr = DataLoader(CropDataset(items, val_tf), batch_size=BATCH, shuffle=False)
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in ldr:
            x = x.to(device)
            p = torch.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist()); labels.extend(y.numpy().tolist())
    probs_arr = np.array(probs); labels_arr = np.array(labels)
    best = {"thr": 0.5, "f05": -1.0}
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs_arr >= thr).astype(int)
        tp = int(((pred == 1) & (labels_arr == 1)).sum())
        fp = int(((pred == 1) & (labels_arr == 0)).sum())
        fn = int(((pred == 0) & (labels_arr == 1)).sum())
        tn = int(((pred == 0) & (labels_arr == 0)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        beta2 = 0.25
        denom = beta2 * prec + rec
        f05 = (1 + beta2) * prec * rec / denom if denom > 0 else 0.0
        acc = (tp + tn) / len(labels_arr)
        if f05 > best["f05"]:
            best = {"thr": float(thr), "f05": f05, "acc": acc,
                    "prec": prec, "rec": rec,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return best


def main() -> None:
    _seed_all()
    items = _items(TRAIN_ROOT)
    pr_items = [it for it in items if it[1] == 0]
    hw_items = [it for it in items if it[1] == 1]
    random.shuffle(pr_items); random.shuffle(hw_items)
    n_pr_val = max(1, len(pr_items) // 5)
    n_hw_val = max(1, len(hw_items) // 5)
    val_items = pr_items[:n_pr_val] + hw_items[:n_hw_val]
    train_items = pr_items[n_pr_val:] + hw_items[n_hw_val:]
    holdout_items = _items(HOLDOUT_ROOT)
    print(f"train={len(train_items)}  val={len(val_items)}  holdout={len(holdout_items)}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    results: list[dict] = []
    for name in ("resnet50", "resnet101", "resnet152"):
        print(f"==== {name} ====")
        _seed_all()
        model, val_acc = _train_one(name, train_items, val_items, device)
        h = _eval_holdout(model, holdout_items, device)
        rec = {"name": name, "best_val_acc": val_acc, "holdout": h}
        results.append(rec)
        print(
            f"  HOLDOUT  thr={h['thr']:.2f}  acc={h['acc']:.2%}  F0.5={h['f05']:.3f}  "
            f"prec={h['prec']:.2%}  rec={h['rec']:.2%}  "
            f"(TP={h['tp']} FP={h['fp']} FN={h['fn']} TN={h['tn']})\n"
        )

    results.sort(key=lambda r: (r["holdout"]["f05"], r["holdout"]["rec"]), reverse=True)
    (OUT_DIR / "resnet_size_results.json").write_text(json.dumps(results, indent=2))

    print("=" * 60)
    print(" RANKING (by holdout F0.5)")
    print("=" * 60)
    print(f"{'variant':12s} {'thr':>5s} {'acc':>7s} {'F0.5':>6s} {'prec':>7s} {'rec':>7s}")
    for r in results:
        h = r["holdout"]
        print(
            f"{r['name']:12s} {h['thr']:>5.2f} {h['acc']:>7.2%} "
            f"{h['f05']:>6.3f} {h['prec']:>7.2%} {h['rec']:>7.2%}"
        )


if __name__ == "__main__":
    main()
