"""Compare backbones × freeze strategies for the HW vs printed classifier.

Trains each (backbone, strategy) combo on data/train/{hw,printed}/ with an
internal val split, then evaluates the best-on-val checkpoint against the
strict holdout at data/holdout_hwjpg/{hw,printed}/ (the 32 hw.jpg crops the
model must NEVER see at train time).

Reports threshold-swept F0.5 / accuracy / precision / recall per variant.

Usage::

    .venv/bin/python scripts/compare_backbones.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

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
RESULTS_PATH = OUT_DIR / "results.json"


@dataclass(frozen=True)
class Variant:
    name: str
    backbone: str  # "resnet18" | "resnet50" | "mobilenet_v3_small"
    unfreeze: str  # "fc_only" | "last_block" | "all"


VARIANTS = [
    Variant("resnet18_frozen", "resnet18", "fc_only"),
    Variant("resnet18_unfreeze_layer4", "resnet18", "last_block"),
    Variant("resnet50_frozen", "resnet50", "fc_only"),
    Variant("resnet50_unfreeze_layer4", "resnet50", "last_block"),
    Variant("mbnv3s_frozen", "mobilenet_v3_small", "fc_only"),
    Variant("mbnv3s_unfreeze_features", "mobilenet_v3_small", "last_block"),
]


def _seed_all(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _load_items(root: Path) -> list[tuple[Path, int]]:
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


def _build_model(v: Variant) -> nn.Module:
    if v.backbone == "resnet18":
        m = M.resnet18(weights="IMAGENET1K_V1")
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, 2)
        last_block_params = list(m.layer4.parameters()) + list(m.fc.parameters())
    elif v.backbone == "resnet50":
        m = M.resnet50(weights="IMAGENET1K_V2")
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, 2)
        last_block_params = list(m.layer4.parameters()) + list(m.fc.parameters())
    elif v.backbone == "mobilenet_v3_small":
        m = M.mobilenet_v3_small(weights="IMAGENET1K_V1")
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 2)
        # last conv block + classifier
        last_block_params = (
            list(m.features[-1].parameters()) + list(m.classifier.parameters())
        )
    else:
        raise ValueError(v.backbone)

    for p in m.parameters():
        p.requires_grad = False

    if v.unfreeze == "fc_only":
        if v.backbone == "mobilenet_v3_small":
            for p in m.classifier[-1].parameters():
                p.requires_grad = True
        else:
            for p in m.fc.parameters():
                p.requires_grad = True
    elif v.unfreeze == "last_block":
        for p in last_block_params:
            p.requires_grad = True
    else:
        for p in m.parameters():
            p.requires_grad = True
    return m


def _trainable_params(m: nn.Module) -> list[nn.Parameter]:
    return [p for p in m.parameters() if p.requires_grad]


def _train_one(v: Variant, train_items, val_items, device) -> tuple[nn.Module, dict]:
    train_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomApply([T.RandomRotation(5)], p=0.5),
        T.ColorJitter(brightness=0.25, contrast=0.25),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ldr = DataLoader(CropDataset(train_items, train_tf), batch_size=BATCH, shuffle=True)
    val_ldr = DataLoader(CropDataset(val_items, val_tf), batch_size=BATCH, shuffle=False)

    model = _build_model(v).to(device)
    n_train = sum(p.numel() for p in _trainable_params(model))
    print(f"  trainable params: {n_train:,}")

    lr = 2e-3 if v.unfreeze == "fc_only" else 5e-4
    opt = torch.optim.AdamW(_trainable_params(model), lr=lr, weight_decay=1e-3)
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
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:2d}  loss={tloss/tcount:.3f}  val_acc={acc:.2%}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val acc: {best_val:.2%}")
    return model, {"best_val_acc": best_val}


def _eval_holdout(model, items, device) -> dict:
    val_tf = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ldr = DataLoader(CropDataset(items, val_tf), batch_size=BATCH, shuffle=False)
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in ldr:
            x = x.to(device)
            p = torch.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(y.numpy().tolist())
    probs_arr = np.array(probs)
    labels_arr = np.array(labels)

    # threshold sweep — pick by F0.5 (precision-weighted)
    best = {"thr": 0.5, "f05": -1.0, "acc": 0.0, "prec": 0.0, "rec": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs_arr >= thr).astype(int)
        tp = int(((pred == 1) & (labels_arr == 1)).sum())
        fp = int(((pred == 1) & (labels_arr == 0)).sum())
        fn = int(((pred == 0) & (labels_arr == 1)).sum())
        tn = int(((pred == 0) & (labels_arr == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        beta2 = 0.25  # F0.5 → β²=0.25
        denom = beta2 * prec + rec
        f05 = (1 + beta2) * prec * rec / denom if denom > 0 else 0.0
        acc = (tp + tn) / len(labels_arr)
        if f05 > best["f05"] or (f05 == best["f05"] and rec > best["rec"]):
            best = {"thr": float(thr), "f05": f05, "acc": acc,
                    "prec": prec, "rec": rec,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return best


def main() -> None:
    _seed_all()
    items = _load_items(TRAIN_ROOT)
    if not items:
        raise SystemExit(f"no training data under {TRAIN_ROOT}")
    n_hw = sum(y for _, y in items)
    n_pr = len(items) - n_hw
    print(f"train data: {n_pr} printed + {n_hw} HW = {len(items)} total")

    # stratified 80/20 split
    pr_items = [it for it in items if it[1] == 0]
    hw_items = [it for it in items if it[1] == 1]
    random.shuffle(pr_items)
    random.shuffle(hw_items)
    n_pr_val = max(1, len(pr_items) // 5)
    n_hw_val = max(1, len(hw_items) // 5)
    val_items = pr_items[:n_pr_val] + hw_items[:n_hw_val]
    train_items = pr_items[n_pr_val:] + hw_items[n_hw_val:]
    print(f"split: train={len(train_items)}  val={len(val_items)}")

    holdout_items = _load_items(HOLDOUT_ROOT)
    n_h_hw = sum(y for _, y in holdout_items)
    n_h_pr = len(holdout_items) - n_h_hw
    print(f"holdout: {n_h_pr} printed + {n_h_hw} HW = {len(holdout_items)} total\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    results: list[dict] = []
    for v in VARIANTS:
        print(f"==== {v.name} ====")
        _seed_all()
        model, train_stats = _train_one(v, train_items, val_items, device)
        holdout = _eval_holdout(model, holdout_items, device)
        weights_path = OUT_DIR / f"{v.name}.pt"
        torch.save(model.state_dict(), weights_path)
        rec = {
            "name": v.name,
            "backbone": v.backbone,
            "unfreeze": v.unfreeze,
            "best_val_acc": train_stats["best_val_acc"],
            "holdout": holdout,
            "weights": str(weights_path),
        }
        results.append(rec)
        print(
            f"  HOLDOUT  thr={holdout['thr']:.2f}  acc={holdout['acc']:.2%}  "
            f"F0.5={holdout['f05']:.3f}  prec={holdout['prec']:.2%}  rec={holdout['rec']:.2%}  "
            f"(TP={holdout['tp']} FP={holdout['fp']} FN={holdout['fn']} TN={holdout['tn']})\n"
        )

    results.sort(key=lambda r: (r["holdout"]["f05"], r["holdout"]["rec"]), reverse=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("=" * 60)
    print(" RANKING (by holdout F0.5)")
    print("=" * 60)
    print(f"{'variant':30s} {'thr':>5s} {'acc':>7s} {'F0.5':>6s} {'prec':>7s} {'rec':>7s}")
    for r in results:
        h = r["holdout"]
        print(
            f"{r['name']:30s} {h['thr']:>5.2f} {h['acc']:>7.2%} "
            f"{h['f05']:>6.3f} {h['prec']:>7.2%} {h['rec']:>7.2%}"
        )
    print(f"\nresults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
