"""Train the ResNet18 HW vs printed classifier used by ocr_inverse.

Inputs:
  data/hw_classifier_finetune/hw/*.png        — handwriting crops
  data/hw_classifier_finetune/printed/*.png   — printed-text crops

Output:
  weights/resnet18_hw_classifier.pt

Approach:
  - ImageNet-pretrained ResNet18, freeze backbone, train only the binary fc head
  - Inputs resized to 96×96 (text crops vary wildly in aspect ratio; the
    network learns the texture difference, not the absolute shape)
  - Light augmentation (rotation, color jitter, slight Gaussian blur) so
    the small dataset doesn't overfit to JPEG/lighting quirks of one page
  - 30 epochs of AdamW + cosine LR; saves the best-on-val checkpoint

To label new crops, run::

    python scripts/extract_crops_from_image.py SOURCE.jpg \
        --hw-text "童年的我,成年的我,..." \
        --printed-text "纸张很黄,我的保姆,..."
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    data_root = Path("data/hw_classifier_finetune")
    items_pr = [(p, 0) for p in (data_root / "printed").glob("*.png")]
    items_hw = [(p, 1) for p in (data_root / "hw").glob("*.png")]
    if not items_pr or not items_hw:
        raise SystemExit(
            f"No data under {data_root}. Populate hw/ and printed/ first."
        )
    print(f"data: {len(items_pr)} printed, {len(items_hw)} HW")

    random.shuffle(items_pr)
    random.shuffle(items_hw)
    n_pr_val = max(1, len(items_pr) // 4)
    n_hw_val = max(1, len(items_hw) // 4)
    val_items = items_pr[:n_pr_val] + items_hw[:n_hw_val]
    train_items = items_pr[n_pr_val:] + items_hw[n_hw_val:]
    print(f"split: train={len(train_items)}, val={len(val_items)}")

    train_tf = T.Compose([
        T.Resize((96, 96)),
        T.RandomApply([T.RandomRotation(5)], p=0.5),
        T.ColorJitter(brightness=0.25, contrast=0.25),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    train_ldr = DataLoader(CropDataset(train_items, train_tf), batch_size=8, shuffle=True)
    val_ldr = DataLoader(CropDataset(val_items, val_tf), batch_size=8, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = M.resnet18(weights="IMAGENET1K_V1")
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.fc.weight.requires_grad_(True)
    model.fc.bias.requires_grad_(True)
    model = model.to(device)

    opt = torch.optim.AdamW(model.fc.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    loss_fn = nn.CrossEntropyLoss()

    out_path = Path("weights/resnet18_hw_classifier.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    for epoch in range(30):
        model.train()
        tloss = 0.0
        tcount = 0
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
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_ldr:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), out_path)
        print(f"epoch {epoch:2d}: train_loss={tloss/tcount:.3f}  val_acc={acc:.2%}")

    print(f"\nbest val acc: {best_val:.2%}  →  saved {out_path}")


if __name__ == "__main__":
    main()
