"""Train any of the 4 methods on the merged RAS-Compound + Ekush + MatriVasha set (256 classes).

    python train_arch.py --arch cnn --epochs 8
    python train_arch.py --arch vit --epochs 8 --limit-per-class 300
    python train_arch.py --arch extended_vit --resume

Data pipeline, split (seed 42) and train loop all come from train_merged.py, so every
arch sees the identical train/val/test samples and the numbers are comparable.
Saves checkpoints/{arch}_merged.pth (+ _last.pth for --resume), same schema as before,
so predict_word.py and evaluate.py work unchanged.
"""
import argparse
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models

from train_merged import (
    IMG_SIZE, CHECKPOINTS, BEST_CHECKPOINT as RESNET_CHECKPOINT, ExtendedViT,
    build_manifest, stratified_split, CharDataset, build_transforms,
    pick_device, run_epoch, build_model, build_extended_vit,
)


class SmallCNN(nn.Module):
    """The from-scratch separable-conv baseline (port of archive-cnn-kaggle.ipynb).

    ponytail: fed the same 3x224x224 tensors as the other archs instead of its original
    1x32x32 — one shared data pipeline, and the comparison stays apples-to-apples. Costs
    a little speed; drop a Resize(64) in front of it if this baseline ever gets slow.
    """

    def __init__(self, num_classes):
        super().__init__()

        def sep(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            sep(16, 32), nn.MaxPool2d(2),
            sep(32, 64), nn.MaxPool2d(2),
            sep(64, 128), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PureViT(nn.Module):
    """ViT-Base/16, ImageNet-pretrained, custom head — the archive-pure-vit-colab method."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        import timm  # only this arch needs it
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit.embed_dim), nn.Dropout(0.3),
            nn.Linear(self.vit.embed_dim, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.vit(x))


def _split_params(model, backbone_attr):
    backbone = list(getattr(model, backbone_attr).parameters())
    ids = {id(p) for p in backbone}
    return backbone, [p for p in model.parameters() if id(p) not in ids]


# build(num_classes, device) -> model ; groups(model, lr) -> AdamW param groups ; default lr
ARCHS = {
    "cnn": dict(
        lr=1e-3,
        build=lambda n, d: SmallCNN(n).to(d),
        groups=lambda m, lr: [{"params": m.parameters(), "lr": lr}],
    ),
    "resnet18": dict(
        lr=3e-4,
        build=lambda n, d: build_model(n, d),
        groups=lambda m, lr: [{"params": list(m.parameters())[:-2], "lr": lr * 0.1},
                              {"params": m.fc.parameters(), "lr": lr}],
    ),
    "vit": dict(
        lr=1e-4,  # full ViT-Base fine-tune diverges at 3e-4
        build=lambda n, d: PureViT(n).to(d),
        groups=lambda m, lr: [{"params": m.vit.parameters(), "lr": lr * 0.1},
                              {"params": m.classifier.parameters(), "lr": lr}],
    ),
    "extended_vit": dict(
        lr=3e-4,
        build=lambda n, d: build_extended_vit(n, d, warm_start_from=RESNET_CHECKPOINT),
        groups=lambda m, lr: [
            {"params": [p for a in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
                        for p in getattr(m, a).parameters()], "lr": lr * 0.1},
            {"params": (list(m.transformer.parameters()) + list(m.head.parameters())
                        + list(m.norm.parameters()) + [m.cls_token, m.pos_embed]), "lr": lr},
        ],
    ),
}


# Bare skeletons for loading a checkpoint — never fetches pretrained weights, since the
# state_dict overwrites them anyway. Used by evaluate.py and predict_word.py.
EVAL_BUILDERS = {
    "cnn": lambda n: SmallCNN(n),
    "resnet18": lambda n: _bare_resnet18(n),
    "vit": lambda n: PureViT(n, pretrained=False),
    "extended_vit": lambda n: ExtendedViT(n, pretrained=False),
}


def _bare_resnet18(n):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m


def build_for_eval(ckpt):
    """Model matching a saved checkpoint. Pre-`arch` checkpoints are plain resnet18."""
    return EVAL_BUILDERS[ckpt.get("arch", "resnet18")](len(ckpt["class_names"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", required=True, choices=list(ARCHS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="overrides the arch default")
    parser.add_argument("--limit-per-class", type=int, default=None, help="cap samples/class before splitting")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing best checkpoint")
    args = parser.parse_args()

    spec = ARCHS[args.arch]
    lr = args.lr if args.lr is not None else spec["lr"]
    # Capped runs get their own filenames so a controlled comparison never overwrites
    # the full-data checkpoints (and the condition is readable off the filename).
    tag = f"_lpc{args.limit_per_class}" if args.limit_per_class else ""
    best_ckpt = CHECKPOINTS / f"{args.arch}_merged{tag}.pth"
    last_ckpt = CHECKPOINTS / f"{args.arch}_merged{tag}_last.pth"
    CHECKPOINTS.mkdir(exist_ok=True)

    # A fresh run restarts best_val_acc at 0, so epoch 1 would clobber a fully trained checkpoint.
    if best_ckpt.exists() and not (args.resume or args.force):
        raise SystemExit(f"{best_ckpt.name} already exists — use --resume to continue it, "
                         f"or --force to retrain from scratch and overwrite it")

    device = pick_device()
    print(f"Arch: {args.arch} | Device: {device} | LR: {lr}")

    print("Scanning datasets...")
    t0 = time.time()
    samples, classes = build_manifest()
    print(f"{len(samples)} images, {len(classes)} classes, scanned in {time.time()-t0:.1f}s")

    train_samples, val_samples, test_samples = stratified_split(samples, limit_per_class=args.limit_per_class)
    print(f"Train {len(train_samples)} | Val {len(val_samples)} | Test {len(test_samples)}")

    train_tf, eval_tf = build_transforms()
    loader = lambda s, tf, sh: DataLoader(CharDataset(s, tf), batch_size=args.batch_size, shuffle=sh,
                                          num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = loader(train_samples, train_tf, True)
    val_loader = loader(val_samples, eval_tf, False)
    test_loader = loader(test_samples, eval_tf, False)

    model = spec["build"](len(classes), device)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(spec["groups"](model, lr), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch, best_val_acc = 1, 0.0
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch, best_val_acc = ckpt["epoch"] + 1, ckpt["best_val_acc"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_val_acc={best_val_acc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"state_dict": deepcopy(model.state_dict()), "class_names": classes,
                        "display_names": classes, "img_size": IMG_SIZE, "arch": args.arch}, best_ckpt)

        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "epoch": epoch,
                    "best_val_acc": best_val_acc}, last_ckpt)

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
              f"Time: {time.time()-t0:.1f}s | Best Val: {best_val_acc:.4f}", flush=True)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, device, criterion)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print(f"Saved {best_ckpt.name}")


if __name__ == "__main__":
    main()
