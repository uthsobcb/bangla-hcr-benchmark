"""Train ExtendedViT (ResNet18 backbone + Transformer encoder) on the same merged
RAS-Compound + Ekush + MatriVasha dataset (256 classes) used by train_merged.py.

Run:
    python train_extended_vit.py                 # full run, defaults below
    python train_extended_vit.py --epochs 8 --resume

Saves extended_vit_merged.pth (same schema as resnet18_merged.pth, plus an "arch" field)
so predict_word.py works unchanged with `--model checkpoints/extended_vit_merged.pth`.
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

from train_merged import (
    IMG_SIZE, BEST_CHECKPOINT as RESNET_CHECKPOINT,
    build_manifest, stratified_split, CharDataset, build_transforms,
    pick_device, run_epoch, build_extended_vit,
)

ROOT = Path(__file__).parent
BEST_CHECKPOINT = ROOT / "checkpoints" / "extended_vit_merged.pth"
LAST_CHECKPOINT = ROOT / "checkpoints" / "extended_vit_merged_last.pth"
ARCH = "extended_vit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit-per-class", type=int, default=None, help="cap samples/class before splitting")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--resume", action="store_true", help="continue from extended_vit_merged_last.pth")
    args = parser.parse_args()

    device = pick_device()
    print("Device:", device)

    print("Scanning datasets...")
    t0 = time.time()
    samples, classes = build_manifest()
    print(f"{len(samples)} images, {len(classes)} classes, scanned in {time.time()-t0:.1f}s")

    train_samples, val_samples, test_samples = stratified_split(samples, limit_per_class=args.limit_per_class)
    print(f"Train {len(train_samples)} | Val {len(val_samples)} | Test {len(test_samples)}")

    train_tf, eval_tf = build_transforms()
    train_loader = DataLoader(CharDataset(train_samples, train_tf), batch_size=args.batch_size,
                               shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = DataLoader(CharDataset(val_samples, eval_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0)
    test_loader = DataLoader(CharDataset(test_samples, eval_tf), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0)

    model = build_extended_vit(len(classes), device, warm_start_from=RESNET_CHECKPOINT)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    backbone_params = (list(model.conv1.parameters()) + list(model.bn1.parameters())
                       + list(model.layer1.parameters()) + list(model.layer2.parameters())
                       + list(model.layer3.parameters()) + list(model.layer4.parameters()))
    new_params = (list(model.transformer.parameters()) + list(model.head.parameters())
                 + [model.cls_token, model.pos_embed] + list(model.norm.parameters()))
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": new_params, "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_acc = 0.0
    if args.resume and LAST_CHECKPOINT.exists():
        ckpt = torch.load(LAST_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_val_acc={best_val_acc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "state_dict": deepcopy(model.state_dict()),
                "class_names": classes,
                "display_names": classes,
                "img_size": IMG_SIZE,
                "arch": ARCH,
            }, BEST_CHECKPOINT)

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
        }, LAST_CHECKPOINT)

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
              f"Time: {time.time()-t0:.1f}s | Best Val: {best_val_acc:.4f}", flush=True)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    best = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, device, criterion)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print(f"Saved {BEST_CHECKPOINT.name}")


if __name__ == "__main__":
    main()
