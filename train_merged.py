"""Train one ResNet18 on RAS-Compound + Ekush + MatriVasha combined (256 unified classes).

Run:
    python train_merged.py                  # full run, defaults below
    python train_merged.py --epochs 8 --batch-size 32 --resume

Saves best_merged_model.pth (same schema as best_bengali_vit.pth: state_dict,
class_names, display_names, img_size) so predict_word.py works unchanged with
`--model best_merged_model.pth`.
"""
import argparse
import csv
import os
import random
import time
import unicodedata
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).parent
IMG_SIZE = 224
SEED = 42
OLD_CHECKPOINT = ROOT / "best_bengali_vit.pth"
BEST_CHECKPOINT = ROOT / "best_merged_model.pth"
LAST_CHECKPOINT = ROOT / "merged_last.pth"


def norm(s):
    return unicodedata.normalize("NFC", s.strip())


def load_mapping(csv_path, key_col, val_col):
    mapping = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = row[val_col].strip()
            if v:
                mapping[row[key_col].strip()] = norm(v)
    return mapping


def build_manifest():
    """Scan all 3 datasets on disk, return [(path, class_idx), ...] and the sorted class list."""
    ras_map = load_mapping(ROOT / "class_mapping.csv", "class_id", "bengali_character")
    ekush_map = load_mapping(ROOT / "ekush-dataset" / "metaData_img.csv", "Folder Name", "Char Name")
    matri_map = load_mapping(ROOT / "matrivasha_mapping_draft.csv", "folder", "character")

    classes = sorted(set(ras_map.values()) | set(ekush_map.values()) | set(matri_map.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = []
    for split in ("train", "test"):
        base = ROOT / "RAS-Compound-character-dataset" / split
        for class_id, char in ras_map.items():
            samples += [(str(p), class_to_idx[char]) for p in (base / class_id).glob("*.png")]

    ekush_base = ROOT / "ekush-dataset"
    for folder, char in ekush_map.items():
        samples += [(str(p), class_to_idx[char]) for p in (ekush_base / folder).glob("*.jpg")]

    for gender_dir in ("male", "feamale"):
        base = ROOT / "MatriVasha_Dataset" / gender_dir
        for folder, char in matri_map.items():
            samples += [(str(p), class_to_idx[char]) for p in (base / folder).glob("*.jpg")]

    return samples, classes


def stratified_split(samples, val_frac=0.1, test_frac=0.1, seed=SEED, limit_per_class=None):
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append(path)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, paths in by_class.items():
        rng.shuffle(paths)
        if limit_per_class:
            paths = paths[:limit_per_class]
        n = len(paths)
        n_test = max(1, int(n * test_frac)) if n > 2 else 0
        n_val = max(1, int(n * val_frac)) if n > 2 else 0
        test += [(p, label) for p in paths[:n_test]]
        val += [(p, label) for p in paths[n_test:n_test + n_val]]
        train += [(p, label) for p in paths[n_test + n_val:]]
    return train, val, test


class PolarityNormalize:
    """Inverts so ink is bright on a dark background, regardless of source convention.

    ponytail: global-mean threshold, not per-blob — good enough for these bimodal
    ink/paper images, would need real binarization for photos with uneven lighting.
    """
    def __call__(self, img):
        arr = np.array(img.convert("L"))
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        if border.mean() > arr.mean():
            arr = 255 - arr
        return Image.fromarray(arr)


class CharDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.transform(img), label


def build_transforms():
    train_tf = transforms.Compose([
        PolarityNormalize(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    eval_tf = transforms.Compose([
        PolarityNormalize(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return train_tf, eval_tf


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes, device):
    model = models.resnet18(weights="IMAGENET1K_V1" if not OLD_CHECKPOINT.exists() else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if OLD_CHECKPOINT.exists():
        old = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
        old_sd, new_sd = old["state_dict"], model.state_dict()
        for k in new_sd:
            if k in old_sd and old_sd[k].shape == new_sd[k].shape:
                new_sd[k] = old_sd[k]
        model.load_state_dict(new_sd)
        print(f"Warm-started backbone from {OLD_CHECKPOINT.name}")

    return model.to(device)


def run_epoch(model, loader, device, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit-per-class", type=int, default=None, help="cap samples/class before splitting")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--resume", action="store_true", help="continue from merged_last.pth")
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

    model = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW([
        {"params": list(model.parameters())[:-2], "lr": args.lr * 0.1},
        {"params": model.fc.parameters(), "lr": args.lr},
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
