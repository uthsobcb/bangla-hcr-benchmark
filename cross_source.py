"""Cross-source generalisation: train on one corpus, test on another.

    python cross_source.py --train matrivasha --test ras --arch resnet18

Restricted to the classes both corpora share, so the label space is identical on each side and
the only thing that changes between train and test is the collection convention — resolution,
ink polarity, writing population. This is what merging corpora implicitly promises to buy, and
it is not measured by a within-corpus split.

Backbones are ImageNet-initialised only; inheriting a Bengali checkpoint would leak the target
corpus into the source model. Appends one row per run to cross_source_results.csv.
"""
import argparse
import csv
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from train_merged import (build_manifest, stratified_split, CharDataset, build_transforms,
                          pick_device, run_epoch)
from train_arch import ARCHS, SOURCE_DIRS

ROOT = Path(__file__).parent
RESULTS = ROOT / "cross_source_results.csv"


def split_by_source(samples, classes, train_src, test_src):
    """Samples of each corpus, restricted to the classes they share, relabelled contiguously."""
    tr_key, te_key = SOURCE_DIRS[train_src], SOURCE_DIRS[test_src]
    tr_raw = [(p, l) for p, l in samples if tr_key in p]
    te_raw = [(p, l) for p, l in samples if te_key in p]

    shared = sorted({l for _, l in tr_raw} & {l for _, l in te_raw})
    remap = {old: i for i, old in enumerate(shared)}
    return ([(p, remap[l]) for p, l in tr_raw if l in remap],
            [(p, remap[l]) for p, l in te_raw if l in remap],
            [classes[i] for i in shared])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, choices=list(SOURCE_DIRS))
    p.add_argument("--test", required=True, choices=list(SOURCE_DIRS))
    p.add_argument("--arch", required=True, choices=list(ARCHS))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit-per-class", type=int, default=300)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if args.train == args.test:
        raise SystemExit("--train and --test must differ")

    device = pick_device()
    samples, classes = build_manifest()
    tr_all, te_all, shared = split_by_source(samples, classes, args.train, args.test)
    if len(shared) < 5:
        raise SystemExit(f"only {len(shared)} shared classes — too few to be meaningful")

    # source corpus supplies train+val (its own test slice is unused); target supplies the test set
    tr, va, _ = stratified_split(tr_all, limit_per_class=args.limit_per_class)
    print(f"{args.train} -> {args.test} | {len(shared)} shared classes | "
          f"train {len(tr)} val {len(va)} | target test {len(te_all)}")

    train_tf, eval_tf = build_transforms()
    mk = lambda s, tf, sh: DataLoader(CharDataset(s, tf), batch_size=args.batch_size, shuffle=sh,
                                      num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader, val_loader, test_loader = mk(tr, train_tf, True), mk(va, eval_tf, False), mk(te_all, eval_tf, False)

    spec = ARCHS[args.arch]
    model = spec["build"](len(shared), device, Namespace(no_warm_start=True))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(spec["groups"](model, spec["lr"]), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        if best_state is None or va_acc > best_val:  # always keep epoch 1, even at 0.0
            best_val = va_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"Epoch [{epoch:02d}/{args.epochs}] Train Acc: {tr_acc:.4f} | "
              f"Val Acc: {va_acc:.4f} (same-source) | Time: {time.time()-t0:.1f}s", flush=True)

    model.load_state_dict(best_state)
    te_loss, te_acc = run_epoch(model, test_loader, device, criterion)
    print(f"\nSame-source val : {best_val:.4f}")
    print(f"Cross-source test: {te_acc:.4f}  ({args.train} -> {args.test})")
    print(f"Generalisation gap: {best_val - te_acc:+.4f}")

    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arch", "train_src", "test_src", "shared_classes", "limit_per_class",
                        "same_source_val", "cross_source_test", "gap"])
        w.writerow([args.arch, args.train, args.test, len(shared), args.limit_per_class,
                    round(best_val, 4), round(te_acc, 4), round(best_val - te_acc, 4)])
    print(f"Appended to {RESULTS.name}")


if __name__ == "__main__":
    main()
