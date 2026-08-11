"""Score every trained method on the same merged test split and print one comparison table.

    python evaluate.py                          # every checkpoints/*_merged.pth found
    python evaluate.py --arch cnn vit           # just these
    python evaluate.py --per-class --confusion  # + sklearn report and confusion-matrix pngs

Uses train_merged.stratified_split with the same seed, so the test set is exactly the one
held out during training. Writes results.csv. CER == top-1 error rate here (single
characters, one label per image).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_merged import (
    CHECKPOINTS, build_manifest, stratified_split,
    CharDataset, build_transforms, pick_device,
)
from train_arch import EVAL_BUILDERS, build_for_eval

SOURCES = {"RAS-Compound-character-dataset": "RAS", "ekush-dataset": "Ekush",
           "MatriVasha_Dataset": "MatriVasha"}


def source_of(path):
    for key, name in SOURCES.items():
        if key in path:
            return name
    return "?"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, y in loader:
        out = model(imgs.to(device))
        preds.append(out.argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def cpu_latency_ms(model, img_size, iters=20):
    """Single-image CPU latency — predict_word.py always runs inference on CPU."""
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(3):
        model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    return (time.perf_counter() - t0) / iters * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", nargs="*", default=None, choices=list(EVAL_BUILDERS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-class", type=int, default=None, help="must match training")
    parser.add_argument("--per-class", action="store_true", help="print sklearn classification_report")
    parser.add_argument("--confusion", action="store_true", help="write confusion_{arch}.png")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    # Same naming rule as train_arch.py: --limit-per-class 300 scores the capped run,
    # no flag scores the full-data run.
    tag = f"_lpc{args.limit_per_class}" if args.limit_per_class else ""
    ckpt_of = lambda a: CHECKPOINTS / f"{a}_merged{tag}.pth"

    archs = args.arch or [a for a in EVAL_BUILDERS if ckpt_of(a).exists()]
    missing = [a for a in archs if not ckpt_of(a).exists()]
    if missing:
        raise SystemExit(f"No checkpoint for: {', '.join(missing)} — expected "
                         f"{', '.join(ckpt_of(a).name for a in missing)}")
    if not archs:
        raise SystemExit(f"No *_merged{tag}.pth checkpoints in checkpoints/ — nothing to evaluate")

    device = pick_device()
    samples, classes = build_manifest()
    _, _, test_samples = stratified_split(samples, limit_per_class=args.limit_per_class)
    print(f"Test set: {len(test_samples)} images, {len(classes)} classes | device {device}\n")

    _, eval_tf = build_transforms()
    loader = DataLoader(CharDataset(test_samples, eval_tf), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)
    sources = np.array([source_of(p) for p, _ in test_samples])

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    rows = []
    for arch in archs:
        ckpt_path = ckpt_of(arch)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        names = ckpt.get("class_names", classes)
        img_size = ckpt.get("img_size", 224)
        model = build_for_eval(ckpt)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        t0 = time.time()
        preds, labels = predict(model, loader, device)
        acc = (preds == labels).mean()

        row = {
            "method": arch,
            "test_acc": round(float(acc), 4),
            "CER": round(float(1 - acc), 4),
            "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "ckpt_MB": round(ckpt_path.stat().st_size / 1e6, 1),
            "cpu_ms_per_img": round(cpu_latency_ms(model, img_size), 1),
            "eval_s": round(time.time() - t0, 1),
        }
        for name in sorted(set(sources)):
            m = sources == name
            row[f"acc_{name}"] = round(float((preds[m] == labels[m]).mean()), 4) if m.any() else float("nan")
        rows.append(row)
        print(f"{arch:14s} acc {acc:.4f}  CER {1-acc:.4f}  macroF1 {row['macro_f1']:.4f}")

        if args.per_class:
            print(f"\n--- {arch} per-class ---")
            print(classification_report(labels, preds, labels=range(len(names)),
                                        target_names=names, zero_division=0))
        if args.confusion:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(14, 12))
            sns.heatmap(confusion_matrix(labels, preds, labels=range(len(names))), cmap="Blues")
            plt.xlabel("Predicted"), plt.ylabel("True")
            plt.title(f"{arch} — CER {1-acc:.4f}")
            plt.tight_layout()
            plt.savefig(f"confusion_{arch}.png", dpi=120)
            plt.close()
            print(f"  wrote confusion_{arch}.png")

    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    print("\n" + df.to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
