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
    CHECKPOINTS, ROOT, build_manifest, stratified_split,
    CharDataset, build_transforms, pick_device,
)
from train_arch import EVAL_BUILDERS, build_for_eval

SOURCES = {"RAS-Compound-character-dataset": "RAS", "ekush-dataset": "Ekush",
           "MatriVasha_Dataset": "MatriVasha"}
RESULTS = ROOT / "results"   # every generated table and matrix lands here


def source_of(path):
    for key, name in SOURCES.items():
        if key in path:
            return name
    return "?"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels, confs, nll = [], [], [], 0.0
    for imgs, y in loader:
        out = model(imgs.to(device))
        p = out.softmax(1).cpu()
        top = p.max(1)
        preds.append(top.indices)
        confs.append(top.values)
        labels.append(y)
        nll += -torch.log(p[range(len(y)), y].clamp_min(1e-12)).sum().item()
    labels = torch.cat(labels)
    return (torch.cat(preds).numpy(), labels.numpy(),
            torch.cat(confs).numpy(), nll / len(labels))


def expected_calibration_error(confs, correct, n_bins=15):
    """ECE: average |accuracy − confidence| across equal-width confidence bins.

    Test loss suggested the hybrid is better calibrated; this measures it directly rather than
    inferring it from cross-entropy, which conflates calibration with accuracy.
    """
    ece, n = 0.0, len(confs)
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - confs[m].mean())
    return ece


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
    parser.add_argument("--suffix", default="", help="extra checkpoint-name suffix, e.g. _imagenet")
    parser.add_argument("--source", choices=["ras", "ekush", "matrivasha"], default=None,
                        help="score single-source runs; must match how they were trained")
    parser.add_argument("--per-class", action="store_true", help="print sklearn classification_report")
    parser.add_argument("--confusion", action="store_true", help="write confusion_{arch}.png")
    parser.add_argument("--out", default="results.csv", help="filename inside results/")
    args = parser.parse_args()

    # Same naming rule as train_arch.py: --limit-per-class 300 scores the capped run,
    # no flag scores the full-data run.
    tag = (f"_{args.source}" if args.source else "")
    tag += (f"_lpc{args.limit_per_class}" if args.limit_per_class else "") + args.suffix
    ckpt_of = lambda a: CHECKPOINTS / f"{a}_merged{tag}.pth"

    archs = args.arch or [a for a in EVAL_BUILDERS if ckpt_of(a).exists()]
    missing = [a for a in archs if not ckpt_of(a).exists()]
    if missing:
        raise SystemExit(f"No checkpoint for: {', '.join(missing)} — expected "
                         f"{', '.join(ckpt_of(a).name for a in missing)}")
    if not archs:
        raise SystemExit(f"No *_merged{tag}.pth checkpoints in checkpoints/ — nothing to evaluate")

    RESULTS.mkdir(exist_ok=True)
    device = pick_device()
    samples, classes = build_manifest()
    if args.source:
        from train_arch import restrict_to_source
        samples, classes = restrict_to_source(samples, classes, args.source)
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
        preds, labels, confs, nll = predict(model, loader, device)
        correct = preds == labels
        acc = correct.mean()

        row = {
            "method": arch,
            "test_acc": round(float(acc), 4),
            "CER": round(float(1 - acc), 4),
            "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
            "NLL": round(float(nll), 4),
            "ECE": round(float(expected_calibration_error(confs, correct)), 4),
            "mean_conf": round(float(confs.mean()), 4),
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

            # saved for the paper's appendix, and to rank confusable pairs
            from sklearn.metrics import precision_recall_fscore_support
            pr, rc, f1, sup = precision_recall_fscore_support(
                labels, preds, labels=range(len(names)), zero_division=0)
            pd.DataFrame({"class": names, "precision": pr.round(4), "recall": rc.round(4),
                          "f1": f1.round(4), "support": sup}).sort_values("support").to_csv(
                RESULTS / f"per_class_{arch}{tag}.csv", index=False)

            cm = confusion_matrix(labels, preds, labels=range(len(names)))
            np.fill_diagonal(cm, 0)
            pairs = [(cm[i, j], names[i], names[j])
                     for i, j in zip(*np.nonzero(cm))]
            pairs.sort(reverse=True)
            print(f"\nMost-confused pairs ({arch}) — true → predicted:")
            for n, t, p_ in pairs[:15]:
                print(f"  {t} → {p_}   {n} images")
            pd.DataFrame(pairs, columns=["count", "true", "predicted"]).to_csv(
                RESULTS / f"confusions_{arch}{tag}.csv", index=False)
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
            plt.savefig(RESULTS / f"confusion_{arch}{tag}.png", dpi=120)
            plt.close()
            print(f"  wrote results/confusion_{arch}{tag}.png")

    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    print("\n" + df.to_string(index=False))
    out = RESULTS / args.out
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
