"""Does post-hoc temperature scaling close the calibration gap?

    python temperature_scaling.py --source ekush --limit-per-class 50

Temperature scaling is the standard cheap calibration fix: divide the logits by a single scalar
fitted on validation data. If it brings ResNet18 down to ExtendedViT's calibration error, then the
transformer encoder is sufficient but not necessary for the effect, and the paper must say so.

Fits T on the validation split (never on test) and reports test ECE and NLL before and after.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_merged import (CHECKPOINTS, build_manifest, stratified_split, CharDataset,
                          build_transforms, pick_device)
from train_arch import build_for_eval, EVAL_BUILDERS, restrict_to_source
from evaluate import expected_calibration_error


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):
    """One scalar, fitted by minimising validation NLL (Guo et al., 2017)."""
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def metrics(logits, labels, T=1.0):
    p = (logits / T).softmax(1)
    conf, pred = p.max(1)
    correct = (pred == labels).numpy()
    nll = F.cross_entropy(logits / T, labels).item()
    return {"acc": correct.mean(), "nll": nll,
            "ece": expected_calibration_error(conf.numpy(), correct),
            "mean_conf": conf.mean().item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="ekush", choices=["ras", "ekush", "matrivasha"])
    ap.add_argument("--limit-per-class", type=int, default=50)
    ap.add_argument("--suffix", default="_imagenet")
    ap.add_argument("--arch", nargs="*", default=["resnet18", "extended_vit"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = pick_device()
    samples, classes = build_manifest()
    samples, classes = restrict_to_source(samples, classes, args.source)
    _, val, test = stratified_split(samples, limit_per_class=args.limit_per_class)
    _, eval_tf = build_transforms()
    mk = lambda s: DataLoader(CharDataset(s, eval_tf), batch_size=64, num_workers=args.workers)
    val_loader, test_loader = mk(val), mk(test)
    print(f"{args.source} @ {args.limit_per_class}/class | val {len(val)} test {len(test)} "
          f"| {len(classes)} classes\n")

    tag = f"_{args.source}_lpc{args.limit_per_class}{args.suffix}"
    rows = []
    for arch in args.arch:
        path = CHECKPOINTS / f"{arch}_merged{tag}.pth"
        if not path.exists():
            print(f"skip {arch}: no {path.name}")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_for_eval(ckpt)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        vl, vy = collect_logits(model, val_loader, device)
        tl, ty = collect_logits(model, test_loader, device)
        T = fit_temperature(vl, vy)
        before, after = metrics(tl, ty, 1.0), metrics(tl, ty, T)

        print(f"{arch}  (T = {T:.3f})")
        print(f"  before:  acc {before['acc']:.4f}  ECE {before['ece']:.4f}  "
              f"NLL {before['nll']:.4f}  conf {before['mean_conf']:.4f}")
        print(f"  scaled:  acc {after['acc']:.4f}  ECE {after['ece']:.4f}  "
              f"NLL {after['nll']:.4f}  conf {after['mean_conf']:.4f}\n")
        rows.append((arch, T, before, after))

    if len(rows) == 2:
        (a1, _, b1, s1), (a2, _, b2, s2) = rows
        print(f"Raw:    {a1} {b1['ece']:.4f} vs {a2} {b2['ece']:.4f} "
              f"({b1['ece']/b2['ece']:.2f}x)")
        print(f"Scaled: {a1} {s1['ece']:.4f} vs {a2} {s2['ece']:.4f} "
              f"({s1['ece']/s2['ece']:.2f}x)")
        if s1["ece"] <= b2["ece"]:
            print(f"\nVERDICT: temperature scaling brings {a1} to or below {a2}'s raw calibration.")
            print(f"         The encoder is SUFFICIENT but NOT NECESSARY for the effect.")
        else:
            print(f"\nVERDICT: even scaled, {a1} stays worse than {a2}'s raw ECE "
                  f"({s1['ece']:.4f} vs {b2['ece']:.4f}).")
            print(f"         The calibration advantage is not reducible to a single scalar.")


if __name__ == "__main__":
    main()
